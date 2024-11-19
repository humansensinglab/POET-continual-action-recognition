import math
import numpy as np
import torch
import torch.nn as nn

'''
POET algorithm (without expansion) uses a fized sized prompt pool in most NTU RGB+R experiments. 

For random prompts w/ freezing, we are following a simpler protocol: we select top k-random prompts based on cosine-sim, 
and use new prompts at the end of sequence (less important positions)
- we do this at both train and test unlike in gestures where we let it select all top_k at test time.
'''

class Prompt(nn.Module):
    def __init__(self, type=0, length=5, embed_dim=128, embedding_key='mean', decouple=True, prompt_init='uniform',
                 prompt_pool=False, random_k=-1, prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False,
                 prompt_key_init='uniform', prev_cmfreq=None, temporal_pe=False, task=0, freeze_prevtask_prompts=False,
                 random_expand=False, force_select_new=False, embed_dim_prompt=128, stack_random_before=False, sort=True):
        super().__init__()

        print(f'\n\n prompt configuration: Prompt type {type}, pool size {pool_size}, length {length}, topk {top_k}, '
              f'temporal pe {temporal_pe}, random_prompts {random_k}, freeze prompts: {freeze_prevtask_prompts}, '
              f'sorting {sort} \n\n')

        self.length = length
        self.embed_dim = embed_dim
        self.embed_dim_prompt = embed_dim_prompt
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.top_k = top_k
        self.required_k = top_k
        self.pool_size = pool_size
        self.batchwise_prompt = batchwise_prompt
        self.prompt_type = type
        self.random_k = random_k
        self.random_prompts = None
        self.cur_task = task
        self.freeze_previous_task_prompts = freeze_prevtask_prompts
        self.random_expand = random_expand
        self.force_select_new = force_select_new
        self.stack_random_before = stack_random_before
        self.sort = sort

        if self.random_k > 0 and self.cur_task > 0:
            self.top_k = self.top_k - self.random_k

        self.decouple_prompting = decouple

        if self.random_expand and self.random_k > 0:
            self.pool_size = pool_size + self.cur_task * self.random_k
            print(f're-init pool size as {pool_size} + {self.cur_task * self.random_k}')

        self.pool_dict = torch.zeros((self.pool_size)).cuda()

        if self.prompt_pool:
            prompt_pool_shape = (self.pool_size, length, self.embed_dim_prompt)
            if prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt)

        # learnable prompts
        key_shape = (self.pool_size, self.embed_dim)
        if prompt_key_init == 'uniform':
            self.prompt_key = nn.Parameter(torch.randn(key_shape))
            nn.init.uniform_(self.prompt_key)

        # revert to 1600 when running cross-attention experiment
        self.multihead_attn = nn.MultiheadAttention(self.embed_dim_prompt, 1, batch_first=True)

        print(f'\n\n checksss inside prompt file pool size {self.prompt.size()}, key size {self.prompt_key.size()}')

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None, train_mode=1):
        out = dict()
        device = x_embed.device
        self.pool_dict = torch.zeros((self.pool_size)).to(device)

        if self.prompt_pool:
            if self.freeze_previous_task_prompts and self.cur_task > 0:
                previous_prompt_tracker = self.pool_size - self.random_k
                prompt = torch.cat((self.prompt[:previous_prompt_tracker, :, :].detach(),
                                  self.prompt[previous_prompt_tracker:, :, :]), dim=0)
                prompt_key = self.prompt_key
            else:
                prompt = self.prompt
                prompt_key = self.prompt_key

            if self.embedding_key == 'query_cls':
                x_embed_mean = cls_features  # torch.Size([32, 128]) for shrec 

            prompt_norm = self.l2_normalize(prompt_key, dim=1)  # torch.Size([10, 128]) for shrec
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)  # torch.Size([32, 128]) for shrec

            # torch.Size([BS, length of prompt])
            similarity = torch.matmul(x_embed_norm, prompt_norm.t())
            B, _ = similarity.size()

            if self.decouple_prompting:
                # L2P style decoupled prompting
                _, idx = torch.topk(similarity, k=self.top_k, dim=1)
                prompt_id, id_counts = torch.unique(idx, return_counts=True)
                batched_prompt_raw = prompt[idx]  # B, top_k, length, C

            elif not self.decouple_prompting:
                Batch_size = similarity.size(0)
                use_similarity = similarity
                selected_keys = torch.zeros(similarity.size()).to(device)

                if self.force_select_new and train_mode == 1 and self.cur_task > 0 and self.random_k > 0:
                    new_idx = torch.arange(self.pool_size)[-self.random_k:].to(device)
                    new_idx = torch.broadcast_to(new_idx, (B, self.random_k))
                    _, idx = torch.topk(use_similarity[:, :-self.random_k], k=self.top_k, dim=1, sorted=True)
                    idx = torch.cat((idx, new_idx), 1)
                else:
                    if self.sort:
                        _, idx = torch.topk(use_similarity, k=self.required_k, dim=1, sorted=True)
                    else:
                        _, idx = torch.topk(use_similarity, k=self.required_k, dim=1, sorted=False)

                prompt_id, id_counts = torch.unique(idx, return_counts=True)
                self.pool_dict[prompt_id] += id_counts.to(device)

                for b in range(Batch_size):
                    selected_keys[b, idx[b]] = 1.0

                selected_keys = (selected_keys-use_similarity).detach() + use_similarity
                batched_prompt_ = selected_keys.unsqueeze(2).unsqueeze(3) * prompt.unsqueeze(0) # torch.Size([32, 16, 22, 128]) for SHREC
                batched_prompt_raw = torch.zeros((Batch_size, self.required_k, self.length, self.embed_dim_prompt)).to(device)

                for b in range(Batch_size):
                    batched_prompt_raw[b, :, :, :] = batched_prompt_[b, idx[b], :, :]

            batch_size, top_k, length, c = batched_prompt_raw.shape

            if self.prompt_type in [7, 17, 100, 200, 300, 27]:
                # batched_prompt_raw (N*M, T, V, Base_channels)
                # L1 output size:(N*M, Base_channels, T, V)
                batched_prompt = batched_prompt_raw.permute(0, 3, 1, 2)

            elif self.prompt_type == 24:
                # mean along top k as length already is req temporal sequence.
                batched_prompt = torch.mean(batched_prompt_raw, 1)

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx]  # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
            sim = batched_key_norm * x_embed_norm  # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0]  # Scalar
            out['reduce_sim'] = reduce_sim

        else:
            '''
            Attach prompt of final shape w/o any selection 
            '''
            batched_prompt = self.prompt.unsqueeze(0)  # for batch dimension
            batched_prompt = batched_prompt.permute(0, 3, 1, 2)  # to match x_embed shape (Batch, Channel_dim, T, V)
            
            out['prompt_idx'] = None
            out['prompt_norm'] = None
            out['x_embed_norm'] = None
            out['similarity'] = None
            out['selected_key'] = None
            out['reduce_sim'] = None

        out['total_prompt_len'] = batched_prompt.shape[1]

        if self.prompt_type in [7, 24]:
            # Proposed solution - simply ADD
            out['prompted_embedding'] = batched_prompt + x_embed

        elif self.prompt_type == 27:
            # single prompt frame, broadcast along time dim
            batched_prompt = torch.broadcast_to(batched_prompt, x_embed.size())
            out['prompted_embedding'] = batched_prompt + x_embed

        elif self.prompt_type == 100:
            # concat along feature dimension (base channel)
            out['prompted_embedding'] = torch.cat((x_embed, batched_prompt), dim=1)

        elif self.prompt_type == 200:
            # concat along time dimension
            out['prompted_embedding'] = torch.cat((x_embed, batched_prompt), dim=2)

        elif self.prompt_type == 300:
            '''
            Apply Cross-attention between prompt & input embed
            attn_output, attn_output_weights = multihead_attn(query, key, value)

            input: (N*M, BC, T, 25) : torch.Size([128, 64, 64, 25])
            required: (N*M, T*25, BC), T*25 is the sequence length
            '''
            B, BC, T, V = batched_prompt.size()
            batched_prompt = batched_prompt.view(B, BC, -1).permute(0, 2, 1)
            x_embed = x_embed.view(B, BC, -1).permute(0, 2, 1)

            attn_output, _ = self.multihead_attn(x_embed, batched_prompt, batched_prompt)
            out['prompted_embedding'] = attn_output.permute(0, 2, 1).view(B, BC, T, V)
        
        elif self.prompt_type == 350:
            '''
            Use cross attention operator instead of addition - POET's selection 
            '''
            batch_size, BC, T, V = x_embed.size()
            batched_prompt = torch.broadcast_to(batched_prompt, (batch_size, BC, T, V))
            
            batched_prompt = batched_prompt.permute(0, 2, 3, 1).reshape(batch_size, T, -1)
            x_embed = x_embed.permute(0, 2, 3, 1).reshape(batch_size, T, -1)  # x_embed is the query; prompts are the key/value

            attn_output, _ = self.multihead_attn(x_embed, batched_prompt, batched_prompt)
            out['prompted_embedding'] = attn_output.view(batch_size, T, BC, V).permute(0, 2, 1, 3)
            
        elif self.prompt_type == 400:
            '''
            Use cross attention operator instead of addition - no selection and addition. cross attention without selection. 
            attention to be applied for temporal frame permutation and ordering 
            input: (N*M, BC, T, 25) : torch.Size([128, 64, 64, 25])
            required: (N*M, T*25, BC), T*25 is the sequence length
            '''
            batch_size, BC, T, V = x_embed.size()
            batched_prompt = torch.broadcast_to(batched_prompt, (batch_size, BC, T, V))
            
            batched_prompt = batched_prompt.permute(0, 2, 3, 1).reshape(batch_size, T, -1)
            x_embed = x_embed.permute(0, 2, 3, 1).reshape(batch_size, T, -1)  # x_embed is the query; prompts are the key/value

            attn_output, _ = self.multihead_attn(x_embed, batched_prompt, batched_prompt)
            out['prompted_embedding'] = attn_output.view(batch_size, T, BC, V).permute(0, 2, 1, 3)
            
        out['selected_prompts_dict'] = self.pool_dict

        return out
