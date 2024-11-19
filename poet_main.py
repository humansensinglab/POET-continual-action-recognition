#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import csv
import numpy as np
import glob
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from argparser_continual import get_parser
from custom_utils import *

# Set resource limits
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

def init_seed(seed):
    """Initialize random seeds for reproducibility"""
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(import_str):
    """Dynamically import a class from a string"""
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                         (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    """Convert string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

class Processor():
    """Processor for Skeleton-based Action Recognition"""

    def __init__(self, cfg, args):
        self.args = args
        self.IL_step = args.IL_step
        self.k_shot = args.k_shot
        self.eval_label_ids = np.arange(args.maxlabelid_curr_step)
        self.num_class = args.maxlabelid_curr_step
        self.train_label_ids = np.arange(args.labels_prev_step, self.num_class)

        # Continual learning protocol parameters
        self.batch_size = 25 if args.IL_step > 0 else 64
        self.test_batch_size = 64
        self.num_epoch = 5 if args.IL_step > 0 else 50

        print(f'Batch size: {self.batch_size}\nNumber of epochs: {self.num_epoch}')
        print(f'Number of classes in classifier head: {self.num_class}')
        print(f'Train label IDs: {self.train_label_ids}')
        print(f'Eval label IDs: {self.eval_label_ids}')

        self.cfg = cfg
        self.save_cfg()
        
        if args.phase == 'train':
            if not cfg.train_feeder_args['debug']:
                save_name = os.path.join(cfg.work_dir, args.save_name_args)
                if os.path.isdir(save_name):
                    shutil.rmtree(save_name)
                    print('Directory removed:', save_name)

                self.train_writer = SummaryWriter(os.path.join(save_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(save_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(save_name, 'test'), 'test')
            self.model_saved_name = save_name
            
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()

        self.lr = self.cfg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0

        self.model = self.model.cuda(self.output_device)
        self.model_query = self.model_query.cuda(self.output_device)

        # Prompt tracking statistics
        self.prev_cumulative_prompt_freq = torch.zeros(self.cfg.prompt.pool_size)
        self.prompt_order_freq = torch.zeros(self.cfg.prompt.pool_size * 10)

        # Input data statistics
        self.input_skeleton_min = []
        self.input_skeleton_max = []
        self.input_skeleton_mean = []

        if type(self.args.device) is list:
            if len(self.args.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.args.device,
                    output_device=self.output_device)
                self.model_query = nn.DataParallel(
                    self.model_query,
                    device_ids=self.args.device,
                    output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.cfg.feeder)
        self.data_loader = dict()
        if self.args.phase == 'train':
            print(f'\n\nLoading all training data')
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.cfg.train_feeder_args,
                               few_shot_data_file_run=self.args.few_shot_data_file,
                               IL_step=self.args.IL_step,
                               k_shot=self.args.k_shot,
                               labels_prev_step=self.args.labels_prev_step,
                               train_label_ids=self.train_label_ids,
                               eval_label_ids=self.eval_label_ids,
                               label_order_protocol=self.args.class_order),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        print(f'\n\nLoading all test data, base+inc')
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.cfg.test_feeder_args,
                           few_shot_data_file_run=self.args.few_shot_data_file,
                           IL_step=self.args.IL_step,
                           k_shot=self.args.k_shot,
                           labels_prev_step=self.args.labels_prev_step,
                           train_label_ids=self.train_label_ids,
                           eval_label_ids=self.eval_label_ids,
                           label_order_protocol=self.args.class_order),
            batch_size=self.cfg.test_batch_size,
            shuffle=False,
            num_workers=self.cfg.num_worker,
            drop_last=True,
            worker_init_fn=init_seed)

        if self.IL_step > 0:
            # load previous and new eval loader
            print(f'\n\nLoading only old class loader')
            self.data_loader['test_prev'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.cfg.test_feeder_args,
                               few_shot_data_file_run=self.args.few_shot_data_file,
                               IL_step=self.args.IL_step,
                               k_shot=self.args.k_shot,
                               labels_prev_step=self.args.labels_prev_step,
                               train_label_ids=self.train_label_ids,
                               eval_label_ids=np.arange(self.args.labels_prev_step),
                               label_order_protocol=self.args.class_order),
                batch_size=self.cfg.test_batch_size,
                shuffle=False,
                num_workers=self.cfg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)

            print(f'\n\nLoading only new class loader')
            self.data_loader['test_new'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.cfg.test_feeder_args,
                               few_shot_data_file_run=self.args.few_shot_data_file,
                               IL_step=self.args.IL_step,
                               k_shot=self.args.k_shot,
                               labels_prev_step=self.args.labels_prev_step,
                               train_label_ids=self.train_label_ids,
                               eval_label_ids=self.train_label_ids,
                               label_order_protocol=self.args.class_order),
                batch_size=self.cfg.test_batch_size,
                shuffle=False,
                num_workers=self.cfg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
            
        if self.args.task_specific_eval:
            def create_task_loader(self, class_range):
                """Helper function to create a task-specific data loader"""
                return torch.utils.data.DataLoader(
                    dataset=Feeder(**self.cfg.test_feeder_args,
                                   few_shot_data_file_run=self.args.few_shot_data_file,
                                   IL_step=self.args.IL_step,
                                   k_shot=self.args.k_shot,
                                   labels_prev_step=self.args.labels_prev_step,
                                   train_label_ids=self.train_label_ids,
                                   eval_label_ids=np.array(class_range),
                                   label_order_protocol=self.args.class_order),
                    batch_size=self.cfg.test_batch_size,
                    shuffle=False,
                    num_workers=self.cfg.num_worker,
                    drop_last=True,
                    worker_init_fn=init_seed
                )

            # Base task loader
            self.data_loader['Base_Task'] = create_task_loader(self, range(40))

            # Task-specific loaders
            task_ranges = {
                'Task1': range(40, 45),
                'Task2': range(45, 50),
                'Task3': range(50, 55),
                'Task4': range(55, 60)
            }
            
            for task_name, class_range in task_ranges.items():
                self.data_loader[task_name] = create_task_loader(self, class_range)

    def load_train_checkpoint(self, saved_model):
        # loads checkpoint without classifier
        dict = {}
        for k, v in saved_model.items():
            if 'weight' != k:
                dict[k] = v

        self.model.load_state_dict(dict, strict=False)

    def load_continual_checkpoint(self, saved_model):
        '''
        This is only for loading cosine classifier model, for regular classifier modify classifier part or ask authors for file. 
        '''
        dict = {}
        for k, v in saved_model.items():
            if k == 'weight':
                continue
            elif self.cfg.prompt.random_expand and 'prompt' in k:
                continue
            else:
                dict[k] = v

        self.model.load_state_dict(dict, strict=False)

        print(f'model.weight size {self.model.weight.size()}')  # 45, 256

        # load parts of classifier
        assert saved_model['weight'].size(
        ) == self.model.weight[0:self.args.labels_prev_step].size()

        with torch.no_grad():
            print(
                f'previous class weights before loading {self.model.weight[0:self.args.labels_prev_step, :].mean()}')
            self.model.weight[0:self.args.labels_prev_step, :].copy_(
                saved_model['weight'])
            print(
                f'previous class weights after loading {self.model.weight[0:self.args.labels_prev_step, :].mean()}')
            #
            self.model.weight[self.args.labels_prev_step:,
                              :] = saved_model['weight'].mean(0)

            if self.cfg.prompt.random_expand > 0:
                print(f'\nHello, this deals with prompt and key init')
                prev_pool_size = self.cfg.prompt.pool_size + \
                    (args.randomized * (self.args.IL_step-1))

                assert saved_model['prompt.prompt'].size(
                ) == self.model.prompt.prompt[0:prev_pool_size, :, :].size()

                print(
                    f'\nold pool weights before loading {self.model.prompt.prompt[0:prev_pool_size, :, :].mean()}')

                self.model.prompt.prompt[0:prev_pool_size, :, :].copy_(
                    saved_model['prompt.prompt'])
                self.model.prompt.prompt_key[0:prev_pool_size, :].copy_(
                    saved_model['prompt.prompt_key'])

                print(
                    f'old pool weights after loading {self.model.prompt.prompt[0:prev_pool_size, :, :].mean()}')

                # trying to init new prompt keys as average of all previous prompt keys - in the hope their cosine sim doesn't suffer.
                if self.args.expand_allpool:
                    self.model.prompt.prompt_key[prev_pool_size:,
                                                 :] = saved_model['prompt.prompt_key'][:self.cfg.prompt.pool_size]
                    self.model.prompt.prompt[prev_pool_size:,
                                             :] = saved_model['prompt.prompt'][:self.cfg.prompt.pool_size]

                else:
                    print(f'\n\n REINITIIALIZING NEW PROMPT POOL AND KEY PARAMS AS AVERAGE OF EXISTING ONES')
                    print('reinit key')
                    self.model.prompt.prompt_key[prev_pool_size:,
                                                 :] = saved_model['prompt.prompt_key'].mean(0)
                    print('reinit prompt')
                    self.model.prompt.prompt[prev_pool_size:,
                                             :] = saved_model['prompt.prompt'].mean(0)

    def load_model(self):
        output_device = self.args.device[0] if type(self.args.device) is list else self.args.device
        self.output_device = output_device
        Model = import_class(self.cfg.model)
        Model_query = import_class(self.cfg.model_query)
        shutil.copy2(inspect.getfile(Model), self.cfg.work_dir)

        self.model = Model(num_class=self.num_class, **self.cfg.model_args,
                           IL_step=self.args.IL_step, do_prompt=True, cfg_prompt=self.cfg.prompt, randomized=self.args.randomized, prompt_layer=self.args.prompting_layer) # classifier_type=self.args.classifier_type if required
        # prompting other layers
        # self.model = Model(num_class=self.num_class, **self.cfg.model_args,
        #                    IL_step=self.args.IL_step, do_prompt=True, cfg_prompt=self.cfg.prompt, randomized=self.args.randomized, Base_channels_prompt=self.args.Base_channels_prompt, layer_to_prompt=self.args.prompting_layer)
        self.model_query = Model_query(num_class=self.num_class, **
                                       self.cfg.model_args)  

        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        # query_checkpoint - must be provided!
        if not self.args.query_checkpoint:
            self.print_log('Provide query checkpoint, using a random init one!!!!!!!!')
            exit()
        self.model_query.load_state_dict(torch.load(self.args.query_checkpoint), strict=False)

        if self.args.weights:
            self.print_log('Load weights from {}.'.format(self.args.weights))

            weights = torch.load(self.args.weights)

            if self.args.IL_step == 0:
                if self.args.load_ckpt_fc:
                    # load checkpoint non strict
                    self.print_log('loading main model vanilla way')
                    self.model.load_state_dict(weights, strict=False)
                else:
                    # load checkpoint strict
                    if self.args.Base_channels_prompt == 32:
                        # prompt along feature dim
                        dict = {}
                        for k, v in weights.items():
                            if 'l1' in k:
                                continue
                            else:
                                dict[k] = v
                        self.model.load_state_dict(dict, strict=False)
                    else:
                        self.load_train_checkpoint(weights)

            # for joint
            elif self.args.phase == 'test':
                self.print_log('loading test main model vanilla way')
                self.model.load_state_dict(weights)

            elif self.args.IL_step > 0 and self.args.phase == 'train':
                self.load_continual_checkpoint(weights)

    def load_optimizer(self):
        # update query fc adaptor based on cfg fc_adaptor
        self.print_log(f'model_args.query {self.cfg.model_args.query}')

        params = list(self.model.named_parameters())
        params_query = list(self.model_query.named_parameters())

        print(
            f'\n\n using self.cfg.model_args.query.query_lr {self.cfg.model_args.query.query_lr},self.cfg.base_lr {self.cfg.base_lr}\n\n')

        for k, v in self.model_query.named_parameters():
            if 'query_adaptor' not in k:
                v.requires_grad = False
            else:
                v.requires_grad = True

        grouped_parameters = [
            {"params": [p for n, p in params_query], 'lr': self.cfg.model_args.query.query_lr},
            {"params": [p for n, p in params]},
        ]

        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                grouped_parameters,
                lr=self.cfg.base_lr,
                momentum=0.9,
                nesterov=self.cfg.nesterov,
                weight_decay=self.cfg.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                grouped_parameters,
                lr=self.cfg.base_lr,
                weight_decay=self.cfg.weight_decay)
        else:
            raise ValueError()

        # conditionally freeze main model parts - don't freeze in base
        if self.args.IL_step > 0:
            for k, v in self.model.named_parameters():
                # add 'sigma' here if training eta
                train_param_list = ['prompt.prompt', 'weight', 'prompt.prompt_key']
                if self.args.eta_cosine:
                    train_param_list += 'sigma'

                if k in train_param_list:
                    v.requires_grad = True
                else:
                    v.requires_grad = False

        self.print_log('using warm up, epoch: {}'.format(self.cfg.warm_up_epoch))

    def print_req_grad_status(self, model):
        for k, v in model.named_parameters():
            self.print_log(f'{k}, {v.requires_grad}')

    def save_cfg(self):
        # save cfg
        cfg_dict = vars(self.cfg)
        if not os.path.exists(self.cfg.work_dir):
            os.makedirs(self.cfg.work_dir)
        with open('{}/config.yaml'.format(self.cfg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(cfg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.args.optimizer == 'SGD' or self.args.optimizer == 'Adam':
            if epoch < self.cfg.warm_up_epoch:
                lr = self.cfg.base_lr * (epoch + 1) / self.cfg.warm_up_epoch
            else:
                lr = self.cfg.base_lr * (
                    self.cfg.lr_decay_rate ** np.sum(epoch >= np.array(self.cfg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.cfg.print_log:
            with open('{}/log.txt'.format(self.cfg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def update_prompt_frequency_stats(self, selected_batch_idx):
        selected_batch_idx = selected_batch_idx.detach().cpu()
        for b in range(selected_batch_idx.size(0)):
            self.prev_cumulative_prompt_freq[selected_batch_idx[b]] += 1.0

    def save_prompt_hist(self, name='train'):
        hist_original = self.prev_cumulative_prompt_freq
        plt.figure(figsize=(8, 4))
        bins = np.arange(self.cfg.prompt.pool_size)
        plt.bar(bins, hist_original, align='center', width=0.2)
        plt.xticks(bins)
        plt.savefig('prompt_analysis_cvpr/' + name + self.args.save_name_args + '.png')

    def update_prompt_ordering_stats(self, selected_batch_idx):
        selected_batch_idx = selected_batch_idx.detach().cpu()
        selected_batch_idx = selected_batch_idx[:, :10]  # plot only 1st 6 in sequence

        batches,  num_prompts = selected_batch_idx.size()

        for b in range(batches):
            for i in range(num_prompts):
                val = selected_batch_idx[b, i]
                self.prompt_order_freq[(self.cfg.prompt.pool_size * i) + val] += 1.0

    def prompt_reinit(self):
        # reinit the prompt histogram, for training stats
        self.prev_cumulative_prompt_freq = torch.zeros(self.cfg.prompt.pool_size)
        # this is recording which prompt indices are being selected as 1st 10, cossim
        self.prompt_order_freq = torch.zeros(self.cfg.prompt.pool_size * 10)

    def save_prompt_ordering_hist(self, name='train'):
        result_dict = {}
        result_dict['experiment'] = self.args.save_name_args
        result_dict['datafile'] = self.args.few_shot_data_file
        result_dict['random_k'] = args.randomized
        result_dict['step'] = self.args.IL_step
        result_dict['hist'] = self.prompt_order_freq.numpy()
        result_dict['pool_size'] = self.cfg.prompt.pool_size

        np.save(self.args.save_name_args+'.npy', self.prompt_order_freq.numpy())
        save_path = 'order_stats.csv'
        print(self.prompt_order_freq.numpy())

        with open(save_path, 'a', newline='') as csvfile:
            file_writer = csv.DictWriter(
                csvfile, fieldnames=list(result_dict.keys()))
            file_writer.writerow(result_dict)
        # exit()

        hist = self.prompt_order_freq
        plt.figure(figsize=(16, 3))
        x_bins = []
        x_all = []
        x_actual = []
        y_values = []
        bar_tick = []
        x_temp = []
        color = []
        max_counter = -1
        for ind, freq in enumerate(hist):
            x_all.append(ind % self.cfg.prompt.pool_size)
            if ind % 72 == 0:
                if ind > 0:
                    print(pos_max_ind % self.cfg.prompt.pool_size, max_counter, pos_max_ind)
                    max_counter = -1
                x_bins.append(ind % self.cfg.prompt.pool_size)
                bar_tick.append(str(ind % self.cfg.prompt.pool_size))
                x_actual.append(ind)
                x_temp.append(ind)
                y_values.append(freq)
                color.append('red')
            elif (name == 'train') or (name == 'eval_all' and freq > 300):
                x_bins.append(ind % self.cfg.prompt.pool_size)
                y_values.append(freq)
                x_actual.append(ind)
                x_temp.append(ind)
                bar_tick.append(str(ind % self.cfg.prompt.pool_size))
                color.append('blue')

            # finding max
            if freq > max_counter:
                pos_max_ind = ind
                max_counter = freq

        x_bins = np.array(x_bins)
        x_all = np.array(x_all)
        x_actual = np.array(x_actual)
        x_temp = np.array(x_temp)
        y_values = np.array(y_values)
        print(f'selected indices: {x_bins}, their frequencies: {y_values}')
        print(f'selected indices: {x_all}, their frequencies: {hist}')

        plt.bar(x_actual, y_values, align='center', width=0.7, tick_label=bar_tick, color=color)
        plt.ylim(0, 50)
        plt.margins(x=0, y=0)
        plt.xticks(x_temp, fontsize=8)  # controls which indices are shown in bar_tick

        plt.savefig('prompt_ordering_cvpr/' + name + 'order' +
                    self.args.save_name_args + '.png', bbox_inches='tight')

    def plot_logits(self, logits, epoch):
        print('plotting logits\n')
        plt.figure(figsize=(8, 4))
        sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})
        g = sns.barplot(y=logits, x=np.arange(self.num_class), hue=self.class_to_task, dodge=False)
        g.set_xticks(np.arange(0, self.num_class, 5))
        plt.xlabel('Class Index', fontsize=14)
        plt.ylabel('Logit Values', fontsize=14)
        sns.move_legend(g, "upper left", fontsize=12)

        plt.savefig('logits/logits_' + self.args.save_name_args +
                    str(epoch) + '.png', bbox_inches="tight")

    def plot_prompt_gradients(self, gradi, ordered_indices, epoch):
        print('plotting prompt gradients\n')
        plt.figure(figsize=(8, 4))
        sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})

        gradi_ordered = gradi[ordered_indices].detach().cpu().numpy()
        ordered_indices = ordered_indices.detach().cpu().numpy()
        ordered_indices_str = ordered_indices.astype(str)

        print(ordered_indices)

        df = pd.DataFrame({'indices': ordered_indices_str,
                           'prompt_gradients': gradi_ordered})

        g = sns.barplot(data=df, x='indices', y='prompt_gradients')

        g.set_xticks(ordered_indices[::2])
        plt.xlabel('Ordered Prompt Selection Indices', fontsize=14)
        plt.ylabel('Gradients', fontsize=14)

        plt.savefig('logits/prompt_grad_' + self.args.save_name_args +
                    str(epoch) + '.png', bbox_inches="tight")

    def train(self, epoch, save_model=False):
        # freeze query, set optimizer and saving and loading mechanism
        # update train toget query then get prompted output then apply extra loss?
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            cls_feat = self.model_query(data)
            output_dict, reduce_sim, selected_idx = self.model(data, cls_feat)
            if self.args.eta_cosine:
                print('using eta cosine')
                output = output_dict['wsigma']
            else:
                output = output_dict['wosigma']
            loss = self.loss(output, label)

            # self.update_prompt_frequency_stats(selected_idx)
            # self.update_prompt_ordering_stats(selected_idx)

            if self.args.prompt_sim_reg and reduce_sim is not None:
                prompt_loss = reduce_sim.sum()
                loss = loss - (self.args.pull_constraint_coeff * prompt_loss)

            # if batch_idx == 0 and self.args.IL_step > 0 and epoch == 4:
            #     self.plot_logits(output.mean(0).detach().cpu().numpy(), epoch)

            self.optimizer.zero_grad()
            loss.backward()

            print(
                f'{self.model_query.query_adaptor.weight.mean()}, {self.model_query.query_adaptor.bias.mean()}')

            # if batch_idx == 0 and epoch in [0, 1, 2, 3, 4]:
            #     # plotting prompt gradients for 1st batch across all 5 epochs
            #     temp = self.model.prompt.prompt.grad.view(72, -1).mean(1)
            #     self.plot_prompt_gradients(temp, selected_idx[0], epoch)

            # if self.args.freeze_cls_parts and self.args.IL_step > 0:
            #     print('zeroing out gradients\n')
            #     if len(self.args.device) > 1:
            #         # zero out prev class gradients
            #         # self.model.module.weight.grad[:self.args.labels_prev_step, :] = 0.0
            #         # self.model.module.bias.grad[:self.args.labels_prev_step] = 0.0
            #         self.model.module.weight.grad[:self.args.labels_prev_step, :] = 0.0
            #     else:
            #         self.model.weight.grad[:self.args.labels_prev_step, :] = 0.0

            self.optimizer.step()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if (epoch+1) % 5 == 0:
            print(f'saving model and model_query')
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()]
                                   for k, v in state_dict.items()])

            torch.save(weights, self.model_saved_name + '-' +
                       str(epoch+1) + '.pt')

            # optimize later, dump query model as well
            state_dict = self.model_query.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()]
                                   for k, v in state_dict.items()])

            torch.save(weights, self.model_saved_name + '-query' + '-' +
                       str(epoch+1) + '.pt')

    def get_accuracy(self, pred_list, target_list):
        acc = (pred_list == target_list).mean()
        print(acc)
        op2 = (pred_list == target_list).sum()/len(pred_list)
        print(op2)
        return (pred_list == target_list).mean()

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None, update_prompt_hist=False, return_acc=False):

        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        all_acc = {}
        for ln in loader_name:
            print(f'\nEvaluating Loader : {ln}')
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln], ncols=40)
            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)

                    cls_feat = self.model_query(data)
                    output_dict, reduce_sim, selected_idx = self.model(data, cls_feat, train_mode=0)

                    if self.args.eta_cosine:
                        output = output_dict['wsigma']
                    else:
                        output = output_dict['wosigma']

                    # if batch_idx == 4:
                        # self.update_prompt_frequency_stats(selected_idx)
                        # self.update_prompt_ordering_stats(selected_idx)
                        # print(selected_idx[:5, :5])
                        # print(f'checking logits {output.mean(0)}, logits old {output[:, :self.args.labels_prev_step].mean()}, logits new {output[:, self.args.labels_prev_step:].mean()}')

                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)

            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            tot_accuracy = self.get_accuracy(pred_list, label_list)

            self.print_log(f'Accuracy: {tot_accuracy}')
            all_acc[ln] = tot_accuracy

            if self.args.save_conf_mat and loader_name[0] == 'test':
                print('\nsaving confusion matrix\n')
                confusion = confusion_matrix(label_list, pred_list)
                np.set_printoptions(threshold=sys.maxsize)

                list_diag = np.diag(confusion)
                list_raw_sum = np.sum(confusion, axis=1)
                each_acc = list_diag / list_raw_sum

                old_acc = each_acc[:self.args.labels_prev_step].mean()
                new_acc = each_acc[self.args.labels_prev_step:].mean()
                print(each_acc)
                print(f'old {old_acc}, new {new_acc}')

                # default ordering
                class_names_mapping = ["drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop", "pickup", "throw", "sitting down",
                                       "standing up (from sitting position)", "clapping", "reading", "writing", "tear up paper", "wear jacket", "take off jacket", "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving", "kicking something", "reach into pocket", "hopping (one foot jumping)", "jump up", "make a phone call/answer phone",
                                       "playing with phone/tablet", "typing on a keyboard", "pointing to something with finger", "taking a selfie", "check time (from watch)", "rub two hands together", "nod head/bow", "shake head", "wipe face", "salute", "put the palms together", "cross hands in front (say stop)", "sneeze/cough", "staggering", "falling", "touch head (headache)", "touch chest (stomachache/heart pain)", "touch back (backache)", "touch neck (neckache)", "nausea or vomiting condition",
                                       "use a fan (with hand or paper)/feeling warm", "punching/slapping other person", "kicking other person", "pushing other person", "pat on back of other person", "point finger at the other person", "hugging other person", "giving something to other person", "touch other persons pocket", "handshaking", "walking towards each other", "walking apart from each other"
                                       ]

                label_mapping = {}
                for ind, name in enumerate(class_names_mapping):
                    label_mapping[ind] = name
                print(f'using label dict: {label_mapping}')

                save_conf_mat_image(
                    confusion,
                    label_mapping,
                    osp.join('/home/prachi/CTR-GCN/confmat_cvpr_supplementary',
                             self.args.save_name_args + '_conf_mat.png'),
                )

        if return_acc:
            return all_acc

    def start(self):

        self.class_to_task = []

        for i in range(self.num_class):
            if i < 40:
                self.class_to_task.append('Base Task class')
            elif i >= 40 and i < 45:
                self.class_to_task.append('Task 1 class')
            elif i >= 45 and i < 50:
                self.class_to_task.append('Task 2 class')
            elif i >= 50 and i < 55:
                self.class_to_task.append('Task 3 class')
            elif i >= 55 and i < 60:
                self.class_to_task.append('Task 4 class')

        if self.args.phase == 'train':
            print('\nTraining\n')
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.cfg))))
            self.global_step = self.cfg.start_epoch * \
                len(self.data_loader['train']) / self.batch_size

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')

            self.prompt_reinit()

            for epoch in range(self.cfg.start_epoch, self.num_epoch):
                self.train(epoch)
            print('\ndone training')

            if self.args.IL_step > 0 and self.args.train_eval:
                # same as weighted average
                # self.eval(epoch, save_score=self.cfg.save_score,
                #           loader_name=['test'], update_prompt_hist=True) 

                print('OLD')
                oldacc = self.eval(epoch=0, save_score=self.cfg.save_score,
                                    loader_name=['test_prev'], return_acc=True)
                oldacc = oldacc['test_prev']
                print('NEW')
                newacc = self.eval(epoch=0, save_score=self.cfg.save_score,
                                    loader_name=['test_new'], return_acc=True)
                newacc = newacc['test_new']
                HM = (2 * oldacc * newacc) / (oldacc + newacc)
                print(f'HM {HM}')
                # compute weoghted average of Old, New - `Avg` metric in paper
                num_new_classes = len(self.train_label_ids)
                num_old_classes = self.num_class - num_new_classes
                print(f'#new = {num_new_classes}, #old = {num_old_classes}, #total_classes = {self.num_class}. Computing weighted average as follows:\n')
                average_all = ((newacc * num_new_classes) +
                                (oldacc * num_old_classes))/self.num_class

                result_dict = {}
                result_dict['experiment'] = self.args.save_name_args
                result_dict['datafile'] = self.args.few_shot_data_file
                result_dict['random_k'] = args.randomized
                result_dict['step'] = self.args.IL_step
                result_dict['layer prompted'] = self.args.prompting_layer
                result_dict['Old'] = oldacc
                result_dict['New'] = newacc
                result_dict['HM'] = HM
                result_dict['average_all'] = average_all

                save_path = f'multiple_runs_{self.args.save_name_args}.csv'
                print(f'writing to the results file {save_path}')
                with open(save_path, 'a', newline='') as csvfile:
                    file_writer = csv.DictWriter(
                        csvfile, fieldnames=list(result_dict.keys()))
                    file_writer.writerow(result_dict)

        elif self.args.phase == 'test':
            wf = self.args.weights.replace('.pt', '_wrong.txt')
            rf = self.args.weights.replace('.pt', '_right.txt')

            if self.args.weights is None:
                raise ValueError('Please appoint --weights.')
            self.cfg.print_log = False
            self.print_log('Model:   {}.'.format(self.cfg.model))
            self.print_log('Weights: {}.'.format(self.args.weights))

            self.prompt_reinit()
            print('Average of all classes')
            self.eval(epoch=0, save_score=self.cfg.save_score,
                      loader_name=['test'], update_prompt_hist=True)
            self.save_prompt_hist(name='eval_all')
            self.save_prompt_ordering_hist(name='eval_all')

            if args.IL_step > 0:
                # eval on old and new separately
                print('eval all previous classes, average of old classes\n')
                print('OLD')
                self.eval(epoch=0, save_score=self.cfg.save_score,
                          loader_name=['test_prev'])
                print('NEW')
                self.eval(epoch=0, save_score=self.cfg.save_score,
                          loader_name=['test_new'])

                if args.task_specific_eval:
                    print(f'Task specific evaluation')
                    task_ids_list = ['Base_Task']
                    for i in range(self.args.IL_step):
                        task = i+1
                        task_ids_list.append("Task" + str(task))
                    print(f'evaluating on these tasks: {task_ids_list}')
                    
                    self.all_acc = self.eval(epoch=0, save_score=self.cfg.save_score,
                                    loader_name=task_ids_list, return_acc=True)
                    task_wise_dict = {'Base_Task': 0.0, 'Task1': 0.0, 'Task2': 0.0, 'Task3': 0.0, 'Task4': 0.0}
                    task_wise_dict['experiment'] = self.args.experiment_name
                    task_wise_dict['datafile'] = self.args.few_shot_data_file
                    task_wise_dict['step'] = self.args.IL_step
                    for k, v in self.all_acc.items():
                        task_wise_dict[k] = v 
                    save_path_BWF = self.args.experiment_name + '.csv'
                    print('writing to the task-wise BWF file')
                    with open(save_path_BWF, 'a', newline='') as csvfile:
                        file_writer = csv.DictWriter(
                            csvfile, fieldnames=list(task_wise_dict.keys()))
                        file_writer.writerow(task_wise_dict)
            self.print_log('Done.\n')


if __name__ == '__main__':
    '''
    separating yaml file (which now has all original params); new argparser contains only continual parameters, dynamically controlled through the bash scripts
    '''
    parser = get_parser()
    args = parser.parse_args()

    print(args)

    with open(args.config, 'rb') as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    init_seed(cfg.seed)
    processor = Processor(cfg, args)
    processor.start()
