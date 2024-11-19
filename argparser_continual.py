import argparse
import os
import sys


def get_parser():
    # modified parameter priority: config file is previous argparser, this argparser will stay as arg in file - these are the dynamic continual parameters being specified in bash scripts.
    parser = argparse.ArgumentParser(
        description='Meta-HSL, FSCIL personalization for NTU60 human activity recognition')
    
    # post ECCV 
    parser.add_argument('--classifier_type', type=str, default='cosine',
                        help='type of classifier to be used : cosine or regular FC')
    
    parser.add_argument('--replay_few_shots', action='store_true',
                        help='use few shot train samples from previous incremental classes for training current or not.')
    
    parser.add_argument('--experiment_name', type=str, default='POET_ours',
                        help='Name of the experiment for logging the task-wise matrix.')

    parser.add_argument('--save_numbers_to_csv', action='store_true',
                        help='Log old new HM average to csv file')
    
    parser.add_argument('--few_shot_data_file', type=str, default='/home/prachi/CTR-GCN/data/NTU60_5shots.npz',
                        help='Few shot data file, default file is this.')
                        
    # pre-ECCV 
    parser.add_argument('--Base_channels_prompt', default=64, type=int,
                       help='base channels, modify for prompting along feature dimension experiment')

    parser.add_argument('--prompting_layer', default=1, type=int,
                       help='attach prompt on this layer!!')

    parser.add_argument('--test_class_index', default=-1, type=int,
                       help='test or run analytics only on this class, make sure pass index not class number')

    parser.add_argument('--save_conf_mat', action='store_true',
                       help='whether to save the confusion matrix.')

    parser.add_argument('--save_name_args', default='base')

    parser.add_argument('--config', default='./config/nturgbd-cross-view/test_bone.yaml',
                       help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train',
                       help='must be train or test')
    
    parser.add_argument('--train_eval', default=False,  action='store_true',
                       help='used to run eval along with train and log results.')

    parser.add_argument('--class_order', default=0, type=int,
                       help='0 for default sequential order, 1 for inc class shuffled order')

    parser.add_argument('--randomized', default=-1, type=int,
                       help='add random prompts, override the cfg.prompt.random_k with this')

    parser.add_argument('--expand_init_protocol', default=1, type=int,
                       help='1 for only prompts, 2 for both')

    parser.add_argument('--task_specific_eval', default=False, action='store_true',
                       help='task-wise eval for step 2 onwards (base is treated as separate task too.)')

    parser.add_argument('--expand_allpool', default=False, action='store_true',
                       help='to init new expanded pool the same as base pool, for running prompt expansion experiment')

    parser.add_argument('--load_ckpt_fc', default=True, action='store_false',
                       help='load ckpt for train, with or without fc')

    parser.add_argument('--eta_cosine', default=False, action='store_true',
                       help='If True, use and update eta. Otherwise just good old basic cosine norm?')

    parser.add_argument('--classifier_lr', type=float, default=0.01,
                       help='tune entire classifier at a reduced lr as compared to prompt components, only applicable in incremental tasks.')

    parser.add_argument('--classifier_average_init', action='store_true',
                       help='init expanded classifier weights as average of existing ones.')

    parser.add_argument('--weights', default=None,
                       help='the weights for network initialization')

    parser.add_argument('--device', default=0, type=int, nargs='+',
                       help='the indexes of GPUs for training or testing')

    parser.add_argument('--experiment', type=str, default='FT',
                       help='experiment name, used for freezing/loading weights')

    parser.add_argument('--query_checkpoint', type=str,
                       help='Pretrained path for initialization of continual training.')

    parser.add_argument('--IL_step', default=0, type=int,
                       help='0 for base session, 1 onwards incremental')

    parser.add_argument('--labels_prev_step', default=0, type=int)

    parser.add_argument('--maxlabelid_curr_step', default=0, type=int)

    parser.add_argument('--k_shot', type=int, default=-1,
                       help='number of few-shots')  # if -1, train on all, else train on few-shots.

    # prompting params, to be used later
    parser.add_argument('--prompt_sim_reg', action='store_true',
                       help='prompt loss during training.')

    # Important! major bug, pass this only for fine-tuning entire classifier
    parser.add_argument('--freeze_cls_parts', default=True, action='store_false',
                       help='freeze old classifier parts.')

    parser.add_argument('--L2P_cls_protocol', default=False, action='store_true',
                       help='make logits of previous classes -infinity like L2P and dualprompt')

    parser.add_argument('--pull_constraint_coeff', type=float, default=0.1,
                       help='prompt loss coefficient')

    # parser.add_argument('--update_query', default=False, action='store_true',
    #                     help='update query function acc to hyperparams in cfg')

    parser.add_argument('--freeze', default=['module.input_map', 'module.s_att', 'module.t_att'],
                       nargs='*', type=list, help='freeze part in backbone model')

    parser.add_argument('--optimizer', default='SGD',
                       help='type of optimizer')

    # main_prompt_lwf.py speccific parameter
    parser.add_argument('--lwf_type', default=1, type=int,
                       help='Exp 1: distill all old, mask for new. Exp 2: distill only old-new, no mask for new.')

    # for L2P, CODAP COMPARISONS
    parser.add_argument('--old_logits_infinity', action='store_true', default=False,
                       help='make previous class logits -inf')

    return parser
