from typing import Any, IO
import pickle
import traceback
import shutil
import os.path as osp
import os
import sys
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.use('agg')


def preproc_conf_mat(conf_mat: np.ndarray) -> np.ndarray:
    # normalize the conf matrix
    conf_mat = conf_mat.astype(np.float32)
    conf_mat = conf_mat/(conf_mat.sum(0) + 1e-6)
    conf_mat = np.round((conf_mat*100)).astype(np.int32)
    return conf_mat


def write_conf_mat(conf_mat: np.ndarray, label_to_name: dict, file_path: str):
    conf_mat = preproc_conf_mat(conf_mat)
    fhand = get_file_handle(file_path, 'w+')

    # write 1st row
    fhand.write(' ')
    for i in range(conf_mat.shape[0]):
        fhand.write(',' + label_to_name[i].ljust(12))
    fhand.write('\n')

    for i in range(conf_mat.shape[0]):
        fhand.write(label_to_name[i].ljust(12))
        for x in conf_mat[i]:
            fhand.write(',' + str(x))
        fhand.write('\n')

    fhand.close()


def save_order_matrix(conf_mat: np.ndarray, file_path: str, cmap: str = 'hot', colorbar: bool = True,
                     include_values: bool = True, fig_size: float = 8.0):
    '''
    Note, this only plots for pool size 64 - its hard coded.
    '''
    font = {'family': 'serif',
            'weight': 'normal',
            'size': 12}

    plt.rc('font', **font)

    fig, ax = plt.subplots()
    fig.tight_layout()
    fig.set_size_inches(fig_size, fig_size)

    conf_mat = conf_mat / conf_mat.astype(float).sum(axis=1)

    pool_size = conf_mat.shape[0]

    print(f'\ncheck, inside save_conf_mat_image() #classes : {pool_size}')

    label_l = [i for i in range(64)]
    print(label_l)

    im_kw = dict(interpolation="nearest", cmap=cmap)
    im_ = ax.imshow(conf_mat, **im_kw)

    cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)

    if colorbar:
        fig.colorbar(im_, ax=ax, shrink=0.8, aspect=30)

    axisticks = np.arange(0, 64, 4)

    ax.set(
        xticks=axisticks,
        yticks=axisticks,
        xticklabels=axisticks,
        yticklabels=axisticks,
    )
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)

    ax.set_ylim((pool_size - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation='vertical')
    plt.savefig(file_path, bbox_inches='tight')


def save_conf_mat_image(conf_mat: np.ndarray, label_to_name: dict, file_path: str, cmap: str = 'hot',
                       colorbar: bool = False, include_values: bool = True, fig_size: float = 10.0):
    # Note. this saves the normalized confusion matrix.
    """Source: https://github.com/scikit-learn/scikit-learn/blob/36958fb240fbe435673a9e3c52e769f01f36bec0/sklearn/metrics/_plot/confusion_matrix.py
    """
    print('Saving confusion matrix to ', file_path)

    fig, ax = plt.subplots()
    fig.tight_layout()
    fig.set_size_inches(fig_size, fig_size)

    n_classes = conf_mat.shape[0]

    print(f'\ncheck, inside save_conf_mat_image() #classes : {n_classes}')

    label_l = [label_to_name[i].capitalize() for i in range(n_classes)]
    print(label_l)

    im_kw = dict(interpolation="nearest", cmap=cmap)
    im_ = ax.imshow(conf_mat, **im_kw)

    cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)

    if include_values:
        text_ = np.empty_like(conf_mat, dtype=object)

        # print text with appropriate color depending on background
        thresh = (conf_mat.max() + conf_mat.min()) / 2.0

        for i in range(n_classes):
            for j in range(n_classes):
                color = cmap_max if conf_mat[i, j] < thresh else cmap_min
                if conf_mat[i, j] > 0:
                    text_[i, j] = ax.text(j, i, conf_mat[i, j], ha="center", va="center",
                                        color=color, fontsize='x-small')

    if colorbar:
        fig.colorbar(im_, ax=ax)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=label_l,
        yticklabels=label_l,
        ylabel="Ground Truth",
        xlabel="Predictions",
    )

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation='vertical')
    plt.savefig(file_path, bbox_inches='tight')


def save_figure(fig, fpath):
    fig.tight_layout()
    fig.savefig(fpath)


def save_pickle(fpath: str, data: Any):
    """Save data in pickle format."""
    with open(fpath, 'wb+') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(fpath: str) -> Any:
    """Load data from pickle file."""
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    return data


def get_file_handle(fpath: str, mode: str) -> IO:
    try:
        fhand = open(fpath, mode)
    except Exception:
        traceback.print_exc()
        sys.exit()
    return fhand


def print_argparser_args(args):
    """prints the argparser args better"""
    for arg in vars(args):
        print(arg, '=', getattr(args, arg))
    print("\n\n")


def get_free_port():
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def mkdir_rm_if_exists(dir_):
    if osp.isdir(dir_):
        shutil.rmtree(dir_)
    os.makedirs(dir_)
