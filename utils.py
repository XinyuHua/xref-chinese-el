import random
import os
import glob
import torch
import math
import numpy as np

DATA_PATH = '/data/model/xinyu/akbc2020_release/to_release/'

def collate_tokens(values, pad_idx):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][:len(v)])
    return res

def get_latest_ckpt_path(ckpt_dir):
    def get_epoch_num(path):
        base = os.path.basename(path)
        base = base.split('.')[0].split('_')[0]
        base = base.split('=')[1]
        return int(base)

    ckpt_list = sorted(
        glob.glob(os.path.join(ckpt_dir, "*.ckpt")),
        key=lambda x: get_epoch_num(x),
        reverse=False
    )
    return ckpt_list[-1]

def move_to_cuda(sample):

    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)

def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            r = {key: _apply(value) for key, value in x.items()}
            return r
            # return {
            #     key: _apply(value)
            #     for key, value in x.items()
            # }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)
