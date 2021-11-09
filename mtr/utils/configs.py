import argparse
import os
import random

import numpy as np
import torch

#from utils.loader import get_data
from utils.loader import get_EAL


def get_device(gpu=0):
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        return 'cuda'
    else:
        return 'cpu'


def get_ddpg_kwargs():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hid_dims', default=[400, 300], type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--tau', default=5e-3, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lr_pi', default=5e-4, type=float)
    parser.add_argument('--lr_vf', default=1e-3, type=float)
    parser.add_argument('--lr_decay', default=0.1, type=float)

    # main variables
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--buffer_size', default=int(5e4), type=int)
    parser.add_argument('--eval_freq', default=200, type=int)
    parser.add_argument('--reward_scale', default=5000, type=int)

    # noise decay
    parser.add_argument('--eta_init', default=0.15, type=float)
    parser.add_argument('--eta_min', default=0.075, type=float)

    args = parser.parse_args()
    kwargs = vars(args)
    kwargs['device'] = get_device()
    return kwargs


def get_configs(algo='ddpg',
                stochastic=False,
                demo=False,
                seed=123):
    parser = argparse.ArgumentParser()

    # Model Params
    parser.add_argument("--algo", default=algo.upper(), type=str)
    parser.add_argument('--seed', default=seed, type=int)

    args, unknown = parser.parse_known_args()
    env_kwargs = dict(data=get_EAL(read=False))
    algo_kwargs = get_ddpg_kwargs()

    if args.seed:
        env_kwargs['seed'] = args.seed + 1

    # get exp_name
    # from mtr.env import TubeEnv
    from env import TubeEnv
    exp_name = ("Demo" if demo else "Main") + (
        "Sto" if stochastic else "Det") + env_kwargs['data']['name'][:3]
    exp_name += f"{TubeEnv.name}_v{TubeEnv.version}"

    return args, algo_kwargs, env_kwargs


def set_global_seeds(seed):
    if seed is None:
        return
    elif np.isscalar(seed):
        # set seeds
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        raise ValueError(f"Invalid seed: {seed} (type {type(seed)})")
