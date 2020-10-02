import random
import pickle
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch


__all__ = 'seed_all', 'rng_state', 'set_rng_state', 'load_rng', 'dump_rng'


def seed_all(seed: int, det_cudnn=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if det_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rng_state():
    return (random.getstate(), np.random.get_state(),
            torch.get_rng_state(), torch.cuda.get_rng_state_all())


def set_rng_state(state: Tuple):
    py_rng, np_rng, torch_rng, torch_cuda_rng = state
    random.setstate(py_rng)
    np.random.set_state(np_rng)
    torch.set_rng_state(torch_rng)
    torch.cuda.set_rng_state_all(torch_cuda_rng)


def load_rng(file: Union[str, Path]):
    with Path(file).open('rb') as f:
        set_rng_state(pickle.load(f))


def dump_rng(file: Union[str, Path]):
    with Path(file).open('wb') as f:
        pickle.dump(rng_state(), f)
