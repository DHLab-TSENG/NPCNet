import random
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = {
    "seed": 42,
    "pretraining_epoch": 3,
    "epoch": 20,
    "batch_size": 64,
    "lr": 1e-4,
    "wd": 1e-7,
    "n_clusters": 4,
    "vocab_size": 406,
    "latent_dim": 32,
    "lambda1": 1,
    "lambda2": 1e-3,
    "lambda3": 1,
    "kappa1": 1,
    "kappa2": 1e-2,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}