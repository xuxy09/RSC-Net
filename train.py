import os
from utils import TrainOptions
from train import Trainer
import torch
import numpy as np
import random
# fix random seeds for reproducibility
seed = 1024
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    trainer = Trainer(options)
    trainer.train()
