import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs')

for i in range(100):
    writer.add_scalar('y=2x', i*2, i)