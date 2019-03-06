#!/usr/bin/env python3

import torch

num_gpus = torch.cuda.device_count()

print('Pytorch version: {}'.format(torch.__version__))

print('Number of GPUs: {}'.format(num_gpus))

for i in range(num_gpus):
    print('GPU {}: {}'.format(i, torch.cuda.get_device_name(i)))
