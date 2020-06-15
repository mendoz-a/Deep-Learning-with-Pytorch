import os
import csv
import time
import torch
import scipy
import numpy as np
from torch.distributions.beta import Beta
from scipy.stats import pearsonr

def pearson(data, label):
    data_t = torch.from_numpy(data)
    label_t = torch.from_numpy(label)

    data_t = data_t.to(device='cuda')
    label_t = label_t.to(device='cuda')
    
    label_t = label_t.unsqueeze(1)
    label_mat = label_t.repeat(1, data_t.shape[1])
    label_mat = label_mat.to('cuda')

    data_mean = data_t.mean(dim=0)
    label_mean = label_mat.mean(dim=0)

    data_sub = data_t - data_mean
    label_sub = label_mat - label_mean

    r = (data_sub * label_sub).sum(dim=0)
    r /= torch.sqrt(data_sub.pow(2).sum(dim=0) * label_sub.pow(2).sum(dim=0))

    n = data_t.shape[0]
    dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)

    p = 2 * dist.cdf(-abs(r.cpu()))
    return r, p

if __name__ == '__main__':
    csv_file = R'C:\Projects\Deep-Learning-with-Pytorch\wines\winequality-white.csv'
    assert os.path.exists(csv_file)

    data = []
    data = np.loadtxt(csv_file, dtype=np.float32, delimiter=';', skiprows=1)

    header = next(csv.reader(open(csv_file), delimiter=';'))
    label = data[:, -1]
    data = data[:, :-1]
    print(pearson(data, label))
