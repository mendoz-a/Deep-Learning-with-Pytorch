import os
import csv
import torch
import numpy as np
from scipy.stats import pearsonr

csv_file = R'C:\Projects\Deep-Learning-with-Pytorch\wines\winequality-white.csv'

if __name__ == '__main__':
    data = []
    assert os.path.exists(csv_file)
    data = np.loadtxt(csv_file, dtype=np.float32, delimiter=';', skiprows=1)
    data_t = torch.from_numpy(data)

    header = next(csv.reader(open(csv_file), delimiter=';'))
    label = data[:, -1]
    data = data[:, :-1]
    data_t = torch.from_numpy(data)
    label_t = torch.from_numpy(label)


    #data_t = (data_t - data_mean) / data_std
    #le_t = torch.le(label_t, 3)
    
    corr_cpu = np.zeros((data.shape[1], 2), dtype=np.float32)
    for i in range(data.shape[1]):
        corr_cpu[i, :] = pearsonr(data[:, i], label)

    data_t = data_t.to(device='cuda')
    label_t = label_t.to(device='cuda')


    label_mat = torch.zeros((label_t.shape[0], data_t.shape[1]), dtype=torch.float32)
    for i in range(label_mat.shape[1]):
        label_mat[:, i] = label_t
    label_mat = label_mat.to('cuda')
    
    data_mean = data_t.mean(dim=0)
    label_mean = label_mat.mean(dim=0)

    data_sub = data_t - data_mean
    label_sub = label_mat - label_mean

    corr_gpu = (data_sub * label_sub).sum(dim=0)
    corr_gpu /= torch.sqrt(data_sub.pow(2).sum(dim=0) * label_sub.pow(2).sum(dim=0))
    print(corr_gpu)