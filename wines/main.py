import os
import csv
import numpy as np

csv_file = R'C:\Projects\Deep-Learning-with-Pytorch\wines\winequality-white.csv'

if __name__ == '__main__':
    data = []
    assert os.path.exists(csv_file)
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        for row in reader:
            data.append([float(elt) for elt in row])
