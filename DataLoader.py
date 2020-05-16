import sys, os, random
import numpy as np
import pandas as pd

class DataLoader(object):
    def __init__(self):
        self.requests = []
        self.operations = []

    def get_requests(self):
        pass
    def get_operations(self):
        pass

class DataLoaderPintos(DataLoader):
    def __init__(self, progs, boot=False):
        super(DataLoaderPintos, self).__init__()

        if isinstance(progs, str): progs = [progs]
        for prog in progs:
            df = pd.read_csv(prog, header=0)
            if not boot: df = df.loc[df['boot/exec'] == 1, :]
            self.requests += list(df['blocksector'])
            self.operations += list(df['read/write'])

    def get_requests(self):
        return self.requests

    def get_operations(self):
        return self.operations

class DataLoaderZipf(DataLoader):
    def __init__(self, num_files, num_samples, param, num_progs=1, operation='random'):
        super(DataLoaderZipf, self).__init__()

        for i in range(num_progs):
            files = np.arange(num_files)
            # Random ranks. Note that it starts from 1.
            ranks = np.random.permutation(files) + 1
            # Distribution
            pdf = 1 / np.power(ranks, param)
            pdf /= np.sum(pdf)
            # Draw samples
            self.requests += np.random.choice(files, size=num_samples, p=pdf).tolist()
            if operation == 'random':
                self.operations += np.random.choice([0, 1], size=num_samples).tolist()
            else:
                self.operations += np.full(num_samples, int(operation)).tolist()

    def get_requests(self):
        return self.requests

    def get_operations(self):
        return self.operations