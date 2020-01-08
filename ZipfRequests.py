
import numpy as np
import pandas as pd
import random

if __name__ == "__main__":
    num_files = 5000
    num_samples = 10000
    save_path = 'data2.0/zipf.csv'
    param = 1.3

    files = np.arange(num_files)
    # Random ranks. Note that it starts from 1.
    ranks = np.random.permutation(files) + 1
    # Distribution
    p = 1 / np.power(ranks, param)
    p /= np.sum(p)
    # Draw samples
    requests = np.random.choice(files, size=num_samples, p=p)
    operations = np.full_like(requests, 0)
    executions = np.full_like(requests, 1)
    # Make dataframe and save .csv
    df = pd.DataFrame({'blocksector': requests, 'read/write': operations, 'boot/exec': executions})
    df.to_csv(save_path, index=False, header=True)