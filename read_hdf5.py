from __future__ import print_function

import numpy as np

import h5py

import pdb 

with h5py.File('training_pair_mean/training_pair0.h5','r') as hf:

    print('List of arrays in this file: \n', hf.keys())

    data = hf.get('data_1')
    sim = hf.get('sim')
    np_data = np.array(data[1999,:,15,:,:])

    pdb.set_trace() 
    print('Shape of the array dataset_1: \n', np_data.sum())
