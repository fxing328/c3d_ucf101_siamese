import numpy as np
import h5py
import pdb
import re


train_pair = 2000
linenum = 0
with open("ucf_siamiese_1.txt") as txt_file1:
   l1 = txt_file1.readlines()
   with open("ucf_siamiese_2.txt") as txt_file2:
       l2 = txt_file2.readlines()


with h5py.File('training_pair_mean/training_pair0.h5','a') as hf:
     print('list of arrays in this file: \n', hf.keys())
     pdb.set_trace()
     if 'label' in hf.keys():
        hf.__delitem__('label')
     if 'sim' in hf.keys():
        hf.__delitem__('sim')
     print('list of arrays in this file: \n', hf.keys())
     
     hf.create_dataset('label',(train_pair,2),dtype="f8")
     hf.create_dataset('sim',(train_pair,2),dtype="f8")
 
     i = 0

     while True:

        line_info_1 = l1[linenum]
        line_info_2 = l2[linenum]
        pdb.set_trace()
        try:
           label_v = np.zeros(2)
           neg_or_pos = int(re.split(' ',line_info_1)[2])
           label_v[neg_or_pos] = 1
           hf['label'][i] = label_v
           hf['sim'][i] = label_v
           i = i+1
           linenum = linenum+1
           if i >= train_pair:
              break
        except:
           pdb.set_trace()
     hf.close()  
