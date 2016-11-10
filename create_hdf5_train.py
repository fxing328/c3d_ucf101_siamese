import matplotlib.pyplot as plt
import numpy as np
import pdb
import cPickle as pkl
from sklearn import preprocessing
import h5py
import copy
import math
import matplotlib
import h5py
import scipy.io as scipy_io
from pylab import *
#from caffe import layers as L
import re
import cv2


if __name__ =='__main__':
   whole_pair = 100000
   #whole_pair = 40000
   train_pair = 2000
   #trainsub_pair = 400
   #batch_pair = 500
   linenum = 0
   #for n in range(train_pair/trainsub_pair):
   #pdb.set_trace()
   meanFile = pkl.load(open("meanFile.p","rb"))
   n = whole_pair/train_pair
   with open("ucf_siamiese_train1.txt") as txt_file1:
       l1 = txt_file1.readlines()
   with open("ucf_siamiese_train2.txt") as txt_file2:
       l2 = txt_file2.readlines()    
   #pdb.set_trace()
  # while linenum < whole_pair:    
     #pdb.set_trace()
   
    
   for s in range(n):
       #o = s*(train_pair/batch_pair)
      # create hdf5 file with data_1 data_2 5-D tensor with dimension (train_pair_num,16,3,224,224)
       #i = 0 
        #pdb.set_trace()
     
     with  h5py.File('training_pair_mean/training_pair'+str(s)+'.h5','w') as f1_pair:
	f1_pair.create_dataset('data_1',(train_pair,3,16,112,112),dtype='f8')
	f1_pair.create_dataset('data_2',(train_pair,3,16,112,112),dtype='f8')
	f1_pair.create_dataset('label',(train_pair,2),dtype="f8")
	f1_pair.create_dataset('sim',(train_pair,2),dtype="f8") 
        i = 0
        # """txt_1 and txt_2 being the previous txt file prepared """
# 	   with open("ucf_siamiese_1.txt") as txt_file1:
#	     with open("ucf_siamiese_2.txt") as txt_file2:
        while True:
		line_info_1 =  l1[linenum]
		#pdb.set_trace()
		line_info_2 =  l2[linenum]  
		try:
		# read video from line info
		      #pdb.set_trace()
		      video_1 = cv2.VideoCapture(re.split(' ',line_info_1)[0]) 
		      video_2 = cv2.VideoCapture(re.split(' ',line_info_2)[0])
		      if  int(re.split(' ',line_info_1)[1]) < 6:
		   	# set video input frame as lineinfo
			  video_1.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, int(re.split(' ',line_info_1)[1]))
		      else:
			  video_1.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, int(re.split(' ',line_info_1)[1])-5)
		      if int(re.split(' ',line_info_2)[1]) < 6:
			  video_2.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,int(re.split(' ', line_info_2)[1]))
		      else:    
			  video_2.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,int(re.split(' ', line_info_2)[1])-5)
		      for frame_num in range(16):
			    # 1)read assigned video frame,2)resize to 224 224(may need further improvement with cropping and padding stuff), 3)swap axes put color info in front 
			  f1_pair['data_1'][i,:,frame_num,:,:] = np.swapaxes(cv2.resize(video_1.read()[1],(112,112)),0,2)-meanFile
			  #pdb.set_trace()
			  f1_pair['data_2'][i,:,frame_num,:,:] = np.swapaxes(cv2.resize(video_2.read()[1],(112,112)),0,2)-meanFile
                      pair_label = np.zeros(2)
                      pos_or_neg = int(re.split(' ',line_info_1)[2])
                      pair_label[pos_or_neg] = 1
		      f1_pair['label'][i] = pair_label
		      f1_pair['sim'][i] = pair_label
		  #if np.array(f1_pair['data_1'][i,:,frame_num,:,:]).sum() == 0 or np.array(f1_pair['data_2'][i,:,frame_num,:,:]).sum() == 0:
		      #print f1_pair['sim'][i]
		      i = i+1
		      linenum = linenum + 1
		      print i,linenum
		      if i >= train_pair:
			 break	
		except:
		     #   print 'something wrong'
		  pdb.set_trace()

		      #continue 
     #n = n-1
       #pdb.set_trace()
      #f1_pair.close()    
     			   
