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
import pdb
import cv2


if __name__ =='__main__':
   
   linenum = 0

  

   mean_file = np.zeros((3,112,112))

   count = 0.0
   i = 0
   j = 0 
   with open("c3d_ucf101_my_train1.txt") as txt_file:
        l = txt_file.readlines()
   
   while True:
	line_info =  l[linenum]

	try:
	# read video from line info
	     
	      video = cv2.VideoCapture(re.split(' ',line_info)[0]) 

	      if  int(re.split(' ',line_info)[1]) < 6:
	   	# set video input frame as lineinfo
		  video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, int(re.split(' ',line_info)[1]))
	      else:
		  video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, int(re.split(' ',line_info)[1])-5)
	     
	      for frame_num in range(16):
		    # 1)read assigned video frame,2)resize to 224 224(may need further improvement with cropping and padding stuff), 3)swap axes put color info in front 
		  mean_file = np.swapaxes(cv2.resize(video.read()[1],(112,112)),0,2).astype('float')*1/(count+1) + mean_file*count/(count+1)
	          #mean_file = (np.swapaxes(cv2.resize(video.read()[1],(112,112)),0,2) + mean_file)/2
	          count = count+1
                  print count	  
	      # i = i+1
	      linenum = linenum + 1
	      #print j,i,linenum 
	      if i >= len(l):
		 break	
	except:
	     #   print 'something wrong'
	  pdb.set_trace()
  
   pkl.dump(mean_file,open("meanFile.p","wb"))


     			   
