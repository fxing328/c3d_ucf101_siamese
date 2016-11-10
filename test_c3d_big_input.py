import matplotlib.pyplot as plt
import numpy as np
import pdb
import cPickle as pkl
from sklearn import preprocessing
import h5py 
import copy
import math
import matplotlib
import caffe, h5py
import scipy.io as scipy_io
from pylab import *
from caffe import layers as L

useSolver = False
if  __name__ == '__main__':
    caffe.set_mode_gpu()
    nvid = 58 #Sunny_MVI_20012
    niter = nvid
    pdb.set_trace()
    if useSolver:
        solver = caffe.SGDSolver('./c3d_solver_big_input.prototxt')
        solver.net.copy_from('../pretrained/test_new_57000.000000.caffemodel')
    else:
        # net = caffe.Net('./c3d_train_test_big_input.prototxt','../pretrained/c3d_ucf101_iter_38000.caffemodel',caffe.TEST)
        net = caffe.Net('./c3d_train_test_big_input.prototxt','test_new_57000.000000',caffe.TEST)        

    for it in range(60):
        print 'iteration:', it
        if useSolver:
            solver.step(1)  ##alway fetch the same batch from h5 regardless of iteration
            # pdb.set_trace()
            solver.test_nets[0].forward()
            frm = solver.test_nets[0].blobs['data'].data
            conv5a = solver.test_nets[0].blobs['conv5a'].data
        else:
            net.forward() ##alway fetch the same batch from h5 regardless of iteration
            conv5a = net.blobs['conv5a'].data            

        # pkl.dump(conv5a,open('conv5a_Sunny_MVI_20012_gop'+str(it).zfill(3)+'.p','wb'))

       # if it==0:
       #     conv5a_0 = conv5a
       #     frm0 = frm
       # if it==1:
       #     conv5a_1 = conv5a
       #     frm1 = frm
       # if it>=2:
       #     print np.sum(frm - frm1)

       # if it==58:
       #     conv5a_58 = conv5a
       # if it==59:
       #     conv5a_59 = conv5a


    # assert conv5a_0==conv5a_58
    # assert conv5a_1==conv5a_59
    pdb.set_trace()

