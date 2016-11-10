import numpy as np
import pdb
import linecache
import re




if __name__ =='__main__':
    
    #dataPath = '/'
    train_pair = 40000
    labelOri = []
    i = 0
    try:

	    with open('ucf_siamiese_test1.txt','w') as filelist_1:
	      with open('ucf_siamiese_test2.txt','w') as filelist_2:
		with open('c3d_ucf101_my_test.txt','rb') as trainOri:
		  list_lines = trainOri.readlines()
		  for lin in list_lines:
		     #pdb.set_trace() 
		     #labelOri_idx = lin.rfind(' ')
		     #labelOri.append(lin[labelOri_idx+1:-1])
		     labelOri_idx = int(re.split(' ',lin)[2])
		     labelOri.append(labelOri_idx)
		
		  while True:
		      neg_or_pos = np.random.random_integers(0,1,1)
		      if neg_or_pos == 1:
		      #same label
		         #pdb.set_trace()
		         pos_label = np.random.random_integers(0,100,1)
		         posOri_start = labelOri.index(pos_label[0])
		         posOriNum = labelOri.count(pos_label[0])
		         postempvector = np.linspace(posOri_start,posOri_start+posOriNum-1, num=posOriNum, dtype=int)
		         np.random.shuffle(postempvector)
		         possel_index = np.random.choice(postempvector,2)
		         posline_tmp_1 = linecache.getline('c3d_ucf101_my_test.txt',possel_index[0]+1)
		         posline_tmp_2 = linecache.getline('c3d_ucf101_my_test.txt',possel_index[1]+1)
		         #poslabel_idx1 = posline_tmp_1.rfind(' ')
		         #poslabel_idx2 = posline_tmp_2.rfind(' ')
		         #posline_1 = posline_tmp_1[:poslabel_idx1] + ' 1' + '\n'
		         #posline_2 = posline_tmp_2[:poslabel_idx2] + ' 1' + '\n'

		         posline_1 = re.split(' ',posline_tmp_1)[0] + ' ' + re.split(' ',posline_tmp_1)[1] + ' '+'1\n'
		         posline_2 = re.split(' ',posline_tmp_2)[0] + ' ' + re.split(' ',posline_tmp_2)[1] + ' '+'1\n'
		         filelist_1.write(posline_1)
		         filelist_2.write(posline_2)
		         
		         #if i <> 0 and len(l1) <5 or len(l2) <5:
		          #  pdb.set_trace()
	 
		      else:
		         #pdb.set_trace()
		         neg_label = np.random.choice(100,2)
		         negOri_start_1 = labelOri.index(neg_label[0])
		         negOriNum_1 = labelOri.count(neg_label[0])
		         negOri_start_2 = labelOri.index(neg_label[1])
		         negOriNum_2 = labelOri.count(neg_label[1])
		         tempvector_1 = np.linspace(negOri_start_1,negOri_start_1+negOriNum_1-1, num=negOriNum_1, dtype=int)
		         tempvector_2 = np.linspace(negOri_start_2,negOri_start_2+negOriNum_2-1, num=negOriNum_2, dtype=int)
		         negsel_index_1 =  np.random.choice(tempvector_1,1)
		         negsel_index_2 =  np.random.choice(tempvector_2,1)
		         negline_tmp_1 = linecache.getline('c3d_ucf101_my_train1.txt',negsel_index_1[0]+1)
		         negline_tmp_2 = linecache.getline('c3d_ucf101_my_train1.txt',negsel_index_2[0]+1)
		         #neglabel_idx1 = negline_tmp_1.rfind(' ')
		         #neglabel_idx2 = negline_tmp_2.rfind(' ')
		         #negline_1 = negline_tmp_1[:neglabel_idx1] + ' 0' + '\n'
		         #negline_2 = negline_tmp_2[:neglabel_idx2] + ' 0' + '\n'
		         negline_1 = re.split(' ',negline_tmp_1)[0] + ' '+ re.split(' ',negline_tmp_1)[1] + ' ' + '0\n'
		         negline_2 = re.split(' ',negline_tmp_2)[0] + ' '+ re.split(' ',negline_tmp_2)[1] + ' ' + '0\n'
		         filelist_1.write(negline_1)
		         filelist_2.write(negline_2)
		         
		         #if i <> 0 and len(l1) <5 or len(l2) <5:
		         #   pdb.set_trace()
		      i+=1;
		      print i
		      if i >= train_pair:
		          break
		      else: 
		          continue
    except:
     #    print 'error somewhere'

      pdb.set_trace()
  
