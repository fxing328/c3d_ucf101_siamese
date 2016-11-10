import numpy as np
import pdb
import re

linenum = 0
with open("ucf_siamiese_train1.txt") as txt_file1:
   l1 = txt_file1.readlines()
   with open("ucf_siamiese_train2.txt") as txt_file2:
       l2 = txt_file2.readlines()

labdif = []
while True:
   try:
        line_info_1 = l1[linenum]
        line_info_2 = l2[linenum]

        label1 = int(re.split(' ',line_info_1)[2])
        label2 = int(re.split(' ',line_info_2)[2])
        diff = label1-label2
        if diff <> 0:
           labdif.append(diff)
           pdb.set_trace()
        linenum = linenum+1
        print linenum
        if linenum >= len(l1):
           break
   except:
        pdb.set_trace()

print len(labdif)
