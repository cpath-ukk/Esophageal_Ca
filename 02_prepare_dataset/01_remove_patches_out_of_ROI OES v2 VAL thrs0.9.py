'''
This scripts goes through folders with extracted patches and remove
those which contain the target class (e.g., TUMOR) less than certain
threshold
'''


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 19:36:01 2021

@author: dr_pusher
"""

import os
from PIL import Image
import numpy as np
#import shutil

base_dir = ''

#Tissue detector: im - patch map, label - tissue class
def detector_tumor(im, thd):
    count = 0
    im_reshape = np.reshape(im, len(im[0])*len(im[1]))
    for i in range(len(im_reshape)):
        if im_reshape[i] == 1:
            count = count + 1
            if (count / len(im_reshape)) > thd:
                return True
    return False

#Tissue detector: im - patch map, label - tissue class
def detector_other(im, thd):
    count = 0
    im_reshape = np.reshape(im, len(im[0])*len(im[1]))
    for i in range(len(im_reshape)):
        if im_reshape[i] != 0:
            count = count + 1
            if (count / len(im_reshape)) > thd:
                return True
    return False


dirnames = sorted(os.listdir(base_dir))


#clean up benign classes
for dir_name in dirnames:
    print("")
    print("#####")
    print("Working with slide:", dir_name)
    print("#####")
    subdirs = sorted(os.listdir(base_dir + dir_name))
    del subdirs[13] #Tumor
    
    
    for sub_dir in subdirs:
        print("Working with class:", sub_dir)
        filenames = sorted(os.listdir(base_dir + dir_name + "/" + sub_dir))
        del filenames [::2]
        print("Number of files to process: ", len(filenames))
        
        list_to_remove = []
        for filename in filenames:
            im = Image.open(base_dir + dir_name + "/" + sub_dir + "/" + filename)
            im = im.resize((20,20), Image.ANTIALIAS)
            im = np.array(im)
            if detector_other (im, 0.9) == False:
                #print(filename, " to be removed")
                list_to_remove = list_to_remove + [filename]
        print("Number of files to remove: ", len(list_to_remove))
        for fname in list_to_remove:
            os.remove(base_dir + dir_name + "/" + sub_dir + "/" + fname)
            #shutil.move(base_dir + dir_name + "/" + sub_dir + "/" + fname, target_dir)
        for fname in list_to_remove:
            os.remove(base_dir + dir_name + "/" + sub_dir + "/" + fname[:-4] + ".jpg")
            #shutil.move(base_dir + dir_name + "/" + sub_dir + "/" + fname[:-13] + ".jpg", target_dir)
        print("Ready")
        print("")
        

#clean up tumor class
for dir_name in dirnames:
    print("")
    print("#####")
    print("Working with slide:", dir_name)
    print("#####")
    
    print("Working with class:", "TUMOR")
    filenames = sorted(os.listdir(base_dir + dir_name + "/TUMOR"))
    del filenames [::2]
    print("Number of files to process: ", len(filenames))
    
    list_to_remove = []
    for filename in filenames:
        im = Image.open(base_dir + dir_name + "/TUMOR/" + filename)
        im = im.resize((20,20), Image.ANTIALIAS)
        im = np.array(im)
        if detector_tumor (im, 0.05) == False:
            #print(filename, " to be removed")
            list_to_remove = list_to_remove + [filename]
    print("Number of files to remove: ", len(list_to_remove))
    for fname in list_to_remove:
        os.remove(base_dir + dir_name + "/TUMOR/" + fname)
    for fname in list_to_remove:
        os.remove(base_dir + dir_name + "/TUMOR/" + fname[:-4] + ".jpg")
    print("Ready")
    print("")


