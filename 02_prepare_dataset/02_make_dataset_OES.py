'''
This script copies patches of single classes from case folders
to one general class folder: preparing for training
'''


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:17:00 2021

@author: dr_pusher
"""


import os
import shutil


base_dir = ''
target_dir = ''


dirnames = sorted(os.listdir(base_dir))

#clean up benign classes
for dir_name in dirnames:
    print("")
    print("#####")
    print("Working with slide:", dir_name)
    print("#####")
    subdirs = sorted(os.listdir(base_dir + dir_name))
    for sub_dir in subdirs:
        print("Working with class:", sub_dir)
        filenames = sorted(os.listdir(base_dir + dir_name + "/" + sub_dir))
        filenames = filenames [::2]
        print("Number of files to process: ", len(filenames))
        for filename in filenames:
            shutil.copy(base_dir + dir_name + "/" + sub_dir + "/" + filename, target_dir + sub_dir)
        print("Files copied")
        print("")
