'''
This script returns number of patches for certain classes
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:40:47 2021

@author: dr_pusher
"""


import os

base_dir = ''


dirnames = sorted(os.listdir(base_dir))

#clean up benign classes
for dir_name in dirnames:
    filenames = sorted(os.listdir(base_dir + dir_name))
    print("Class:", dir_name, "Number of files: ", len(filenames))
   
    