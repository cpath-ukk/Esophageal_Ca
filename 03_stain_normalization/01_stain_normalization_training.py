'''
This script stain normalizes patches of training
dataset according to a reference image and
Macenko principle
'''


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 18:36:02 2021

@author: dr_pusher
"""

import numpy as np
import staintools
import os
from PIL import Image

st = staintools.read_image('')

standardizer = staintools.BrightnessStandardizer()
stain_norm = staintools.StainNormalizer(method='macenko')
stain_norm.fit(st)

base_dir = ''
target_dir = ''

dirnames = sorted(os.listdir(base_dir))


for dir_name in dirnames:
    print("")
    print("#####")
    print("Working with slide:", dir_name)
    print("#####")
    fnames = sorted(os.listdir(base_dir + dir_name))

    for filename in fnames:
        im = Image.open(base_dir + dir_name + "/" + filename)
        im = np.array(im)
        im = standardizer.transform(im)
        try:
            im = stain_norm.transform(im)
            print(filename, "ready")
        except:
            print("There was problem normalizing")
        im = Image.fromarray(im)
        filename_new = target_dir + "/" + dir_name + "/" + filename
        im.save(filename_new, quality = 90)


