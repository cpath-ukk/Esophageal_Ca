'''
Prepare images for CycleGAN stale transfer models
training
'''


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 00:19:23 2020

@author: dr_pusher
"""
from os import listdir
from numpy import asarray
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from numpy import savez_compressed, vstack


#Load images
def load_images(path, size=(256,256)):
    im_list = list()
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = np.array(pixels)
        # split into satellite and map
        im_list.append(pixels)
    return asarray(im_list)


### Load and save dataset
# dataset path
#A - TRAIN, B - WNS
path_A = '/media/dr_pusher/WORK1/1_dl/3_projects/02_OES/08_Cycle_GAN/01_TRAIN_VALSET2_WNS/01_patches_TRAIN_SN_ukk3/'
path_B = '/media/dr_pusher/WORK1/1_dl/3_projects/02_OES/08_Cycle_GAN/05_TRAIN_UKKL2/'
# load dataset
dataA1 = load_images(path_A + 'train/')
dataA2 = load_images(path_A + 'test/')

dataB1 = load_images(path_B + 'train/')
dataB2 = load_images(path_B + 'test/')

dataA = vstack((dataA1, dataA2))
print('Loaded dataA: ', dataA.shape)

dataB = vstack((dataB1, dataB2))
print('Loaded dataB: ', dataB.shape)


# save as compressed numpy array
filename = '/media/dr_pusher/WORK1/1_dl/3_projects/02_OES/08_Cycle_GAN/05_TRAIN_UKKL2/train_A_UKKL2_B_256.npz'
savez_compressed(filename, dataA, dataB)
print('Saved dataset: ', filename)


### Load and plot the prepared dataset
from numpy import load
from matplotlib import pyplot
# load the dataset
data = load('/media/dr_pusher/WORK1/1_dl/3_projects/02_OES/08_Cycle_GAN/05_TRAIN_UKKL2/train_A_UKKL2_B_256.npz')
dataA, dataB = data['arr_0'], data['arr_1']
print('Loaded: ', dataA.shape, dataB.shape)
# plot source images
n_samples = 5
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(dataA[i].astype('uint8'))
# plot target image
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + n_samples + i)
	pyplot.axis('off')
	pyplot.imshow(dataB[i].astype('uint8'))
pyplot.show()