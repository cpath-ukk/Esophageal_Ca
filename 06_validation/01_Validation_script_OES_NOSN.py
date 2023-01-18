'''
Validation script for test datasets
Without any forms of stain normalization
'''


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dr_pusher
"""

#0. SET PARAMETERS
###Path to directory with models 
model_dir = ''
###DIRECTORY WITH IMAGES
base_dir = ''
###OUTPUT DIRECTORY FOR RESULT FILES
result_dir = ''
### List of model patch sizes for models in model_dir (alphabetically sorted)
m_p_s_list = [220, 220, 220, 220, 220]

#1. IMPORT LIBRARIES
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps
from statistics import median


#2. GENERATE LISTS AND NAMES
###GENERATE LIST OF MODELS
model_names = sorted(os.listdir(model_dir))
###MODEL PATCH SIZES
class_names = sorted(os.listdir(base_dir))


#4. FUNCTIONS
#C8 main
def gateway_median (patch):
    #base
    base = patch #1
    #rotations
    r90 = patch.rotate(90) #2
    r180 = patch.rotate(180) #3
    r270 = patch.rotate(270) #4
    #flip/rotations
    r90_VF = ImageOps.flip(r90) #5
    r270_VF = ImageOps.flip(r270) #6
    #flips
    VF = ImageOps.flip(base) #7
    HF = base.transpose(Image.FLIP_LEFT_RIGHT) #8 
    #calculate score as arythmetic mean
    pred_stack = np.vstack((pred(base),pred(r90),pred(r180),pred(r270),pred(r90_VF),pred(r270_VF),pred(VF),pred(HF)))
    pred_1 = median(pred_stack[0:8,0])
    pred_2 = median(pred_stack[0:8,1])
    pred_3 = median(pred_stack[0:8,2])
    pred_4 = median(pred_stack[0:8,3])
    pred_5 = median(pred_stack[0:8,4])
    pred_6 = median(pred_stack[0:8,5])
    pred_7 = median(pred_stack[0:8,6])
    pred_8 = median(pred_stack[0:8,7])
    pred_9 = median(pred_stack[0:8,8])
    pred_10 = median(pred_stack[0:8,9])
    pred_11 = median(pred_stack[0:8,10])
    preds_med = np.array([pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9, pred_10, pred_11])
    return preds_med 

#Function for prediction for single patches (used in C8)
def pred (patch):
    #IMAGE TO ARRAY, PREPROCESSING
    patch = np.array(patch)
    patch = np.expand_dims(patch, axis = 0)
    patch /= 255.
    #prediction from model
    preds = model.predict(patch)
    return preds

def processor (m_p_s, class_name, output_C1, output_ALL):
    work_dir = base_dir + class_name
    
    fnames = sorted(os.listdir(work_dir))
    
    counter = [0,0,0,0,0,0,0,0,0,0,0]
	
    for fname in fnames:
        filename = os.path.join(work_dir, fname)
        im = image.load_img(filename)
        #im = im.resize((m_p_s,m_p_s), Image.ANTIALIAS)
        im = np.array(im)
                
        im = np.float32(im)
        
        x = np.expand_dims(im, axis = 0)
        x /= 255.
        
        #prediction
        preds = model.predict(x)
        pr_1 = str(round(preds[0,0],3))
        pr_2 = str(round(preds[0,1],3))
        pr_3 = str(round(preds[0,2],3))
        pr_4 = str(round(preds[0,3],3))
        pr_5 = str(round(preds[0,4],3))
        pr_6 = str(round(preds[0,5],3))
        pr_7 = str(round(preds[0,6],3))
        pr_8 = str(round(preds[0,7],3))
        pr_9 = str(round(preds[0,8],3))
        pr_10 = str(round(preds[0,9],3))
        pr_11 = str(round(preds[0,10],3))
		
        entity = np.argmax(preds)
        #Output of C1
        output = fname + "\t" + pr_1 + "\t" + pr_2 + "\t" + pr_3 + "\t"
        output = output + pr_4 + "\t" + pr_5 + "\t" + pr_6 + "\t"
        output = output + pr_7 + "\t" + pr_8 + "\t" + pr_9 + "\t"
        output = output + pr_10 + "\t" + pr_11 + "\t" + str(entity+1) + "\n"

		
        counter [entity] = counter [entity] + 1
        #Write down output of C1
        results = open (output_C1, "a+")
        results.write(output)
        results.close()
    
    stat_ALL = str(counter) + "\t" + class_name + "\n"
    results = open (output_ALL, "a+")
    results.write(stat_ALL)
    results.close()
    return counter

def accuracy_calc (counter_pool, output_ACCU): #[height, width]
	
	ALL = np.sum(counter_pool[:, :])
	
	TP_tu = counter_pool [9,9]
	FP_tu = np.sum(counter_pool[0:9,9]) + counter_pool[10,9]
	FN_tu = np.sum(counter_pool[9,0:9]) + counter_pool[9,10]
	ALL_tu = np.sum(counter_pool[9,:])
	TN_tu = ALL - TP_tu - FN_tu - FP_tu
	
	
	TP_regr = counter_pool [4,4]
	FP_regr = np.sum(counter_pool[0:4,4]) + np.sum(counter_pool[5:,4])
	FN_regr = np.sum(counter_pool[4,0:4]) + np.sum(counter_pool[4,5:])
	ALL_regr = np.sum(counter_pool[4,:])
	TN_regr = ALL - TP_regr - FN_regr - FP_regr
	
	
	write_down_ACCU_head = "Class" + "\t" + "TP"
	write_down_ACCU_head = write_down_ACCU_head + "\t" + "FP"
	write_down_ACCU_head = write_down_ACCU_head + "\t" + "FN"
	write_down_ACCU_head = write_down_ACCU_head + "\t" + "TN" 
	write_down_ACCU_head = write_down_ACCU_head + "\t" + "ALL_class" 
	write_down_ACCU_head = write_down_ACCU_head + "\t" + "ALL_dataset"
	write_down_ACCU_head = write_down_ACCU_head + "\t" + "Sensitivity"
	write_down_ACCU_head = write_down_ACCU_head + "\t" + "Specificity"
	write_down_ACCU_head = write_down_ACCU_head + "\n"
	
	write_down_ACCU = write_down_ACCU_head + "TUMOR"
	write_down_ACCU = write_down_ACCU + "\t" + str(TP_tu)
	write_down_ACCU = write_down_ACCU + "\t" + str(FP_tu)
	write_down_ACCU = write_down_ACCU + "\t" + str(FN_tu)
	write_down_ACCU = write_down_ACCU + "\t" + str(TN_tu)
	write_down_ACCU = write_down_ACCU + "\t" + str(ALL_tu)
	write_down_ACCU = write_down_ACCU + "\t" + str(ALL)
	write_down_ACCU = write_down_ACCU + "\t" + str(TP_tu/(TP_tu + FN_tu)) # sensitivity
	write_down_ACCU = write_down_ACCU + "\t" + str(TN_tu/(TN_tu + FP_tu)) # specificity
	write_down_ACCU = write_down_ACCU + "\n"
	
	write_down_ACCU = write_down_ACCU + "TU_REGR"
	write_down_ACCU = write_down_ACCU + "\t" + str(TP_regr)
	write_down_ACCU = write_down_ACCU + "\t" + str(FP_regr)
	write_down_ACCU = write_down_ACCU + "\t" + str(FN_regr)
	write_down_ACCU = write_down_ACCU + "\t" + str(TN_regr)
	write_down_ACCU = write_down_ACCU + "\t" + str(ALL_regr)
	write_down_ACCU = write_down_ACCU + "\t" + str(ALL)
	write_down_ACCU = write_down_ACCU + "\t" + str(TP_regr/(TP_regr + FN_regr)) # sensitivity
	write_down_ACCU = write_down_ACCU + "\t" + str(TN_regr/(TN_regr + FP_regr)) # specificity
	write_down_ACCU = write_down_ACCU + "\n"
	
	results = open (output_ACCU, "a+")
	results.write(write_down_ACCU)
	results.close()
	

  
#MAIN LOOP
i = 0
for model_name in model_names:
   
    print("Loading model: ", model_name, " ...")
    path_model = os.path.join(model_dir, model_name)
    model = load_model(path_model)
    print("Model loaded")
    os.mkdir(result_dir + model_name)
    counter_pool = []
    y = 0
    for class_name in class_names:
        output_C1 = result_dir + model_name + "/" + class_name + "_C1.txt"
        output_ALL = result_dir + model_name + "/" + model_name + "_ALL.txt"
        output_ACCU = result_dir + model_name + "/" + model_name + "_ACCURACY.txt"
        counter = processor(m_p_s_list[i], class_name, output_C1, output_ALL)
        counter = np.array(counter)
        if y == 0:
            counter_pool = counter
        else:
            counter_pool = np.vstack([counter_pool,counter])
			
        y = y + 1	   
    
    accuracy_calc (counter_pool, output_ACCU)
	
	
	#Increment of i for m_p_s
    i = i+1
    print("Ready! Going to the next model.")
    
    
    
