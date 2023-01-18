'''
Script for processing of whole slide images
with a trained algorithm
wsi_process is the main script for analysis
For details of implementation and additional
details to script construction see our repositorium:
https://github.com/gagarin37/deep_learning_pca
'''



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: drpusher
"""


###PATCH SIZE
#Now defined automatically
#MODEL PATCH SIZE
m_p_s = 220
#HEATMAP PATCH SIZE
hmap_p_s = 10
###PATH_TO_SLIDE_DIR
slide_dir = ''
###CASE NAME (for statistics)
case_name = ''
###PATH_TO_OUTPUT_DIR
output_dir = ''
###MODEL_PATH Tumor vs Normal
model_tvn_dir = ''
model_tvn_name = ''
###OVERLAY FACTOR: REDUCTION OF ORIGINAL SLIDE
overlay_factor = 4
###FLAGs
env_flag = True  #Single environment analysis (OPTIONAL)
C1_flag = True #Basic mode of analysis of single patches
C8_flag = False #C8 strategy (OPTIONAL)

#Maps
maps_dir = ''


# =============================================================================
# 1. LIBRARIES
# =============================================================================
#from tensorflow.keras.models import load_model
from openslide import open_slide
from PIL import Image
import os
from wsi_slide_info import slide_info
#from wsi_process import slide_process
from wsi_maps import make_wsi_map_bin, make_wsi_heatmap, make_overlay
import numpy as np
import timeit
import copy
from wsi_single_env import check_single_environment  


# =========
# ====================================================================
# Inference Loop
# =============================================================================

slide_names = sorted(os.listdir(slide_dir))

path_result = output_dir + case_name + "_stats_per_slide.txt"

output_header = "slide_name" + "\t" + "obj_power" + "\t" + "mpp" + "\t" 
output_header = output_header + "patch_n_h_l0" + "\t" + "patch_n_w_l0" + "\t" 
output_header = output_header + "patch_overall" + "\t" + "patch_tissue" + "\t" 
output_header = output_header + "height" + "\t" + "width" + "\t"
output_header = output_header + "patch_tumor" + "\t" + "patch_regr" + "\t"
output_header = output_header + "patch_tumor_SINGLE" + "\t" + "patch_regr_SINGLE" + "\t"
output_header = output_header + "time"

output_header = output_header + "\n"


results = open (path_result, "a+")
results.write(output_header)
results.close()
try:
    os.mkdir(output_dir + '/tvn_maps')
    os.mkdir(output_dir + '/tvn_c1_heatmap')
    os.mkdir(output_dir + '/tvn_c1_overlay')     
    
    if env_flag == True:
        os.mkdir(output_dir + '/tvn_heatmap_SINGLE')
        os.mkdir(output_dir + '/tvn_overlay_SINGLE')
        
    if C8_flag == True:
        os.mkdir(output_dir + '/tvn_c8')

except:
    print ('Directories already there')

for slide_name in slide_names:
    start = timeit.default_timer()
    

    print("Processing:", slide_name)
    # =============================================================================
    # 3. OPEN/PROCESS WSI, EXTRACT DATA, PRESENT BASIC DATA ABOUT SLIDE
    # =============================================================================
    #Open slide
    path_slide = os.path.join(slide_dir, slide_name)
    slide = open_slide(path_slide)
    
    ###Print Meta-Data, Calculate number of patches (width and height), generate thumbnail
    #thumbnail is a numpy array
    #where every pixel = area of patch size
    #We standardize brightness to extend upper pixel
    #values in grayscale image to 255
    #In this situation <250 is a well threshold, even for fat tisue
    #to discriminate between background and tissue
    p_s, thumbnail, patch_n_w_l0, patch_n_h_l0, mpp, obj_power, count_patch_process = slide_info(slide)
    
     
    # =============================================================================
    # 4. WSI PROCESSING TO GENERATE END IMAGE
    # =============================================================================
    #Return end_image and map with predictions
  
    map_path = maps_dir + slide_name + "_map_C1_preds.npy"
    wsi_map_preds_C1 = np.load(map_path)
    
    
    if C1_flag == True:
        # =============================================================================
        # 5. MAKE BIN MAP FROM PREDS MAP
        # =============================================================================
        wsi_map_bin_C1 = make_wsi_map_bin (wsi_map_preds_C1, patch_n_w_l0, patch_n_h_l0)
        
        wsi_map_preds_name_C1 = output_dir + "tvn_maps/" + slide_name + "_map_C1_preds.npy"
        np.save(wsi_map_preds_name_C1, wsi_map_preds_C1)
        
        wsi_map_bin_C1_name = output_dir + "tvn_maps/" + slide_name + "_map_C1_bin.npy"
        np.save(wsi_map_bin_C1_name, wsi_map_bin_C1)
        
        # =============================================================================
        # 7. MAKE AND SAVE HEATMAP FROM BINARY MAP for C8
        # =============================================================================
        #Load BIN map
        #wsi_map_bin = np.load('/media/dr_pusher/DATA4TB/_WSI_scans/DL/PCA/PCA Cases Stanford/_maps/part2/21689 D4.svs_map_C8_bin.npy')
        
        wsi_heatmap_C1, counter_tumor, counter_regr = make_wsi_heatmap (wsi_map_bin_C1, hmap_p_s)
                            
        #Save WSI HEATMAP native
        #wsi_heatmap = np.uint8(wsi_heatmap)
        wsi_heatmap_im_C1 = Image.fromarray(wsi_heatmap_C1)
        wsi_heatmap_im_C1_name = output_dir + "tvn_c1_heatmap/" + slide_name + "_heatmap_C1.png"
        wsi_heatmap_im_C1.save(wsi_heatmap_im_C1_name)
        
        
        # =============================================================================
        # 8. MAKE AND SAVE OVERLAY for C8: HEATMAP ON REDUCED AND CROPPED SLIDE CLON
        # =============================================================================
        overlay_C1 = make_overlay (slide, wsi_heatmap_im_C1, p_s, patch_n_w_l0, patch_n_h_l0, overlay_factor)
        
        #Save overlaid image
        overlay_im_C1 = Image.fromarray(overlay_C1)
        overlay_im_C1_name = output_dir + "tvn_c1_overlay/" + slide_name + "_overlay_C1.jpg"
        overlay_im_C1.save(overlay_im_C1_name)
        
    
    # =============================================================================
    # 9. CHECK SINGLES (ENVIRONMENT CHECK)
    # =============================================================================
    if env_flag == True:
        print("Starting Single Environment Check") 
        wsi_map_bin_single = copy.copy(wsi_map_bin_C1)
        wsi_map_bin_single = check_single_environment (slide, wsi_map_bin_single, p_s, m_p_s)
    
        #Save tumor map (binary) for SINGLE
        wsi_map_bin_single_name = output_dir + "tvn_maps/" + slide_name + "_map_bin_SINGLE.npy"
        np.save(wsi_map_bin_single_name, wsi_map_bin_single)
        
        
        # =============================================================================
        # 10. MAKE AND SAVE HEATMAP FROM BINARY MAP for C8_SINGLE
        # =============================================================================
        #Load BIN map SINGLE
        #wsi_map_bin_single = np.load('prostate-024.svs_map_C8_binary_single.npy')
        wsi_heatmap_single, counter_tumor_SINGLE, counter_regr_SINGLE = make_wsi_heatmap (wsi_map_bin_single, hmap_p_s)
                            
        #Save WSI HEATMAP native
        #wsi_heatmap_single = np.uint8(wsi_heatmap_single)
        wsi_heatmap_single_im = Image.fromarray(wsi_heatmap_single)
        wsi_heatmap_single_im_name = output_dir + "tvn_heatmap_SINGLE/" + slide_name + "_heatmap_SINGLE.png"
        wsi_heatmap_single_im.save(wsi_heatmap_single_im_name)
        
        # =============================================================================
        # 11. MAKE AND SAVE OVERLAY for C8_SINGLE: HEATMAP ON REDUCED AND CROPPED SLIDE CLON
        # =============================================================================
        overlay_single = make_overlay (slide, wsi_heatmap_single_im, p_s, patch_n_w_l0, patch_n_h_l0, overlay_factor)
        
        #Save overlaid image
        overlay_single_im = Image.fromarray(overlay_single)
        overlay_single_im_name = output_dir + "tvn_overlay_SINGLE/" + slide_name + "_overlay_SINGLE.jpg"
        overlay_single_im.save(overlay_single_im_name)
  

    #Timer stop
    stop = timeit.default_timer()
       
    #Write down per slide result
    #Basic data about slide (size, pixel size, objective power, height, width)
    output_temp = slide_name + "\t" + str(obj_power) + "\t" + str(mpp) + "\t" 
    output_temp = output_temp + str(patch_n_h_l0) + "\t" + str(patch_n_w_l0) + "\t" 
    output_temp = output_temp + str(patch_n_h_l0 * patch_n_w_l0) + "\t" + str(count_patch_process) + "\t" 
    output_temp = output_temp + str(patch_n_h_l0 * p_s) + "\t" + str(patch_n_w_l0 * p_s) + "\t" 
    output_temp = output_temp + str(counter_tumor) + "\t" + str(counter_regr) + "\t"
    output_temp = output_temp + str(counter_tumor_SINGLE) + "\t" + str(counter_regr_SINGLE) + "\t"
    output_temp = output_temp + str(round((stop - start)/60, 1))
    
    output_temp = output_temp + "\n"
    
    results = open (path_result, "a+")
    results.write(output_temp)
    results.close()




