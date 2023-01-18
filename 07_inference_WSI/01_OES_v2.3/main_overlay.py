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
slide_dir = '/media/dr_pusher/OES_1/02_OES_test_cases_P2_with_AI/ALL/'
###CASE NAME (for statistics)
case_name = 'P1_P4_SN_Old_V50_V5E2_thresh_TU0.90_REGR0.05_LARGE'
###PATH_TO_OUTPUT_DIR
output_dir = '/home/dr_pusher/Dropbox/temp_OES/24_100cases_SNold_LARGE_FILES/02_OUTPUT/'
###MODEL_PATH Tumor vs Normal
model_tvn_dir = '/home/dr_pusher/Dropbox/temp_OES/24_100cases_SNold/03_P3_V50_V5E2_thresh_TU0.98_REGR0.2/v50/'
model_tvn_name = 'V50_V5E2.h5'
###OVERLAY FACTOR: REDUCTION OF ORIGINAL SLIDE
overlay_factor = 4
###FLAGs
env_flag = True
C1_flag = True
C8_flag = False   

#Maps
maps_dir = '/home/dr_pusher/Dropbox/temp_OES/24_100cases_SNold_LARGE_FILES/01_P1_P4_V50_V5E2_thresh_TU0.90_REGR0.05_MAPS/'


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
# LOOOOOOOOOOOP
# =============================================================================

#import tensorflow as tf


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
  
    #map_path = maps_dir + slide_name + "_map_C1_preds.npy"
    #wsi_map_preds_C1 = np.load(map_path)
    
    
    # =============================================================================
    # 9. CHECK SINGLES (ENVIRONMENT CHECK)
    # =============================================================================
    
    print("Starting Single Environment Check") 
    #wsi_map_bin_single = copy.copy(wsi_map_bin_C1)
    map_path_single = maps_dir + slide_name + "_map_bin_SINGLE.npy"
    wsi_map_bin_single = np.load(map_path_single)
    #wsi_map_bin_single = check_single_environment (slide, wsi_map_bin_single, p_s, m_p_s)

    #Save tumor map (binary) for SINGLE
    #wsi_map_bin_single_name = output_dir + "tvn_maps/" + slide_name + "_map_bin_SINGLE.npy"
    #np.save(wsi_map_bin_single_name, wsi_map_bin_single)
    
    
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
  
