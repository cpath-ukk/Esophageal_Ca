import numpy as np
from wsi_heatmaps import gen_heatmaps
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2

#RECORD FUNCTION FOR PREDS MAPS
def record_map_preds (wsi_map_preds, hi, wi, preds):
    wsi_map_preds [hi,wi] = preds


#MAKE BINARY MAP
def make_wsi_map_bin (wsi_map_preds, patch_n_w_l0, patch_n_h_l0):
    wsi_map_bin = np.zeros((patch_n_h_l0, patch_n_w_l0, 1) , dtype=np.int16)
    for he in range(wsi_map_preds.shape[0]):
        for wi in range(wsi_map_preds.shape[1]):
            if sum(wsi_map_preds [he,wi,0:3]) > 0:
                entity = np.argmax(wsi_map_preds [he,wi])
                if entity == 9:
                    if wsi_map_preds [he,wi,9] < 0.90:
                        entity = 9999
                if entity != 4 and entity != 9:
                    if wsi_map_preds [he,wi,4] > 0.05:
                        entity = 4
            else:
                entity = 9999 # background without tissue
            wsi_map_bin [he,wi] = entity
    return wsi_map_bin


#MAKE HEATMAP AS NUMPY IMAGE
def make_wsi_heatmap (wsi_map_bin, hmap_p_s):
    tumor, regress, advent, lam_prop, mucosa, musc, submucosa, blank_patch = gen_heatmaps(hmap_p_s)
    heatmap_blank = np.array(blank_patch)
    counter_tumor = 0
    counter_regr = 0
    for he in range(wsi_map_bin.shape[0]):
            for wi in range(wsi_map_bin.shape[1]):
                if wsi_map_bin [he,wi] == 9999:
                    heatmap = heatmap_blank
                elif wsi_map_bin [he,wi] == 0: #advent
                    heatmap = heatmap_blank
                elif wsi_map_bin [he,wi] == 1: #lamina propria
                    heatmap = heatmap_blank
                elif wsi_map_bin [he,wi] == 2: #MUSC_MUC
                    heatmap = heatmap_blank
                elif wsi_map_bin [he,wi] == 3: #MUSC_PROP
                    heatmap = heatmap_blank
                elif wsi_map_bin [he,wi] == 4: #regression
                    heatmap = regress
                    counter_regr += 1
                elif wsi_map_bin [he,wi] == 5: #SH_MAG
                    heatmap = heatmap_blank
                elif wsi_map_bin [he,wi] == 6: #SH_OES
                    heatmap = heatmap_blank
                elif wsi_map_bin [he,wi] == 7: #SUB_GL
                    heatmap = heatmap_blank
                elif wsi_map_bin [he,wi] == 8: #SUBMUC
                    heatmap = heatmap_blank
                elif wsi_map_bin [he,wi] == 9: #TUMOR
                    heatmap = tumor
                    counter_tumor += 1
                elif wsi_map_bin [he,wi] == 10: #ULCUS
                    heatmap = heatmap_blank            
                
                if (wi==0):
                    temp_image = heatmap
                else:
                    temp_image = np.concatenate((temp_image, heatmap), axis=1)
           
            if (he==0):
                end_image = temp_image
            else:
                end_image = np.concatenate((end_image, temp_image), axis=0)
            
            del temp_image
    return end_image, counter_tumor, counter_regr




#MAKE OVERLAY: HEATMAP ON REDUCED AND CROPPED SLIDE CLON
def make_overlay (slide, wsi_heatmap_im, p_s, patch_n_w_l0, patch_n_h_l0, overlay_factor):
    
    w_l0, h_l0 = slide.level_dimensions[0]
    
    slide_reduced = slide.get_thumbnail((w_l0/overlay_factor,h_l0/overlay_factor))
    
    hei = patch_n_h_l0 * p_s / overlay_factor
    wid = patch_n_w_l0 * p_s / overlay_factor
    
    area = (0,0, wid, hei)
    slide_reduced_crop = slide_reduced.crop(area)
    
    heatmap_temp = wsi_heatmap_im.resize(slide_reduced_crop.size, Image.ANTIALIAS)          
    overlay = cv2.addWeighted(np.array(slide_reduced_crop),0.7,np.array(heatmap_temp),0.3,0)
    return (overlay)
