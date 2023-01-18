#MAIN LOOP TO PROCESS WSI
import numpy as np
from PIL import Image
from wsi_maps import record_map_preds
import staintools
from wsi_stain_norm import standardizer, stain_norm
#from wsi_c8_functions import gateway_median

# =============================================================================
# C8 version
# =============================================================================

def slide_process (model, slide, patch_n_w_l0, patch_n_h_l0, p_s, m_p_s, thumbnail):
        
    #CREATE CHUNK FOR PRED_MAP
    wsi_map_preds = np.zeros((patch_n_h_l0, patch_n_w_l0, 11) , dtype=np.float32)
    wsi_map_preds_C1 = np.zeros((patch_n_h_l0, patch_n_w_l0, 11) , dtype=np.float32)
    
    #INITIALIZE STAIN NORMALIZER
    st = staintools.read_image('images/ukk_3.jpg')
    stain_norm.fit(st)   
    
    #Start loop
    for hi in range(patch_n_h_l0):
        h = hi*p_s + 1
        if (hi==0):
            h = 0
        print("Current cycle ", hi+1, " of ", patch_n_h_l0)
        for wi in range(patch_n_w_l0):
            w = wi*p_s+1
            if (wi==0):
                w = 0
            
            if thumbnail [hi,wi] < 248 and thumbnail [hi,wi] > 70:
            
                #Generate patch
                work_patch = slide.read_region((w,h), 0, (p_s,p_s))
                work_patch = work_patch.convert('RGB')
        
                #Resize to model patch size
                work_patch = work_patch.resize((m_p_s,m_p_s), Image.ANTIALIAS)
                
                wp_temp = np.array(work_patch)
        
                try:
                    wp_temp = standardizer.transform(wp_temp)
                    wp_temp = stain_norm.transform(wp_temp)
                except:
                    print("Stain normalization failed")
                finally:    
                    #im_sn = Image.fromarray(wp_temp)
                    
                    wp_temp = np.float32(wp_temp)
                
                
                #PREPROCESSING
                wp_temp = np.expand_dims(wp_temp, axis = 0)
                wp_temp /= 255.

                #prediction from model
                preds = model.predict(wp_temp)
                
                #record predictions into map
                record_map_preds(wsi_map_preds, hi, wi, preds)
                record_map_preds(wsi_map_preds_C1, hi, wi, preds)
                '''
                #tumor regression (tumor = 9, regression = 4)
                if (preds [0,9]) < 0.2 or (preds [0,4]) < 0.2:
                    print("making C8")
                    preds_C8 = gateway_median(model, im_sn)
                    record_map_preds(wsi_map_preds, hi, wi, preds_C8)
                '''        

    return (wsi_map_preds, wsi_map_preds_C1)