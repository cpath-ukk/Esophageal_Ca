#EXTRACTION OF META-DATA FROM SLIDE

from PIL import Image
from wsi_stain_norm import standardizer
import numpy as np

def slide_info (slide):
    #Objective power
    obj_power = slide.properties ["openslide.objective-power"]
    
    #Microne per pixel
    mpp = float(slide.properties ["openslide.mpp-x"])
    p_s = 880
    
    #Vendor
    vendor = slide.properties ["openslide.vendor"]
    
    #Extract and save dimensions of level [0]
    dim_l0 = slide.level_dimensions[0]
    w_l0 = dim_l0 [0]
    h_l0 = dim_l0 [1]
    
    #Calculate number of patches to process
    patch_n_w_l0 = int(w_l0 / p_s)
    patch_n_h_l0 = int(h_l0 / p_s)
    
    #Output BASIC DATA
    print ("")
    print ("Basic data about processed whole-slide image")
    print ("")
    print ("Vendor: ", vendor)
    print ("Scan magnification: ", obj_power)
    print ("Microns per pixel:", mpp)
    print ("Height: ", h_l0)
    print ("Width: ", w_l0)
    print ("Patch size: ", p_s, "x", p_s)
    print ("Width: number of patches: ", patch_n_w_l0)
    print ("Height: number of patches: ", patch_n_h_l0)
    print ("Overall number of patches: ", patch_n_w_l0 * patch_n_h_l0)
    
    #Thumbnail, where every pixel = area of patch size
    #We standardize brightness to extend upper pixel
    #values in grayscale image to 255 through BrightnessStandardizer in staintools
    #In this situation <245 is a well threshold, even for fat tisue
    #to discriminate between background and tissue
    thumbnail = slide.get_thumbnail((int(w_l0/p_s),int(h_l0/p_s)))
    thumbnail = np.array(thumbnail)
    thumbnail = standardizer.transform(thumbnail)
    thumbnail = Image.fromarray(thumbnail)
    thumbnail = thumbnail.convert('L')
    thumbnail = np.array(thumbnail)
    
    for i in range(0,10):
        if thumbnail.shape [0] != patch_n_h_l0:
            th_last_row = thumbnail [thumbnail.shape [0]-1,:]
            th_last_row = np.expand_dims (th_last_row, axis=0)        
            thumbnail = np.append(thumbnail, th_last_row, axis = 0)
            print ("Thumbnail size was optimized")
            print ("Thumbnail size:", thumbnail.shape)
        
        if thumbnail.shape [1] != patch_n_w_l0:
            th_last_row = thumbnail [:,thumbnail.shape [1]-1]
            th_last_row = np.expand_dims (th_last_row, axis=1)        
            thumbnail = np.append(thumbnail, th_last_row, axis = 1)
            print ("Thumbnail size was optimized")
            print ("Thumbnail size:", thumbnail.shape)

    
    
    #Count number of patches to process using the threshold (currently 248)
    thumb_count = np.reshape(thumbnail, thumbnail.shape[0]*thumbnail.shape[1])
    count = 0
    for i in range(len(thumb_count)):
        if thumb_count[i] < 248:
            count = count + 1
    
    print ("Number of patches to process: ", count)        
        
    #return(thumbnail as array, patch_n_w_l0, patch_n_h_l0)
    return(p_s, thumbnail, patch_n_w_l0, patch_n_h_l0, mpp, obj_power, count)
