import numpy as np
from PIL import Image

#FUNCTION: analysis of environment of single tumor patches to generate new binary map
def check_single_environment (slide, wsi_map_bin_f, p_s, m_p_s):
    #Write down coordinates of single tumor patches
    for he in range(wsi_map_bin_f.shape[0]-1): # -1 because we can not analyse environment if "tumor" patches are in the last row
        for wi in range(wsi_map_bin_f.shape[1]-1):
            if wsi_map_bin_f [he,wi] == 9:
                if test_env (wsi_map_bin_f, he, wi) == False: # False = no further tumor patches in environment
                    #Starting re-evaluation of patch 
                    wsi_map_bin_f [he, wi] = 9999
            if wsi_map_bin_f [he,wi] == 4:
                if test_env_reg (wsi_map_bin_f, he, wi) == False: # False = no further tumor patches in environment
                    #Starting re-evaluation of patch 
                    wsi_map_bin_f [he, wi] = 9999
    return wsi_map_bin_f
                    


#FUNCTION: Test if there are further tumor patches in environment.
def test_env (wsi_map_bin_f, he, wi):
    #counter of positive neigbours
    counter = 0
    #define coordinates of neighboor patches
    he_list = [he - 1, he, he + 1, he]
    wi_list = [wi, wi + 1, wi, wi - 1]
    #Get status of neigbours
    for i in range(4):
        if is_tumor(wsi_map_bin_f, he_list[i], wi_list[i]) == True:
            counter = counter + 1
    #Return True is environment is positive for tumor patches
    if counter > 0:
        return True
    else:
        return False

#FUNCTION support for test_env(): is patch a tumor?
def is_tumor (wsi_map_bin_f, he, wi):
    if wsi_map_bin_f [he, wi] == 9:
        return True
    else:
        return False

def test_env_reg (wsi_map_bin_f, he, wi):
    #counter of positive neigbours
    counter = 0
    #define coordinates of neighboor patches
    he_list = [he - 1, he, he + 1, he]
    wi_list = [wi, wi + 1, wi, wi - 1]
    #Get status of neigbours
    for i in range(4):
        if is_reg(wsi_map_bin_f, he_list[i], wi_list[i]) == True:
            counter = counter + 1
    #Return True is environment is positive for tumor patches
    if counter > 0:
        return True
    else:
        return False

#FUNCTION support for test_env(): is patch a tumor?
def is_reg (wsi_map_bin_f, he, wi):
    if wsi_map_bin_f [he, wi] == 4:
        return True
    else:
        return False