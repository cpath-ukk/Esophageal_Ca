from PIL import Image, ImageOps
from statistics import median
import numpy as np


#Function for processing of 8 patch derivates
def gateway_median (model, patch):
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
    pred_stack = np.vstack((pred(model, base),
                            pred(model, r90),
                            pred(model, r180),
                            pred(model, r270),
                            pred(model, r90_VF),
                            pred(model, r270_VF),
                            pred(model, VF),
                            pred(model, HF)))
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
    preds_med = np.array([pred_1, pred_2, pred_3,\
                          pred_4, pred_5, pred_6,\
                              pred_7, pred_8, pred_9,\
                                  pred_10, pred_11])
    return preds_med

#Function for prediction for single patches
def pred (model, patch):
    #IMAGE TO ARRAY, PREPROCESSING
    patch = np.array(patch).astype(float)
    patch = np.expand_dims(patch, axis = 0)
    patch /= 255.
    #prediction from model
    preds = model.predict(patch)
    return preds
###############################