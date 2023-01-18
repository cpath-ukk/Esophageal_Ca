#HEATMAP IMAGES

from tensorflow.keras.preprocessing import image
import numpy as np

def gen_heatmaps (m_p_s):
    #tumor
    tumor = image.load_img("images/tumor.jpg", target_size=(m_p_s, m_p_s))
    tumor = np.array(tumor)
    
    #
    regress = image.load_img("images/regress.jpg", target_size=(m_p_s, m_p_s))
    regress = np.array(regress)
    
    advent = image.load_img("images/advent.jpg", target_size=(m_p_s, m_p_s))
    advent = np.array(advent)
    
    lam_prop = image.load_img("images/lam_prop.jpg", target_size=(m_p_s, m_p_s))
    lam_prop = np.array(lam_prop)
    
    mucosa = image.load_img("images/mucosa.jpg", target_size=(m_p_s, m_p_s))
    mucosa = np.array(mucosa)
    
    musc = image.load_img("images/musc.jpg", target_size=(m_p_s, m_p_s))
    musc = np.array(musc)
    
    submucosa = image.load_img("images/submucosa.jpg", target_size=(m_p_s, m_p_s))
    submucosa = np.array(submucosa)
    
    #blank
    blank_patch = image.load_img("images/blank_patch.jpg", target_size=(m_p_s, m_p_s))
    
    return (tumor, regress, advent, lam_prop, mucosa, musc, submucosa, blank_patch)

