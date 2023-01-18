
'''
Validation script for test datasets
CycleGAN version of stain normalization
'''


################################


import tensorflow as tf


gpus = tf.config.list_physical_devices('GPU')


if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[1], 
            [tf.config.LogicalDeviceConfiguration(memory_limit=4000)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUS.")
    except RuntimeError as e:
        print(e)



with tf.device('/device:GPU:1'):
        
    
    #0. SET PARAMETERS
    ###Path to directory with models 
    model_dir = ''
    ###DIRECTORY WITH IMAGES
    base_dir = ''
    ###OUTPUT DIRECTORY FOR RESULT FILES
    result_dir = ''
    ###
    m_p_s_list = [220]
    ###
    cg_model_dir = ''
    cg_model_name = 'g_model_BtoA_026400.h5'
    
    from tensorflow_addons.layers import InstanceNormalization
    
    #1. IMPORT LIBRARIES
    from tensorflow.keras.models import load_model
    import os
    from tensorflow.keras.preprocessing import image
    import numpy as np
    from PIL import Image, ImageOps
    import cv2
    
    
    #2. GENERATE LISTS AND NAMES
    ###GENERATE LIST OF MODELS
    model_names = sorted(os.listdir(model_dir))
    ###MODEL PATCH SIZES
    class_names = sorted(os.listdir(base_dir))
    
    #Load CycleGAN model
    cust = {'InstanceNormalization': InstanceNormalization}
    path_cg_model = os.path.join(cg_model_dir, cg_model_name)
    cg_model = load_model(path_cg_model, cust)
    
    
    
    #4. FUNCTIONS
    def processor (m_p_s, class_name, output_C1, output_ALL):
        work_dir = base_dir + class_name
        
        fnames = sorted(os.listdir(work_dir))
        
        counter = [0,0,0,0,0,0,0,0,0,0,0]
    	
        for fname in fnames:
            filename = os.path.join(work_dir, fname)
            im = image.load_img(filename)
            #im = im.resize((256,256), Image.ANTIALIAS)
            im = np.array(im)
            #stain normalization
            im = (im - 127.5) / 127.5
            # reshape to 1 sample
            im = np.expand_dims(im, 0)
            gen_im = cg_model.predict(im)
                
            x = gen_im[0] * 127.5 + 127.5
            #resize to model size
                
            x = cv2.resize(x, dsize=(m_p_s,m_p_s), interpolation=cv2.INTER_CUBIC)
            
            x = np.expand_dims(x, axis = 0)
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
 




