# example of using saved cycleGAN models for image translation
#based on https://machinelearningmastery.com/cyclegan-tutorial-with-keras/
from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import os
import numpy as np
from tifffile import imread, imwrite
from sklearn.metrics import mutual_info_score
from skimage.measure import compare_ssim as ssim
from sklearn.metrics.cluster import normalized_mutual_info_score
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

def predict(model_dir, images_all_dir, save_dir_2dmasks, save_dir_3dmasks, mode_, patch_size, _step):

    # load the models
    cust = {'InstanceNormalization': InstanceNormalization}

    print('Mode: {}'.format(mode_))

    if mode_ == 'train':

        images_names = ['PBS_new_ret1', 'sFLT1_ret1', 'VEGF_ret1', 'WT_AngII_ret1', 'PBS_new_ret1b', 'PBS_old_ret1','WT_AngII_ret2', 'VEGF_ret1b', 'Hemaxi_ICAM2', 'PBS_ret1-02', 
        'PBS_ret4', 'sFLT1_ret2', 'sFLT1_ret3', 
        'VEGF_ret3', 'WT_retina_5', 'Captopril_new_ret1_stiched.tif', 'Captopril_new_ret2_stiched.tif',
        'Captopril_new_ret3_stitched.tif', 'iROCK_ret3_stiched.tif','Maria_Captopril_ret6_stiched.tif', 'PBS_ret1-02-stiched.tif',
        'Ret4_Maria_Captopril_1-stiched.tif', 'Ret5_Maria_Captopril_2-stiched.tif',
        'RGDS-stitched-ret1.tif', 'RGDS-stitched-ret2.tif', 'RGDS-stitched-ret3.tif','RGDS-stitched-ret4.tif','sFlt1_ret4.tif',
        'sFlt1_ret5.tif', 'VEGF_ret4.tif', 'VEGF_ret5.tif', 'VEGF_ret6.tif', 'VEGF_ret7.tif', 
        'FLIJ10_TG_stiching_2ndExp_1.tif', 'FLIJ11_TG_stiching_2ndExp_1.tif', 
          'FLIJ7_TG_stiching_1.tif', 'FLIJ7_TG_stiching_2ndExp_1.tif', 
          'FLIJ8_WT_stiching_2ndExp_1.tif', 'FLIJ9_TG_stiching_1.tif', 
          'FLIJ9_TG_stiching_2ndExp_1.tif', 'FLIO1a_TG_stiching_2ndExp_2.tif', 
          'FLIO1b_TG_stiching_2ndExp_1.tif', 'FLIO4_TG_stiching_2ndExp_1.tif', 
          'FLIO5_TG_stiching_2ndExp_1.tif', 'FLIO6_TG_stiching_1.tif',
          'FLIO6_TG_stiching_2.tif', 'FLIO6_TG_stiching_3.tif', 
           'FLIO8_TG_stiching_1.tif', 
          'FLIP1_TG_stiching_1.tif', 'FLIP1_TG_stiching_1Exp_1.tif',
          'FLIP1_TG_stiching_1Exp_2.tif', 'FLIP1_TG_stiching_1Exp_3.tif', 
          'FLIP1_TG_stiching_2.tif', 'FLIP5_WT_stiching_2ndExp_1.tif', 
          'yesretinas_a_WT1_C1.tif', 'yesretinas_b_WT2_C1.tif', 
          'yesretinas_c_WT3_C1.tif','yesretinas_e_KO1_C1.tif',
          'yesretinas_f_KO2_C1.tif','yesretinas_g_KO3_C1.tif',
          'yesretinas_h_KO4_C1.tif']

        masks2d_names = ['PBS_new_ret1_stiched_mask', 'sFlt1_ret2-02_mask', 'VEGF_ret1-02_mask', 'Wnt5aWT_AngII_Ret7_tile1_mask', 'PBS_new_ret1b_stiched_mask', 
        'PBS_old_ret1_mask','Wnt5aWT_AngII_Ret7_tile2_mask',
        'VEGF_Ret1_tile1_stitched_mask', 'hemaxi_icam_2d_mask', 'PBS_ret1-02-stiched_mask',  'PBS_ret4_stiched_mask', 
        'sFlt1_ret2-03_mask', 'sFLT1_ret3_mask',
        'VEGF_ret1-03_mask', 'Wnt5aWT_PBS_Ret5_tile1_mask', 'Captopril_new_ret1_mask.tif','Captopril_new_ret2_mask.tif',
        'Captopril_new_ret3_mask.tif', 'iROCK_ret3_mask.tif','Maria_Captopril_ret6_mask.tif', 'PBS_ret1-02_mask.tif','Ret4_Maria_Captopril_1_mask.tif',
        'Ret5_Maria_Captopril_2_mask.tif', 'RGDS-ret1_mask.tif', 'RGDS-ret2_mask.tif','RGDS-ret3_mask.tif', 'RGDS-ret4_mask.tif',
        'sFlt1_ret4_mask.tif','sFlt1_ret5_mask.tif', 'VEGF_ret4_mask.tif','VEGF_ret5_mask.tif', 'VEGF_ret6_mask.tif', 'VEGF_ret7_mask.tif',
        'FLIJ10_TG_stiching_2ndExp_1.tif', 'FLIJ11_TG_stiching_2ndExp_1.tif', 
          'FLIJ7_TG_stiching_1.tif', 'FLIJ7_TG_stiching_2ndExp_1.tif', 
          'FLIJ8_WT_stiching_2ndExp_1.tif', 'FLIJ9_TG_stiching_1.tif', 
          'FLIJ9_TG_stiching_2ndExp_1.tif', 'FLIO1a_TG_stiching_2ndExp_2.tif', 
          'FLIO1b_TG_stiching_2ndExp_1.tif', 'FLIO4_TG_stiching_2ndExp_1.tif', 
          'FLIO5_TG_stiching_2ndExp_1.tif', 'FLIO6_TG_stiching_1.tif',
          'FLIO6_TG_stiching_2.tif', 'FLIO6_TG_stiching_3.tif', 
           'FLIO8_TG_stiching_1.tif', 
          'FLIP1_TG_stiching_1.tif', 'FLIP1_TG_stiching_1Exp_1.tif',
          'FLIP1_TG_stiching_1Exp_2.tif', 'FLIP1_TG_stiching_1Exp_3.tif', 
          'FLIP1_TG_stiching_2.tif', 'FLIP5_WT_stiching_2ndExp_1.tif',
          'yesretinas_a_WT1_C1.tif', 'yesretinas_b_WT2_C1.tif', 
          'yesretinas_c_WT3_C1.tif','yesretinas_e_KO1_C1.tif',
          'yesretinas_f_KO2_C1.tif','yesretinas_g_KO3_C1.tif',
          'yesretinas_h_KO4_C1.tif']

        crop_sizes = [[4000,2700],  [-1000,-2200], [2000,3500], [0,0], [0,-2000], [-500, 3000], [0,0], [-1000, 4000], [0,0], [-3000, 0], [-2500,0],
         [0, 5000], [-500,2200], [-2500,0], [0, -3000], [-500,3500], [-500, -3000], [3700,3100],
              [-750, -1200], [-1600,0], [-2750,0],
              [0,2000], [0,3200], [0000,0000], [0,-1050], 
              [1850,2650],[2750, 3200],[0000,0000],[0,-1000], 
              [-1600,0], [0, 3450], [-2000,0], [3750, 3000], [1100, 1700,  0, -1], [1000, 2000, 0, 4400], 
              [3000,-1,0,-1], [500, 2500, 0, 1000],
              [1000, 1700, 1100, 2300],
              [500, 1700, 0, 2300],
              [1000, 2900, 1800, 3600],
              [1000, -1, 690, 1920],
              [0, 2000, 1000, -1], [1200, 2500, 780, -1],
              [1000, -1, 1860, 3093],
              [2500, -1, 600, -1],
              [1300, 3900, 1250, -1],
              [2000, -1, 770, 3400],
              [0, 3000, 250, 2900],
              [500, 2500, 0, -1],
              [500, 1750, 180, 2000],
              [500, 3000, 0, 2050],
              [3000, 4000, 0, 2700],
              [1000, 3000, 0, 3700],
              [1000,2000,0,3000], [0, 2800, 0, 2900],
              [2050, 3600, 600, 1900],
              [0, 3400, 1600, 2500],
              [1400, 3000, 0, 2000],
              [2400, -1, 1300, 3060],               
              [0, 2000, 1500, 3200],
              [0, 2000, 1300, 4700]]

        folds_ = [1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4]


    _patch_size = patch_size[1]
    _nbslices = patch_size[0]


    for img, mask2d, fold in zip(images_names, masks2d_names, folds_):

        print('Image Name: {}'.format(img))

        img_dir = images_all_dir[fold-1]

        #crop the image (according to the overlap between gt 2d mask and image)


        if fold==1:
            image = imread(os.path.join(img_dir, img + '.tif'))
        elif fold==2 or fold==3 or fold==4:
            image = imread(os.path.join(img_dir, img))


        if image.shape[2]>64: #happens for hemaxi_icam2
                image = image[:,:,0:64]

        mip_img = np.max(image, axis=2) #maximum intensity projection

        print('Image Shape: {}'.format(image.shape))
        print('----------------------------------------')

        #image size
        size_y = np.shape(image)[0]
        size_x = np.shape(image)[1]
        size_depth = np.shape(image)[2]
        aux_sizes_or = [size_y, size_x]

        #patch size
        new_size_y = int((size_y/_patch_size) + 1) * _patch_size
        new_size_x = int((size_x/_patch_size) + 1) * _patch_size

        aux_sizes = [new_size_y, new_size_x]
          
        ## zero padding
        aux_img = np.random.randint(1,50,(aux_sizes[0], aux_sizes[1], 64))
        aux_img[0:aux_sizes_or[0], 0:aux_sizes_or[1],0:np.shape(image)[2]] = image
        image = aux_img
        del aux_img

        print('Padded Image Shape: {}'.format(image.shape))
        print('----------------------------------------')

        final_mask_foreground = np.zeros((np.shape(image)[0], np.shape(image)[1], np.shape(image)[2]))
        final_mask_background = np.zeros((np.shape(image)[0], np.shape(image)[1], np.shape(image)[2]))
        final_mask_background = final_mask_background.astype('uint8')
        final_mask_foreground = final_mask_foreground.astype('uint8')

        i=0

        #load the model
        model_BtoA = load_model(model_dir, cust)

        while i+_patch_size<=image.shape[0]:
            j=0
            while j+_patch_size<=image.shape[1]:
                B_real = np.zeros((1,_nbslices,_patch_size,_patch_size,patch_size[3]),dtype='float32')
                _slice = image[i:i+_patch_size, j:j+_patch_size,:]
                _slice = _slice.transpose(2,0,1)
                _slice = np.expand_dims(_slice, axis=-1)
                B_real[0,:]=(_slice-127.5) /127.5   

                A_generated  = model_BtoA.predict(B_real)

                A_generated = (A_generated + 1)/2 #from [-1,1] to [0,1]

                A_generated = A_generated[0,:,:,:,0]
                A_generated = A_generated.transpose(1,2,0)

                #print(np.unique(A_generated))
                A_generated = (A_generated>0.5)*1

                A_generated = A_generated.astype('uint8')

                final_mask_foreground[i:i+_patch_size, j:j+_patch_size,:] = final_mask_foreground[i:i+_patch_size, j:j+_patch_size,:] + A_generated
                final_mask_background[i:i+_patch_size, j:j+_patch_size,:] = final_mask_background[i:i+_patch_size, j:j+_patch_size,:] + (1-A_generated)

                  
                j=j+_step
            i=i+_step

        del _slice
        del A_generated
        del B_real
        del model_BtoA

        final_mask = (final_mask_foreground>=final_mask_background)*1

        image = image[0:aux_sizes_or[0], 0:aux_sizes_or[1],0:size_depth]
        print('Image Shape: {}'.format(image.shape))
        print('----------------------------------------')

        final_mask = final_mask[0:aux_sizes_or[0], 0:aux_sizes_or[1],0:size_depth]
        print('Mask Shape: {}'.format(final_mask.shape))
        print('----------------------------------------')
        final_mask = final_mask*255.0
        final_mask = final_mask.astype('uint8')
        imwrite(os.path.join(save_dir_3dmasks, img+'.tif'), final_mask)
        seg = np.max(final_mask, axis=2)  #####
        imwrite(os.path.join(save_dir_2dmasks, img + '.tif'), seg)