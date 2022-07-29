#import packages
import os
import numpy as np
from tifffile import imread, imwrite
from sklearn.metrics import mutual_info_score
from skimage.measure import compare_ssim as ssim
from sklearn.metrics.cluster import normalized_mutual_info_score
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from skimage.measure import label, regionprops
from skimage import morphology


def post_processing(msk_dir, save_dir_masks3d, save_dir_masks2d, mode_):

    print('Mode: {}'.format(mode_))

    if mode_ == 'train':

        images_names = ['PBS_new_ret1', 'PBS_new_ret1b', 'PBS_old_ret1', 'PBS_ret1-02', 
        'PBS_ret4', 'sFLT1_ret1', 'sFLT1_ret2', 'sFLT1_ret3', 'VEGF_ret1', 
        'VEGF_ret3', 'WT_retina_5']

        masks2d_names = ['PBS_new_ret1_stiched_mask', 'PBS_new_ret1b_stiched_mask', 
        'PBS_old_ret1_mask', 'PBS_ret1-02-stiched_mask',  'PBS_ret4_stiched_mask', 
        'sFlt1_ret2-02_mask', 'sFlt1_ret2-03_mask', 'sFLT1_ret3_mask', 'VEGF_ret1-02_mask',
         'VEGF_ret1-03_mask', 'Wnt5aWT_PBS_Ret5_tile1_mask']

        crop_sizes =  [[4000,2700], [0,-2000], [-500, 3000], [-3000, 0], [-2500,0], [-1000,-2200],
         [0, 5000], [-500,2200], [2000,3500], [-2500,0], [0, -3000]]

    else:

        images_names = ['WT_AngII_ret1','WT_AngII_ret2', 'VEGF_ret1b', 'Hemaxi_ICAM2']

        masks2d_names = ['Wnt5aWT_AngII_Ret7_tile1_mask','Wnt5aWT_AngII_Ret7_tile2_mask',
        'VEGF_Ret1_tile1_stitched_mask', 'hemaxi_icam_2d_mask']

        crop_sizes = [[0,0], [0,0], [-1000, 4000], [0,0]]


    for img, mask2d, cropsize in zip(images_names, masks2d_names, crop_sizes):
        print('Image Name: {}'.format(img))
        final_mask = imread(os.path.join(msk_dir, img + '.tif'))
        final_mask = final_mask/255.0
        final_mask = final_mask.astype('uint8')

        ##post-processing
        #remove outliers
        print('Remove the Outliers')
        labeled_image = label(final_mask)
        properties = regionprops(labeled_image)

        max_ = 0
        for region in properties:
            if region.area > max_:
                max_ = region.area

        properties = [a for a in properties if a.area>= max_-10]

        ## remove small regions from the mask as well
        mask_aux = np.zeros(np.shape(final_mask))
        print('size regions')
        print(len(properties))
        for r in properties:
            mask_aux[r._label_image==r.label]=1
            print(r.area)
        final_mask = mask_aux.copy()
        final_mask = final_mask.astype('uint8')
        del mask_aux
        del properties
        print('Objects Removed')


        print('Mask Shape: {}'.format(final_mask.shape))
        print('----------------------------------------')
        final_mask = final_mask*255.0
        final_mask = final_mask.astype('uint8')
        imwrite(os.path.join(save_dir_masks3d, img+'.tif'), final_mask)


        seg = np.max(final_mask, axis=2)  
        imwrite(os.path.join(save_dir_masks2d, img + '.tif'), seg)
