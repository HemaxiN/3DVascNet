#import packages
import os
import numpy as np
from tifffile import imread, imwrite
import warnings
warnings.filterwarnings("ignore")
from skimage.measure import label, regionprops
import pandas as pd

def post_processing(msk_dir, save_dir_masks3d, save_dir_masks2d, resolution_file):
    resolution = pd.read_excel(resolution_file)
    for msk_name in os.listdir(msk_dir):
        if '.tif' in msk_name:
            print('here: {}'.format(msk_name))
            final_mask = imread(os.path.join(msk_dir, msk_name))
            final_mask = final_mask/255.0
            final_mask = final_mask.astype('uint8')

            aux_res = resolution[resolution['Image'] == msk_name]
            dim_x = aux_res['resx'].values[0]
            dim_y = aux_res['resy'].values[0]
            dim_z = aux_res['resz'].values[0]

            ##post-processing
            #remove outliers
            print('Remove the Outliers')
            labeled_image = label(final_mask)
            properties = regionprops(labeled_image, spacing=(dim_x, dim_y, dim_z))

            mask = np.zeros(np.shape(final_mask)).astype('uint8')

            for prop in properties:
                area_ = prop.area
                print(area_)
                if area_ < 50000 and area_ >= 1000:
                    conv = prop.area/prop.area_convex
                    if conv<=0.2:
                        mask[labeled_image==prop.label] = 255 #region to include
                elif area_ > 50000:
                    mask[labeled_image==prop.label] = 255 #region to include

            final_mask = mask.copy()
            final_mask = final_mask.astype('uint8')

            del mask
            del properties
            print('Objects Removed')


            print('Mask Shape: {}'.format(final_mask.shape))
            print('----------------------------------------')
            imwrite(os.path.join(save_dir_masks3d, msk_name), final_mask)

            seg = np.max(final_mask, axis=2)  
            imwrite(os.path.join(save_dir_masks2d, msk_name), seg)
