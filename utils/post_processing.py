#import packages
import os
import numpy as np
from tifffile import imread, imwrite
import warnings
warnings.filterwarnings("ignore")
from skimage.measure import label, regionprops


def post_processing(msk_dir, save_dir_masks3d, save_dir_masks2d, mode_):

    for msk_name in os.listdir(msk_dir):
        final_mask = imread(os.path.join(msk_dir, msk_name))
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

        properties = [a for a in properties if a.area>= max_*0.5]

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
        imwrite(os.path.join(save_dir_masks3d, msk_name), final_mask)

        seg = np.max(final_mask, axis=2)  
        imwrite(os.path.join(save_dir_masks2d, msk_name), seg)
