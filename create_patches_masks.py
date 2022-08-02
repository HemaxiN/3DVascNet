from tifffile import imwrite, imread
import os
import numpy as np
from skimage import morphology
import os

save_dir = r'/mnt/2TBData/hemaxi/cycleGAN/ds/masks'

k=0 #iterator for the patches

masks_names = ['mask_PBS_new_ret1', 'mask_PBS_new_ret1b', 'mask_PBS_old_ret1']

msk_dir = r'/mnt/2TBData/hemaxi/Downloads/masks'

_patch_size = 128
_step = 128

for msk in masks_names:
    #msk = msk.replace('mask_','mask_resized_xy_')
    print('Mask Name: {}'.format(msk))

    mask = imread(os.path.join(msk_dir, msk +'.tif'))
    mask = mask[:-3,:-3,:-3]

    print('Mask Shape: {}'.format(mask.shape))
    print('----------------------------------------')

    #image size
    size_y = np.shape(mask)[0]
    size_x = np.shape(mask)[1]
    aux_sizes_or = [size_y, size_x]

    #patch size
    new_size_y = int((size_y/_patch_size) + 1) * _patch_size
    new_size_x = int((size_x/_patch_size) + 1) * _patch_size

    aux_sizes = [new_size_y, new_size_x]
    
    ## zero padding
    aux_img = np.zeros((aux_sizes[0], aux_sizes[1], 64))
    aux_img[0:aux_sizes_or[0], 0:aux_sizes_or[1],0:np.shape(mask)[2]] = mask
    mask = aux_img
    #print('Unique numbers: {}'.format(np.unique(mask)))

    print('Mask Padding Shape: {}'.format(mask.shape))
    print('----------------------------------------')

    i=0
    while i+_patch_size<=mask.shape[0]:
        j=0
        while j+_patch_size<=mask.shape[1]:
            _slice = mask[i:i+_patch_size, j:j+_patch_size,0:64]
            #_slice = _slice/255.0
            _slice = _slice.astype('uint8')


            #if random.uniform(0,1)>0.5:
            #if True:
                #print('Perform binary_erosion')
                #print(str(k))
            #    rad = np.random.choice([7,9,11])
            #    _slice = morphology.binary_erosion(_slice, morphology.ball(radius=rad))


            #_slice = _slice*255.0
            #_slice = _slice.astype('uint8')
            _slice = _slice.transpose(2,0,1)

            imwrite(os.path.join(save_dir, str(k)+'.tif'), _slice)
            k=k+1
            j=j+_step
        i=i+_step
