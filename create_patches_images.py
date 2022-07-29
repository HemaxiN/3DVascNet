from tifffile import imwrite, imread
import os
import numpy as np
from skimage import morphology
import os

save_dir = r'/mnt/2TBData/hemaxi/cycleGAN/ds/images'

k=0 #iterator for the patches

images_names = ['PBS_new_ret1', 'PBS_new_ret1b', 'sFLT1_ret2']

img_dir = r'/mnt/2TBData/hemaxi/Downloads/percentile'

_patch_size = 128
_step = 128

for img in images_names:
    print('Image Name: {}'.format(img))

    image = imread(os.path.join(img_dir, img + '.tif'))
    #image = image[:,:,:,2]

    print('Image Shape: {}'.format(image.shape))
    print('----------------------------------------')

    #image size
    size_y = np.shape(image)[0]
    size_x = np.shape(image)[1]
    aux_sizes_or = [size_y, size_x]

    #patch size
    new_size_y = int((size_y/_patch_size) + 1) * _patch_size
    new_size_x = int((size_x/_patch_size) + 1) * _patch_size

    aux_sizes = [new_size_y, new_size_x]
    
    ## zero padding
    aux_img = np.random.randint(1,50,(aux_sizes[0], aux_sizes[1], 64))
    aux_img[0:aux_sizes_or[0], 0:aux_sizes_or[1],0:np.shape(image)[2]] = image
    image = aux_img
    #print('Unique numbers: {}'.format(np.unique(mask)))

    print('Image Padding Shape: {}'.format(image.shape))
    print('----------------------------------------')

    i=0
    while i+_patch_size<=image.shape[0]:
        j=0
        while j+_patch_size<=image.shape[1]:
            _slice = image[i:i+_patch_size, j:j+_patch_size,0:64]
            _slice = _slice.astype('uint8')

            _slice = _slice.transpose(2,0,1)

            imwrite(os.path.join(save_dir, str(k)+'.tif'), _slice)
            k=k+1
            j=j+_step
        i=i+_step
