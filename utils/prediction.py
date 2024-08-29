# example of using saved cycleGAN models for image translation
#based on https://machinelearningmastery.com/cyclegan-tutorial-with-keras/
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import os
import numpy as np
import tifffile
from scipy.ndimage import zoom
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def auximread(filepath):

    image = tifffile.imread(filepath)
        
    #the output image should be (X,Y,Z)
    original_0 = np.shape(image)[0]
    original_1 = np.shape(image)[1]
    original_2 = np.shape(image)[2]

    index_min = np.argmin([original_0, original_1, original_2])

    if index_min == 0:
        image = image.transpose(1,2,0)
    elif index_min == 1:
        image = image.transpose(0,2,1)

    return image

def predict(model_dir, img_dir, save_dir_2dmasks, save_dir_3dmasks, path_res, _resize, norm_, mode_, patch_size, _step, _step_z=32, _patch_size_z=64):

    #load the file with resolution information and extract features
    resolution = pd.read_excel(path_res)

    cust = {'InstanceNormalization': InstanceNormalization}
    #load the model
    model_BtoA = load_model(model_dir, cust)

    print('Mode: {}'.format(mode_))

    _patch_size = patch_size[1]
    _nbslices = patch_size[0]

    for img_name in os.listdir(img_dir):

        aux_res = resolution[resolution['Image'] == img_name]
        dim_x = aux_res['resx'].values[0]
        dim_y = aux_res['resy'].values[0]
        dim_z = aux_res['resz'].values[0]
        perceqmin = aux_res['perceqmin'].values[0]
        perceqmax = aux_res['perceqmax'].values[0]


        print('Image Name: {}'.format(img_name))
        image = auximread(os.path.join(img_dir, img_name))

        image = ((image/(np.max(image)))*255).astype('uint8')

        print('Image Shape: {}'.format(image.shape))
        print('----------------------------------------')

        initial_image_x = np.shape(image)[0]
        initial_image_y = np.shape(image)[1]
        initial_image_z = np.shape(image)[2]

        #percentile equalization
        if norm_:
            minval = np.percentile(image, perceqmin) 
            maxval = np.percentile(image, perceqmax)
            image = np.clip(image, minval, maxval)
            image = (((image - minval) / (maxval - minval)) * 255).astype('uint8')

        if _resize:
            image = zoom(image, (dim_x/0.333, dim_y/0.333, dim_z/0.5), order=3, mode='nearest')
            image = ((image/np.max(image))*255.0).astype('uint8')


        #image size
        size_y = np.shape(image)[0]
        size_x = np.shape(image)[1]
        size_depth = np.shape(image)[2]
        aux_sizes_or = [size_y, size_x, size_depth]
        

        #patch size
        new_size_y = int((size_y/_patch_size) + 1) * _patch_size
        new_size_x = int((size_x/_patch_size) + 1) * _patch_size
        new_size_z = int((size_depth/_patch_size_z) + 1) * _patch_size_z
        aux_sizes = [new_size_y, new_size_x, new_size_z]
        
        ## zero padding
        aux_img = np.random.randint(1,50,(aux_sizes[0], aux_sizes[1], aux_sizes[2]))
        aux_img[0:aux_sizes_or[0], 0:aux_sizes_or[1],0:aux_sizes_or[2]] = image
        image = aux_img
        del aux_img
            
        final_mask_foreground = np.zeros((np.shape(image)[0], np.shape(image)[1], np.shape(image)[2]))
        final_mask_background = np.zeros((np.shape(image)[0], np.shape(image)[1], np.shape(image)[2]))
        final_mask_background = final_mask_background.astype('uint8')
        final_mask_foreground = final_mask_foreground.astype('uint8')
        

        total_iterations = int(image.shape[0]/_patch_size)

        with tqdm(total=total_iterations) as pbar:
            i=0
            while i+_patch_size<=image.shape[0]:
                j=0
                while j+_patch_size<=image.shape[1]:
                    k=0
                    while k+_patch_size_z<=image.shape[2]:
                    
                        B_real = np.zeros((1,_nbslices,_patch_size,_patch_size,1),dtype='float32')
                        _slice = image[i:i+_patch_size, j:j+_patch_size, k:k+_patch_size_z]
                        
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
            
                        final_mask_foreground[i:i+_patch_size, j:j+_patch_size, k:k+_patch_size_z] = final_mask_foreground[i:i+_patch_size, j:j+_patch_size, k:k+_patch_size_z] + A_generated
                        final_mask_background[i:i+_patch_size, j:j+_patch_size, k:k+_patch_size_z] = final_mask_background[i:i+_patch_size, j:j+_patch_size, k:k+_patch_size_z] + (1-A_generated)
                        
                        k=k+_step_z
                    j=j+_step
                i=i+_step
                pbar.update(1)


        del _slice
        del A_generated
        del B_real

        final_mask = (final_mask_foreground>=final_mask_background)*1

        image = image[0:aux_sizes_or[0], 0:aux_sizes_or[1],0:size_depth]
        print('Image Shape: {}'.format(image.shape))
        print('----------------------------------------')

        final_mask = final_mask[0:aux_sizes_or[0], 0:aux_sizes_or[1],0:aux_sizes_or[2]]


        if _resize:
            final_mask = zoom(final_mask, (0.333/dim_x, 0.333/dim_y, 0.5/dim_z), order=3, mode='nearest')
            final_mask = (final_mask*255.0).astype('uint8')

            final_size_x = np.shape(final_mask)[0]
            final_size_y = np.shape(final_mask)[1]
            final_size_z = np.shape(final_mask)[2]

            aux_mask = np.zeros((initial_image_x, initial_image_y, initial_image_z)).astype('uint8')
            aux_mask[0:min(initial_image_x, final_size_x),0:min(initial_image_y, final_size_y),0:min(initial_image_z, final_size_z)] = final_mask[0:min(initial_image_x, final_size_x),0:min(initial_image_y, final_size_y),0:min(initial_image_z, final_size_z)]

            final_mask = aux_mask.copy()


        print('Mask Shape: {}'.format(final_mask.shape))
        print('----------------------------------------')
        final_mask = final_mask/np.max(final_mask)
        final_mask = final_mask*255.0
        final_mask = final_mask.astype('uint8')
        tifffile.imwrite(os.path.join(save_dir_3dmasks, img_name), final_mask)
        seg = np.max(final_mask, axis=2) 
        tifffile.imwrite(os.path.join(save_dir_2dmasks, img_name), seg)

        del seg
        del final_mask
        del image