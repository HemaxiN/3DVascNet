# example of using saved cycleGAN models for image translation
#based on https://machinelearningmastery.com/cyclegan-tutorial-with-keras/
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import os
import numpy as np
from tifffile import imread, imwrite
import warnings
warnings.filterwarnings("ignore")

def predict(model_dir, img_dir, save_dir_2dmasks, save_dir_3dmasks, mode_, patch_size, _step, _step_z=32, _patch_size_z=64):

    cust = {'InstanceNormalization': InstanceNormalization}
    #load the model
    model_BtoA = load_model(model_dir, cust)

    print('Mode: {}'.format(mode_))

    _patch_size = patch_size[1]
    _nbslices = patch_size[0]

    for img_name in os.listdir(img_dir):

        print('Image Name: {}'.format(img_name))
        image = imread(os.path.join(img_dir, img_name))

        print('Image Shape: {}'.format(image.shape))
        print('----------------------------------------')

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


        del _slice
        del A_generated
        del B_real

        final_mask = (final_mask_foreground>=final_mask_background)*1

        image = image[0:aux_sizes_or[0], 0:aux_sizes_or[1],0:size_depth]
        print('Image Shape: {}'.format(image.shape))
        print('----------------------------------------')

        final_mask = final_mask[0:aux_sizes_or[0], 0:aux_sizes_or[1],0:aux_sizes_or[2]]
        print('Mask Shape: {}'.format(final_mask.shape))
        print('----------------------------------------')
        final_mask = final_mask*255.0
        final_mask = final_mask.astype('uint8')
        imwrite(os.path.join(save_dir_3dmasks, img_name), final_mask)
        seg = np.max(final_mask, axis=2) 
        imwrite(os.path.join(save_dir_2dmasks, img_name), seg)