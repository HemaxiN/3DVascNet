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

def predict(model_dir, img_dir, save_dir_2dmasks, save_dir_3dmasks, mode_, patch_size, _step):

	# load the models
	cust = {'InstanceNormalization': InstanceNormalization}

	print('Mode: {}'.format(mode_))

	if mode_ == 'train':

	    images_names = ['PBS_new_ret1', 'PBS_new_ret1b', 'PBS_old_ret1']

	    masks2d_names = ['PBS_new_ret1_stiched_mask', 'PBS_new_ret1b_stiched_mask', 
	    'PBS_old_ret1_mask']

	    crop_sizes =  [[4000,2700], [0,-2000], [-500, 3000]]

	else:

	    images_names = ['WT_AngII_ret1','WT_AngII_ret2', 'VEGF_ret1b', 'Hemaxi_ICAM2', 'PBS_ret1-02', 
	    'PBS_ret4', 'sFLT1_ret1', 'sFLT1_ret2', 'sFLT1_ret3', 'VEGF_ret1', 
	    'VEGF_ret3', 'WT_retina_5']

	    masks2d_names = ['Wnt5aWT_AngII_Ret7_tile1_mask','Wnt5aWT_AngII_Ret7_tile2_mask',
	    'VEGF_Ret1_tile1_stitched_mask', 'hemaxi_icam_2d_mask', 'PBS_ret1-02-stiched_mask',  'PBS_ret4_stiched_mask', 
	    'sFlt1_ret2-02_mask', 'sFlt1_ret2-03_mask', 'sFLT1_ret3_mask', 'VEGF_ret1-02_mask',
	     'VEGF_ret1-03_mask', 'Wnt5aWT_PBS_Ret5_tile1_mask']

	    crop_sizes = [[0,0], [0,0], [-1000, 4000], [0,0], [-3000, 0], [-2500,0], [-1000,-2200],
	     [0, 5000], [-500,2200], [2000,3500], [-2500,0], [0, -3000]]


	_patch_size = patch_size[1]
	_nbslices = patch_size[0]


	for img, mask2d, cropsize in zip(images_names, masks2d_names, crop_sizes):
	    #results_df = pd.read_csv('results.csv', sep=';')

	    print('Image Name: {}'.format(img))

	    image = imread(os.path.join(img_dir, img + '.tif'))

	    if image.shape[2]>64: #happens for hemaxi_icam2
	        image = image[:,:,0:64,:]

	    #crop the image (according to the overlap between gt 2d mask and image)
	    #image = image[:, :, :, 2]


	    if cropsize[0] !=0:
	      if cropsize[0]>0:
	        image = image[:cropsize[0],:,:]
	      elif cropsize[0]<0:
	        image = image[-cropsize[0]:,:,:]

	    if cropsize[1]!=0:
	      if cropsize[1]>0:
	        image = image[:,:cropsize[1],:]
	      elif cropsize[1]<0:
	        image = image[:,-cropsize[1]:,:]



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

	    final_mask = np.zeros((np.shape(image)[0], np.shape(image)[1], np.shape(image)[2]))
	    final_mask = final_mask.astype('uint8')

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

	            A_generated[-5:,:,:] = 0
	            A_generated[:,-5:,:] = 0
	            A_generated[:,:,-4:] = 0
	            A_generated[:5,:,:] = 0
	            A_generated[:,:5,:] = 0
	            A_generated[:,:,:4] = 0

	            A_generated = A_generated.astype('uint8')

	            final_mask[i:i+_patch_size, j:j+_patch_size,:] = np.logical_or(final_mask[i:i+_patch_size, j:j+_patch_size,:], A_generated)
	            
	            j=j+_step
	        i=i+_step

	    del _slice
	    del A_generated
	    del B_real
	    del model_BtoA

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
