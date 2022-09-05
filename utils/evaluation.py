from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
import os
import numpy as np
from tifffile import imread, imwrite
from sklearn.metrics import mutual_info_score
from skimage.measure import compare_ssim as ssim
#from sklearn.metrics.cluster import normalized_mutual_info_score
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from skimage.measure import label, regionprops
from skimage import morphology
from scipy import ndimage

def normalized_mutual_information(x,y):
    #joint histogram
    hgram, x_edges, y_edges = np.histogram2d(x.ravel(), y.ravel(), bins=20)
    hgram = hgram+EPS
    #joint probability distribution
    pxy = hgram/float(np.sum(hgram))
    #marginal x
    px = np.sum(pxy, axis=1)
    #marginal y
    py = np.sum(pxy, axis=0)
    #broadcast to multiply marginals
    px_py = px[:,None] * py[None, :]
    nzs = pxy>0 #consider only non zero values
    return (np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))) / (np.sum(-pxy*np.log(pxy)))

def evaluate(img_dir, masks_dir, gt_dir_2dmasks, mode_):

    #create the dataframe to save the results
    results_df = pd.DataFrame(columns = ["Image", "NMI3D", "MI2D", "NMI2D", "DC", "TP", "FP", "FN", "TN", "SP", "SN"])
    results_df.to_csv('results.csv', sep=';', index=False)

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


    for img, mask2d, cropsize in zip(images_names, masks2d_names, crop_sizes):
        results_df = pd.read_csv('results.csv', sep=';')

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

        size_depth = image.shape[2]


        image = image[:,:,0:size_depth]


        mip_img = np.max(image, axis=2) #maximum intensity projection
        #print(np.max(mip_img))

        #minval = np.percentile(mip_img, 20) 
        #maxval = np.percentile(mip_img, 80)
        #mip_img = np.clip(mip_img, minval, maxval)
        #mip_img = ((mip_img - minval) / (maxval - minval)) * 255    
        mip_img = mip_img.astype('uint8')


        print('Image Shape: {}'.format(image.shape))
        print('----------------------------------------')

        #image size
        size_y = np.shape(image)[0]
        size_x = np.shape(image)[1]
        size_depth = np.shape(image)[2]
        aux_sizes_or = [size_y, size_x]
       

        seg3d = imread(os.path.join(masks_dir, img + '.tif'))

        seg = np.max(seg3d, axis=-1) #2D projection of the 3D mask

        print('3D Evaluation')

        #mis3d = mutual_info_score(image.ravel(), final_mask.ravel())
        #print('Mutual Info Score 3D: {}'.format(mis3d))

        nmis3d = normalized_mutual_information(image.ravel(), seg3d.ravel())
        print('Normalized Mutual Info Score 3D: {}'.format(nmis3d))
        print('----------------------------------------')

        print('2D Evaluation')

        gt = imread(os.path.join(gt_dir_2dmasks, mask2d +'.tif'))

        #crop the mask
        gt = gt[:,:,0]

        if cropsize[0] !=0:
          if cropsize[0]>0:
            gt = gt[:cropsize[0],:]
          elif cropsize[0]<0:
            gt = gt[-cropsize[0]:,:]

        if cropsize[1]!=0:
          if cropsize[1]>0:
            gt = gt[:,:cropsize[1]]
          elif cropsize[1]<0:
            gt = gt[:,-cropsize[1]:]


        #mask inversion
        gt[gt==255] = 100
        gt[gt!=100] = 255
        gt[gt==100] = 0

        k = 255

        dice = np.sum(seg[gt==k]==k)*2.0 / (np.sum(seg[seg==k]==k) + np.sum(gt[gt==k]==k))
        print('Dice Similarity Score {}'.format(dice))

        mis2d = mutual_info_score(mip_img.ravel(), seg.ravel())
        print('Mutual Info Score 2D: {}'.format(mis2d))

        nmis2d = normalized_mutual_information(mip_img.ravel(), seg.ravel())
        print('Normalized Mutual Info Score 2D: {}'.format(nmis2d))
        print('----------------------------------------')
        
        tp = np.sum(seg[gt==k]==k)
        tn = np.sum(seg[gt==0]==0)
        fn = np.sum(seg[gt==k]==0)
        fp = np.sum(seg[gt==0]==k)

        sens = tp/(tp+fn) #sensitivity
        spec = tn/(tn+fp) #specificity

        res = {"Image": img, "NMI3D": nmis3d, "MI2D": mis2d, "NMI2D": nmis2d, "DC": dice, 
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "SP": spec, "SN": sens}
        row = len(results_df)
        results_df.loc[row] = res

        results_df.to_csv('results.csv', sep=';', index=False)

    print(results_df)


