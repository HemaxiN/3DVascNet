import os
import numpy as np
from tifffile import imread
from sklearn.metrics import mutual_info_score
#from sklearn.metrics.cluster import normalized_mutual_info_score
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from skimage.measure import label, regionprops
from skimage import morphology
from scipy import ndimage

EPS = np.finfo(float).eps

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

    for img, mask2d in zip(sorted(os.listdir(img_dir)), sorted(os.listdir(masks_dir))):
        results_df = pd.read_csv('results.csv', sep=';')

        print('Image Name: {}'.format(img))

        image = imread(os.path.join(img_dir, img))

        size_depth = image.shape[2]

        mip_img = np.max(image, axis=2) #maximum intensity projection
        mip_img = mip_img.astype('uint8')

        print('Image Shape: {}'.format(image.shape))
        print('----------------------------------------')

        #image size
        size_y = np.shape(image)[0]
        size_x = np.shape(image)[1]
        size_depth = np.shape(image)[2]
        aux_sizes_or = [size_y, size_x]

        seg3d = imread(os.path.join(masks_dir, img))
        print('Mask 3D Shape: {}'.format(seg3d.shape))
        print('----------------------------------------')

        seg = np.max(seg3d, axis=-1) #2D projection of the 3D mask


        nmis3d = normalized_mutual_information(image.ravel(), seg3d.ravel())
        print('Normalized Mutual Info Score 3D: {}'.format(nmis3d))
        print('----------------------------------------')

        print('2D Evaluation')

        gt = imread(os.path.join(gt_dir_2dmasks, mask2d))

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