### post-process the mask
from tifffile import imread, imwrite
from skimage.measure import label, regionprops
import numpy as np
from skimage import morphology
from skimage.morphology import square, erosion, disk
# Erosion shrinks bright regions https://homepages.inf.ed.ac.uk/rbf/HIPR2/erode.htm
import os

dir_ = r'/mnt/2TBData/hemaxi/cycleGAN/26_05_2022/models_02_07/masks2'
save_dir_3dmasks = r'/mnt/2TBData/hemaxi/cycleGAN/26_05_2022/models_02_07/masks2_proc'
save_dir_2dmasks = r'/mnt/2TBData/hemaxi/cycleGAN/26_05_2022/models_02_07/results2_proc'

for msk in os.listdir(dir_):
	bw = imread(os.path.join(dir_, msk))
	bw = bw/255.0
	bw = bw.astype('uint8')
	#erosion operation
	print('Perform Erosion')
	aux_img = np.zeros(np.shape(bw))
	for z in range(0,np.shape(bw)[2]):
		aux_img[:,:,z] = erosion(bw[:,:,z], square(6))
	print('Erosion Done')
	bw = aux_img
	bw = bw*255.0
	bw = bw.astype('uint8')
	imwrite(os.path.join(save_dir_3dmasks, msk), bw)

	bw = np.max(bw, axis=-1)
	imwrite(os.path.join(save_dir_2dmasks, msk), bw)