#Percentile Equalization in 3D

import tifffile
import numpy as np
import os

images_dir = r'/mnt/2TBData/hemaxi/Downloads/tif' #original images
save_dir = r'/mnt/2TBData/hemaxi/Downloads/percentile' #images after percentile equalization

for img_name in os.listdir(images_dir): 
	mip_img = tifffile.imread(os.path.join(images_dir, img_name))
	mip_img = mip_img[:,:,:,2] #blue channel (vessels channel)
	print(np.shape(mip_img))
	minval = np.percentile(mip_img, 1) 
	maxval = np.percentile(mip_img, 99)
	mip_img = np.clip(mip_img, minval, maxval)
	mip_img = (((mip_img - minval) / (maxval - minval)) * 255).astype('uint8')	

	#tifffile.imwrite(r'/mnt/2TBData/hemaxi/Downloads/aa.tif', np.max(mip_img, axis=2))
	tifffile.imwrite(os.path.join(save_dir, img_name), mip_img)