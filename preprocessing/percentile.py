#Percentile Normalization in 3D
#https://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm
import tifffile
import numpy as np
import os

images_dir = r'C:\Users\hemax\Desktop\testing\images' #original images
save_dir = r'C:\Users\hemax\Desktop\testing\perc' #images after percentile normalization

for img_name in os.listdir(images_dir): 
	mip_img = tifffile.imread(os.path.join(images_dir, img_name))
	print(np.shape(mip_img))
	minval = np.percentile(mip_img, 1) 
	maxval = np.percentile(mip_img, 99)
	mip_img = np.clip(mip_img, minval, maxval)
	mip_img = (((mip_img - minval) / (maxval - minval)) * 255).astype('uint8')	
	tifffile.imwrite(os.path.join(save_dir, img_name), mip_img)
