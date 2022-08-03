from tifffile import imread, imwrite
import os
import numpy as np

img_dir = r'/dev/shm/dataset3d/images'
msk_dir = r'/dev/shm/dataset3d/masks'

for i in range(3093, 3500):
	image = np.random.randint(0,50,(64,128,128))
	mask = np.zeros((64,128,128))

	image = image.astype('uint8')
	mask = mask.astype('uint8')

	imwrite(os.path.join(img_dir, str(i)+'.tif'), image)
	imwrite(os.path.join(msk_dir, str(i)+'.tif'), mask)

