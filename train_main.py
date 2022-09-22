from utils.cycleGAN import *

#directory with the training images and masks with the following tree structure
'''
training dataset
   ├── images  0.tif, 1.tif, ..., N.tif (patches of microscopy images of vessels)
   └── masks   0.tif, 1.tif, ..., N.tif (patches of segmentation masks of vessels)
'''    

train_dir = r'/dev/shm/dataset3d' 

# define input and output image shape
image_shape = (64,128,128,1)

#number of patches in the training set
n_patches = 4987

#number of epochs and batch size
nbepochs, batchsize = 5, 1

# generator: A -> B
g_model_AtoB = define_generator(image_shape)
# generator: B -> A
g_model_BtoA = define_generator(image_shape)
# discriminator: A -> [real/fake]
d_model_A = define_discriminator(image_shape)
# discriminator: B -> [real/fake]
d_model_B = define_discriminator(image_shape)
# composite: A -> B -> [real/fake, A]
c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
# composite: B -> A -> [real/fake, B]
c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
# train models
train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, n_patches, train_dir, image_shape, nbepochs, batchsize)
