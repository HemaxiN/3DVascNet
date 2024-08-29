from utils.prediction import *

model_dir = '/mnt/2TBData/hemaxi/cycleGAN/26_05_2022/models_02_07/1st/g_model_AtoB_000003.h5'
img_dir = r'/mnt/2TBData/hemaxi/Downloads/percentile'
save_dir_2dmasks = r'/mnt/2TBData/hemaxi/cycleGAN/26_05_2022/models_02_07/results2'
save_dir_3dmasks = r'/mnt/2TBData/hemaxi/cycleGAN/26_05_2022/models_02_07/masks2'
path_res = r'/mnt/2TBData/hemaxi/Downloads/resolution.xlsx'
mode_ = 'test'
_step = 64
norm_ = False #percentile based normalization
_resize = False #image resizing to match the resolution of the training images
patch_size = (64,128,128,1)

predict(model_dir, img_dir, save_dir_2dmasks, save_dir_3dmasks, path_res, _resize, norm_, mode_,  patch_size, _step)

