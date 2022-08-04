from utils.evaluation import *

img_dir = r'/mnt/2TBData/hemaxi/Downloads/tif'
masks_dir = r'/mnt/2TBData/hemaxi/cycleGAN/26_05_2022/models_02_07/masks_proc'
gt_dir_2dmasks = r'/mnt/2TBData/hemaxi/Downloads/masks2d/'

mode_ = 'test'

evaluate(img_dir, masks_dir, gt_dir_2dmasks, mode_)