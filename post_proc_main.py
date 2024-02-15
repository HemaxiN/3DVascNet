from utils.post_processing import *

msk_dir = r'/mnt/2TBData/hemaxi/cycleGAN/26_05_2022/models_01_07/masks'  #3D masks (CycleGAN)
save_dir_masks2d = r'/mnt/2TBData/hemaxi/cycleGAN/26_05_2022/models_01_07/results_proc'
save_dir_masks3d = r'/mnt/2TBData/hemaxi/cycleGAN/26_05_2022/models_01_07/masks_proc'
resolution_file = r'/mnt/2TBData/hemaxi/Downloads/resolution.xlsx'

post_processing(msk_dir, save_dir_masks3d, save_dir_masks2d, resolution_file)
