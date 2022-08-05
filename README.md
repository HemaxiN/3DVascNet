# 3D Vessel Segmentation
Semi-Supervised 3D Vessel Segmentation in Microscopy Images of Mouse Retinas 

# Dataset

Our dataset contains 3D microscopy images of mouse retinas and the corresponding 2D segmentation masks (annotated manually based on the maximum intensity projection (MIP) imagees of the 3D microscopy images. Moreover, we have the 3D segmentation masks obtained based on the 2D segmentation masks using [PolNet](https://github.com/mobernabeu/polnet).





# Acknowledgements

* Code based on the [2D implementation of the cycleGAN](https://machinelearningmastery.com/cyclegan-tutorial-with-keras/).


create_patches_images.py
create_patches_masks.py

create_background_patches.py (optional)

train_main.py

predict_main.py

post_proc_main.py

erosion.py (optional)

eval_main.py
