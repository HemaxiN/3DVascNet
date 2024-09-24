# 3DVascNet: An Automated Software for 3D Retinal Vascular Networks Segmentation and Quantification

This repository contains the Python implementation of a [3D CycleGAN model to segment blood vessels in 3D microscopy images of mouse retinas](https://github.com/HemaxiN/3DVascNet/tree/main#segmentation). It also contains the code of the automatic pipeline developed for the [quantification of the vascular network based on the segmentation masks](https://github.com/HemaxiN/3DVascNet/tree/main#vasculature-quantification). Moreover, the instructions for downloading and using our interface can be found in [this repository](https://github.com/HemaxiN/3DVascNet/tree/main#graphical-user-interface---3dvascnet).

This is the official implementation of the paper: [3DVascNet](https://www.ahajournals.org/doi/abs/10.1161/ATVBAHA.124.320672). 

## Acknowledgements ✨

The code is continuously improved thanks to the valuable feedback from fellow researchers: Nekane Maritorena Hualde, Olya Oppenheim, and Tom Van de Kemp. I'm truly grateful for their feedback and comments!

Happy coding, and please keep the feedback coming! 💻🚀✉️ I greatly appreciate it!

Note: Check [this repository](https://github.com/HemaxiN/transpose_masks_3DVascNet) if you want to visualize the output 3D masks in **Fiji** or **Imaris**.

## How to cite

```bibtex
@article{narotamo20243dvascnet,
  title={3DVascNet: An Automated Software for Segmentation and Quantification of Mouse Vascular Networks in 3D},
  author={Narotamo, Hemaxi and Silveira, Margarida and Franco, Cl{\'a}udio A},
  journal={Arteriosclerosis, Thrombosis, and Vascular Biology},
  year={2024},
  publisher={Am Heart Assoc}
}
```

## Segmentation

### Architecture

![](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/images/overview.png)

![](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/images/architecturegit3.png)



### Dataset

Our dataset contains 3D microscopy images of mouse retinas and the corresponding 2D segmentation masks (annotated manually based on the maximum intensity projection (MIP) images of the 3D microscopy images). Moreover, we have the 3D segmentation masks obtained based on the 2D segmentation masks using [PolNet](https://github.com/mobernabeu/polnet).
The dataset *will be soon* made publicly available [here](https://huggingface.co/datasets/Hemaxi/3DVesselSegmentation/tree/main).

### Requirements

The code was initially developed in Python 3.5.2, using Keras 2.2.4. 

Now it has been updated to work in Python 3.10, the required packages are listed in [requirements.txt](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/utils/requirements.txt).

### Pre-Processing

To apply the pre-processing vessel enhancement method run the file [percentile.py](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/preprocessing/percentile.py). Moreover, the code to convert the ```.stl``` objects (3D segmentation masks obtained with [PolNet](https://github.com/mobernabeu/polnet)) into 3D ```.tif``` files is provided in the jupyter notebook [STL2Voxel.ipynb](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/preprocessing/STL2Voxel.ipynb). Note that while percentile normalization with values of 1 and 99 is effective for single-layer vasculature (P6 retinas), it may not be optimal for complex vasculatures with substantial intensity variations between different vessel layers as observed in adult retinas. In such cases, we have found that using percentiles of 0.5 and 99.5 yields better results. Thus, percentile normalization parameters may need to be adjusted according to the user's specific dataset requirements.


### Testing the Pre-trained Model on Your Own Dataset

Firstly, download the weights of the [pre-trained model](https://huggingface.co/Hemaxi/3DCycleGAN/tree/main).

This implementation supports the segmentation of images with any dimensions along x and y, and z.
The images should be .tif files, the images' names should have a prefix denoting its group (for instance 'GroupName_Image1.tif).
If the pre-trained model does not work well on your images you can train the cycleGAN model using the images of your dataset, and [masks of our dataset](https://huggingface.co/datasets/Hemaxi/3DVesselSegmentation/tree/main).

### Training on Your Own Dataset


To create the training patches (images and masks) use the files [create_patches_images.py](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/preprocessing/create_patches_images.py) and [create_patches_masks.py](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/preprocessing/create_patches_masks.py), respectively. Furthermore, patches representing the background can be created running the file [create_background_patches.py](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/preprocessing/create_background_patches.py).

Run the file [train_main.py](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/train_main.py) after changing the parameters defined in this file.
The `train_dir` contains the following tree structure:

```
train_dir
   ├── images  0.tif, 1.tif, ..., N.tif (patches of microscopy images of vessels  (Z_slices, X_dim, Y_dim))
   └── masks   0.tif, 1.tif, ..., N.tif (patches of segmentation masks of vessels  (Z_slices, X_dim, Y_dim))
```


### Prediction

To test the trained model on new images specify the directories in file [predict_main.py](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/predict_main.py) and run it.

*Update August 29, 2024*
- The resolution file is now required as input for the prediction model. This file is used to obtain percentile equalization values (pre-processing step) and to resize images if the ```_resize``` flag is True.

The resolution.xlsx file should have the following structure:

![](https://github.com/HemaxiN/3DVascNet/blob/main/images/resolutionfile.PNG)

- Added a ```norm_``` parameter (True or False) to control whether percentile equalization is applied. When set to True, percentile equalization parameters will be automatically extracted from the resolution file.
- Added a ```_resize``` parameter (True or False) to determine if images should be resized to match the resolution of the training images.
- Implemented a progress bar to monitor the prediction process.
- Image channels are now automatically transposed so that the output dimensions are (X, Y, Z).


### Post-Processing and Evaluation

Post-Processing methods can be applied to the predicted segmentation masks using the file [post_proc_main.py](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/post_proc_main.py); and the performance of the model can be evaluated using [eval_main.py](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/eval_main.py), which writes a ```results.csv```file:

```
| Image         | NMI3D | MI2D | NMI2D | DC | TP | FP | FN | TN | SP | SN |
|---------------|-------|------|-------|----|----|----|----|----|----|----|
| GroupA_Image1 |       |      |       |    |    |    |    |    |    |    |
| GroupB_Image2 |       |      |       |    |    |    |    |    |    |    |
| GroupA_Image3 |       |      |       |    |    |    |    |    |    |    |
```


* NMI3D: normalized mutual information in 3D
* MI2D: mutual information in 2D
* NMI2D: normalized mutual information in 2D
* DC: dice coefficient
* TP: true positive pixels
* FP: false positive pixels
* FN: false negative pixels
* TN: true negative pixels
* SP: specificity
* SN: sensitivity


### Model

The model that achieved the best performance is made publicly available [here](https://huggingface.co/Hemaxi/3DCycleGAN/tree/main).

## Vasculature Quantification

The code for feature extraction from the segmentation masks of vessels is provided in this [script](https://github.com/HemaxiN/3DVascNet/blob/main/quantification.py). 
First, you should change the ```masks_dir``` and ```resolution_file``` variables, which correspond respectively to the directory where the 3D post-processed masks are saved and the path to a resolution.xlsx file containing details about the voxel's physical sizes for each image.
The resolution.xlsx file should have the following structure:

![](https://github.com/HemaxiN/3DVascNet/blob/main/images/resolutionfile.PNG)

As stated above, the images' names should have a prefix denoting its group (for instance 'GroupName_Image1.tif). This is important to later visualize the distributions of the vascular features by grouping images belonging to the same group.
After running [quantification.py](https://github.com/HemaxiN/3DVascNet/blob/main/quantification.py), a features3d.csv file will be generated containing the computed 3D features for each mask, it will have the following structure:

```
| Image        | Group | Branching Points Density | Vessel Density % | Avascular Area % | Mean Branch Length | Mean Vessel Radius |
|--------------|-------|--------------------------|------------------|------------------|--------------------|--------------------|
| Image_Name_1 |       |                          |                  |                  |                    |                    |  
| Image_Name_2 |       |                          |                  |                  |                    |                    |    
| Image_Name_3 |       |                          |                  |                  |                    |                    |    
```

To define the region of interest we computed the concave hull for each segmentation mask using this [implementation](https://github.com/sebastianbeyer/concavehull).
More details are presented here [ConcaveHull](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/ConcaveHull).

## Graphical User Interface - 3DVascNet

3DVascNet is available as a GUI that allows to automatically analyse 3D microscopy images of retinal blood vessels.
Instructions for downloading and using our 3DVascNet software can be found [here](https://github.com/HemaxiN/3DVascNet/wiki/Downloading-and-Running-3DVascNet).

![](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/images/interface.png)










## Acknowledgements

* Code based on the [2D implementation of the cycleGAN](https://machinelearningmastery.com/cyclegan-tutorial-with-keras/);
* [STL2Voxel](https://github.com/cpederkoff/stl-to-voxel).
