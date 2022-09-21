# 3D Vessel Segmentation
A Deep Learning-Based Approach for 3D Vessel Segmentation in Microscopy Images of Mouse Retinas 

This repository contains the Python implementation of a 3D cycleGAN model to segment blood vessels in 3D microscopy images of mouse retinas.

## Architecture

![](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/images/architecturegit2.png)



## Dataset

Our dataset contains 3D microscopy images of mouse retinas and the corresponding 2D segmentation masks (annotated manually based on the maximum intensity projection (MIP) images of the 3D microscopy images). Moreover, we have the 3D segmentation masks obtained based on the 2D segmentation masks using [PolNet](https://github.com/mobernabeu/polnet).
The dataset is made publicly available [here](https://huggingface.co/datasets/Hemaxi/3DVesselSegmentation/tree/main).


## Pre-Processing

To apply the pre-processing vessel enhancement method run the file [percentile.py](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/preprocessing/percentile.py). Moreover, the code to convert the ```.stl``` objects (3D segmentation masks obtained with [PolNet](https://github.com/mobernabeu/polnet)) into 3D ```.tif``` files is provided in the jupyter notebook [STL2Voxel.ipynb](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/preprocessing/STL2Voxel.ipynb).

## Training on Your Own Dataset


To create the training patches (images and masks) use the files [create_patches_images.py](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/preprocessing/create_patches_images.py) and [create_patches_masks.py](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/preprocessing/create_patches_masks.py), respectively. Furthermore, patches representing the background can be created running the file [create_background_patches.py](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/preprocessing/create_background_patches.py).

Run the file [train_main.py](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/train_main.py) after changing the parameters defined in this file.
The `train_dir` contains the following tree structure:

```
train_dir
   ├── images  0.tif, 1.tif, ..., N.tif (patches of microscopy images of vessels  (Z_slices, X_dim, Y_dim))
   └── masks   0.tif, 1.tif, ..., N.tif (patches of segmentation masks of vessels  (Z_slices, X_dim, Y_dim))
```


## Prediction

To test the trained model on new images specify the directories in file [predict_main.py](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/predict_main.py) and run it.

## Post-Processing and Evaluation

Post-Processing methods can be applied to the predicted segmentation masks using the files [post_proc_main.py](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/post_proc_main.py) and [erosion.py](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/erosion.py); and the performance of the model can be evaluated using [eval_main.py](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/eval_main.py), which writes a ```results.csv```file:

```
| Image        | NMI3D | MI2D | NMI2D | DC | TP | FP | FN | TN | SP | SN |
|--------------|-------|------|-------|----|----|----|----|----|----|----|
| Image_Name_1 |       |      |       |    |    |    |    |    |    |    |
| Image_Name_2 |       |      |       |    |    |    |    |    |    |    |
| Image_Name_3 |       |      |       |    |    |    |    |    |    |    |
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


## Model

The model that achieved the best performance is made publicly available [here](https://huggingface.co/Hemaxi/3DCycleGAN/tree/main).

## Requirements

Python 3.5.2, Keras 2.2.4 and other packages listed in [requirements.txt](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/utils/requirements.txt).


## Acknowledgements

* Code based on the [2D implementation of the cycleGAN](https://machinelearningmastery.com/cyclegan-tutorial-with-keras/);
* [STL2Voxel](https://github.com/cpederkoff/stl-to-voxel).
