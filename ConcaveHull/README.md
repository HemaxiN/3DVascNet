# Concave Hull

Concave Hull Computation from the Segmentation Masks to Define the Region of Interest For Feature Extraction

To compute the concave hull for the binary segmentation masks follow the code provided in the jupyter notebook [ConcaveHull.ipynb](https://github.com/HemaxiN/3DVesselSegmentation/blob/main/ConcaveHull/ConcaveHull.ipynb).
You may need to change some parameters, for instance:

* ```rx and ry: the dimensions of the resized mask (to compute the concave hull), we performed resizing to reduce computational costs, it is not mandatory```

* ```k: parameter of the Concave Hull algorithm``` check the [original paper](https://repositorium.sdum.uminho.pt/handle/1822/6429)

## Acknowledgements

* https://github.com/sebastianbeyer/concavehull
