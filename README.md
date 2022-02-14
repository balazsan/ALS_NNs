# ALS_CNN
Comparison of k-Nearest Neighbors (k-NN) method and Neural Networks (NN) to predict forest stand variables with help of ALS data

---------
k-NN:
lidR_lasfeat.R:
Functions to calculate features using ALS point cloud files.

ALS_feature_calc.R:
Wrapper script to call lidR_lasfeat.R.

---------
ANN (Artificial Neural Network)

---------
2D-CNN:

2D-CNN_alexnet2_larger_multiout_lindecay_sepmod.py:
AlexNet-type 2D CNN for forest variable prediction. Input 4D numpy array (voxelized ALS data) and csv table with response variables (field data, e.g. volume, mean height, diameter, etc.)

Example data files:
voxel_data.npy
forest_attributes.csv