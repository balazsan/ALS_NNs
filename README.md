# ALS_CNN
 Convolutional Neural Networks (CNN) to predict forest stand variables with help of voxelized ALS data

2D-CNN_alexnet2_larger_multiout_lindecay_sepmod.py:
AlexNet-type 2D CNN for forest variable prediction. Input 4D numpy array (voxelized ALS data) and csv table with response variables (field data, e.g. volume, mean height, diameter, etc.)

Example data files:
voxel_data.npy
forest_attributes.csv