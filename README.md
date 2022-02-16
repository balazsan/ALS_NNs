# ALS_CNN
Comparison of k-Nearest Neighbors (k-NN) method and Neural Networks (NN) to predict forest stand variables with help of ALS data

---------
sample_plot _data:
CSV files containing mean diameter in cm (d), mean height in m (h) and total growing stock in m3/ha (v).

---------
features:
CSV files with features calculated using ALS_feature_calc.R. Note, no augmentation done here as it would produce the same values for most of the features.

---------
voxel_data:
4D-numpy arrays with voxel data for 2D and 3D-CNN. Numeric values represent the number of points within each voxel. Pre-augmented training data is used in 2D-CNN, augmentation carried out on-the-fly in 3D-CNN.

---------
k-NN:

lidR_lasfeat.R:
Functions to calculate features using ALS point cloud files.

ALS_feature_calc.R:
Wrapper script to call lidR_lasfeat.R. Showing the use of the relevant functions, no real data provided due to data distribution restrictions.

ALS_feature_sel.R:
Feature selection and feature weight search.

knn_funcs_group_28052019.R:
k-NN and genalg functions.

---------
ANN (Artificial Neural Network):

ANN.R:
Predicting forest stand variables using simple ANN based on ALS features.

keras_tf_funcs.R:
Functions to estimate forest attributes based on field measurements and remote sensing features using Keras with TensorFlow (TF) back-end.

---------
2D-CNN:

2D-CNN_alexnet2_larger_multiout_lindecay_sepmod.py:
AlexNet-type 2D CNN for forest variable prediction. Input 4D numpy array (voxelized ALS data) and csv table with response variables (field data, e.g. volume, mean height, diameter, etc.)

Example data files:
voxel_data.npy
forest_attributes.csv