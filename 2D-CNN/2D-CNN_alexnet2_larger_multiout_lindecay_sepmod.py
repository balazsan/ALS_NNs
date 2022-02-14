"""
Created on Sep 18 2019

@author: Eero Liski

Script for constructing a AlexNet style neural network from point cloud data

loss: mse
optimizer: sgd
batch_normalization
Linear rate scheduler: linear, init_lr = 1e-5
Architecture: Alexnet largest version



What you need:
    - A saved numpy array of voxelized point cloud training, validation and test
    data with dimensions (n,40,40,105) produced by the
    point_cloud_voxelizer.py script.
    - A CSV containing a column with field measured values for training,
    validation and test in n plots

Saving models based on each forest attribute separately
"""
import platform, sys, os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras import optimizers
from keras import backend
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Model
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#%%
####################################################################################
## user class
class PolynomialDecay:
    def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
        # store the maximum number of epochs, base learning rate,
        # and power of the polynomial
        self.maxEpochs = maxEpochs
        self.initAlpha = initAlpha
        self.power = power

    def __call__(self, epoch):
        # compute the new learning rate based on polynomial decay
        decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
        alpha = self.initAlpha * decay

        # return the new learning rate
        return float(alpha)

####################################################################################
# # check if using gpu
# device_name = tf.test.gpu_device_name()
# if device_name == '':
#     print('Using CPU')
# else:
#     print('Using GPU device', device_name)

#print(tf.test.gpu_device_name())
#print(backend.tensorflow_backend._get_available_gpus())

# define user metric
def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
#%%
################################################################################
# loss weights for response variables (v,h,d)
#l_weights=[0.5,0.25,0.25]
#l_weights=[0.6,0.2,0.2]
l_weights=[0.7,0.15,0.15]
#l_weights=[0.8,0.1,0.1]
#l_weights=[1.,1.,1.]

# should scaling of y variables be done?
scale_y=False
#scale_y=True

# using standardized voxels
#use_std=False
use_std=True

# binary voxel or not (if number in voxel>=1 -> 1, if voxel=0 -> 0)
use_binary=False
#use_binary=True

# only for selecting input file name and path --->
# use augmented data
augmented="aug"
#augmented="noaug"

# dataset to be used
#dataset="AV-MK"
dataset="AV"

# different path if run on laptop (Windows) than on CSC (Unix)
path="3D_CNN/" if platform.system()=="Windows" else ""
# <--- only for selecting input file name and path

# load datasets (x and y; predictors and response)
# training dataset
if augmented=="aug":
    x_train=np.load("../"+path+dataset+"_leafon/"+dataset+"_leafon_train_voxels_aug_np_circ.npy")
    y_train_whole=pd.read_csv("../"+path+dataset+"_leafon/"+dataset+".leaf.on.train.aug.csv", sep=',')
else:
    x_train=np.load("../"+path+dataset+"_leafon/"+dataset+"_leafon_train_voxels_np_circ.npy")
    y_train_whole=pd.read_csv("../"+path+dataset+"_leafon/"+dataset+".leaf.on.train.csv", sep=',')

# load validation data
x_val = np.load("../"+path+dataset+"_leafon/"+dataset+"_leafon_val_voxels_np_circ.npy")
y_val_whole = pd.read_csv("../"+path+dataset+"_leafon/"+dataset+".leaf.on.val.csv", sep=',')

# load test data
x_test = np.load("../"+path+dataset+"_leafon/"+dataset+"_leafon_test_voxels_np_circ.npy")
y_test_whole = pd.read_csv("../"+path+dataset+"_leafon/"+dataset+".leaf.on.test.csv", sep=',')

# extracting relevant columns from csv files --->
v_train = y_train_whole.v.values
h_train = y_train_whole.h.values
d_train = y_train_whole.d.values
v_train = v_train.reshape(-1,1)
h_train = h_train.reshape(-1,1)
d_train = d_train.reshape(-1,1)

v_val = y_val_whole.v.values
h_val = y_val_whole.h.values
d_val = y_val_whole.d.values
v_val = v_val.reshape(-1,1)
h_val = h_val.reshape(-1,1)
d_val = d_val.reshape(-1,1)

v_test = y_test_whole.v.values
h_test = y_test_whole.h.values
d_test = y_test_whole.d.values
v_test = v_test.reshape(-1,1)
h_test = h_test.reshape(-1,1)
d_test = d_test.reshape(-1,1)
# extracting relevant columns from csv files <---

# scaling y
if scale_y:
    # scaling each set of data to the training set's attributes' means and standard deviations
    y_train_mean=[y.mean() for y in [v_train,h_train,d_train]]
    y_train_sd=[y.std() for y in [v_train,h_train,d_train]]
    v_train,h_train,d_train=[(y-y_mean)/y_sd for y,y_mean,y_sd in zip([v_train,h_train,d_train],y_train_mean,y_train_sd)]
    v_val,h_val,d_val=[(y-y_mean)/y_sd for y,y_mean,y_sd in zip([v_val,h_val,d_val],y_train_mean,y_train_sd)]
    v_test,h_test,d_test=[(y-y_mean)/y_sd for y,y_mean,y_sd in zip([v_test,h_test,d_test],y_train_mean,y_train_sd)]

# converting to binary voxels
if use_binary:
    x_train=np.where(x_train!=0,1,0)
    x_val=np.where(x_val!=0,1,0)
    x_test=np.where(x_test!=0,1,0)

# standardizing voxels using the training set's maximum
if use_std:
    x_train_max = np.max(x_train)
    x_train = x_train / x_train_max
    x_val = x_val / x_train_max
    x_test = x_test / x_train_max

# define output directory
# input/output directory
w_dirs=["./Best_Model_Saves_multi_"+dataset+"_leafon_np_"+augmented+"_"+sys.argv[1]+"_"+sys.argv[2]+"/"+for_attr+"/" for for_attr in ["v","h","d"]]
for w_dir in w_dirs:
    os.makedirs(w_dir)
#%%
################################################################################
## build custom AlexNet type of network from scratch

## define a functional model structure

# 1st convolutional layers
input_ = layers.Input(shape = (40, 40, 105))
x = layers.Conv2D(filters=1050, input_shape=(40, 40, 105), kernel_size=(2,2), strides=(1,1), padding='same')(input_)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)

# 2nd Convolutional Layer
x = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same')(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)

# 3rd Convolutional Layer
x = layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same')(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)

# 4th Convolutional Layer
x = layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same')(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)

# 5th Convolutional Layer
x = layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same')(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(x)

# passing int to a dense layer
x = layers.Flatten()(x)
x = layers.Dense(4096)(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)

# 2nd Dense Layer
x = layers.Dense(4096)(x)
x = layers.Activation('relu')(x)
x = layers.BatchNormalization()(x)

# 3rd Dense Layer
x = layers.Dense(1000)(x)
x = layers.Activation('relu')(x)

# output layer
output1 = layers.Dense(1, activation = 'relu', name = "output1")(x)
output2 = layers.Dense(1, activation = 'relu', name = "output2")(x)
output3 = layers.Dense(1, activation = 'relu', name = "output3")(x)

# model
model = Model(inputs = input_, outputs = [output1, output2, output3])

#%%
#################################################

# no of epochs
epochs = 1000

# init learning rate
init_lr = 1e-5

# optimizer
optimizer = optimizers.SGD(lr = init_lr, decay = 0)

# losses
losses = {
    'output1': 'mse',
    'output2': 'mse',
    'output3': 'mse'
}

# compile model
model.compile(loss = losses, loss_weights=l_weights, optimizer = optimizer, metrics = [rmse])

# simple early stopping
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=250)

# model checkpoint, saving only best preforming model (one for each response variable)
fa_losses = ["val_output1_rmse","val_output2_rmse","val_output3_rmse"]
mod_outs = [w_dir+"AlexNet."+dataset+"."+augmented+".h5" for w_dir in w_dirs]
mc_v,mc_h,mc_d = [ModelCheckpoint(mod_out, monitor=fa_loss, mode='min', verbose=0, save_best_only=True) for fa_loss,mod_out in zip(fa_losses,mod_outs)]

# linear learning rate scheduler
schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=init_lr, power=1)
lin = LearningRateScheduler(schedule)

# fit model
history = model.fit(x_train,
                    {"output1": v_train, "output2": h_train, "output3": d_train},
                    validation_data=(x_val,{"output1": v_val, "output2": h_val, "output3": d_val}),
                    verbose=2,
                    epochs=epochs,
                    batch_size=32,
                    callbacks=[mc_v, mc_h, mc_d, lin])
#%%
# load the saved model
dependencies = {
    'rmse': rmse
}
saved_models = [load_model(mod_out, custom_objects = dependencies) for mod_out in mod_outs]
#%%
# loading non-augmented training dataset for evaluation (augmentation only used for training)
if augmented=="aug":
    x_train=np.load("../"+path+dataset+"_leafon/"+dataset+"_leafon_train_voxels_np.npy")
    # using binary voxels if necessary
    if use_binary:
        x_train=np.where(x_train!=0,1,0)
    # standardizing voxels if necessary
    if use_std:
        x_train_max = np.max(x_train)
        x_train = x_train / x_train_max

# Evaluate the model using predict
for d_set, x_in, in zip(["train","val","test"],[x_train,x_val,x_test]):
    y_preds = [saved_model.predict(x_in) for saved_model in saved_models]
    # if scale_y:
    #     # reverse scaling if needed (augmented or non-augmented training data has same sd and mean)
    #     y_preds=[y_preds[i]*y_train_sd[i]+y_train_mean[i] for i in range(len(y_preds))]
    out_files=[dataset+".leaf.on.np."+augmented+"."+d_set+".pred.multi."+for_attr+"."+sys.argv[1]+"."+sys.argv[2]+".csv" for for_attr in ["v","h","d"]]
    _=[np.savetxt(out_file, np.column_stack((y_pred)), delimiter=";", header="v;h;d", fmt="%1.2f", comments='') for out_file,y_pred in zip(out_files,y_preds)]
