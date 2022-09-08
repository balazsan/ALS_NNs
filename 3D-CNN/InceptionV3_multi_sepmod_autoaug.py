# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 13:06:58 2017

@author: Elias Ayrey
@edited: Andras Balazs (on Fri May 10 11:55:30 2019)

The original script can be found at:
https://github.com/eayrey/3D-Convolutional-Neural-Networks-with-LiDAR/tree/master/InceptionV3-3D_Neural_Network

This is the main script for developing a neural network from point cloud and field inventory data
What you need:
    - A saved numpy array of voxelized point cloud data with dimensions (n,40,40,105) produced by the 
    point_cloud_voxelizer.py script.
    - A CSV containing a column with field measured values (biomass or otherwise) in n plots
    - A CSV with withheld indices for model validation
    - Tensorflow-GPU must be installed, you need a good NVIDIA GPU if you expect these to ever train

This is a multi-output version which saves a model for each estimated variable separately
(estimating all forest variables at the same time and saving best model based on the RMSE of each variable)
Training can be carried out with all variables at the time or alternatingly (each variable separately during each epoch)

Added automatic augmentation for training before feeding training data batch to training
Removed using augmented validation data
"""
import glob, os
import tensorflow as tf
import numpy as np
import pandas as pd
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#%%
#################################################################
########################### LOAD DATA ###########################
#################################################################

#############################################
# --> the following variables can be modified
# intensity or number-of-points voxels
v_type="np"
#v_type="i"

# batch size is duplicated due to augmentation during training
batch_size=8
# learning rate to be used during training
l_rate=0.0001
#l_rate=0.001

# should alternate model training be used?
# model trained one variable at the time in each iteration
#train_alt=True
train_alt=False

# use augmented training data
augmented="aug"
#augmented="noaug"

# using standardized voxels
use_std=False
# use_std=True

# binary voxel or not
#use_binary=False
use_binary=True

# intensity voxel cannot be binary and voxel standardization makes no sense for binary voxels
if v_type=="i" or use_std:
    use_binary=False

# forest attributes to estimate (dependent variables)
for_attrs=["v","h","d"]

# should scaling of y variables be done?
scale_y=True
#scale_y=False
# <-- the above variables can be modified
#############################################

# loading datasets (x and y; predictors and response)
x_all=np.load("../voxel_data/voxel_data.npz")

#training dataset
train_xs=x_all['train']
train_ys_orig=pd.read_csv("../sample_plot_data/sp_data_train.aug.csv", sep=',')
#training data includes augmented data, keeping only observerd data as augmentation will be done on-the-fly
train_xs=train_xs[0:1044]
train_ys_orig=train_ys_orig[0:1044]

#validation dataset
validation_xs=x_all['val']
validation_ys_orig=pd.read_csv("../sample_plot_data/sp_data_val.csv", sep=',')

#test dataset
test_xs=pd.read_csv("../sample_plot_data/sp_data_test.csv", sep=',')

# putting forest attribute columns into a list of numpy arrays
train_ys=[train_ys_orig[for_attr].to_numpy().astype("float32").reshape(-1,1) for for_attr in for_attrs]
validation_ys=[validation_ys_orig[for_attr].to_numpy().astype("float32").reshape(-1,1) for for_attr in for_attrs]

if scale_y:
    # scaling each set of data to the training set's attributes' means and standard deviations
    train_ys_mean=[train_ys[i].mean() for i in range(len(train_ys))]
    train_ys_sd=[train_ys[i].std() for i in range(len(train_ys))]
    train_ys=[(train_ys[i]-train_ys_mean[i])/train_ys_sd[i] for i in range(len(train_ys))]
    validation_ys=[(validation_ys[i]-train_ys_mean[i])/train_ys_sd[i] for i in range(len(validation_ys))]

# mean values for %RMSE calculation
val_ys_means=validation_ys_orig[for_attrs].mean().to_numpy()

# converting to binary voxels
if use_binary:
    train_xs=np.where(train_xs!=0,1,0)
    validation_xs=np.where(validation_xs!=0,1,0)
    test_xs=np.where(test_xs!=0,1,0)

# standardizing voxels using the training set's maximum
if use_std:
    train_xs_max=np.max(train_xs)
    train_xs=np.where(train_xs!=0,train_xs/train_xs_max,0)
    validation_xs=np.where(validation_xs!=0,validation_xs/train_xs_max,0)
    test_xs=np.where(test_xs!=0,test_xs/train_xs_max,0)

# input/output directory
v_type_orig=v_type
if use_binary:
    v_type="bin"
w_dirs=["./Best_Model_Saves_"+v_type+"_"+augmented+"_multi"+"/"+for_attr+"/" for for_attr in for_attrs]

for w_dir in w_dirs:
    os.makedirs(w_dir)

#defining weight variables from random normal curve with a shape of the input
def weight_variable(shape):
    initial= tf.truncated_normal(shape, stddev=.05)
    return tf.Variable(initial)

#defining bias variables as all starting as .1, with shape of the input    
def bias_variable(shape):
    initial=tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#A simple convolution layer in our network
def Conv_layer(inputs,kernal,stride,shape,pad='VALID'):
    #training was copied here from line def model, otherwise not defined for Conv_layer
    #training switch, used with batch norm. Switch to false during validation.
    training = tf.placeholder_with_default(True, shape=())
    weights=weight_variable(kernal+shape)
    biases=bias_variable([shape[1]])
    conv=tf.nn.conv3d(inputs, weights, strides=[1]+stride+[1], padding=pad)
    added =tf.nn.bias_add(conv, biases)
    norm=tf.layers.batch_normalization(added,momentum=0.99, center=True, scale=True,training=training, name='BN'+str(int(np.random.randint(low=0, high=1000000, size=1))))
    # original activation function
    h_conv=tf.nn.elu(norm)
    return h_conv

#An inception layer comprised of many convolutions
def Inception_layer1(intensor, outshapes):
    shape = intensor.get_shape().as_list()
    #first bit
    Ia_pool1=tf.nn.max_pool3d(intensor, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='SAME')
    Ia_conv1=Conv_layer(Ia_pool1,[1,1,1],[1,1,1],[shape[4],outshapes[3]],pad='SAME')
    #second bit
    Ib_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],64],pad='SAME')
    Ib_conv2=Conv_layer(Ib_conv1,[2,2,3],[1,1,1],[64,96],pad='SAME')
    Ib_conv3=Conv_layer(Ib_conv2,[2,2,3],[1,1,1],[96,outshapes[2]],pad='SAME')
    #third bit
    Ic_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],48],pad='SAME')
    Ic_conv2=Conv_layer(Ic_conv1,[3,3,4],[1,1,1],[48,outshapes[1]],pad='SAME')
    #forth bit
    Id_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],outshapes[0]],pad='SAME')
    
    I_concat=tf.concat(axis=4, values=[Ia_conv1,Ib_conv3,Ic_conv2,Id_conv1])
    return I_concat

#The second inception layer
def Inception_layer2(intensor, outshapes, x):   
    shape = intensor.get_shape().as_list()
    #first bit
    Ia_pool1=tf.nn.max_pool3d(intensor, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='SAME')
    Ia_conv1=Conv_layer(Ia_pool1,[1,1,1],[1,1,1],[shape[4],outshapes[3]],pad='SAME')
    #second bit
    Ib_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],x],pad='SAME')
    Ib_conv2=Conv_layer(Ib_conv1,[1,1,6],[1,1,1],[x,x],pad='SAME')
    Ib_conv3=Conv_layer(Ib_conv2,[1,5,1],[1,1,1],[x,x],pad='SAME')
    Ib_conv4=Conv_layer(Ib_conv3,[5,1,1],[1,1,1],[x,x],pad='SAME')
    Ib_conv5=Conv_layer(Ib_conv4,[1,1,6],[1,1,1],[x,outshapes[2]],pad='SAME')
    #third bit
    Ic_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],x],pad='SAME')
    Ic_conv2=Conv_layer(Ic_conv1,[1,1,6],[1,1,1],[x,x],pad='SAME')
    Ic_conv3=Conv_layer(Ic_conv2,[5,1,1],[1,1,1],[x,x],pad='SAME')
    Ic_conv4=Conv_layer(Ic_conv3,[1,5,1],[1,1,1],[x,outshapes[1]],pad='SAME')    
    #forth bit
    Id_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],outshapes[0]],pad='SAME')
    
    I_concat=tf.concat(axis=4, values=[Ia_conv1,Ib_conv5,Ic_conv4,Id_conv1])
    return I_concat

#Final inception layer
def Elephant_foot(intensor):
    shape = intensor.get_shape().as_list()

    Ia_pool1=tf.nn.max_pool3d(intensor, ksize=[1,2,2,2,1], strides=[1,1,1,1,1], padding='SAME') 
    Ia_conv1=Conv_layer(Ia_pool1,[1,1,1],[1,1,1],[shape[4],192],pad='SAME')

    Ib_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],448],pad='SAME')
    Ib_conv2=Conv_layer(Ib_conv1,[2,2,2],[1,1,1],[448,384],pad='SAME')
    Ib1_conv1=Conv_layer(Ib_conv2,[1,1,3],[1,1,1],[384,256],pad='SAME')
    Ib2_conv1=Conv_layer(Ib_conv2,[2,1,1],[1,1,1],[384,256],pad='SAME')
    Ib3_conv1=Conv_layer(Ib_conv2,[1,2,1],[1,1,1],[384,256],pad='SAME')

    Ic_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],384],pad='SAME')
    Ic1_conv1=Conv_layer(Ic_conv1,[1,1,3],[1,1,1],[384,256],pad='SAME')
    Ic2_conv1=Conv_layer(Ic_conv1,[2,1,1],[1,1,1],[384,256],pad='SAME')
    Ic3_conv1=Conv_layer(Ic_conv1,[1,2,1],[1,1,1],[384,256],pad='SAME')
    
    Id_conv1=Conv_layer(intensor,[1,1,1],[1,1,1],[shape[4],320],pad='SAME') 
    
    I_concat=tf.concat(axis=4, values=[Ia_conv1,Ib1_conv1,Ib2_conv1,Ib3_conv1,Ic1_conv1,Ic2_conv1,Ic3_conv1,Id_conv1])
    return I_concat

def model(train_xs,train_ys,validation_xs,validation_ys,step,batchS,record_low,optimizer):
    #specify placeholders for tensorflow in the shape of our input data
    xs=tf.placeholder(tf.float32, [None,40,40,105], name='Xinput') 
    #ys=tf.placeholder(tf.float32, [None,1], name='Yinput') 
    ys1=tf.placeholder(tf.float32, [None,1], name='Yinput1') 
    ys2=tf.placeholder(tf.float32, [None,1], name='Yinput2') 
    ys3=tf.placeholder(tf.float32, [None,1], name='Yinput3') 
    #training switch, used with batch norm. Switch to false during validation.
    training = tf.placeholder_with_default(True, shape=())
    
    #Our model input reshaped into tensorflow's prefered standard
    x_image=tf.reshape(xs, [-1,40,40,105,1])
    ####################################################################
    ########################### DEFINE MODEL ###########################
    ####################################################################

    ################initial conv layers######################    
    W_conv1=weight_variable([2,2,3,1,32])#filter size 5x5, 1 band, 32 convolutions out
    b_conv1=bias_variable([32])#biases for each of the 32 convolutions    
    conv1= tf.nn.conv3d(x_image, W_conv1, strides=[1,2,2,2,1], padding='VALID')#convolutions, stride equals 1, 
    a_conv1 = tf.nn.bias_add(conv1, b_conv1)    
    h_norm1=tf.layers.batch_normalization(a_conv1,momentum=0.99, center=True, scale=True,training=training, name='BN'+str(int(np.random.randint(low=0, high=1000000, size=1))))
    r_conv1=tf.nn.relu(h_norm1)

    W_conv2=weight_variable([2,2,3,32,32])#filter size 5x5, 1 band, 32 convolutions out
    b_conv2=bias_variable([32])#biases for each of the 32 convolutions    
    conv2= tf.nn.conv3d(r_conv1, W_conv2, strides=[1,1,1,1,1], padding='VALID')#convolutions, stride equals 1, 
    a_conv2 = tf.nn.bias_add(conv2, b_conv2)    
    h_norm2=tf.layers.batch_normalization(a_conv2,momentum=0.99, center=True, scale=True,training=training, name='BN'+str(int(np.random.randint(low=0, high=1000000, size=1))))
    r_conv2=tf.nn.relu(h_norm2)

    W_conv3=weight_variable([2,2,3,32,64])#filter size 5x5, 1 band, 32 convolutions out
    b_conv3=bias_variable([64])#biases for each of the 32 convolutions    
    conv3= tf.nn.conv3d(r_conv2, W_conv3, strides=[1,1,1,1,1], padding='SAME')#convolutions, stride equals 1, 
    a_conv3 = tf.nn.bias_add(conv3, b_conv3)
    h_norm3=tf.layers.batch_normalization(a_conv3,momentum=0.99, center=True, scale=True,training=training, name='BN'+str(int(np.random.randint(low=0, high=1000000, size=1))))
    r_conv3=tf.nn.relu(h_norm3)
    #Dimensionality Redux
    h_pool3=tf.nn.max_pool3d(r_conv3, ksize=[1,1,1,2,1], strides=[1,1,1,1,1], padding='SAME')

    W_conv4=weight_variable([1,1,1,64,80])#filter size 5x5, 1 band, 32 convolutions out
    b_conv4=bias_variable([80])#biases for each of the 32 convolutions    
    conv4= tf.nn.conv3d(h_pool3, W_conv4, strides=[1,1,1,1,1], padding='SAME')#convolutions, stride equals 1, 
    a_conv4 = tf.nn.bias_add(conv4, b_conv4)    
    h_norm4=tf.layers.batch_normalization(a_conv4,momentum=0.99, center=True, scale=True,training=training, name='BN'+str(int(np.random.randint(low=0, high=1000000, size=1))))
    r_conv4=tf.nn.relu(h_norm4)

    W_conv5=weight_variable([2,2,3,80,192])#filter size 5x5, 1 band, 32 convolutions out
    b_conv5=bias_variable([192])#biases for each of the 32 convolutions    
    conv5= tf.nn.conv3d(r_conv4, W_conv5, strides=[1,1,1,1,1], padding='SAME')#convolutions, stride equals 1, 
    a_conv5 = tf.nn.bias_add(conv5, b_conv5)    
    h_norm5=tf.layers.batch_normalization(a_conv5,momentum=0.99, center=True, scale=True,training=training, name='BN'+str(int(np.random.randint(low=0, high=1000000, size=1))))
    r_conv5=tf.nn.relu(h_norm5)

    h_pool5=tf.nn.max_pool3d(r_conv5, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

    #Inception layers
    In1=Inception_layer1(h_pool5, [64,64,96,32])
    In2=Inception_layer1(In1, [64,64,96,64])
    In3=Inception_layer1(In2, [64,64,96,64])
    
    #Dimensionality Redux
    #first side channel
    In4a_pool1=tf.nn.max_pool3d(In3, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='VALID')
    #second side channel
    In4b_conv1=Conv_layer(In3,[1,1,1],[1,1,1],[288,64],pad='SAME')
    In4b_conv2=Conv_layer(In4b_conv1,[2,2,2],[1,1,1],[64,96],pad='SAME')
    In4b_conv3=Conv_layer(In4b_conv2,[2,2,2],[2,2,2],[96,96],pad='VALID')
    #third side channel
    In4c_conv1=Conv_layer(In3,[2,2,2],[2,2,2],[288,384],pad='VALID')
    #concat
    In4_concat=tf.concat(axis=4, values=[In4a_pool1,In4b_conv3,In4c_conv1])
    
    #Inception layers
    In5=Inception_layer2(In4_concat, [192,192,192,192], 128)
    In6=Inception_layer2(In5, [192,192,192,192], 160)
    In7=Inception_layer2(In6, [192,192,192,192], 160)
    In8=Inception_layer2(In7, [192,192,192,192], 192)
    
    #Dimensionality Redux
    #first side channel
    In9a_pool1=tf.nn.max_pool3d(In8, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
    #second side channel
    In9b_conv1=Conv_layer(In8,[1,1,1],[1,1,1],[768,192],pad='SAME')
    In9b_conv2=Conv_layer(In9b_conv1,[1,1,6],[1,1,1],[192,192],pad='SAME')
    In9b_conv3=Conv_layer(In9b_conv2,[1,5,1],[1,1,1],[192,192],pad='SAME')
    In9b_conv4=Conv_layer(In9b_conv3,[5,1,1],[1,1,1],[192,192],pad='SAME')
    In9b_conv5=Conv_layer(In9b_conv4,[2,2,2],[2,2,2],[192,192],pad='SAME')
    #third side channel
    In9c_conv1=Conv_layer(In8,[1,1,1],[1,1,1],[768,192],pad='SAME')
    In9c_conv2=Conv_layer(In9c_conv1,[2,2,2],[2,2,2],[192,320],pad='SAME')
    #concat
    In9_concat=tf.concat(axis=4, values=[In9a_pool1,In9b_conv5,In9c_conv2])    
    
    #Final inception layers
    In10=Elephant_foot(In9_concat)
    In11=Elephant_foot(In10)
    
    #Dimensionality Redux
    FC_pool=tf.nn.avg_pool3d(In11, ksize=[1,3,3,6,1], strides=[1,1,1,1,1], padding='VALID')
    
    #Fully connected layer leading into prediction
    shape = FC_pool.get_shape().as_list()
    FC_flat=tf.reshape(FC_pool, [-1, shape[1] * shape[2] * shape[3]* shape[4]])
#    W_FCo=weight_variable([2048, 1])
    W_FCo1=weight_variable([2048, 1])
    W_FCo2=weight_variable([2048, 1])
    W_FCo3=weight_variable([2048, 1])
#    b_FCo=bias_variable([1])
    b_FCo1=bias_variable([1])
    b_FCo2=bias_variable([1])
    b_FCo3=bias_variable([1])
#    prediction=(tf.matmul(FC_flat,W_FCo) + b_FCo)#output
    prediction1=(tf.matmul(FC_flat,W_FCo1) + b_FCo1)#output
    prediction2=(tf.matmul(FC_flat,W_FCo2) + b_FCo2)#output
    prediction3=(tf.matmul(FC_flat,W_FCo3) + b_FCo3)#output
    
    # Our loss
    loss1=tf.reduce_mean(tf.square(ys1-prediction1))
    loss2=tf.reduce_mean(tf.square(ys2-prediction2))
    loss3=tf.reduce_mean(tf.square(ys3-prediction3))

    # Necessary crap for batch normalization, cuz tensorflow
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        #our optimizer
        if optimizer=="adam":
            train_step1=tf.train.AdamOptimizer(step, name='AdamOptimizer').minimize(loss1)
            train_step2=tf.train.AdamOptimizer(step, name='AdamOptimizer').minimize(loss2)
            train_step3=tf.train.AdamOptimizer(step, name='AdamOptimizer').minimize(loss3)
        else:
            train_step1=tf.train.RMSPropOptimizer(step, name='RMSPropOptimizer').minimize(loss1)
            train_step2=tf.train.RMSPropOptimizer(step, name='RMSPropOptimizer').minimize(loss2)
            train_step3=tf.train.RMSPropOptimizer(step, name='RMSPropOptimizer').minimize(loss3)

    ########################################################################
    ########################### INITIALIZE MODEL ###########################
    ########################################################################
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    init=tf.global_variables_initializer()
    savers = [tf.train.Saver(max_to_keep=1),tf.train.Saver(max_to_keep=1),tf.train.Saver(max_to_keep=1)]
    sess.run(init)
    
    ################################################################
    ########################### TRAINING ###########################
    ################################################################
    #overtrain indicator to show when loss/accuracy is not improving any more
    overtrain_indicator=[0,0,0]
    niter=20000
    for i in range(niter):
        #for each training step, withhold a batch of data, train 
        indices=np.random.randint(low=0, high=len(train_ys[0]), size=[batchS,])
        batch_xs=train_xs[indices]
        batch_ys=[train_ys[ii][indices] for ii in range(len(train_ys))]
        # do augmentation on-the-fly
        # augmenting data by rotating the arrays by 90/180/270 degrees randomly clock-wise around the Z axis
        # creating vector for magnitude of rotation randomly (1=90deg,2=180deg,3=270deg)
        if augmented=="aug":
            # define magnitude of rotations
            rot_dir=np.random.randint(low=1,high=4,size=len(batch_xs))
            # rotate voxel spaces
            batch_xs_rot=[np.rot90(batch_xs[ii],k=rot_dir[ii],axes=(1,0)) for ii in range(len(batch_xs))]
            batch_xs_rot=np.array(batch_xs_rot)
            # adding augmented part to training batch
            batch_xs=np.concatenate((batch_xs,batch_xs_rot))
            # duplicating forest attributes for training batch
            batch_ys=[np.tile(batch_ys[ii],(2,1)) for ii in range(len(batch_ys))]
        # separate losses, alternate training
        if train_alt:
            # training with one response variable at the same time
            sess.run(train_step1, feed_dict={xs:batch_xs,ys1:batch_ys[0]})
            sess.run(train_step2, feed_dict={xs:batch_xs,ys2:batch_ys[1]})
            sess.run(train_step3, feed_dict={xs:batch_xs,ys3:batch_ys[2]})
        # separate losses, trained with all response variables at the same time
        else:
            sess.run([train_step1,train_step2,train_step3], feed_dict={xs:batch_xs,ys1:batch_ys[0],ys2:batch_ys[1],ys3:batch_ys[2]})
        
        #run a validation test every 100 steps
        if i%100==0 or i==(niter-1):
            train_acc = sess.run([loss1,loss2,loss3], feed_dict={xs:batch_xs,ys1:batch_ys[0],ys2:batch_ys[1],ys3:batch_ys[2],training: False})
            train_acc = [np.around(train_acc[ii],5) for ii in range(len(train_acc))]
            
            # assess model accuracy on validation data in small pieces (don't want to overload GPU mem)
            # instead of calculating means of MSE and RMSE over batches, get all predictions
            # and calculate MSE and RMSE of the entire validation set
            preds1=[]; preds2=[]; preds3=[]
            val_bat = 25
            for n in range(validation_ys[0].shape[0]//val_bat):
                if n<(validation_ys[0].shape[0]//val_bat-1):
                    batch_valid_xs=validation_xs[val_bat*n:val_bat*(n+1)]
                    batch_valid_ys=[validation_ys[0][val_bat*n:val_bat*(n+1)],
                                    validation_ys[1][val_bat*n:val_bat*(n+1)],validation_ys[2][val_bat*n:val_bat*(n+1)]]
                else:
                    batch_valid_xs=validation_xs[val_bat*n:]
                    batch_valid_ys=[validation_ys[0][val_bat*n:],validation_ys[1][val_bat*n:],validation_ys[2][val_bat*n:]]

                #obtain prediction and loss (MSE) from model, training set to false
                pred1, pred2, pred3 = sess.run([prediction1,prediction2,prediction3],
                                               feed_dict={xs:batch_valid_xs,ys1:batch_valid_ys[0],
                                                          ys2:batch_valid_ys[1],ys3:batch_valid_ys[2],training: False})
                # output is a nested list
                preds1.append(pred1); preds2.append(pred2); preds3.append(pred3)

            # merging predictions over batches
            preds1=np.concatenate(preds1); preds2=np.concatenate(preds2); preds3=np.concatenate(preds3)
            preds=[preds1, preds2, preds3]
            # accuracy measures (using relative RMSE)
            if scale_y:
                RMSEs = [np.around(np.sqrt(np.mean(((validation_ys[ii].flatten()*train_ys_sd[ii]+train_ys_mean[ii])-(preds[ii].flatten()*train_ys_sd[ii]+train_ys_mean[ii]))**2))/val_ys_means[ii]*100,2) for ii in range(len(preds))]
            else:
                RMSEs = [np.around(np.sqrt(np.mean((validation_ys[ii].flatten()-preds[ii].flatten())**2))/val_ys_means[ii]*100,2) for ii in range(len(preds))]
            #save model if its the best so far
            if i>1000:
                # using relative RMSEs to make decision for model saving
                for ii in range(len(record_low)):
                    if RMSEs[ii] < record_low[ii]:
                        savers[ii].save(sess, w_dirs[ii]+str(i)+'_'+"{:.2f}".format(RMSEs[0])+'_'+
                                        "{:.2f}".format(RMSEs[1])+'_'+"{:.2f}".format(RMSEs[2]))
                        record_low[ii]=RMSEs[ii]
                        overtrain_indicator[ii]=0
                    else:
                        overtrain_indicator[ii]+=1
            print(i, train_acc[0], train_acc[1], train_acc[2], RMSEs[0], RMSEs[1], RMSEs[2],
                  overtrain_indicator[0], overtrain_indicator[1], overtrain_indicator[2], flush=True)
    sess.close()
    return record_low

# running the model
model(train_xs,train_ys,validation_xs,validation_ys,step=l_rate,batchS=batch_size,record_low=[50.,25.,30.],optimizer="adam")

#%% 
# Calculating predictions of training, validation and test datasets

#Specify if you want to use the GPU or CPU. The GPU is required for training. For validation or prediction a powerful CPU will do.
num_cores = 6
GPU=False
CPU=True
if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

xs_list=[train_xs,validation_xs,test_xs]

# output directories for predictions
outpaths=["preds_"+for_attr for for_attr in for_attrs]

# iterating through input voxels and labels and calculating predictions using best model
for i in range(len(w_dirs)):
    w_dir=w_dirs[i]
    ############################ LOAD DATA ############################
    # getting saved model from output directory and making sure only one model is selected (last is the best based on validation loss)
    in_meta=glob.glob(w_dir+"*.meta")[-1]
        
    for in_xs, d_set in zip(xs_list,["train","val","test"]):
        ############################ Initialize Model ######################################
        config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
                 inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
                 device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    
        sess = tf.Session(config=config)
        init=tf.global_variables_initializer()
        sess.run(init)

        # importing saved model
        saver = tf.train.import_meta_graph(in_meta)
        saver.restore(sess, in_meta[:-5])
        graph=tf.get_default_graph()

        # extract the model inputs from the graph
        xs0=graph.get_tensor_by_name("Xinput:0")
    
        # switching training off for prediction
        training=graph.get_tensor_by_name("PlaceholderWithDefault/input:0")
    
        # extract the model output from the graph
        prediction1=graph.get_tensor_by_name("add:0")
        prediction2=graph.get_tensor_by_name("add_1:0")
        prediction3=graph.get_tensor_by_name("add_2:0")
    
        # calculating predictions for input data
        preds1=[]; preds2=[]; preds3=[]
        val_bat = 50
        for n in range(in_xs.shape[0]//val_bat):
            if n<(in_xs.shape[0]//val_bat-1):
                batch_in_xs=in_xs[val_bat*n:val_bat*(n+1)]
            else:
                batch_in_xs=in_xs[val_bat*n:]
            # extracting predictions
            pred1, pred2, pred3 = sess.run([prediction1,prediction2,prediction3],feed_dict={xs0: batch_in_xs, training: False})
            # appending batches of predictions to lists
            preds1.append(pred1); preds2.append(pred2); preds3.append(pred3)
        preds1=np.concatenate(preds1); preds2=np.concatenate(preds2); preds3=np.concatenate(preds3)
        preds=[preds1.flatten(), preds2.flatten(), preds3.flatten()]
        if scale_y:
            # reverse scaling if needed (augmented or non-augmented training data has same sd and mean)
            preds=[preds[ii]*train_ys_sd[ii]+train_ys_mean[ii] for ii in range(len(preds))]
        out_file=outpaths[i]+"/pred.multi."+v_type+"."+augmented+"."+d_set+".csv"
        # exporting field observations vs predictions
        np.savetxt(out_file, np.column_stack((preds)), delimiter=";", header="v;h;d", fmt="%1.1f", comments='')
        del sess, in_xs
