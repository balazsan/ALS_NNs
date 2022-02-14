# Functions to estimate forest attributes based on field measurements
# and remote sensing features using Keras with TensorFlow (TF) back-end.
# Using densely connected layers, input-1 hidden layer-output. (need for 2 hidden layers?)
# Using relative RMSE as custom metrics
# Andras Balazs, Luke, 22.08.2019

# Changes:
# 27.11.2019: added option to feed in one attribute at the time
# 03.04.2020: added option to scale x and y here (also de-scale predictions) using training data and labels mean and sd

# the keras library need to be installed with TF back-end
library(keras)

#' Calculate relative RMSE for predictions
#' 
#' @param pred Table of predictions
#' @param obs Table of observations
#' @param n.decim Number of decimals for output
rel.rmse <- function(obs,pred,n.decim=2) {
  out <- mapply(function(x,y) {
    round(sqrt(mean(x^2))/mean(y)*100,n.decim)
  },pred-obs,obs)
  names(out) <- names(obs)
  return(out)
}

#' Calculate relative bias for predictions
#' Underestimation gives negative bias
#' 
#' @param pred Table of predictions
#' @param obs Table of observations
#' @param n.decim Number of decimals for output
rel.bias <- function(obs,pred,n.decim=2) {
  out <- mapply(function(x,y) {
    round(mean(x)/mean(y)*100,2)
  },pred-obs,obs)
  names(out) <- names(obs)
  return(out)
}

#' Creating custom metrics (relative RMSE) (if not defined batch_size=32)
#' During training if batch size is smaller than test sample size RMSE is calculated over batches
#' and will be slightly different from the RMSE calculated over the entire dataset
metric_rmse <- custom_metric("rmse",function(y_true,y_pred) {
  k_sqrt(k_mean(k_square(y_true-y_pred)))/k_mean(y_true)*100
})
# in case the above doesn't work out leaving the deprecated method
# metric_rmse <- function(y_true,y_pred) {
#   k_sqrt(k_mean(k_square(y_true-y_pred)))/k_mean(y_true)*100
# }

#' Custom activation functions (swish/e-swish) (https://github.com/EricAlcaide/E-swish)
#' With beta=1 called swish and beta>1 (e.g. 1.25,1.50,1.75,2.00) e-swish
#' s.beta has to be declared like that as the function is not callable in layer_dense with parameters
swish_activation <- function(x) {
  s.beta <- 1
  s.beta*x*k_sigmoid(x)
}

#' Building predictions using built-in activation function
#' this is needed to build a network with multiple outputs (predictions)
build.pred <- function(pred,act,inputs,b.i) {
  inputs %>%
    layer_dense(units=n.neur,activation=act,use_bias=T,bias_initializer=initializer_constant(b.i)) %>%
    layer_dense(units=1,name=pred)
}

#' Building predictions using custom activation function
#' this is needed to build a network with multiple outputs (predictions)
build.pred.swish <- function(pred,inputs,b.i) {
  inputs %>%
    layer_dense(units=n.neur,activation=swish_activation,use_bias=T,bias_initializer=initializer_constant(b.i)) %>%
    layer_dense(units=1,name=pred)
}

#' Creating sequential model with one densely connected hidden layer
#' 
#' @param train.data Vector/matrix of features for training (needs scaling if multiple variables have different value range)
#' @param train.labels Vector/matrix of observations for training
#' @param val.data Vector/matrix of features for validation (needs scaling if multiple variables have different value range)
#' @param val.labels Vector/matrix of observations for validation
#' @param n.neur Integer, number of neurons in the hidden layer
#' @param preds.w Weights for variables to predict (has to sum up to 1), NULL if only one variable to predict
#         s.beta Float, parameter to be passed on to the swish activation function (not in use at the moment)
#' @param scl Character, should x (.data) or y (.labels) or both be scaled? ("x","y","both")
#' @param act Character, name of activation function to be used in hidden layer (e.g. "swish","relu","sigmoid")
#'        for a list of available functions see https://keras.rstudio.com/reference/index.html#section-activations
#' @param opt Character, name of optimizer to be used in the model (e.g. "adam","rmsprop")
#'        for a list of available optimizers see https://keras.rstudio.com/reference/index.html#section-optimizers
#' @param loss Character, name of loss function to be used during model training (e.g. "mean_squared_error","mean_absolute_error")
#'        for a list of available functions see https://keras.rstudio.com/reference/index.html#section-losses
#' @param metric Character, name of metric function to be used during model training (e.g. "rmse","mean_squared_error","mean_absolute_error")
#'        for a list of available functions see https://keras.rstudio.com/reference/index.html#section-metrics
#'        for relative RMSE set this parameter to "rmse"
#' @param b.i Integer, constant used in bias initializer
#' @param epochs Integer, maximum number of epochs to train the model
#' @param patience Integer, number of epochs with no improvement after which training will be stopped
#' @param batch_size Integer, number of samples per gradient update
#' @param view_metrics Boolean, view real-time plot of training metrics (by epoch)
#' @param outdir Character, name of output directory
#' @param fit.verbose Integer, should a progress bar be displayed for every epoch; 0=don't show, 1=show
#'        rmse and bias for training and test data, training history and best model will be saved here
#' @param out Character, "pred" for returning predictions and anything else for accuracy measures (relative RMSE and bias)
tf.model.res <- function(train.data,train.labels,val.data,val.labels,n.neur=NULL,preds.w=NULL,scl="both",act="swish",opt="adam",
                         loss="mean_squared_error",metric="rmse",b.i=0,epochs=100,patience=20,batch_size=25,view_metrics=F,
                         outdir="tf.results",fit.verbose=0,out="") {
  # calculating n.neur if not defined
  if (is.null(n.neur)) {
    n.neur <- ceiling((ncol(train.data)+ncol(train.labels))/2)
    cat(paste0("Number of nodes used in hidden layer: ",n.neur))
  }

  # stop if inputs are not of the same type
  if ((!is.vector(train.labels)&is.vector(val.labels))|(is.vector(train.labels)&!is.vector(val.labels))) {
    stop("Train and test labels are not the same type (vector/matrix)!")
  }
  if (!is.vector(train.labels)) {
    # stop if numbers of columns of train and test labels not matching
    if (ncol(train.labels)!=ncol(val.labels)) stop("Train and test labels don't have the same number of columns!")
    # stop if weights don't sum up to 1
    if (!is.null(preds.w)&sum(preds.w)!=1) stop("Sum of prediction wieghts not 1!")
    # stop if weights and columns in train.labels don't match
    if (!is.null(preds.w)&length(preds.w)!=ncol(train.labels)) stop("Length of pred.w doesn't match number of columns in train.labels!")
  } else {
    if (!is.null(preds.w)&length(preds.w)>1) preds.w <- 1
  }
  
  # creating input layer
  if (is.vector(train.data)) in.dim <- 1 else in.dim <- ncol(train.data)
  inputs <- layer_input(shape=in.dim)
  
  # creating list of variable names to be predicted
  # dots trigger errors in some python process
  if (is.vector(train.labels)) {preds.names <- "pred_var"} else {preds.names <- gsub("\\.","_",colnames(train.labels))}
  
  if (is.vector(train.labels)) {
    # converting vector to data frame
    train.labels <- as.data.frame(train.labels)
    val.labels <- as.data.frame(val.labels)
  }
    
  # scaling x and/or y if necessary
  if (scl %in% c("x","both")) {
    train.data <- scale(train.data)
    mean.train <- attr(train.data,"scaled:center")
    sd.train <- attr(train.data,"scaled:scale")
    val.data <- scale(val.data,center=mean.train,scale=sd.train)
  }
  if (scl %in% c("y","both")) {
    train.labels.sc <- scale(train.labels)
    mean.train <- attr(train.labels.sc,"scaled:center")
    sd.train <- attr(train.labels.sc,"scaled:scale")
    # model fit doesn't accept the output of scale() for labels
    val.labels.sc <- as.data.frame(scale(val.labels,center=mean.train,scale=sd.train))
    train.labels.sc <- as.data.frame(train.labels.sc)
  } else {
    val.labels.sc <- val.labels
    train.labels.sc <- train.labels
  }
  
  # setting prediction weights' names
  if (!is.null(preds.w)) names(preds.w) <- preds.names
  
  # creating network structure
  if (act=="swish") {
    preds <- lapply(preds.names,function(x) build.pred.swish(pred=x,inputs=inputs,b.i=b.i))
  } else {
    preds <- lapply(preds.names,function(x) build.pred(pred=x,act=act,inputs=inputs,b.i=b.i))
  }
  
  # creating and compiling model
  tf.model <- keras_model(inputs=inputs,outputs=preds)
  # for further process original value needs to be preserved
  metric.orig <- metric
  if (metric=="rmse") metric <- metric_rmse
  if (!is.null(preds.w)&length(preds.w)==1) preds.w <- list(preds.w)
  tf.model %>% compile(
    optimizer=opt,
    loss=loss,
    loss_weights=preds.w,
    metrics=metric
    # deprecated method
    # metrics=list(rmse=metric_rmse)
  )
  
  if (!dir.exists(outdir)) dir.create(outdir,recursive=T)
  cp.outdir <- paste0(outdir,"/checkp_dir"); if (!dir.exists(cp.outdir)) dir.create(cp.outdir)
  filepath <- file.path(cp.outdir,"model.{epoch:03d}-{loss:.2f}-{val_loss:.2f}.hdf5")

  # creating checkpoint callback
  # model saved after each epoch if the loss is smaller than previous min
  cp_callback <- callback_model_checkpoint(
    filepath=filepath,monitor="val_loss",
    save_weights_only=F,save_best_only=T,
    mode="min",verbose=0
  )
  
  # the patience parameter is the amount of epochs to check for improvement
  early.stop <- callback_early_stopping(monitor="val_loss",patience=patience)
  # fit the model and store training stats
  history <- fit(tf.model,train.data,train.labels.sc,epochs=epochs,
                 batch_size=batch_size,validation_data=list(val.data,val.labels.sc),
                 view_metrics=view_metrics,verbose=fit.verbose,callbacks=list(early.stop,cp_callback)
  )
  
  # removing model to avoid confusion
  rm(tf.model)
  
  # saving training history
  # creating unique output name for batch executing function on the same data
  f.count <- length(list.files(outdir,"history"))+1
  history.out <- paste0(outdir,"/TensorFlow_history_run",f.count,".RDS")
  saveRDS(history,history.out)
  
  # copying best model to output directory with unique name
  f.count <- length(list.files(outdir,"*.hdf5$"))+1
  best.models <- list.files(cp.outdir,"*.hdf5$",full.names=T)
  best.model.path <- best.models[length(best.models)]
  best.model.copy <- paste0(tools::file_path_sans_ext(best.model.path),"_run",f.count,".hdf5")
  best.model.copy <- gsub("checkp_dir/","",best.model.copy)
  file.rename(best.model.path,best.model.copy)
  # removing checkpoint directory and its contents
  unlink(cp.outdir,recursive=T)
  
  # loading best model to calculate predictions and validation results
  # due to custom functions that need to be declared in load_model_hdf5
  # it is important to check which functions were used
  if (metric.orig=="rmse") {
    if (act=="swish") {
      best.model <- load_model_hdf5(best.model.copy,custom_objects=c("rmse"=metric_rmse,"python_function"=swish_activation))
    } else {
      best.model <- load_model_hdf5(best.model.copy,custom_objects=c("rmse"=metric_rmse))
    }
  } else {
    if (act=="swish") {
      best.model <- load_model_hdf5(best.model.copy,custom_objects=c("python_function"=swish_activation))
    } else {
      best.model <- load_model_hdf5(best.model.copy)
    }
  }
  
  # calculating relative RMSE and bias for training and test sets
  # training accuracy
  train.predictions <- best.model %>% predict(train.data)
  if (is.list(train.predictions)) {
    train.predictions <- as.data.frame(do.call(cbind,train.predictions))
  } else {
    train.predictions <- as.data.frame(train.predictions)
  }
  
  # de-scaling predictions if necessary
  if (scl %in% c("y","both")) {
    train.predictions <- sapply(1:ncol(train.predictions),function(i) (train.predictions[i]*sd.train[i])+mean.train[i])
    train.predictions <- as.data.frame(do.call(cbind,train.predictions))
  }
  
  # relative RMSE and bias
  train.rmse <- rel.rmse(train.labels,train.predictions)
  train.bias <- rel.bias(train.labels,train.predictions)
  
  # validation accuracy
  val.predictions <- best.model %>% predict(val.data)
  if (is.list(val.predictions)) {
    val.predictions <- as.data.frame(do.call(cbind,val.predictions))
  } else {
    val.predictions <- as.data.frame(val.predictions)
  }
  
  # de-scaling predictions if necessary
  if (scl %in% c("y","both")) {
    val.predictions <- sapply(1:ncol(val.predictions),function(i) (val.predictions[i]*sd.train[i])+mean.train[i])
    val.predictions <- as.data.frame(do.call(cbind,val.predictions))
  }
  
  # relative RMSE and bias
  val.rmse <- rel.rmse(val.labels,val.predictions)
  val.bias <- rel.bias(val.labels,val.predictions)
  
  if (out=="pred") {
    return(list(train.predictions,val.predictions))
  } else {
    return(rbind(train.rmse,train.bias,val.rmse,val.bias))
  }
}

#' Calculating predictions based on TensorFlow model in hdf5 format
#' The input data has to be in the same format as the training data and scaled if training was scaled
#' 
#' @param in.data Vector/matrix of features for prediction (needs scaling if multiple variables have different value range)
#' @param in.labels Vector/matrix of observations for accuracy calculations
#' @param train.data Vector/matrix of features used for model training (only needed if scaling is done before training 
#'        using the mean and sd of training set)
#' @param train.labels Vector/matrix of observations used for model training (only needed if scaling is done before training 
#'        using the mean and sd of training set)
#' @param hdf5.path Character, path to model
#' @param scl Character, should x (.data) or y (.labels) or both be scaled? ("x","y","both")
#' @param act Character, name of activation function used in hidden layer (only to use with "swish")
#'        only relevant if it is a custom function; the function has to be loaded in the environment
#' @param metric Character, name of metric function used during model training (only to use with "rmse")
#'        only relevant if it is a custom function; the function has to be loaded in the environment
#'        for relative RMSE set this parameter to "rmse"
#' @param out Character, "pred" for returning predictions and anything else for accuracy measures (relative RMSE and bias)
tf.model.pred <- function(in.data,in.labels,train.data,train.labels,hdf5.path="",scl="both",act="",metric="",out="") {
  # some error checks for input parameters
  if (!file.exists(hdf5.path)) stop("TensorFlow model doesn't exist!")
  # loading model to calculate predictions
  # due to custom functions that need to be declared in load_model_hdf5
  # it is important to check which functions were used
  if (metric=="rmse") {
    if (act=="swish") {
      best.model <- load_model_hdf5(hdf5.path,custom_objects=c("rmse"=metric_rmse,"python_function"=swish_activation))
    } else {
      best.model <- load_model_hdf5(hdf5.path,custom_objects=c("rmse"=metric_rmse))
    }
  } else {
    if (act=="swish") {
      best.model <- load_model_hdf5(hdf5.path,custom_objects=c("python_function"=swish_activation))
    } else {
      best.model <- load_model_hdf5(hdf5.path)
    }
  }
  
  # scaling x if necessary
  if (scl %in% c("x","both")) {
    mean.train <- sapply(train.data,mean)
    sd.train <- sapply(train.data,sd)
    in.data <- scale(in.data,center=mean.train,scale=sd.train)
  }

  # calculating relative RMSE and bias for input
  # training accuracy
  predictions <- best.model %>% predict(in.data)
  if (is.list(predictions)) {
    predictions <- as.data.frame(do.call(cbind,predictions))
  } else {
    predictions <- as.data.frame(predictions)
  }
  # de-scaling predictions if necessary
  if (scl %in% c("y","both")) {
    mean.train <- sapply(train.labels,mean)
    sd.train <- sapply(train.labels,sd)
    predictions <- sapply(1:ncol(predictions),function(i) (predictions[i]*sd.train[i])+mean.train[i])
    predictions <- as.data.frame(do.call(cbind,predictions))
    colnames(predictions) <- colnames(train.labels)
  }
  
  # relative RMSE and bias
  pred.rmse <- rel.rmse(in.labels,predictions)
  pred.bias <- rel.bias(in.labels,predictions)
  
  if (out=="pred") {
    return(predictions)
  } else {
    return(rbind(pred.rmse,pred.bias))
  }
}
