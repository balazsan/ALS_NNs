# Predicting forest stand variables using simple ANN based on ALS features.

# loading required libraries
library(keras); library(ggplot2); library(parallel)
source("keras_tf_funcs.R")

# loading plot information divided into training/validation/test sets
sp.data.train <- read.csv("path/to/plot_data_train.csv",as.is=T)
sp.data.val <- read.csv("path/to/plot_data_val.csv",as.is=T)
sp.data.test <- read.csv("path/to/plot_data_test.csv",as.is=T)

# loading features created with ALS_feature_calc.R
feat <- readRDS("als.feat.RDS")

# subsetting features the same way as sample plots (this is done here using sample plot IDs)
feat.train <- feat[feat$sampleplotid%in%sp.data.train$sampleplotid,]
feat.val <- feat[feat$sampleplotid%in%sp.data.val$sampleplotid,]
feat.test <- feat[feat$sampleplotid%in%sp.data.test$sampleplotid,]

# adjusting column names (keras doesn't like dots in column names)
names(sp.data.train) <- gsub("\\.","_",names(sp.data.train))
names(sp.data.val) <- gsub("\\.","_",names(sp.data.val))
names(sp.data.test) <- gsub("\\.","_",names(sp.data.test))

# forest attributes
for.attrs <- c("v","h","d")

# creating input for TF
if (grepl("id|sampleplotid",names(feat.train)[1])) train.data <- feat.train[,-1] else train.data <- feat.train
if (grepl("id|sampleplotid",names(feat.val)[1])) val.data <- feat.val[,-1] else val.data <- feat.val
if (grepl("id|sampleplotid",names(feat.test)[1])) test.data <- feat.test[,-1] else test.data <- feat.test
train.labels <- sp.data.train[,for.attrs]
val.labels <- sp.data.val[,for.attrs]
test.labels <- sp.data.test[,for.attrs]

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
# as described above 1 hidden layer should be enough
# number of neurons of the hidden layer can vary

# # Ns/(a*(Ni+No)) (Ns: number of samples; a: scaling factor (2-10);
# # Ni: number of input neurons (features); No: number of output neurons (dependent variables))
# a <- 2
# n.neur <- ceiling(nrow(train.data)/(a*(ncol(train.data)+ncol(train.labels))))
# paste0("Number of nodes: ",n.neur)

# # number of neurons of hidden layer can be Ni*2/3+No
# n.neur <- ceiling(ncol(train.data)*(2/3)+ncol(train.labels))
# paste0("Number of nodes: ",n.neur)

# # number of neurons of hidden layer can be (Ni+No)/2
# n.neur <- ceiling((ncol(train.data)+ncol(train.labels))/2)
# paste0("Number of nodes: ",n.neur)

n.neur <- 4

# declaring functions used during training
opt <- "adam"; act <- "swish"
# output directory
out.dir <- paste0("AV_3layers_",n.neur,"neur_",opt,"_",act,"_multi_circ")

# training 10 times, best models and training history saved to output directory
preds.w <- c(0.6,0.2,0.2)
tf.runs <- lapply(1:10,function(i) {
  cat("\r",ifelse(i<10,paste0("Running iteration #",i),paste0("Running iteration #",i,"\n")))
  out.tmp <- tf.model.res(train.data,train.labels,val.data,val.labels,n.neur=n.neur,
                          scl="both",preds.w=preds.w,act=act,opt=opt,epochs=200,
                          patience=20,batch_size=25,metric="rmse",outdir=out.dir)
  row.names(out.tmp) <- paste0(row.names(out.tmp),".run",i)
  return(out.tmp)
})

tf.runs <- do.call(rbind,tf.runs)
# exporting training results
out.f <- paste0(out.dir,"/tf.train.runs.csv")
write.table(tf.runs[c(T,T,F,F),],out.f,quote=F,row.names=T,col.names=NA,sep=";")
# exporting validation results
out.f <- paste0(out.dir,"/tf.val.runs.csv")
write.table(tf.runs[c(F,F,T,T),],out.f,quote=F,row.names=T,col.names=NA,sep=";")

# predictions for test set (not used during training at all) with best saved model/run
# listing models
in.list <- list.files(out.dir,"*.hdf5",full.names=T)
# sorting paths by run#
in.list <- in.list[sapply(paste0("run",1:10,"\\."),function(x) grep(x,in.list))]
# calculating predictions for test data using each model
test.runs <- lapply(1:length(in.list),function(i) {
  cat("\r",paste0("Running iteration #",i,"/",length(in.list)))
  in.path <- in.list[i]
  tf.model.pred(in.data=test.data,in.labels=test.labels,train.data=train.data,train.labels=train.labels,
                scl="both",hdf5.path=in.path,act="swish",metric="rmse")
}); cat("\n")
test.runs <- data.frame(do.call(rbind,test.runs))
row.names(test.runs) <- paste0(c("test.rmse.run","test.bias.run"),rep(1:10,each=2))
# exporting results for each run
out.f <- paste0(out.dir,"/tf.test.runs.csv")
write.table(test.runs,out.f,quote=F,row.names=T,col.names=NA,sep=";")

# calculating means of 10 runs for test set
test.runs.means <- lapply(1:2,function(i) {
  sel.row <- rep(F,2); sel.row[i] <- T
  round(colMeans(abs(test.runs[sel.row,,drop=F])),2)
})
test.runs.means <- data.frame(do.call(rbind,test.runs.means))
row.names(test.runs.means) <- c("test.rmse","test.bias")

# calculating SD of 10 runs, test set
test.runs.sd <- lapply(1:2,function(i) {
  sel.row <- rep(F,2); sel.row[i] <- T
  round(apply(test.runs[sel.row,,drop=F],2,sd),2)
})
test.runs.sd <- data.frame(do.call(rbind,test.runs.sd))
row.names(test.runs.sd) <- c("test.rmse","test.bias")

# calculating means of 10 runs, training and validation sets
tf.runs.means <- lapply(1:4,function(i) {
  sel.row <- rep(F,4); sel.row[i] <- T
  round(colMeans(abs(tf.runs[sel.row,,drop=F])),2)
})
tf.runs.means <- data.frame(do.call(rbind,tf.runs.means))
row.names(tf.runs.means) <- c("train.rmse","train.bias","val.rmse","val.bias")
tf.runs.means <- rbind(tf.runs.means,test.runs.means)
write.table(tf.runs.means,paste0(out.dir,"/tf.runs.means.csv"),quote=F,row.names=T,col.names=NA,sep=";")

# calculating SD of 10 runs, training and validation sets
tf.runs.sd <- lapply(1:4,function(i) {
  sel.row <- rep(F,4); sel.row[i] <- T
  round(apply(tf.runs[sel.row,,drop=F],2,sd),2)
})
tf.runs.sd <- data.frame(do.call(rbind,tf.runs.sd))
row.names(tf.runs.sd) <- c("train.rmse","train.bias","val.rmse","val.bias")
tf.runs.sd <- rbind(tf.runs.sd,test.runs.sd)
write.table(tf.runs.sd,paste0(out.dir,"/tf.runs.sd.csv"),quote=F,row.names=T,col.names=NA,sep=";")

