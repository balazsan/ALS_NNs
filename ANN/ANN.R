# Predicting forest stand variables using simple ANN based on ALS features.
# Keras with Tensorflow backend necessary for this to run.

# loading required libraries
library(keras)
source("keras_tf_funcs.R")

# loading plot information divided into training/validation/test sets
train.labels <- read.csv("../sample_plot_data/sp_data_train.csv",as.is=T)[,c("v","h","d")]
val.labels <- read.csv("../sample_plot_data/sp_data_val.csv",as.is=T)[,c("v","h","d")]
test.labels <- read.csv("../sample_plot_data/sp_data_test.csv",as.is=T)[,c("v","h","d")]

# subsetting features the same way as sample plots (this is done here using sample plot IDs)
train.data <- read.csv("../features/features_train.csv",as.is=T)
val.data <- read.csv("../features/features_val.csv",as.is=T)
test.data <- read.csv("../features/features_test.csv",as.is=T)

# adjusting column names (keras doesn't like dots in column names)
names(train.data) <- gsub("\\.","_",names(train.data))
names(val.data) <- gsub("\\.","_",names(val.data))
names(test.data) <- gsub("\\.","_",names(test.data))

# dropping columns w/o useful information (standard deviation is 0 or NaN/NA)
train.data <- train.data[,apply(train.data,2,function(x) !(sd(x)==0|is.na(sd(x))))]
val.data <- val.data[,apply(val.data,2,function(x) !(sd(x)==0|is.na(sd(x))))]
test.data <- test.data[,apply(test.data,2,function(x) !(sd(x)==0|is.na(sd(x))))]
# keeping only features common in the two sets
feat.common <- Reduce(intersect,list(names(train.data),names(val.data),names(test.data)))
train.data <- train.data[,feat.common]
val.data <- val.data[,feat.common]
test.data <- test.data[,feat.common]; rm(feat.common)

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
# as described above 1 hidden layer should be enough
# number of neurons of the hidden layer can vary:

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
out.dir <- "out_dir"

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

# table of RMSE% and bias% of each run for predicted attributes
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
# table of RMSE% and bias% of each run for predicted attributes
test.runs <- data.frame(do.call(rbind,test.runs))
row.names(test.runs) <- paste0(c("test.rmse.run","test.bias.run"),rep(1:10,each=2))
# exporting results for each run
out.f <- paste0(out.dir,"/tf.test.runs.csv")
write.table(test.runs,out.f,quote=F,row.names=T,col.names=NA,sep=";")

