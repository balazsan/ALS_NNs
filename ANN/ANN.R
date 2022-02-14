# Comparing Keras & TensorFlow to knn & genalg (parallelized)
# multivariate: estimating all variables at the time

# loading required libraries
library(keras); library(ggplot2); library(parallel)
source("functions/keras_tf_funcs.R")

# loading plot information
sp.data.train <- read.csv("../../../3D_CNN/AV_leafon/AV.leaf.on.train.csv",as.is=T)
sp.data.val <- read.csv("../../../3D_CNN/AV_leafon/AV.leaf.on.val.csv",as.is=T)
sp.data.test <- read.csv("../../../3D_CNN/AV_leafon/AV.leaf.on.test.csv",as.is=T)

# loading features
feat <- readRDS("../../feature_calc/las.feat.AV.leaf.on.circ.RDS")

feat.train <- feat[feat$sampleplotid%in%sp.data.train$sampleplotid,]
feat.val <- feat[feat$sampleplotid%in%sp.data.val$sampleplotid,]
feat.test <- feat[feat$sampleplotid%in%sp.data.test$sampleplotid,]

# dropping columns w/o useful information (standard deviation is 0 or NaN/NA)
feat.train <- feat.train[,apply(feat.train,2,function(x) !(sd(x)==0|is.na(sd(x))))]
feat.val <- feat.val[,apply(feat.val,2,function(x) !(sd(x)==0|is.na(sd(x))))]
feat.test <- feat.test[,apply(feat.test,2,function(x) !(sd(x)==0|is.na(sd(x))))]
# keeping only features common in the two sets
feat.common <- Reduce(intersect,list(names(feat.train),names(feat.val),names(feat.test)))
feat.train <- feat.train[,feat.common]
feat.val <- feat.val[,feat.common]
feat.test <- feat.test[,feat.common]; rm(feat.common,feat)

# adjusting column names
names(sp.data.train) <- gsub("\\.","_",names(sp.data.train))
names(sp.data.val) <- gsub("\\.","_",names(sp.data.val))
names(sp.data.test) <- gsub("\\.","_",names(sp.data.test))

# plot(sp.data$h,feat$hq85.f)
cor(sp.data.train$h,feat.train$hq85.f) #OK
cor(sp.data.val$h,feat.val$hq85.f) #OK
cor(sp.data.test$h,feat.test$hq85.f) #OK

# forest attributes
for.attrs <- c("v","h","d")

# datasets
d.sets <- c("train","val","test")

# creating input for TF
if (grepl("id|sampleplotid",names(feat.train)[1])) train.data <- feat.train[,-1] else train.data <- feat.train
if (grepl("id|sampleplotid",names(feat.val)[1])) val.data <- feat.val[,-1] else val.data <- feat.val
if (grepl("id|sampleplotid",names(feat.test)[1])) test.data <- feat.test[,-1] else test.data <- feat.test
train.labels <- sp.data.train[,for.attrs]
val.labels <- sp.data.val[,for.attrs]
test.labels <- sp.data.test[,for.attrs]

all.labels <- rbind(train.labels,test.labels,val.labels)

rbind(mean=round(colMeans(all.labels),1),sd=round(sapply(all.labels,sd),1),max=sapply(all.labels,max))

# # checking distribution of traning and test data
dev.off()
par(mfrow=c(3,1))
hist(train.labels$v,breaks=seq(from=0,to=900,by=50))
hist(val.labels$v,breaks=seq(from=0,to=900,by=50))
hist(test.labels$v,breaks=seq(from=0,to=900,by=50))
dev.off()
hist(all.labels$v,breaks=seq(from=0,to=900,by=50))

library(plotly)
p <- plot_ly(x=all.labels$v,type="histogram",nbinsx=18,
             marker=list(color="orange",line=list(color="black",width=2))) %>%
  layout(xaxis=list(title="<b>Total growing stock class (m<sup>3</sup>/ha)</b>",
                    tickvals=seq(from=0,to=900,by=50)),
         yaxis=list(title="<b>Number of sample plots</b>"),font=list(size=26),
         margin=list(l=120,b=140))
p

p <- plot_ly(x=test.labels$v,type="histogram",nbinsx=18,name="test set",
             marker=list(color="#d62828",line=list(color="black",width=2))) %>%
  add_trace(x=val.labels$v,type="histogram",nbinsx=18,name="validation set",
            marker=list(color="#f77f00",line=list(color="black",width=2))) %>%
  add_trace(x=train.labels$v,type="histogram",nbinsx=18,name="training set",
            marker=list(color="#fcbf49",line=list(color="black",width=2))) %>%
  layout(xaxis=list(title="<b>Total growing stock class (m<sup>3</sup>/ha)</b>",
                    tickvals=seq(from=0,to=900,by=50)),
         yaxis=list(title="<b>Number of sample plots</b>"),font=list(size=26),
         barmode="stack",margin=list(l=120,b=140),legend=list(x=0.7,y=0.9,font=list(size=26)))
p

orca(p,"stacked_bar2.png",scale=3,width=1200,height=600)


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

# calculating SD of 10 runs, training and valiation sets
tf.runs.sd <- lapply(1:4,function(i) {
  sel.row <- rep(F,4); sel.row[i] <- T
  round(apply(tf.runs[sel.row,,drop=F],2,sd),2)
})
tf.runs.sd <- data.frame(do.call(rbind,tf.runs.sd))
row.names(tf.runs.sd) <- c("train.rmse","train.bias","val.rmse","val.bias")
tf.runs.sd <- rbind(tf.runs.sd,test.runs.sd)
write.table(tf.runs.sd,paste0(out.dir,"/tf.runs.sd.csv"),quote=F,row.names=T,col.names=NA,sep=";")

#################### merging results ####################
##########################
# based on summary_SNN.xlsx (minimum values out of 10 runs)
# v = 5 neurons
# h,d = 4 neurons
##########################
# selecting best results based on test RMSEs
in.dirs <- list.dirs(recursive=F); in.dirs <- in.dirs[grep("circ",in.dirs)]
in.files <- list.files(in.dirs,"tf.test.runs.csv",recursive=T,full.names=T)
test.rmse <- lapply(in.files,function(x) {
  # only importing RMSE
  tmp <- read.table(x,header=T,sep=";",as.is=T)[c(T,F),]
  rownames(tmp) <- tmp[1] <- NULL
  return(tmp)
})

# best results (RMSE and bias) and predictions based on best test RMSEs for each forest attribute
best.preds.res <- lapply(for.attrs,function(fa) {
  # tmp <- dplyr::bind_cols(lapply(test.rmse,function(x) x[fa]))
  tmp <- do.call(cbind,lapply(test.rmse,function(x) x[fa]))
  # getting indices of lowest test RMSEs, row=# of run, column=method (in this case amount of neurons used)
  best.fa <- which(tmp==min(tmp),arr.ind=T)[1,]
  # directory of best method
  best.fa.dir <- paste0(dirname(in.files[best.fa["col"]]),"/")
  # input model path for calculating predictions
  in.path <- list.files(best.fa.dir,paste0("run",best.fa["row"],".hdf5"),full.names=T)
  # iterating through datasets, extracting accuracy scores and calculating predictions
  fa.preds.res <- lapply(d.sets,function(d.set) {
    preds.d.set <- tf.model.pred(in.data=get(paste0(d.set,".data")),in.labels=get(paste0(d.set,".labels")),
                                 train.data=train.data,train.labels=train.labels,
                                 scl="both",hdf5.path=in.path,act="swish",metric="rmse",out="pred")[fa]
    fa.d.set <- paste0(best.fa.dir,"tf.",d.set,".runs.csv")
    fa.d.set <- read.table(fa.d.set,header=T,sep=";",as.is=T); row.names(fa.d.set) <- fa.d.set[,1]; fa.d.set[1] <- NULL
    fa.d.set <- fa.d.set[fa]
    return(list(fa.d.set,preds.d.set))
  })
  fa.preds <- lapply(fa.preds.res,"[[",2)
  names(fa.preds) <- d.sets
  fa.res <- lapply(fa.preds.res,"[[",1)
  # getting mean accuracy scores of method for forest attribute
  fa.means <- paste0(best.fa.dir,"tf.runs.means.csv")
  fa.means <- read.table(fa.means,header=T,sep=";",as.is=T)
  row.names(fa.means) <- fa.means[,1]; fa.means[1] <- NULL; fa.means <- fa.means[fa]
  return(list(fa.means,fa.res,fa.preds))
})
# extracting and merging predictions by d.sets
best.preds <- lapply(best.preds.res,"[[",3)
best.preds <- lapply(1:length(best.preds),function(i) dplyr::bind_cols(lapply(best.preds,"[[",i)))
names(best.preds) <- d.sets
saveRDS(best.preds,"ann.best.preds.cric.RDS")

# extracting means
best.means <- dplyr::bind_cols(lapply(best.preds.res,"[[",1))
# rownames(best.means) <- paste0(rep(d.sets,each=2),c(".rmse",".bias"))
write.table(best.means,"ann.runs.means.circ.csv",quote=F,row.names=T,col.names=NA,sep=";")

# extracting runs' accuracy scores by d.sets
best.res <- lapply(best.preds.res,"[[",2)
best.res <- lapply(1:length(best.res),function(i) dplyr::bind_cols(lapply(best.res,"[[",i)))
names(best.res) <- d.sets
saveRDS(best.res,"ann.best.runs.circ.RDS")

# best test predictions' covariance
ann.cov <- apply(combn(1:length(for.attrs),2),2,function(x) cov(best.preds$test[x[1]],best.preds$test[x[2]]))
names(ann.cov) <- apply(combn(1:length(for.attrs),2),2,function(x) paste0(for.attrs[x[1]],".vs.",for.attrs[x[2]]))
# data.table::fwrite(t(ann.cov),"ann.test.cov.circ.csv",quote=F,row.names=F,col.names=T,sep=";")

# reference data's covariance
test.cov <- apply(combn(1:length(for.attrs),2),2,function(x) cov(test.labels[x[1]],test.labels[x[2]]))
names(test.cov) <- apply(combn(1:length(for.attrs),2),2,function(x) paste0(for.attrs[x[1]],".vs.",for.attrs[x[2]]))

#########################################################
model.weights <- get_weights(best.model)
model.weights.1 <- model.weights[[1]]
model.weights.2 <- model.weights[[2]]
model.weights.3 <- model.weights[[3]]
model.weights.4 <- model.weights[[4]]
model.weights.5 <- model.weights[[5]]
model.weights.6 <- model.weights[[6]]

hist(model.weights.1)

###################################################################################################
##################################### parallelized knn+genalg #####################################
###################################################################################################

source("functions/knn_funcs_group_28052019.R")
source("functions/keras_tf_funcs.R")
library(beepr); library(plotly); library(parallel)

outname <- "AV"
outdir <- "feat_sel_output_AV_circ/"

mx <- train.data
# continuous forest attributes
my <- sp.data.train[,for.attrs]
# weights for forest attributes
w.my <- c(0.6,0.2,0.2)

# scaling also "y" if multiple attributes estimated at the same time
if (ncol(my)>1) sca <- "both" else sca <- "x"

search.k.g <- function(run) {
  fs.c2 <- ffeatsel.con(mx,my,sca=sca,wrmse=w.my,wmin=0.3,maxpen=0,k=c(4,6),g=c(0,3),popSize=100,iters=30)
  # using "x" in result calculation as my is in original units if not scaled
  fs.c2.res <- ffeatsel.con.results(mx,my,rbga.res=fs.c2,sca="x")
  return(c(k=fs.c2.res$k,g=fs.c2.res$g,fs.c2.res$rmse.pct))
}

cl <- makeCluster(5L)
clusterExport(cl,c("mx","my","sca","w.my","ffeatsel.con","ffeatsel.con.results",
                   "fknncv.fs","knn","fknnestcon","fknncv","fknncv.base","rmse"))
k.g <- parLapply(cl,1:5,search.k.g)
stopCluster(cl); rm(cl)

saveRDS(k.g,paste0(outdir,outname,"_kg_multi_circ.RDS"))

beep(sound=1)

# summarizing results
k.g <- as.data.frame(do.call(rbind,k.g))

# k.g <- cbind(k.g,mean=rowSums(k.g[-(1:2)]))
# k.g <- k.g[order(k.g$mean),]
k.g <- k.g[order(k.g[,3]),]
k.g

k <- 6
g <- 2.0
fname <- paste0(outname,"_k",k,"g",sprintf("%.1f",g),"_multi_circ")

# searching for best feature combinations
feat.sel <- function(run) {
  fs.tmp <- ffeatsel.bin(mx,my,sca=sca,wrmse=w.my,k=k,g=g,popSize=100,iters=30)
  # using "x" in result calculation as my is in original units if not scaled
  res.fs.tmp <- ffeatsel.bin.results(mx,my,rbga.res=fs.tmp,sca="x")
  return(list(fs.tmp,rmse.pct=res.fs.tmp$rmse.pct,bias.pct=res.fs.tmp$bias.pct))
}

cl <- makeCluster(5L)
clusterExport(cl,c("mx","my","sca","w.my","k","g","ffeatsel.bin","ffeatsel.bin.results",
                   "fknncv.fs","knn","fknnestcon","fknncv","fknncv.base","rmse"))
fs <- parLapply(cl,1:10,feat.sel)
stopCluster(cl); rm(cl)

saveRDS(fs,file=paste0(outdir,"fs_",fname,"_10runs.RDS"))
beep(sound=1)

# fs <- readRDS(paste0(outdir,"fs_",fname,"_10runs_circ.RDS"))

fs.rmse.bias <- lapply(fs,function(x) cbind(rmse.pct=x[[2]],bias.pct=x[[3]]))
fs.rmse.bias <- as.data.frame(do.call(cbind,fs.rmse.bias))
names(fs.rmse.bias) <- paste(names(fs.rmse.bias),rep(1:10,each=2),sep=".")
fs.rmse.bias
best.mean.rmse <- which.min(colMeans(fs.rmse.bias[,c(T,F)]))
best.mean.bias <- which.min(colMeans(abs(fs.rmse.bias[,c(F,T)])))

# weight search for selected features
wei.sel <- function(i) {
  x <- fs[[i]]
  # extracting results of feature selection
  res.fs.tmp <- ffeatsel.bin.results(mx,my,rbga.res=x[[1]],sca="x")
  # keeping only selected features for weight search
  mx.subs <- mx[,res.fs.tmp$in.use]
  # # setting seed for weight search
  # set.seed(1976)
  # weights search for selected features
  ws.tmp <- ffeatsel.con(mx.subs,my,sca=sca,wrmse=w.my,k=k,g=g,wmin=0.0,maxpen=0,popSize=100,iters=30)
  # extracting results of weight search
  # using "x" in result calculation as my is in original units if not scaled
  res.ws.tmp <- ffeatsel.con.results(mx.subs,my,rbga.res=ws.tmp,sca="x")
  return(list(wei.sel=ws.tmp,rmse.pct=res.ws.tmp$rmse.pct,bias.pct=res.ws.tmp$bias.pct,wei.sel.res=res.ws.tmp))
}

cl <- makeCluster(5L)
clusterExport(cl,c("fs","mx","my","sca","w.my","k","g","ffeatsel.con","ffeatsel.con.results","ffeatsel.bin.results",
                   "fknncv.fs","knn","fknnestcon","fknncv","fknncv.base","rmse"))
ws <- parLapply(cl,1:length(fs),wei.sel)
stopCluster(cl); rm(cl)

saveRDS(ws,file=paste0(outdir,"ws_",fname,"_10runs.RDS"))
beep(sound=1)

# ws <- readRDS(paste0(outdir,"ws_",fname,"_10runs.RDS"))

ws.rmse.bias <- lapply(ws,function(x) cbind(rmse.pct=x$rmse.pct,bias.pct=x$bias.pct))
ws.rmse.bias <- as.data.frame(do.call(cbind,ws.rmse.bias))
names(ws.rmse.bias) <- paste(names(ws.rmse.bias),rep(1:10,each=2),sep=".")
ws.rmse.bias
best.mean.rmse <- which.min(colMeans(ws.rmse.bias[,c(T,F)]))
best.mean.bias <- which.min(colMeans(abs(ws.rmse.bias[,c(F,T)])))

# for compatibility with CNN and the above fully connected shallow network
# predictions and accuracy metrics are calculated for all datasets (training, validation and test sets)
# (scaling info for mx and *.data as well as k and g in fs.res)
preds <- lapply(ws,function(x) {
  train <- fknnreg(mx,my,train.data,fs.res=x$wei.sel.res,verbose=F)
  val <- fknnreg(mx,my,val.data,fs.res=x$wei.sel.res,verbose=F)
  test <- fknnreg(mx,my,test.data,fs.res=x$wei.sel.res,verbose=F)
  return(list(train=train,val=val,test=test))
})
saveRDS(preds,file=paste0(outdir,"preds_",fname,"_10runs.RDS"))

# training results
train.rmse.bias.out <- as.data.frame(rbind(train.rmse=rowMeans(ws.rmse.bias[,c(T,F)]),train.bias=rowMeans(abs(ws.rmse.bias[,c(F,T)]))))

# extracting predictions for validation sets
val.predictions.knn <- lapply(preds,"[[",2)

# relative RMSEs and biases of validation set
val.rmse.bias <- lapply(val.predictions.knn,function(x) {
  as.data.frame(rbind(val.rmse=rel.rmse(val.labels[names(my)],x),val.bias=rel.bias(val.labels[names(my)],x)))
})
val.rmse.bias <- do.call(rbind,val.rmse.bias)
val.rmse.bias.out <- as.data.frame(rbind(val.rmse=colMeans(val.rmse.bias[c(T,F),,drop=F]),
                                         val.bias=colMeans(abs(val.rmse.bias[c(F,T),,drop=F]))))

# extracting predictions for test sets
test.predictions.knn <- lapply(preds,"[[",3)

# relative RMSEs and biases of test set
test.rmse.bias <- lapply(test.predictions.knn,function(x) {
  as.data.frame(rbind(test.rmse=rel.rmse(test.labels[names(my)],x),test.bias=rel.bias(test.labels[names(my)],x)))
})
test.rmse.bias <- do.call(rbind,test.rmse.bias)
test.rmse.bias.out <- as.data.frame(rbind(test.rmse=colMeans(test.rmse.bias[c(T,F),,drop=F]),
                                          test.bias=colMeans(abs(test.rmse.bias[c(F,T),,drop=F]))))

# merging results
rmse.bias.knn <- round(rbind(train.rmse.bias.out,val.rmse.bias.out,test.rmse.bias.out),2)
write.table(rmse.bias.knn,paste0(outdir,"knn.runs.means.multi.csv"),quote=F,row.names=T,col.names=NA,sep=";")

# exporting training, validation and test results for all runs
knn.runs <- t(ws.rmse.bias)
row.names(knn.runs) <- paste0(c("train.rmse.run","train.bias.run"),rep(1:10,each=2))
write.table(knn.runs,paste0(outdir,"knn.train.runs.multi.csv"),quote=F,row.names=T,col.names=NA,sep=";")

row.names(val.rmse.bias) <- paste0(c("val.rmse.run","val.bias.run"),rep(1:10,each=2))
write.table(val.rmse.bias,paste0(outdir,"knn.val.runs.multi.csv"),quote=F,row.names=T,col.names=NA,sep=";")

row.names(test.rmse.bias) <- paste0(c("test.rmse.run","test.bias.run"),rep(1:10,each=2))
write.table(test.rmse.bias,paste0(outdir,"knn.test.runs.multi.csv"),quote=F,row.names=T,col.names=NA,sep=";")

############################################################################################
##################################### PLOTTING RESULTS #####################################
############################################################################################

############################################################################################
########################################### MIN ############################################
############################################################################################
library(plotly)
source("functions/keras_tf_funcs.R")

# dataset to be used
dataset <- "AV"

# forest attributes
for.attrs <- c("v","h","d")

# subset of data to be used
d.sets <- c("train","val","test")

# importing results of each run and selecting runs with best performance (RMSE/bias)
# importing SNN results
tf.runs <- readRDS("ann.best.runs.circ.RDS")

# importing knn+genalg results
knn.runs <- lapply(d.sets,function(d.set) {
  tmp <- read.table(paste0("feat_sel_output_",dataset,"_circ/knn.",d.set,".runs.multi.csv"),header=T,sep=";",as.is=T)
  row.names(tmp) <- tmp$X; tmp$X <- NULL; return(t(tmp))
})
names(knn.runs) <- d.sets
# importing knn+genalg predictions
knn.preds <- readRDS("feat_sel_output_AV_circ/preds_AV_k6g2.0_multi_circ_10runs.RDS")

# selecting best results for each forest attribute based on best test RMSE
# row numbers (i.e. runs) of biases relevant to min test RMSEs selected
# multiplying by 2 to get the row number of the original matrices (c(T,F))
run.nums <- apply(tf.runs$test[c(T,F),],2,which.min)*2
min.tf <- lapply(1:length(tf.runs),function(i) {
  tmp <- lapply(1:length(run.nums),function(ii) {
    tf.runs[[i]][c(run.nums[ii]-1,run.nums[ii]),ii]
  }); tmp <- do.call(rbind,tmp)
  rownames(tmp) <- for.attrs; colnames(tmp) <- paste0(d.sets[i],c(".rmse",".bias"))
  return(tmp)
}); names(min.tf) <- d.sets

# selecting best results for each forest attribute based on best test RMSE
# column numbers (i.e. runs) of biases relevant to min test RMSEs selected
# multiplying by 2 to get the column number of the original matrices (c(T,F))
run.nums <- apply(knn.runs$test[,c(T,F)],1,which.min)*2
min.knn <- lapply(1:length(knn.runs),function(i) {
  tmp <- lapply(1:length(run.nums),function(ii) {
    knn.runs[[i]][ii,c(run.nums[ii]-1,run.nums[ii])]
  }); tmp <- do.call(rbind,tmp)
  rownames(tmp) <- for.attrs; colnames(tmp) <- paste0(d.sets[i],c(".rmse",".bias"))
  return(tmp)
}); names(min.knn) <- d.sets
# selecting best predictions for each forest attribute based on best test RMSE
run.nums <- run.nums/2
best.preds <- lapply(for.attrs,function(fa) {
  cn <- run.nums[fa]
  tmp <- knn.preds[[cn]]
  lapply(tmp,function(x) x[fa])
})
best.preds <- lapply(1:length(best.preds),function(i) dplyr::bind_cols(lapply(best.preds,"[[",i)))
names(best.preds) <- d.sets
# saveRDS(best.preds,"feat_sel_output_AV_circ/best_preds_AV_k6g2.0_multi_circ.RDS")

# 2D-CNN
cnn.2d.runs <- readRDS("cnn.best.2d.cric.RDS")
row.nums <- sapply(cnn.2d.runs[,grep("test.rmse",names(cnn.2d.runs))],which.min)
names(row.nums) <- for.attrs
min.cnn.2d <- lapply(d.sets,function(d.set) {
  tmp <- lapply(for.attrs,function(for.attr) {
    f.a.rmse <- cnn.2d.runs[row.nums[for.attr],grep(paste0(for.attr,".",d.set,".rmse"),names(cnn.2d.runs))]
    f.a.bias <- cnn.2d.runs[row.nums[for.attr],grep(paste0(for.attr,".",d.set,".bias"),names(cnn.2d.runs))]
    tmp <- c(f.a.rmse,f.a.bias); names(tmp) <- paste0(d.set,c(".rmse",".bias"))
    return(tmp)
  }); tmp <- do.call(rbind,tmp); rownames(tmp) <- for.attrs
  return(tmp)
}); names(min.cnn.2d) <- d.sets

# 3D-CNN
cnn.3d.runs <- readRDS("cnn.best.3d.circ.aa.RDS")
row.nums <- sapply(cnn.3d.runs[,grep("test.rmse",names(cnn.3d.runs))],which.min)
names(row.nums) <- for.attrs
min.cnn.3d <- lapply(d.sets,function(d.set) {
  tmp <- lapply(for.attrs,function(for.attr) {
    f.a.rmse <- cnn.3d.runs[row.nums[for.attr],grep(paste0(for.attr,".",d.set,".rmse"),names(cnn.3d.runs))]
    f.a.bias <- cnn.3d.runs[row.nums[for.attr],grep(paste0(for.attr,".",d.set,".bias"),names(cnn.3d.runs))]
    tmp <- c(f.a.rmse,f.a.bias); names(tmp) <- paste0(d.set,c(".rmse",".bias"))
    return(tmp)
  }); tmp <- do.call(rbind,tmp); rownames(tmp) <- for.attrs
  return(tmp)
}); names(min.cnn.3d) <- d.sets


# # best test predictions' covariance
# ann.cov <- apply(combn(1:length(for.attrs),2),2,function(x) cov(best.preds$test[x[1]],best.preds$test[x[2]]))
# names(ann.cov) <- apply(combn(1:length(for.attrs),2),2,function(x) paste0(for.attrs[x[1]],".vs.",for.attrs[x[2]]))
# # data.table::fwrite(t(ann.cov),"ann.test.cov.circ.csv",quote=F,row.names=F,col.names=T,sep=";")

######################
# covariance
# reference data's covariance
test.cov <- apply(combn(1:length(for.attrs),2),2,function(x) cov(test.labels[x[1]],test.labels[x[2]]))
names(test.cov) <- apply(combn(1:length(for.attrs),2),2,function(x) paste0(for.attrs[x[1]],".vs.",for.attrs[x[2]]))

# ann covariance
ann.preds <- readRDS("ann.best.preds.cric.RDS")$test
ann.cov <- apply(combn(1:length(for.attrs),2),2,function(x) cov(ann.preds[x[1]],ann.preds[x[2]]))
names(ann.cov) <- apply(combn(1:length(for.attrs),2),2,function(x) paste0(for.attrs[x[1]],".vs.",for.attrs[x[2]]))
# data.table::fwrite(t(ann.cov),"ann.test.cov.circ.csv",quote=F,row.names=F,col.names=T,sep=";")

# knn covariance
knn.preds <- readRDS("feat_sel_output_AV_circ/best_preds_AV_k6g2.0_multi_circ.RDS")$test
knn.cov <- apply(combn(1:length(for.attrs),2),2,function(x) cov(knn.preds[x[1]],knn.preds[x[2]]))
names(knn.cov) <- apply(combn(1:length(for.attrs),2),2,function(x) paste0(for.attrs[x[1]],".vs.",for.attrs[x[2]]))

# 3D CNN covariance
cnn.3d.preds <- readRDS("cnn.best.3d.circ.aa.preds.RDS")$test
cnn.3d.cov <- apply(combn(1:length(for.attrs),2),2,function(x) cov(cnn.3d.preds[x[1]],cnn.3d.preds[x[2]]))
names(cnn.3d.cov) <- apply(combn(1:length(for.attrs),2),2,function(x) paste0(for.attrs[x[1]],".vs.",for.attrs[x[2]]))

# 2D CNN covariance
cnn.2d.preds <- readRDS("cnn.best.2d.cric.preds.RDS")$test
cnn.2d.cov <- apply(combn(1:length(for.attrs),2),2,function(x) cov(cnn.2d.preds[x[1]],cnn.2d.preds[x[2]]))
names(cnn.2d.cov) <- apply(combn(1:length(for.attrs),2),2,function(x) paste0(for.attrs[x[1]],".vs.",for.attrs[x[2]]))

as.data.frame(rbind(obs=test.cov,knn=knn.cov,ann=ann.cov,cnn.2d=cnn.2d.cov,cnn.3d=cnn.3d.cov))
#########################################

# plotting correlation of h/d, covariance
preds <- list("ground reference"=test.labels,"knn&genalg"=knn.preds,ANN=ann.preds,CNN.2D=cnn.2d.preds,CNN.3D=cnn.3d.preds)
model.names <- c("ground reference","k-NN","ANN","2D-CNN","3D-CNN")
preds <- lapply(preds,function(x) round(x,1))
dev.off()
par(mfrow=c(3,2))
for (i in 1:length(preds)) {
  x <- preds[[i]]
  plot(x$h,x$d,main=paste0(names(preds)[i],"\n","Covariance h vs d: ",round(cov(x$h,x$d),2)))
  # readline(prompt="Press [enter] to continue")
}
rm(i,x)

# defining maximum values for plot
max.p <- lapply(preds,function(x) c(h=max(x$h),d=max(x$d)))
max.p <- do.call(rbind,max.p)
# rounding up to nearest 5
max.p <- 5*ceiling(sapply(1:2,function(i) max(max.p[,i]))/5)
# ticks of axes
t.vals.h <- seq(0,max.p[1],5); t.vals.d <- seq(5,max.p[2],5)

# font size of subplot title
a.font.splot <- 26
# font size of trace and axis labels
t.font <- 24 # for png
# font size for axis
a.title.font <- 24 # for png

for (i in 1:length(preds)) {
  if (i%in%c(4,5)) x.title <- "<b>mean height [m]</b>" else x.title <- ""
  x <- preds[[i]]
  # calculating covariance h/d
  pred.cov <- round(cov(x$h,x$d),2)
  p.title <- paste0(model.names[i],"<br>covariance h vs d: ",
                    sprintf("%.2f",pred.cov))
  p <- plot_ly()
  p <- p %>% add_trace(x=x$h,y=x$d,type="scatter",mode="markers",marker=list(color="black"),name=model.names[i])
  p <- p %>% layout(annotations=list(text=paste0("<b>",p.title,"</b>"),
                                     font=list(size=a.font.splot),xref="paper",yref="paper",yanchor="bottom",
                                     xanchor="center",align="center",x=0.5,y=0.9,showarrow=FALSE),
                    width=600,height=600,showlegend=F,
                    xaxis=list(title=x.title,rangemode="tozero",range=c(0,max.p[1]),tickvals=t.vals.h,
                               tickfont=list(size=t.font),titlefont=list(size=a.title.font)),
                    yaxis=list(title="<b>mean diameter [cm]</b>",rangemode="tozero",range=c(0,max.p[2]),tickvals=t.vals.d,
                               tickfont=list(size=t.font),titlefont=list(size=a.title.font)))
  assign(paste0("p.",names(preds)[i]),p)
}

p <- subplot(`p.ground reference`,`p.knn&genalg`,p.ANN,p.CNN.2D,p.CNN.3D,nrows=3,margin=0.04,titleX=T,titleY=T) %>%
  # layout(annotations=list(text=paste0("<b>Mean height vs mean diameter scatter plots<br>of sample plot level obseravtions and predictions (test set)</b>"),
  #                         font=list(size=22),xref="paper",yref="paper",yanchor="bottom",
  #                         xanchor="center",align="center",x=0.5,y=1.01,showarrow=FALSE),
  #        margin=list(t=80,b=100,l=100,r=40),width=1400,height=1600)#; p
  layout(margin=list(t=80,b=100,l=100,r=40),width=1400,height=1600)#; p

orca(p,"preds_h_vs_d_cov.png",scale=3,width=1400,height=1600)

# plotting correlation of h/v, covariance
dev.off()
par(mfrow=c(3,2))
for (i in 1:length(preds)) {
  x <- preds[[i]]
  plot(x$h,x$v,main=paste0(names(preds)[i],"\n","Covariance h vs v: ",round(cov(x$h,x$v),2)))
  # readline(prompt="Press [enter] to continue")
}
rm(i,x)

# defining maximum values for plot
max.p <- lapply(preds,function(x) c(h=max(x$h),v=max(x$v)))
max.p <- do.call(rbind,max.p)
# rounding up to nearest 5
max.p <- 5*ceiling(sapply(1:2,function(i) max(max.p[,i]))/5)
# ticks of axes
t.vals.h <- seq(0,max.p[1],5); t.vals.v <- c(seq(0,600,100),650)
t.vals.v <- seq(0,600,100)
# font size of subplot title
a.font.splot <- 26
# font size of trace and axis labels
t.font <- 24 # for png
# font size for axis
a.title.font <- 24 # for png

for (i in 1:length(preds)) {
  if (i%in%c(4,5)) x.title <- "<b>mean height [m]</b>" else x.title <- ""
  x <- preds[[i]]
  # calculating covariance h/v
  pred.cov <- round(cov(x$h,x$v),2)
  p.title <- paste0(model.names[i],"<br>covariance h vs v: ",
                    sprintf("%.2f",pred.cov))
  p <- plot_ly()
  p <- p %>% add_trace(x=x$h,y=x$v,type="scatter",mode="markers",marker=list(color="black"),name=model.names[i])
  p <- p %>% layout(annotations=list(text=paste0("<b>",p.title,"</b>"),
                                     font=list(size=a.font.splot),xref="paper",yref="paper",yanchor="bottom",
                                     xanchor="center",align="center",x=0.5,y=0.9,showarrow=FALSE),
                    width=600,height=600,showlegend=F,
                    xaxis=list(title=x.title,rangemode="tozero",range=c(0,max.p[1]),tickvals=t.vals.h,
                               tickfont=list(size=t.font),titlefont=list(size=a.title.font)),
                    yaxis=list(title="<b>total growing stock [m<sup>3</sup>/ha]</b>",rangemode="tozero",range=c(0,650),tickvals=t.vals.v,
                               tickfont=list(size=t.font),titlefont=list(size=a.title.font)))
  assign(paste0("p.",names(preds)[i]),p)
}

p <- subplot(`p.ground reference`,`p.knn&genalg`,p.ANN,p.CNN.2D,p.CNN.3D,nrows=3,margin=0.04,titleX=T,titleY=T) %>%
  # layout(annotations=list(text=paste0("<b>Mean height vs mean diameter scatter plots<br>of sample plot level obseravtions and predictions (test set)</b>"),
  #                         font=list(size=22),xref="paper",yref="paper",yanchor="bottom",
  #                         xanchor="center",align="center",x=0.5,y=1.01,showarrow=FALSE),
  #        margin=list(t=80,b=100,l=100,r=40),width=1400,height=1600)#; p
  layout(margin=list(t=80,b=100,l=100,r=40),width=1400,height=1600)#; p

orca(p,"preds_h_vs_v_cov.png",scale=3,width=1400,height=1600)

# plotting correlation of v/d, covariance
dev.off()
par(mfrow=c(3,2))
for (i in 1:length(preds)) {
  x <- preds[[i]]
  plot(x$v,x$d,main=paste0(names(preds)[i],"\n","Covariance v vs d: ",round(cov(x$v,x$d),2)))
  # readline(prompt="Press [enter] to continue")
}
rm(i,x)

# defining maximum values for plot
max.p <- lapply(preds,function(x) c(v=max(x$v),d=max(x$d)))
max.p <- do.call(rbind,max.p)
# rounding up to nearest 5
max.p <- 5*ceiling(sapply(1:2,function(i) max(max.p[,i]))/5)
# ticks of axes
t.vals.d <- seq(0,max.p[2],5); t.vals.v <- seq(0,600,100)

# font size of subplot title
a.font.splot <- 26
# font size of trace and axis labels
t.font <- 24 # for png
# font size for axis
a.title.font <- 24 # for png

for (i in 1:length(preds)) {
  if (i%in%c(4,5)) x.title <- "<b>total growing stock [m<sup>3</sup>/ha]</b></b>" else x.title <- ""
  x <- preds[[i]]
  # calculating covariance h/d
  pred.cov <- round(cov(x$v,x$d),2)
  p.title <- paste0(model.names[i],"<br>covariance v vs d: ",
                    sprintf("%.2f",pred.cov))
  p <- plot_ly()
  p <- p %>% add_trace(x=x$v,y=x$d,type="scatter",mode="markers",marker=list(color="black"),name=model.names[i])
  p <- p %>% layout(annotations=list(text=paste0("<b>",p.title,"</b>"),
                                     font=list(size=a.font.splot),xref="paper",yref="paper",yanchor="bottom",
                                     xanchor="center",align="center",x=0.5,y=0.9,showarrow=FALSE),
                    width=600,height=600,showlegend=F,
                    xaxis=list(title=x.title,rangemode="tozero",range=c(0,650),tickvals=t.vals.v,
                               tickfont=list(size=t.font),titlefont=list(size=a.title.font)),
                    yaxis=list(title="<b>mean diameter [cm]</b>",rangemode="tozero",range=c(0,45),tickvals=t.vals.d,
                               tickfont=list(size=t.font),titlefont=list(size=a.title.font)))
  assign(paste0("p.",names(preds)[i]),p)
}

p <- subplot(`p.ground reference`,`p.knn&genalg`,p.ANN,p.CNN.2D,p.CNN.3D,nrows=3,margin=0.04,titleX=T,titleY=T) %>%
  # layout(annotations=list(text=paste0("<b>Mean height vs mean diameter scatter plots<br>of sample plot level obseravtions and predictions (test set)</b>"),
  #                         font=list(size=22),xref="paper",yref="paper",yanchor="bottom",
  #                         xanchor="center",align="center",x=0.5,y=1.01,showarrow=FALSE),
  #        margin=list(t=80,b=100,l=100,r=40),width=1400,height=1600)#; p
  layout(margin=list(t=80,b=100,l=100,r=40),width=1400,height=1600)#; p

orca(p,"preds_v_vs_d_cov.png",scale=3,width=1400,height=1600)

#########################################
# merging results
min.all <- list(min.knn,min.tf,min.cnn.2d,min.cnn.3d)
names(min.all) <- c("knn","ann","cnn.2d","cnn.3d")

# plotting results

# defining value limits for rendering plots
val.lims <- lapply(d.sets,function(d.set) {
  tmp <- lapply(min.all,function(x) {
    x[[d.set]]
  }); tmp <- do.call(rbind,tmp)
  c(max.rmse=5*ceiling(max(tmp[,1])/5),min.bias=floor(min(tmp[,2]))-0.5,max.bias=ceiling(max(tmp[,2]))+0.5)
}); names(val.lims) <- d.sets
val.lim <- as.data.frame(do.call(rbind,val.lims))
# val.lim <- c(max(val.lim$max.rmse),min(val.lim$min.bias),max(val.lim$max.bias))

# font size of subplot title
a.font.splot <- 14
# font size of trace and axis labels
t.font <- 14 # for html
t.font <- 24 # for png
# font size for axis
a.title.font <- 14 # for html
a.title.font <- 24 # for png
# trace names for plotting
trace.name <- c("k-NN","ANN","2D-CNN","3D-CNN")
# trace colors for plotting
trace.color <- c("olive","orange","brown","orangered")
# subset names for plotting
d.sets.p <- c("training","validation","test")
# subset sizes for plotting
n.plots <- c(1044,225,225)

a.format <- list(font=list(size=a.font.splot),xref="paper",yref="paper",yanchor="bottom",
                 xanchor="center",align="center",x=0.5,y=0.92,showarrow=FALSE)

############## RMSE ##############
for (i in 1:length(d.sets)) {
  # creating header
  a.rmse <- append(list(text=paste0("<b>",d.sets.p[i]," set (",n.plots[i]," plots)</b>")),a.format)
  
  p.name <- paste0("p.",d.sets[i],".rmse")
  assign(p.name,plot_ly(type="bar"))
  for (ii in 1:length(min.all)) {
    min.tmp <- min.all[[ii]][[d.sets[i]]]
    assign(p.name,add_trace(get(p.name),x=1:nrow(min.tmp),y=min.tmp[,1],name=paste0("<b>RMSE/bias/R<sup>2</sup> (",trace.name[ii],")</b>"),
                            marker=list(color=trace.color[ii]),constraintext="none",text=sprintf("<b>%.1f</b>",min.tmp[,1]),
                            textposition="outside",textfont=list(size=t.font-1),hovertext=trace.name[ii],hoverinfo="text"))
  }
  assign(p.name,layout(get(p.name),#annotations=a.rmse,
                       xaxis=list(tickvals=1:length(for.attrs),ticktext=sprintf("<b>%s</b>",for.attrs),
                                  tickfont=list(size=t.font),titlefont=list(size=a.title.font),title="<b>forest attribute</b>"),
                       yaxis=list(range=c(0,val.lim["test",1]),tickfont=list(size=t.font),titlefont=list(size=a.title.font),
                                  title="<b>RMSE %</b>"),barmode="group",
                       legend=list(xanchor="right",x=1.0,font=list(size=t.font-2))))
}

############## bias ##############
for (i in 1:length(d.sets)) {
  p.name <- paste0("p.",d.sets[i],".bias")
  assign(p.name,plot_ly(type="bar"))
  for (ii in 1:length(min.all)) {
    min.tmp <- min.all[[ii]][[d.sets[i]]]
    assign(p.name,add_trace(get(p.name),x=1:nrow(min.tmp),y=min.tmp[,2],name=paste0("<b>RMSE/bias (",trace.name[ii],")</b>"),
                            marker=list(color=trace.color[ii]),constraintext="none",text=sprintf("<b>%.1f</b>",min.tmp[,2]),
                            textposition="outside",textfont=list(size=t.font-1),hovertext=trace.name[ii],hoverinfo="text"))
  }
  assign(p.name,layout(get(p.name),
                       xaxis=list(tickvals=1:length(for.attrs),ticktext=sprintf("<b>%s</b>",for.attrs),
                                  tickfont=list(size=t.font),titlefont=list(size=a.title.font),title="<b>forest attribute</b>"),
                       yaxis=list(range=c(val.lim["test",2],val.lim["test",3]),tickfont=list(size=t.font),
                                  titlefont=list(size=a.title.font),title="<b>bias %</b>"),barmode="group"))
}

############## R2 ##############

rsq <- function(obs,preds) {
  # total sum of squares
  tss <- sum((obs-mean(obs))^2)
  # residual sum of squares
  rss <- sum((obs-preds)^2)
  # R-squared
  round(1-(rss/tss),2)
}

r2 <- lapply(2:5,function(z) {
  tmp <- sapply(for.attrs,function(fa) {
    rsq(preds[[1]][,fa],preds[[z]][,fa])
  })
})
r2 <- as.data.frame(do.call(rbind,r2))
row.names(r2) <- names(preds)[-1]

p.test.r2 <- plot_ly(type="bar")
for (ii in 1:nrow(r2)) {
  min.tmp <- r2[ii,]
  p.test.r2 <- add_trace(p.test.r2,x=1:ncol(min.tmp),y=unlist(min.tmp[1,]),name=paste0("<b>R<sup>2</sup> (",trace.name[ii],")</b>"),
                         marker=list(color=trace.color[ii]),constraintext="none",text=sprintf("<b>%.2f</b>",unlist(min.tmp[1,])),
                         textposition="outside",textfont=list(size=t.font-1),hovertext=trace.name[ii],hoverinfo="text")
}
p.test.r2 <- layout(p.test.r2,
                    xaxis=list(tickvals=1:length(for.attrs),ticktext=sprintf("<b>%s</b>",for.attrs),
                               tickfont=list(size=t.font),titlefont=list(size=a.title.font),title="<b>forest attribute</b>"),
                    yaxis=list(range=c(0.75,1),tickfont=list(size=t.font),
                               titlefont=list(size=a.title.font),title="<b>R<sup>2</sup></b>"),barmode="group")

# a.plot <- list(text=paste0("<b>Comparison of model performances",
#                            "<br>(minimum test RMSE values of 10 runs with corresponding biases)</b>"),
#                font=list(size=16),xref="paper",yref="paper",yanchor="bottom",
#                xanchor="center",align="center",x=0.5,y=1.03,showarrow=FALSE)
# a.plot <- list(text="<b>Comparison of model performances<br>test set (225 plots)</b>",
#                font=list(size=26),xref="paper",yref="paper",yanchor="bottom",
#                xanchor="center",align="center",x=0.5,y=1.0,showarrow=FALSE)

p.test <- subplot(p.test.rmse,style(p.test.bias,showlegend=F),style(p.test.r2,showlegend=F),
                  heights=c(0.4,0.3,0.2),nrows=3,shareX=T,shareY=T) %>%
  layout(margin=list(t=80,b=100,l=100,r=40),width=1200,height=1300)

# html.out <- paste0("knn&g_ANN_2D3DCNN_test.best.html")
# htmlwidgets::saveWidget(p.test,file=html.out)
# browseURL(html.out)

orca(p.test,"_knn&g_ANN_2D3DCNN_test.best.png",scale=3,width=1200,height=1200)

min.rmse <- as.data.frame(t(sapply(2:5,function(i) p.test.rmse$x$attrs[[i]]$y)))
row.names(min.rmse) <- trace.name
min.bias <- as.data.frame(t(sapply(2:5,function(i) p.test.bias$x$attrs[[i]]$y)))
row.names(min.bias) <- trace.name

# a.plot <- list(text=paste0("<b>Comparison of model performances",
#                            "<br>(minimum test RMSE values of 10 runs with corresponding biases and training results)</b>"),
#                font=list(size=16),xref="paper",yref="paper",yanchor="bottom",
#                xanchor="center",align="center",x=0.5,y=1.03,showarrow=FALSE)
# 
# p.train.test <- subplot(p.test.rmse,style(p.train.rmse,showlegend=F),style(p.test.bias,showlegend=F),
#                        style(p.train.bias,showlegend=F),nrows=2,shareX=T,shareY=T) %>%
#   layout(annotations=a.plot,margin=list(t=60,b=20))
# 
# html.out <- paste0("knn&g_SNN_3D2DCNN_test.train.best.html")
# htmlwidgets::saveWidget(p.train.test,file=html.out)
# browseURL(html.out)

# png.out <- paste0(out,".png")
# orca(p,png.out,scale=2,height=1080,width=1920)

########################################
# creating plots from best predictions #
########################################
# name of file with predictions
in.p.files <- c("feat_sel_output_AV_circ/best_preds_AV_k6g2.0_multi_circ.RDS",
                "ann.best.preds.cric.RDS","cnn.best.2d.cric.preds.RDS",
                "cnn.best.3d.circ.aa.preds.RDS")
p.titles <- c("k-NN","ANN","2D-CNN","3D-CNN")

library(plotly)
source("functions/keras_tf_funcs.R")

# forest attributes
for.attrs <- c("v","h","d")

# subset of data to be used
d.sets <- c("train","val","test")

obs <- lapply(d.sets,function(d.set) read.csv(paste0("../../../3D_CNN/AV_leafon/AV.leaf.on.",d.set,".csv"),as.is=T))
names(obs) <- d.sets
cor(obs$test$h,obs$test$d)
cov(obs$test$h,obs$test$d)

invisible(lapply(1:length(in.p.files),function(i) {
  in.p.file <- in.p.files[i]
  preds <- readRDS(in.p.file)$test
  cat(in.p.file,"\n")
  cat(cor(preds$h,preds$d),"\n")
  cat(cov(preds$h,preds$d),"\n")
  cat("------\n")
}))

# # saving observations and predictions
# preds <- lapply(1:length(in.p.files),function(i) {
#   in.p.file <- in.p.files[i]
#   preds <- readRDS(in.p.file)$test
# })
# obs.preds <- list()
# obs.preds[[1]] <- obs
# obs.preds <- c(obs.preds,preds)
# names(obs.preds) <- c("obs","knn","ann","2dcnn","3dcnn")
# saveRDS(obs.preds,"obs.preds.RDS")

# checking R-squared (coefficient of determination)
# see: https://stackoverflow.com/questions/40901445/function-to-calculate-r2-r-squared-in-r
# it is said that R-squared should not be used for estimating goodness of a model on test data
# ("there is no justification that it can measure the goodness of out-of-sample prediction")
# however it could be ok/useful here since the test/validation data was selected from the sample in a systematic way
# using general equation 1-(rss/tss) where rss=residual sum of squares, tss=total sum of squares
rsq <- function(obs,preds) {
  # total sum of squares
  tss <- sum((obs-mean(obs))^2)
  # residual sum of squares
  rss <- sum((obs-preds)^2)
  # R-squared
  round(1-(rss/tss),2)
}

# there is no consensus whether x=pred or y=pred
# https://www.researchgate.net/publication/230692926_How_to_Evaluate_Models_Observed_vs_Predicted_or_Predicted_vs_Observed
# above authors suggest to use x=pred for linear regression (might not be relevant for visual interpretation)
# y=pred however makes visual interpretation a bit easier: underestimated data points are under the 1:1 line and overestimated vice versa
# shall predicted values plotted on the x axis?
x.pred <- T

# font size of plot title
a.font.plot <- 16 # for html
a.font.plot <- 32 # for png
# font size of subplot title
a.font.splot <- 14 # for html
a.font.splot <- 28 # for png
# font size for axis labels and ticks
axis.font <- 14 # for html
axis.font <- 28 # for png

# plotting observed vs predicted (or vice versa)
# iterating through input files (best predictions of various models)
# warnings "Specifying width/height in layout() is now deprecated" can be ignored
for (i in 1:length(in.p.files)) {
  in.p.file <- in.p.files[i]
  # importing predictions
  preds <- readRDS(in.p.file)
  # adding scatter plot for each forest attribute
  for (for.attr in for.attrs) {
    # defining maximum values for plot
    max.p <- ceiling(max(cbind(preds$test[,for.attr],obs$test[,for.attr])))
    max.p <- ifelse(for.attr=="v",max.p+50,max.p+5)
    if (for.attr=="v") {
      t.vals.x <- seq(0,max.p,100); t.vals.y <- seq(100,max.p,100)
    } else {
      t.vals.x <- seq(0,max.p,5); t.vals.y <- seq(5,max.p,5)
    }
    # calculating covariance for predictions and observations
    if (for.attr=="v") {
      preds.cov <- cov(preds$test$v,preds$test$h); obs.cov <- cov(obs$test$v,obs$test$h)
      a.title <- "Total growing stock (m<sup>3</sup>/ha)"}
      # a.cov <- paste0(c("Prediction","Observation")," covariance, v vs h: ")}
    if (for.attr=="h") {
      preds.cov <- cov(preds$test$h,preds$test$d); obs.cov <- cov(obs$test$h,obs$test$d)
      a.title <- "Mean height (m)"}
      # a.cov <- paste0(c("Prediction","Observation")," covariance, h vs d: ")}
    if (for.attr=="d") {
      preds.cov <- cov(preds$test$d,preds$test$v); obs.cov <- cov(obs$test$d,obs$test$v)
      a.title <- "Mean diameter (cm)"}
      # a.cov <- paste0(c("Prediction","Observation")," covariance, d vs v: ")}
    p <- plot_ly()
    if (x.pred) {
      p <- p %>% add_trace(x=preds$test[,for.attr],y=obs$test[,for.attr],type="scatter",mode="markers",marker=list(color="black"),name=for.attr)
    } else {
      p <- p %>% add_trace(x=obs$test[,for.attr],y=preds$test[,for.attr],type="scatter",mode="markers",marker=list(color="black"),name=for.attr)
    }
    # adding abline (1:1)
    p <- p %>% add_trace(x=c(0,max.p),y=c(0,max.p),type="scatter",mode="lines",line=list(color="red"),name="abline")
    p <- p %>% layout(annotations=list(text=paste0("<b>",a.title,"</b>"),
                                       font=list(size=a.font.splot),xref="paper",yref="paper",yanchor="bottom",
                                       xanchor="center",align="center",x=0.5,y=0.95,showarrow=FALSE),
                      width=600,height=600,showlegend=F,
                      xaxis=list(title=ifelse(x.pred,"<b>predictions</b>","<b>observations</b>"),rangemode="tozero",
                                 range=c(0,max.p),titlefont=list(size=axis.font),tickvals=t.vals.x,tickfont=list(size=axis.font)),
                      yaxis=list(title=ifelse(x.pred,"<b>observations</b>","<b>predictions</b>"),rangemode="tozero",
                                 range=c(0,max.p),titlefont=list(size=axis.font),tickvals=t.vals.y,tickfont=list(size=axis.font)))
    assign(paste0("p.",for.attr),p)
  }
  
  p <- subplot(p.v,p.h,p.d,nrows=1,titleX=T,titleY=F) %>%
    layout(annotations=list(text=paste0("<b>",p.titles[i],"</b>"),
                            font=list(size=a.font.plot),xref="paper",yref="paper",yanchor="bottom",
                            xanchor="center",align="center",x=0.5,y=1.1,showarrow=FALSE),
           yaxis=list(title=ifelse(x.pred,"<b>observations</b>","<b>predictions</b>")),
           margin=list(t=100,b=110,l=100,r=40),width=1800,height=600)
  
  if (dirname(in.p.file)==".") png.out <- paste0(tools::file_path_sans_ext(in.p.file),".png") else png.out <- "knn.best.preds.circ.png"
  orca(p,png.out,scale=3,width=1800,height=600)
}

###########################################
# WITH R-SQUARED
# plotting observed vs predicted (or vice versa)
# iterating through input files (best predictions of various models)
# warnings "Specifying width/height in layout() is now deprecated" can be ignored
for (in.p.file in in.p.files) {
  # importing predictions
  preds <- readRDS(in.p.file)
  # adding scatter plot for each forest attribute
  for (for.attr in for.attrs) {
    # defining maximum values for plot
    max.p <- ceiling(max(cbind(preds$test[,for.attr],obs$test[,for.attr])))
    max.p <- ifelse(for.attr=="v",max.p+50,max.p+5)
    if (for.attr=="v") t.vals <- seq(0,max.p,100) else t.vals <- seq(0,max.p,5)
    # calculating R-squared for each forest attribute
    fa.rsq <- rsq(obs$test[,for.attr],preds$test[,for.attr])
    # calculating covariance for predictions and observations
    if (for.attr=="v") {
      preds.cov <- cov(preds$test$v,preds$test$h); obs.cov <- cov(obs$test$v,obs$test$h)
      a.title <- "Total growing stock (m<sup>3</sup>/ha)<br>R<sup>2</sup>: "}
    # a.cov <- paste0(c("Prediction","Observation")," covariance, v vs h: ")}
    if (for.attr=="h") {
      preds.cov <- cov(preds$test$h,preds$test$d); obs.cov <- cov(obs$test$h,obs$test$d)
      a.title <- "Mean height (m)<br>R<sup>2</sup>: "}
    # a.cov <- paste0(c("Prediction","Observation")," covariance, h vs d: ")}
    if (for.attr=="d") {
      preds.cov <- cov(preds$test$d,preds$test$v); obs.cov <- cov(obs$test$d,obs$test$v)
      a.title <- "Mean diameter (cm)<br>R<sup>2</sup>: "}
    # a.cov <- paste0(c("Prediction","Observation")," covariance, d vs v: ")}
    p.title <- paste0(a.title,sprintf("%.2f",fa.rsq))
    # p.title <- paste0(a.title,sprintf("%.2f",fa.rsq),"<br>",a.cov[1],sprintf("%.2f",preds.cov),
    #                   "<br>",a.cov[2],sprintf("%.2f",obs.cov))
    p <- plot_ly()
    if (x.pred) {
      p <- p %>% add_trace(x=preds$test[,for.attr],y=obs$test[,for.attr],type="scatter",mode="markers",marker=list(color="black"),name=for.attr)
    } else {
      p <- p %>% add_trace(x=obs$test[,for.attr],y=preds$test[,for.attr],type="scatter",mode="markers",marker=list(color="black"),name=for.attr)
    }
    # adding abline (1:1)
    p <- p %>% add_trace(x=c(0,max.p),y=c(0,max.p),type="scatter",mode="lines",line=list(color="red"),name="abline")
    p <- p %>% layout(annotations=list(text=paste0("<b>",p.title,"</b>"),
                                       font=list(size=16),xref="paper",yref="paper",yanchor="bottom",
                                       xanchor="center",align="center",x=0.5,y=1.0,showarrow=FALSE),
                      width=600,height=600,showlegend=F,
                      xaxis=list(title=ifelse(x.pred,"<b>predictions</b>","<b>observations</b>"),rangemode="tozero",range=c(0,max.p),tickvals=t.vals),
                      yaxis=list(title=ifelse(x.pred,"<b>observations</b>","<b>predictions</b>"),rangemode="tozero",range=c(0,max.p),tickvals=t.vals))
    assign(paste0("p.",for.attr),p)
  }
  
  p <- subplot(p.v,p.h,p.d,nrows=1,titleX=T,titleY=T) %>%
    layout(margin=list(t=60,b=10),width=1800,height=600)
  
  if (dirname(in.p.file)==".") png.out <- paste0(tools::file_path_sans_ext(in.p.file),".png") else png.out <- "knn.best.preds.circ.png"
  orca(p,png.out,scale=3,width=1800,height=600)
}
##################################

# plotting predicted vs residuals
for (in.p.file in in.p.files) {
  # importing predictions
  preds <- readRDS(in.p.file)
  preds.resid <- preds$test-obs$test[for.attrs]
  for (for.attr in for.attrs) {
    max.x <- ceiling(max(preds$test[,for.attr]))
    max.y <- ceiling(max(preds.resid[,for.attr]))
    min.y <- floor(min(preds.resid[,for.attr]))
    max.x <- ifelse(for.attr=="v",max.x+50,max.x+5)
    max.y <- ifelse(for.attr=="v",max.y+50,max.y+1); min.y <- ifelse(for.attr=="v",min.y-50,min.y-1)
    if (for.attr=="v") t.vals <- seq(0,max.x,50) else t.vals <- seq(0,max.x,5)
    p.title <- ifelse(for.attr=="v","Residual plot of total growing stock (m<sup>3</sup>/ha)",
                      ifelse(for.attr=="h","Residual plot of mean height (m)","Residual plot of mean diameter (cm)"))
    p <- plot_ly()
    p <- p %>% add_trace(x=preds$test[,for.attr],y=preds.resid[,for.attr],type="scatter",mode="markers",marker=list(color="red"),name=for.attr)
    p <- p %>% layout(annotations=list(text=paste0("<b>",p.title,"</b>"),
                                       font=list(size=16),xref="paper",yref="paper",yanchor="bottom",
                                       xanchor="center",align="center",x=0.5,y=1,showarrow=FALSE),
                      width=600,height=600,showlegend=F,
                      xaxis=list(title="<b>predicted value</b>",rangemode="tozero",range=c(0,max.x),tickvals=t.vals),
                      yaxis=list(title="<b>residuals</b>",range=c(min.y,max.y)))
    assign(paste0("p.",for.attr),p)
  }
  
  p <- subplot(p.v,p.h,p.d,nrows=1,titleX=T,titleY=T) %>%
    layout(margin=list(t=60,b=10),width=1800,height=600)
  #p
  
  if (dirname(in.p.file)==".") png.out <- paste0(tools::file_path_sans_ext(in.p.file),".residuals.png") else png.out <- "knn.best.preds.circ.residuals.png"
  orca(p,png.out,scale=3,width=1800,height=600)
}

#################################
# # testing error bars (min, mean and max on the same chart)
# 
# # selecting best results for each forest attribute based on best RMSE
# rmse.range.tf <- lapply(1:nrow(rmse.bias.tf.runs),function(i) {
#   rmse.min <- min(rmse.bias.tf.runs[i,c(T,F)])
#   rmse.max <- max(rmse.bias.tf.runs[i,c(T,F)])
#   rmse.mean <- round(mean(rmse.bias.tf.runs[i,c(T,F)]),2)
#   cbind(rmse.mean-rmse.min,rmse.max-rmse.mean)
# })
# rmse.range.tf <- do.call(rbind,rmse.range.tf); rownames(rmse.range.tf) <- c("v","h","d")
# 
# p.rmse.test <- plot_ly(type="bar") %>%
#   add_trace(x=1:nrow(rmse.bias.knn),y=rmse.bias.knn[,5],name="Relative RMSE (knn & genalg)",marker=list(color="olive"),constraintext="none",
#             text=sprintf("<b>%.2f</b>",rmse.bias.knn[,5]),textposition="outside",hovertext="knn & genalg",hoverinfo="text") %>%
#   add_trace(x=1:nrow(rmse.bias.tf),y=rmse.bias.tf[,5],name="Relative RMSE (SNN)",marker=list(color="orange"),constraintext="none",
#             text=sprintf("<b>%.2f</b>",rmse.bias.tf[,5]),textposition="outside",hovertext="SNN",hoverinfo="text",
#             error_y=list(arrayminus=rmse.range.tf[,1],array=rmse.range.tf[,2],color="black")) %>%
#   add_trace(x=1:nrow(rmse.bias.cnn),y=rmse.bias.cnn[,5],name="Relative RMSE (CNN)",marker=list(color="orangered"),constraintext="none",
#             text=sprintf("<b>%.2f</b>",rmse.bias.cnn[,5]),textposition="outside",hovertext="CNN",hoverinfo="text") %>%
#   layout(annotations=a.rmse.test,barmode="group")
# 
# p.train.test <- subplot(p.rmse.train,style(p.rmse.test,showlegend=F),p.bias.train,
#                         style(p.bias.test,showlegend=F),nrows=2,shareX=T,shareY=T) %>%
#   layout(margin=list(t=60,b=10))
# 
# p.train.test


############################################################################################
########################################### MEAN ###########################################
############################################################################################
library(plotly)
source("functions/keras_tf_funcs.R")
source("functions/CNN_funcs.R")

mean.tf <- read.table("snn.runs.means.csv",header=T,sep=";",as.is=T)
row.names(mean.tf) <- mean.tf$X; mean.tf$X <- NULL

mean.knn <- read.table("feat_sel_output_AV/knn.runs.means.multi.csv",header=T,sep=";",as.is=T)
row.names(mean.knn) <- mean.knn$X; mean.knn$X <- NULL

mean.knn <- t(mean.knn)
mean.tf <- t(mean.tf)

# 3D-CNN
# set if CNN input should be augmented or non-augmented
aug <- T; h.text <- "augmented training data"
#aug <- F; h.text <- "non-augmented training data"

# forest attributes
for.attrs <- c("v","h","d")

# datasets
d.sets <- c("train","val","test")

# methods.com <- list.files("../../../3D_CNN/InceptionV3/AV/multi/number_of_points/common/",include.dirs=T,full.names=T); methods.com
methods.sep <- list.files("../../../3D_CNN/InceptionV3/AV/multi/number_of_points/separate/",include.dirs=T,full.names=T)#; methods.sep

res.3d <- eval.separate(methods=methods.sep,for.attrs=for.attrs,obs.path="../../../3D_CNN/AV_leafon/",d.sets=d.sets,out.name="cnn.best.3d")

cnn.best.3d <- res.3d$best.rmse.bias
saveRDS(cnn.best.3d,"cnn.best.3d.RDS")

mean.cnn.3d <- lapply(for.attrs,function(for.attr) {
  colMeans(abs(cnn.best.3d[grep(paste0("^",for.attr,"."),names(cnn.best.3d))]))
})
mean.cnn.3d <- do.call(rbind,mean.cnn.3d)
colnames(mean.cnn.3d) <- colnames(mean.knn)
row.names(mean.cnn.3d) <- row.names(mean.knn)

# 2D-CNN

# listing training methods used (only augmented)
methods.sep <- list.files("../../../2D_CNN/AlexNet/AV/multi/number_of_points/separate/","^[aug]",include.dirs=T,full.names=T)
res.2d <- eval.separate(methods=methods.sep,for.attrs=for.attrs,obs.path="../../../3D_CNN/AV_leafon/",d.sets=d.sets,out.name="cnn.best.2d")

cnn.best.2d <- res.2d$best.rmse.bias
saveRDS(cnn.best.2d,"cnn.best.2d.RDS")

mean.cnn.2d <- lapply(for.attrs,function(for.attr) {
  colMeans(abs(cnn.best.2d[grep(paste0("^",for.attr,"."),names(cnn.best.2d))]))
})
mean.cnn.2d <- do.call(rbind,mean.cnn.2d)
colnames(mean.cnn.2d) <- colnames(mean.knn)
row.names(mean.cnn.2d) <- row.names(mean.knn)

# merging results
mean.all <- lapply(list(mean.knn,mean.tf,mean.cnn.3d,mean.cnn.2d),function(x) {
  tmp <- lapply(d.sets,function(d.set) x[,grep(d.set,colnames(x))])
  names(tmp) <- d.sets; return(tmp)
})
names(mean.all) <- c("knn","snn","cnn.3d","cnn.2d")

# plotting results

# defining value limits for rendering plots
val.lims <- lapply(d.sets,function(d.set) {
  tmp <- lapply(mean.all,function(x) {
    x[[d.set]]
  }); tmp <- do.call(rbind,tmp)
  c(max.rmse=10*ceiling(max(tmp[,1])/10)+5,min.bias=floor(min(tmp[,2]))-0.5,max.bias=ceiling(max(tmp[,2]))+0.5)
}); names(val.lims) <- d.sets
val.lim <- as.data.frame(do.call(rbind,val.lims))
val.lim <- c(max(val.lim$max.rmse),min(val.lim$min.bias),max(val.lim$max.bias))

# font size of subplot title
a.font.splot <- 14
# font size of trace labels
t.font <- 11
# trace names for plotting
trace.name <- c("knn & genalg","SNN","3D-CNN","2D-CNN")
# trace colors for plotting
trace.color <- c("olive","orange","orangered","brown")
# subset names for plotting
d.sets.p <- c("training","validation","test")
# subset sizes for plotting
n.plots <- c(1044,225,225)

a.format <- list(font=list(size=a.font.splot),xref="paper",yref="paper",yanchor="bottom",
                 xanchor="center",align="center",x=0.5,y=0.92,showarrow=FALSE)

############## RMSE ##############
for (i in 1:length(d.sets)) {
  # creating header
  a.rmse <- append(list(text=paste0("<b>",d.sets.p[i]," set (",n.plots[i]," plots)</b>")),a.format)
  
  p.name <- paste0("p.",d.sets[i],".rmse")
  assign(p.name,plot_ly(type="bar"))
  for (ii in 1:length(mean.all)) {
    min.tmp <- mean.all[[ii]][[d.sets[i]]]
    assign(p.name,add_trace(get(p.name),x=1:nrow(min.tmp),y=min.tmp[,1],name=paste0("RMSE/bias (",trace.name[ii],")"),
                            marker=list(color=trace.color[ii]),constraintext="none",text=sprintf("<b>%.2f</b>",min.tmp[,1]),
                            textposition="outside",textfont=list(size=t.font),hovertext=trace.name[ii],hoverinfo="text"))
  }
  assign(p.name,layout(get(p.name),annotations=a.rmse,
                       xaxis=list(tickvals=1:length(for.attrs),ticktext=sprintf("<b>%s</b>",for.attrs),
                                  title="<b>forest attribute</b>"),
                       yaxis=list(range=c(0,val.lim[1]),title="<b>RMSE %</b>"),barmode="group"))
}

############## bias ##############
for (i in 1:length(d.sets)) {
  p.name <- paste0("p.",d.sets[i],".bias")
  assign(p.name,plot_ly(type="bar"))
  for (ii in 1:length(mean.all)) {
    min.tmp <- mean.all[[ii]][[d.sets[i]]]
    assign(p.name,add_trace(get(p.name),x=1:nrow(min.tmp),y=min.tmp[,2],name=paste0("RMSE/bias (",trace.name[ii],")"),
                            marker=list(color=trace.color[ii]),constraintext="none",text=sprintf("<b>%.2f</b>",min.tmp[,2]),
                            textposition="outside",textfont=list(size=t.font),hovertext=trace.name[ii],hoverinfo="text"))
  }
  assign(p.name,layout(get(p.name),
                       xaxis=list(tickvals=1:length(for.attrs),ticktext=sprintf("<b>%s</b>",for.attrs),
                                  title="<b>forest attribute</b>"),
                       yaxis=list(range=c(val.lim[2],val.lim[3]),title="<b>bias %</b>"),barmode="group"))
}

a.plot <- list(text=paste0("<b>Comparison of model performances<br>(mean absolute values of 10 runs)</b>"),
               font=list(size=16),xref="paper",yref="paper",yanchor="bottom",
               xanchor="center",align="center",x=0.5,y=1.03,showarrow=FALSE)

p.train.test <- subplot(p.test.rmse,style(p.train.rmse,showlegend=F),style(p.test.bias,showlegend=F),
                        style(p.train.bias,showlegend=F),nrows=2,shareX=T,shareY=T) %>%
  layout(annotations=a.plot,margin=list(t=60,b=20))

html.out <- paste0("knn&g_SNN_3D2DCNN_test.train.mean.html")
htmlwidgets::saveWidget(p.train.test,file=html.out)
browseURL(html.out)

# png.out <- paste0(out,".png")
# orca(p,png.out,scale=2,height=1080,width=1920)

