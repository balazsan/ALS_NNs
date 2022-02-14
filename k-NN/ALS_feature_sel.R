###################################################################################################
##################################### parallelized knn+genalg #####################################
###################################################################################################

# Predicting forest stand variables using k-NN and genalg based on ALS features.

# loading required libraries
library(parallel)

# loading k-NN/genalg functions
source("knn_funcs_group_28052019.R")

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

# forest attributes
for.attrs <- c("v","h","d")

outname <- "output_filename_of_your_choice"
outdir <- "output_directory_of_your_choice/"

# independent variables
mx <- train.data
# continuous forest attributes (dependent variables)
my <- sp.data.train[,for.attrs]
# weights for forest attributes (set to 1.0 if only one attribute predicted)
w.my <- c(0.6,0.2,0.2)

# scaling also "y" if multiple attributes estimated at the same time
if (ncol(my)>1) sca <- "both" else sca <- "x"

# first searching for optimal number of neighbors (k) and weight of neighbors (g)
# g>0 inverse distance weighing of neighbors
# g=0 equal weighing of neighbors
search.k.g <- function(run) {
  fs.c2 <- ffeatsel.con(mx,my,sca=sca,wrmse=w.my,wmin=0.3,maxpen=0,k=c(4,6),g=c(0,3),popSize=100,iters=30)
  # using "x" in result calculation as my is in original units if not scaled
  fs.c2.res <- ffeatsel.con.results(mx,my,rbga.res=fs.c2,sca="x")
  return(c(k=fs.c2.res$k,g=fs.c2.res$g,fs.c2.res$rmse.pct))
}

# parallel processing, number of cores used=5
cl <- makeCluster(5L)
clusterExport(cl,c("mx","my","sca","w.my","ffeatsel.con","ffeatsel.con.results",
                   "fknncv.fs","knn","fknnestcon","fknncv","fknncv.base","rmse"))
k.g <- parLapply(cl,1:5,search.k.g)
stopCluster(cl); rm(cl)

saveRDS(k.g,paste0(outdir,outname,"_kg_search.RDS"))

# summarizing results
k.g <- as.data.frame(do.call(rbind,k.g))
k.g

# select values for k and g
k <- 6
g <- 2.0
fname <- paste0(outname,"_k",k,"g",sprintf("%.1f",g))

# searching for best feature combinations
feat.sel <- function(run) {
  fs.tmp <- ffeatsel.bin(mx,my,sca=sca,wrmse=w.my,k=k,g=g,popSize=100,iters=30)
  # using "x" in result calculation as my is in original units if not scaled
  res.fs.tmp <- ffeatsel.bin.results(mx,my,rbga.res=fs.tmp,sca="x")
  return(list(fs.tmp,rmse.pct=res.fs.tmp$rmse.pct,bias.pct=res.fs.tmp$bias.pct))
}

# running feature search 10 times
cl <- makeCluster(5L)
clusterExport(cl,c("mx","my","sca","w.my","k","g","ffeatsel.bin","ffeatsel.bin.results",
                   "fknncv.fs","knn","fknnestcon","fknncv","fknncv.base","rmse"))
fs <- parLapply(cl,1:10,feat.sel)
stopCluster(cl); rm(cl)

# saving the results of all runs in the same file
saveRDS(fs,file=paste0(outdir,"fs_",fname,"_10runs.RDS"))

# printing RMSE and bias of each run
fs.rmse.bias <- lapply(fs,function(x) cbind(rmse.pct=x[[2]],bias.pct=x[[3]]))
fs.rmse.bias <- as.data.frame(do.call(cbind,fs.rmse.bias))
names(fs.rmse.bias) <- paste(names(fs.rmse.bias),rep(1:10,each=2),sep=".")
fs.rmse.bias

# weight search for selected features
# selected features can be weighted if it improves the results
wei.sel <- function(i) {
  x <- fs[[i]]
  # extracting results of feature selection
  res.fs.tmp <- ffeatsel.bin.results(mx,my,rbga.res=x[[1]],sca="x")
  # keeping only selected features for weight search
  mx.subs <- mx[,res.fs.tmp$in.use]
  # weights search for selected features (set wmin to >0.0 to reduce number of selected features)
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

# printing RMSE and bias of each run
# compare these results to fs.rmse.bias and check if RMSE and bias are improved
# if not, it's better to stick with the object "fs" instead of "ws"
fs.rmse.bias
ws.rmse.bias <- lapply(ws,function(x) cbind(rmse.pct=x$rmse.pct,bias.pct=x$bias.pct))
ws.rmse.bias <- as.data.frame(do.call(cbind,ws.rmse.bias))
names(ws.rmse.bias) <- paste(names(ws.rmse.bias),rep(1:10,each=2),sep=".")
ws.rmse.bias
# selecting best run based on e.g. min RMSE or min bias
best.mean.rmse <- which.min(colMeans(ws.rmse.bias[,c(T,F)]))
best.mean.bias <- which.min(colMeans(abs(ws.rmse.bias[,c(F,T)])))
best.run <- best.mean.rmse

# for each run $wei.sel.res contains the results of the weight search for the selected features (see $in.use, $in.use.weights)

# for compatibility with NNs
# predictions and accuracy metrics are calculated for all datasets (training, validation and test sets)
# replace here "ws" with "fs" if weight search didn't improve the results
preds.train <- fknnreg(mx,my,train.data,fs.res=ws[[best.run]]$wei.sel.res,verbose=F)
preds.val <- fknnreg(mx,my,val.data,fs.res=ws[[best.run]]$wei.sel.res,verbose=F)
preds.test <- fknnreg(mx,my,test.data,fs.res=ws[[best.run]]$wei.sel.res,verbose=F)

