# Calculating LiDAR features from point cloud files (*.las) clipped to the shape of sample plots measured in the field

# loading packages
library(lidR)

# loading functions
source("lidR_lasfeat.R")

# loading plot information (field observations)
sp.data <- read.table("path/to/plot_data.csv",header=T,sep=",",as.is=T)

# directory with normalized LiDAR point clouds clipped to circular sample plots (9 m radius)
las.dir <- "path/to/ALS_data/"
# creating LiDAR catalog
las.cat <- catalog(las.dir)

sp.las.feat <- lapply(1:nrow(sp.data),function(i) {
  cat("\r",paste0("Processing item #",i,"/",nrow(sp.data)))
  las.tmp <- readLAS(paste0(las.dir,sp.data$sampleplotid[i],".las"))
  feat.first <- lasfeat.4.plots(sp.data[i,],las.in=las.tmp,h.max=37.0,h.veg=1.3,first.only=T,clip.las=F,calc.csm=F,normalize=F,progress=F)
  feat.last <- lasfeat.4.plots(sp.data[i,],las.in=las.tmp,h.max=37.0,h.veg=1.3,last.only=T,clip.las=F,calc.csm=T,normalize=F,progress=F)
  feat.text <- textfeat.4.plots(sp.data[i,],las.in=las.tmp,h.max=37.0,first.only=T,r.res=0.5,r.lag=2.5,clip.las=F,normalize=F,progress=F)
  return(cbind(feat.first,feat.last,feat.text))
})
sp.las.feat <- do.call(rbind,sp.las.feat)
sp.las.feat <- cbind(sp.data["sampleplotid"],sp.las.feat)

# dropping columns w/o useful or w/ missing information (standard deviation is 0 or NaN/NA)
sp.las.feat <- sp.las.feat[,apply(sp.las.feat,2,function(x) !(sd(x)==0|is.na(sd(x))))]

saveRDS(sp.las.feat,"als.feat.RDS")
