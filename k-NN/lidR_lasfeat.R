# Using lidR and own functions to calculate 3D features
# Andras Balazs, Luke, 9.10.2018
# added function for textural feature calculation using heights above groud and/or laser intensity
# Andras Balazs, Luke, 12.12.2018
# fixed functions after major lidR update (2.0.0) https://github.com/Jean-Romain/lidR/blob/master/NEWS.md
# and added progress indicator
# Andras Balazs, Luke, 24.01.2019
# added option to not normalize in case it's already done
# Andras Balazs, Luke,30.05.2019
# added height thresholds for percentage of returns above each threshold
# Andras Balazs, Luke,10.07.2019
# replaced lasmetrics (deprecated) with cloud_metrics
# Andras Balazs, Luke,21.01.2020
# replaced all deprecated functions; csm-features calculated first returns only
# Andras Balazs, Luke,02.08.2021

#' Calculate vertical features from 3D point clouds (LAScatalog,*.las file) around sample plot centers with given radius
#' 
#' @param sp Data frame of sample plot data containing plot center coordinates (x,y)
#' @param x.name Character, name of column with plot coordinates x
#' @param y.name Character, name of column with plot coordinates y
#' @param radius Number or vector, Radius of sample plot in meters, if not set, 9 meters is used, if vector, length(radius)=nrow(sp)
#' @param las.in Laser catalog or *.las file containing 3D point data
#' @param h.max Maximum normalized height in meters allowed in laser data (if set, points above h.max excluded)
#' @param h.veg Vegetation height in meters (if set, points below h.veg excluded)
#' @param t.hold Thresholds for percentage of returns above each threshold (default from 5 to 45 by 5m steps)
#' @param first.only Boolean, if set to TRUE, only first returns included
#' @param last.only Boolean, if set to TRUE, only last returns included
#' @param firstlast.only Boolean, if set to TRUE, only first and last returns included
#' @param calc.csm Boolean, if set to TRUE, Canopy Surface Model and related features are calculated (rumple index, inner/outer volume and gap area)
#' @param clip.las Option to clip laser returns from a bigger dataset (True) or use the entire input file (False)
#' @param normalize Option to skip normalization if it has been done already
#' @param dem RasterLayer or a lasmetrics object computed with grid_terrain, if not provided kriging is used to create ground surface on the fly
#' @param progress should there be a progress indicator?
lasfeat.4.plots <- function(sp,x.name="x",y.name="y",radius=NULL,las.in,h.max=NULL,h.veg=NULL,t.hold=seq(5,45,5),
                            first.only=F,last.only=F,firstlast.only=F,calc.csm=F,clip.las=T,normalize=T,dem=NULL,progress=T) {
  require(lidR); require(Lmoments); require(fBasics); require(lsr); require(raster)
  if (clip.las && !any(names(sp)%in%x.name)) stop("x.name not in sample plot data!")
  if (clip.las && !any(names(sp)%in%y.name)) stop("y.name not in sample plot data!")
  if (!class(las.in)%in%c("LAS","LAScatalog")) stop("las.in has to be class LAS or LAScatalog!")
  if (!clip.las&!class(las.in)=="LAS") stop("las.in has to be class LAS if no clipping is performed!")
  if (!is.null(dem) && !class(dem)%in%c("RasterLayer","lasmetrics")) stop("dem has to be a RasterLayer or lasmetrics object!")
  if (!is.null(radius) && !is.numeric(radius)) stop("radius is not a number!")
  if (!is.null(radius) && length(radius)>1) stopifnot(length(radius)==nrow(sp))
  if (clip.las && is.null(radius)) {radius <- 9;cat("Radius not set, using 9 meters!\n")}
  if (!is.null(h.max) && !is.numeric(h.max)) stop("h.max is not a number!")
  if (!is.null(h.veg) && !is.numeric(h.veg)) stop("h.veg is not a number!")
  if (is.null(h.veg)) {h.veg <- 2.0;cat("Vegetation height not set, using 2.0 meters!\n")}
  if (sum(first.only,last.only,firstlast.only)>1) (stop("Only one of first.only/last.only/firstlast.only can be set to TRUE!"))

  # sending messages of lasfilter to file, otherwise nr of points below 0 printed
  # for warnings and errors check the file
  tmp <- file("lidR_lasfeat_msg.txt",open="wt"); sink(tmp,type="message")
  
  var.names <- c("zmax","zmean","zsd","zskew","zkurt","zentropy","pzabovezmean",paste0("pzabove",t.hold),"zq5","zq10","zq15","zq20","zq25","zq30","zq35",
                 "zq40","zq45","zq50","zq55","zq60","zq65","zq70","zq75","zq80","zq85","zq90","zq95","zpcum1","zpcum2",
                 "zpcum3","zpcum4","zpcum5","zpcum6","zpcum7","zpcum8","zpcum9","zcv","pveg","pcg","p20","p40","p60","p80","p95",
                 "canopy.rr","L1","L2","L3","L4","lmom.cov","lmom.skew","lmom.kurt","mean.quad","mean.cub","aad","mad.med",
                 "itot","imax","imean","isd","iskew","ikurt","ipground","ipcumzq10","ipcumzq30","ipcumzq50","ipcumzq70","ipcumzq90")
  
  # number of sample plots
  n.row <- nrow(sp)
  # increasing radius for surface model calculation if necessary
  if (calc.csm&clip.las) radius.csm <- radius+(radius/2)
  out <- lapply(1:n.row,function(i) {
    # progress indicator
    if (progress) cat("\r",paste0("Processing item #",i,"/",nrow(sp)))
	  # laser returns can be clipped from a bigger dataset or the entire file can be used
	  if (clip.las) {
	    # plot center coordinates
	    x <- sp[i,x.name]; y <- sp[i,y.name]
	    # obtaining radius for sample plot if radius provided is vector
	    if (length(radius)>1) {if (calc.csm) radius.csm.i <- radius.csm[i]; radius.i <- radius[i]} else {if (calc.csm) radius.csm.i <- radius.csm; radius.i <- radius}
	    # extracting laser returns for plots (with increased radius if required)
	    if (calc.csm) las.plot <- lasclipCircle(las.in,x,y,radius=radius.csm.i) else las.plot <- lasclipCircle(las.in,x,y,radius=radius.i)
	  } else {
	    las.plot <- las.in
	  }
    if (nrow(las.plot@data)!=0) {
      # removing duplicates (LiDAR tiles might overlap)
      las.plot <- filter_duplicates(las.plot)
      # normalizing heights if necessary
      if (normalize) {
        # normalizing heights by using Digital Terrain Model if available
        if (is.null(dem)) las.plot <- normalize_height(las.plot,kriging(k=5)) else las.plot <- las.plot-dem
      }
      # excluding outliers
      if (!is.null(h.max)) las.plot <- filter_poi(las.plot,Z<=h.max)
      if (calc.csm) {
        # filtering first returns for csm
        las.plot.csm <- filter_first(las.plot)
        if (clip.las) las.plot <- clip_circle(las.plot,x,y,radius=radius.i)
      }
      if (nrow(las.plot@data)>0) {
        # filtering laser points if required
        if (first.only) las.plot <- filter_first(las.plot)
        if (last.only) las.plot <- filter_last(las.plot)
        if (firstlast.only) las.plot <- filter_firstlast(las.plot)
        # extracting number of returns before excluding non-vegetation returns
        n.all <- length(las.plot$Z)
        # dropping non-vegetation returns
        las.plot <- filter_poi(las.plot,Z>=h.veg)
        if (nrow(las.plot@data)>0) {
          # extracting number of vegetation returns
          n.veg <- length(las.plot$Z)
          # calculating vertical features
          las.feat.tmp.z <- stdmetrics_z(las.plot$Z,th=t.hold)
          # coefficient of variation of laser canopy heights (%)
          las.feat.tmp.z$zcv <- las.feat.tmp.z$zsd/las.feat.tmp.z$zmean*100
          # proportion of vegetation returns
          las.feat.tmp.z$pveg <- n.veg/n.all*100
          # proportion of vegetation returns versus ground returns
          las.feat.tmp.z$pcg <- n.veg/(n.all-n.veg)*100
          # canopy height densities
          las.feat.tmp.z <- c(las.feat.tmp.z,cloud_metrics(las.plot,chp(Z,prec=c(0.2,0.4,0.6,0.8,0.95))))
          # canopy relief ratio
          if (max(las.plot@data$Z)!=min(las.plot@data$Z)) {
            las.feat.tmp.z$canopy.rr <- (mean(las.plot@data$Z)-min(las.plot@data$Z))/(max(las.plot@data$Z)-min(las.plot@data$Z))
          } else {
            las.feat.tmp.z$canopy.rr <- 0.0
          }
          # L-moments (1-4)
          lmom.tmp <- Lmoments(las.plot@data$Z)[1,]
          if (length(lmom.tmp)<4) {
            lmom.tmp <- c(lmom.tmp,rep(NA,4-length(lmom.tmp)))
            names(lmom.tmp) <- paste0("L",1:4)
          }
          # L-moment coefficient of variation
          lmom.c <- lmom.tmp["L2"]/lmom.tmp["L1"]; names(lmom.c) <- NULL
          if (is.na(lmom.c)) lmom.c <- 100.0
          las.feat.tmp.z$lmom.cov <- lmom.c
          # L-moment skewness
          lmom.s <- lmom.tmp["L3"]/lmom.tmp["L2"]; names(lmom.s) <- NULL
          if (is.na(lmom.s)) lmom.s <- 100
          las.feat.tmp.z$lmom.skew <- lmom.s
          # L-moment kurtosis
          lmom.k <- lmom.tmp["L4"]/lmom.tmp["L2"]; names(lmom.k) <- NULL
          if (is.na(lmom.k)) lmom.k <- 100
          las.feat.tmp.z$lmom.kurt <- lmom.k
          # setting NAs in Lmoments results to 100
          lmom.tmp[is.na(lmom.tmp)] <- 100
          las.feat.tmp.z <- c(las.feat.tmp.z,lmom.tmp)
          # quadratic mean
          las.feat.tmp.z$mean.quad <- (sum(las.plot@data$Z^2)/length(las.plot@data$Z))^(1/2)
          # cubic mean
          las.feat.tmp.z$mean.cub <- (sum(las.plot@data$Z^3)/length(las.plot@data$Z))^(1/3)
          # Average Absolute Deviation
          las.feat.tmp.z$aad <- aad(las.plot@data$Z)
          # Median of the absolute deviations from the overall median
          las.feat.tmp.z$mad.med <- mad(las.plot@data$Z,constant=1)
          
          # calculating intensity features
          las.feat.tmp.i <- cloud_metrics(las.plot,.stdmetrics_i)
          las.feat.tmp <- data.frame(c(las.feat.tmp.z,las.feat.tmp.i))
          # adjusting column names
          las.feat.tmp <- adj.names(las.feat.tmp,first.only,last.only,firstlast.only)
       } else {
          # no vegetation returns after filtering
          las.feat.tmp <- as.data.frame(t(c(rep(0.0,(43+length(t.hold))),0.0,rep(100,11),rep(0,12))))
          names(las.feat.tmp) <- var.names
          # adjusting column names
          las.feat.tmp <- adj.names(las.feat.tmp,first.only,last.only,firstlast.only)
       }
      } else {
        # no laser returns at all after clipping
        # this means there are returns in the buffered area but not on the original plot
        las.feat.tmp <- as.data.frame(t(rep(NA_real_,length(var.names))))
        names(las.feat.tmp) <- var.names
        # adjusting column names
        las.feat.tmp <- adj.names(las.feat.tmp,first.only,last.only,firstlast.only)
      }
      if (calc.csm) {
        # rumple index
        # creating Canopy Surface Model (buffered with radius/2 if clipping was done)
        # no significant difference using thresholds and max_edge
        # in order to have a robust function errors handled by try
        # (there might be errors due to sparse 3D data)
        # setting resolution
        csm.res <- 0.5
        csm <-  try(grid_canopy(las.plot.csm,res=csm.res,dsmtin()))
        if(is(csm,"try-error")) {
          las.feat.tmp$rmpl.ind <- NA; las.feat.tmp$volin <- NA; las.feat.tmp$volout <- NA; las.feat.tmp$gap.area <- NA
        } else {
          # transforming to data frame (x, y, z coordinates) for easy cropping
          csm <- data.frame(cbind(coordinates(csm),extract(csm,coordinates(csm))))
          names(csm) <- c("X","Y","Z")
          # cropping csm to plot size if clipping was performed
          if (clip.las) {
            del.buff <- ((csm$X-x)^2+(csm$Y-y)^2) <= radius.i^2
            csm <- csm[del.buff,]
          }
          # calculating rumple index (transforming csm back to raster on-the-fly)
          las.feat.tmp$rmpl.ind <- rumple_index(rasterFromXYZ(csm,res=c(csm.res,csm.res)))

          # dropping NAs
          csm <- csm[!is.na(csm$Z),]
          # inner volume
          volin.tmp <- sum(csm$Z[csm$Z>=h.veg]*(csm.res^2))
          las.feat.tmp$volin <- volin.tmp
          
          # outer volume = volmax-volmin
          volmax.tmp <- sum(rep(max(csm$Z),times=sum(csm$Z>=h.veg))*(csm.res^2))
          las.feat.tmp$volout <- volmax.tmp-volin.tmp
          
          # gap area
          las.feat.tmp$gap.area <- sum(csm$Z<h.veg)*(csm.res^2)
        }
      }
    } else {
      # plot not covered by the laser dataset (marked with NaN istead of NA)
      las.feat.tmp <- as.data.frame(t(rep(NaN,length(var.names))))
      names(las.feat.tmp) <- var.names
      # adjusting column names
      las.feat.tmp <- adj.names(las.feat.tmp,first.only,last.only,firstlast.only)
      if (calc.csm) {las.feat.tmp$rmpl.ind <- NaN; las.feat.tmp$volin <- NaN; las.feat.tmp$volout <- NaN; las.feat.tmp$gap.area <- NaN}
    }
    return(las.feat.tmp)
  })
  out <- do.call(rbind,out)
  sink(type="message"); close.connection(tmp)
  return(out)
}

#' Calculate canopy height densities
#' Canopy densitiy pxx is the proportion of laser returns
#' accumulated at the corresponding proportion of height
#' @param Z vector of normalized laser return heights
#' @param n number of returns
#' @param prec precentiles where canopy densities are calculated
chp <- function(Z,prec=c(0.2,0.4,0.6,0.8,0.95)) {
  n <- length(Z)
  chp.out <- lapply(prec,function(x) sum(Z>=sort(Z)[(cumsum(sort(Z))/sum(Z)>=x)][1])/n*100)
  names(chp.out) <- paste0("p",as.integer(prec*100))
  return(chp.out)
}

#' Calculate textural features from 3D point clouds (LAScatalog,*.las file) around sample plot centers in a given window
#' 
#' @param sp Data frame of sample plot data containing plot center coordinates (x,y)
#' @param x.name Character, name of column with plot coordinates x
#' @param y.name Character, name of column with plot coordinates y
#' @param calc.mode Character, either "canopy" or "intensity" or both
#' @param w.size Number or vector, window size in meters, if not set, 16x16 meters window is used, if vector, length(w.size)=nrow(sp)
#' @param las.in Laser catalog or *.las file containing 3D point data
#' @param h.max Maximum normalized height in meters allowed in laser data (if set, points above h.max excluded)
#' @param first.only Boolean, if set to TRUE, only first returns included (otherwise all)
#' @param r.res resolution of rasterized heights/intensity data
#' @param r.lag lag used for textural feature calculation in meters
#' @param clip.las Option to clip laser returns from a bigger dataset (True) or use the entire input file (False)
#' @param normalize Option to skip normalization if it has been done already
#' @param dem RasterLayer or a lasmetrics object computed with grid_terrain, if not provided kriging is used to create ground surface on the fly
textfeat.4.plots <- function(sp,x.name="x",y.name="y",calc.mode=c("canopy","intensity"),w.size=NULL,
                             las.in,h.max=NULL,first.only=T,r.res=0.5,r.lag=2.5,clip.las=T,normalize=T,dem=NULL,progress=T) {
  require(lidR,quietly=T); require(radiomics,quietly=T); require(raster,quietly=T)
  if (all(!c("canopy","intensity")%in%calc.mode)) stop("calc.mode has to contain either canopy or intensity!")
  if (clip.las && !any(names(sp)%in%x.name)) stop("x.name not in sample plot data!")
  if (clip.las && !any(names(sp)%in%y.name)) stop("y.name not in sample plot data!")
  if (!class(las.in)%in%c("LAS","LAScatalog")) stop("las.in has to be class LAS or LAScatalog!")
  if (!clip.las&!class(las.in)=="LAS") stop("las.in has to be class LAS if no clipping is performed!")
  if (!is.null(w.size) && !is.numeric(w.size)) stop("window size is not a number!")
  if (!is.null(w.size) && length(w.size)>1) stopifnot(length(w.size)==nrow(sp))
  if (clip.las && is.null(w.size)) {w.size <- 16;cat("Radius not set, using 16 meters!\n")}
  if (!is.null(h.max) && !is.numeric(h.max)) stop("h.max is not a number!")

  # sending messages of lasfilter to file, otherwise nr of points below 0 printed
  # for warnings and errors check the file
  tmp <- file("lidR_textfeat_msg.txt",open="wt"); sink(tmp,type="message")
  
  # output variable names
  var.names.csm <- c("csm.mean","csm.variance","csm.autoCorrelation","csm.cProminence","csm.cShade","csm.cTendency",
                     "csm.contrast","csm.correlation","csm.differenceEntropy","csm.dissimilarity",
                     "csm.energy","csm.entropy","csm.homogeneity1","csm.homogeneity2","csm.IDMN",
                     "csm.IDN","csm.inverseVariance","csm.maxProb","csm.sumAverage","csm.sumEntropy","csm.sumVariance")
  var.names.i <- c("i.mean","i.variance","i.autoCorrelation","i.cProminence","i.cShade","i.cTendency",
                   "i.contrast","i.correlation","i.differenceEntropy","i.dissimilarity",
                   "i.energy","i.entropy","i.homogeneity1","i.homogeneity2","i.IDMN",
                   "i.IDN","i.inverseVariance","i.maxProb","i.sumAverage","i.sumEntropy","i.sumVariance")
  # number of sample plots
  n.row <- nrow(sp)
  out <- lapply(1:n.row,function(i) {
    # progress indicator
    if (progress) cat("\r",paste0("Processing item #",i,"/",nrow(sp)))
    # obtaining window size and calculating AOI for sample plot if clipping is performed
    if (clip.las) {
      # plot center coordinates
      x <- sp[i,x.name]; y <- sp[i,y.name]
      if (length(w.size)>1) w.size.i <- w.size[i] else w.size.i <- w.size
      # calcuating area of interest (window is buffered by 5*r.res on all sides to avoid edge effect)
      # rounding to neasrest resolution unit
      x.min <- round(x/r.res)*r.res-((w.size.i/2)+(5*r.res)); x.max <- round(x/r.res)*r.res+((w.size.i/2)+(5*r.res))
      y.min <- round(y/r.res)*r.res-((w.size.i/2)+(5*r.res)); y.max <- round(y/r.res)*r.res+((w.size.i/2)+(5*r.res))
      if (first.only) {orig.filter <- opt_filter(las.in); opt_filter(las.in) <- "-keep_first"}
      # extracting laser returns for plots
      las.plot <- clip_rectangle(las.in,x.min,y.min,x.max,y.max)
      # setting filter back to original
      if (first.only) opt_filter(las.in) <- orig.filter
    } else {
      las.plot <- las.in
      if (first.only) las.plot <- filter_first(las.plot)
    }
    if (nrow(las.plot@data)>0) {
      # removing duplicates (LiDAR tiles might overlap)
      las.plot <- filter_duplicates(las.plot)
      # normalizing heights if necessary
      if (normalize) {
        # normalizing heights by using Digital Terrain Model if available
        if (is.null(dem)) las.plot <- normalize_height(las.plot,kriging(k=5)) else las.plot <- las.plot-dem
      }
      # excluding outliers
      if (!is.null(h.max)) las.plot <- filter_poi(las.plot,Z<=h.max)
      # lag for Grey Level Co-occurence Matrix (glcm)
      r.lag <- r.lag/r.res
      if (any(calc.mode=="canopy")) {
        # creating canopy surface model
        # no significant difference using thresholds and max_edge
        # in order to have a robust function errors handeled by try
        # (there might be errors due to scarse 3D data)
        # setting resolution
        csm.res <- 0.5
        csm <-  try(grid_canopy(las.plot,res=csm.res,dsmtin()))
        if(is(csm,"try-error")) {
          csm.textfeat <- rep(NA,21)
          names(csm.textfeat) <- var.names.csm
        } else {
          # transforming CSM to matrix and dropping buffer if clipping was performed
          csm <- getValues(csm,format="matrix")
          if (clip.las) csm <- csm[6:(nrow(csm)-5),6:(ncol(csm)-5)]
          # the funciton for texture calculation doesn't accept negative values; setting those to 1mm
          csm[csm<0] <- 0.001
          # calling function for texture calculation
          csm.textfeat <- textfeat(csm,d=r.lag)
          # setting names
          names(csm.textfeat) <- var.names.csm
        }
      }
      if (any(calc.mode=="intensity")) {
        # for rasterizing intensity height values replaced by intensity values
        las.plot@data$Z <- las.plot@data$Intensity
        # rasterizing intensity values
        # i.rast <-  try(grid_tincanopy(las.plot,thresholds=0,max_edge=0)) #due to major update function not available
        # setting resolution
        csm.res <- 0.5
        i.rast <-  try(grid_canopy(las.plot,res=csm.res,dsmtin()))
        if(is(csm,"try-error")) {
          i.textfeat <- rep(NA,21)
          names(i.textfeat) <- var.names.i
        } else {
          # transforming CSM to matrix and dropping buffer if clipping was performed
          i.rast <- getValues(i.rast,format="matrix")
          if (clip.las) i.rast <- i.rast[6:(nrow(i.rast)-5),6:(ncol(i.rast)-5)]
          # the funciton for texture calculation doesn't accept negative values; setting those to 0.001
          i.rast[i.rast<0] <- 0.001
          # calling function for texture calculation
          i.textfeat <- textfeat(i.rast,d=r.lag)
          # setting names
          names(i.textfeat) <- var.names.i
        }
      }
    } else {
      # plot not covered by the laser dataset (marked with NaN istead of NA)
      if (any(calc.mode=="canopy")) {csm.textfeat <- rep(NA,21); names(csm.textfeat) <- var.names.csm}
      if (any(calc.mode=="intensity")) {i.textfeat <- rep(NA,21); names(i.textfeat) <- var.names.i}
    }
    if (all(c("canopy","intensity")%in%calc.mode)) {
      return(c(csm.textfeat,i.textfeat))
    } else if ("canopy"%in%calc.mode) {
      return(csm.textfeat)
    } else {
      return(i.textfeat)
    }
  })
  out <- do.call(rbind,out)
  sink(type="message"); close.connection(tmp)
  return(as.data.frame(out))
}

#' Calculating textural features from rasterized data using matrix as input
#' @param in.matrix input raster data as matrix
#' @param d lag used for textural feature calculation in pixels

require(radiomics)

textfeat <- function(in.matrix,d) {
  # creating Grey Level Co-occurrence Matrices for 4 cardinal directions
  # if number of unique values is in.matrix is less than 20 n_grey is automatically set to that value
  # sometimes this triggers an error, n_grey will be set to unique values-1
  ngrey <- 20
  ngrey.alt <- length(table(in.matrix))-1
  cf <- lapply(c(0,45,90,135),function(a) {
    r.glcm <- try(glcm(in.matrix,angle=a,d=d,n_grey=ngrey,normalize=T))
    # if error occurs setting n_grey lower
    if (is(r.glcm,"try-error")) {
      ngrey <<- ngrey.alt
      r.glcm <- glcm(in.matrix,angle=a,d=d,n_grey=ngrey,normalize=T)
      cat(paste0("Using n_grey=",ngrey.alt+1," caused an error, using ",ngrey.alt," instead\n"))
    }
    # calculating features
    calc_features(r.glcm)
  })
  cf <- do.call(rbind,cf)
  # averaging features of 4 directions
  cf <- colMeans(cf)
  return(cf)
}

#' Adjusting column names
#' Based on first, last, firstlast suffixes .f, .l or .fl are added to the column names respetively
adj.names <- function(in.table,first=F,last=F,firstlast=F) {
  if (first) {
    names(in.table) <- paste0(names(in.table),".f")
  }
  if (last) {
    names(in.table) <- paste0(names(in.table),".l")
  }
  if (firstlast) {
    names(in.table) <- paste0(names(in.table),".fl")
  }
  names(in.table) <- gsub("^z","h",names(in.table))
  names(in.table) <- gsub("^pz","p",names(in.table))
  names(in.table) <- gsub("zmean","hmean",names(in.table))
  return(in.table)
}