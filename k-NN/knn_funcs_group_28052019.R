# Revision 1.1  2011/01/17 15:42:19  japitkan
# Initial revision
#
# $Id: knn_funcs.R,v 1.28 2016/05/24 14:05:07 japitkan Exp japitkan $
# 2016/11/11 balazs, added option to query one data set with another (ffeatsel.con/bin, ffeatsel.con/bin.results, fknncv.base, fknncv, fknncv.fs)
# Input data format:
# mx: list of two tables with features for two data sets, first being the data and second the query
# the nearest neighbours are selected for the query data set items from the data items
# my: table of data variables
# my.query: table of query variables
#
# 2016/09/09 balazs, added grouping option to feature selection, enhanced speed with package nabor (knn)
# 2018/06/20 balazs: added check for standard deviation = 0, stop script if any column in mx (or mx[[1]] or mx[[2]]) has sd=0
# 2019/05/28 balazs: added option to select any best result in ffeatsel.bin.results after checking rbga.res

# Using nabor package's knn instead of RANN's nn2 due to speed enhancment in knn
library(RANN); library(nabor)

fknnestcon <- function(idx, wei, my, k="all", g=1, zerodist=NULL, onlyzeros=F)
   #  idx:    matrix of nn-indices, (n of query data) x (k of the nn search)
   #  wei:    matrix of nn-distances, (n of query data) x (k of the nn search).
   #          Distances are transformed to weights later in this function.
   #  my:     matrix of continuos variables of reference data. Nn-indices
   #          refer to row index of my
   #  k:      number of nearest neighbours used in calculation. If "all",
   #          use all (i.e. the number of columns) in nn index-matrix.
   #  g:      weighting of the nearest neighbours: larger g gives less weight to
   #          an nn on a larger distance
   #          g=0 : equal weighting of the nn
   #          g=1 : inverse distance weighting of the nn
   #  zerodist:   value to add in distance in inverse distance based weighting to
   #          avoid division by zero: weight = 1/(zerodist+distance). Default
   #          is to replace only zero distances with the half of the smallest
   #          non-zero distance
   #  onlyzeros:  only replace zero distances with the value of zerodist (> 0)
   #          without modifying other distances
   #
   #  Knn estimation for continuos variables using inverse distance based (g>0)
   #  or equal (g=0) weighting of the nn.
   #
   #  22 September 2010 Juho Pitkänen
{
   if (k == "all") {
      k <- ncol(idx)
   } else if (k < ncol(idx)) {
      # only keep k nearest neighbours; subset gives matrix even if k is 1
      wei <- subset(wei, select=1:k)
      idx <- subset(idx, select=1:k)
   } else if (k > ncol(idx)) {
      stop("k is larger than number of nn in the matrix of nn-indices (idx)")
   }

   if (g == 0 | k == 1) {
      # equal weighting of the nn
      wei[,] <- 1/k
   } else {
      # non-equal weighting of the nn: have to handle zero distances to 
      # avoid division by zero
      if (is.null(zerodist)) {
         if (max(wei) > 0) {
            # zero distances to half (now) of smallest non-zero distance
            if (any(wei == 0)) {
               smalldist <- min(wei[wei > 0]) / 2
               wei[wei==0] <- smalldist
            }
         } else {
            # only zero distances
            wei[,] <- 1/k
         }
      } else {
         stopifnot(zerodist > 0)
         if (onlyzeros) {
            # add zerodist only in zero distances
            wei[wei==0] <- zerodist
         } else {
            # add zerodist in all distances
            wei <- wei + zerodist
         }
      }

      # calculate weights from distances
      if (g == 1) {
         wei <- 1/wei
      } else {
         wei <- 1/wei^g
      }
      # fast row sum
      wsum <- as.vector(wei %*% rep.int(1,k))
      wei <- wei/wsum
   }

   if (! is.matrix(my)) {
      # a vector or data frame?
      my <- as.matrix(my)
   }
   cols <- ncol(my)
   rows <- nrow(idx)
   # result matrix
   y <- matrix(nrow=rows, ncol=cols)

   # calculate by variables
   for (i in 1:cols) {
      # variable values of nn in a k*rows vector
      nnvar <- my[idx,i]
      # to a matrix
      dim(nnvar) <- c(rows, k)
      # multiply by weights
      nnvar <- nnvar*wei
      res <- nnvar %*% rep.int(1, k)
      y[,i] <- res
   }

   if (! is.null(colnames(my)) ) {
      colnames(y) <- colnames(my)
   }
   return(y)
}

fknnestdisc <- function(idx, wei, my, k="all", g=1, zerodist=NULL, onlyzeros=F)
   #  idx:    matrix of nn-indices, (n of query data) x (k of the nn search)
   #  wei:    matrix of nn-distances, (n of query data) x (k of the nn search).
   #          Distances are transformed to weights later in this function.
   #  my:     matrix of discrete variables of reference data. Nn-indices
   #          refer to row index of my
   #  k:      number of nearest neighbours used in calculation. If "all",
   #          use all (i.e. the number of columns) in nn index-matrix.
   #  g:      weighting of the nearest neighbours: larger g gives less weight to
   #          an nn on a larger distance
   #          g=0 : equal weighting of the nn
   #          g=1 : inverse distance weighting of the nn
   #  zerodist:   value to add in distance in inverse distance based weighting to
   #          avoid division by zero: weight = 1/(zerodist+distance). Default
   #          is to replace zero distances with the half of the smallest
   #          non-zero distance
   #  onlyzeros:  only replace zero distances with the value of zerodist (> 0)
   #          without modifying other distances
   #
   #  Knn estimation for discrete variables based on mode within the nn. Mode
   #  can be calculated according to the mode of the variable values directly
   #  (g=0), or according to the sum of weights of the variable values (g>0).
   #  Discrete variables have to be numerically coded.
   #
   #  23 September 2010 Juho Pitkänen
{
   if (k == "all") {
      k <- ncol(idx)
   } else if (k < ncol(idx)) {
      # only keep k nearest neighbours; subset gives matrix even if k is 1
      wei <- subset(wei, select=1:k)
      idx <- subset(idx, select=1:k)
   } else if (k > ncol(idx)) {
      stop("k is larger than number of nn in the matrix of nn-indices (idx)")
   }

   if (g == 0 | k == 1) {
      # equal weighting of the nn
      wei[,] <- 1/k
   } else {
      # non-equal weighting of the nn: have to handle zero distances to 
      # avoid division by zero
      if (is.null(zerodist)) {
         if (max(wei) > 0) {
            if (any(wei == 0)) {
               # zero distances to half (now) of smallest non-zero distance
               smalldist <- min(wei[wei > 0]) / 2
               wei[wei==0] <- smalldist
            }
         } else {
            # only zero distances
            wei[,] <- 1/k
         }
      } else {
         stopifnot(zerodist > 0)
         if (onlyzeros) {
            # add zerodist only in zero distances
            wei[wei==0] <- zerodist
         } else {
            # add zerodist in all distances
            wei <- wei + zerodist
         }
      }

      # calculate weights from distances
      if (g == 1) {
         wei <- 1/wei
      } else {
         wei <- 1/wei^g
      }
      # fast row sum
      wsum <- as.vector(wei %*% rep.int(1,k))
      wei <- wei/wsum
   }

   if (! is.matrix(my)) {
      # a vector or data frame?
      my <- as.matrix(my)
   }
   cols <- ncol(my)
   rows <- nrow(idx)
   # result matrix
   y <- matrix(nrow=rows, ncol=cols)

   # to get same results in different runs, if there are even cases
   set.seed(100)

   # calculate by variables
   for (i in 1:cols) {
      # variable values of nn in a k*rows vector
      nnvar <- my[idx,i]
      # to a matrix
      dim(nnvar) <- c(rows, k)

      # initialize vectors for result, maximum sum of weights and
      # count of equal sums
      res <- rep.int(-1, rows)
      maxsum <- rep.int(-1, rows)
      eqcount <- rep.int(1, rows)
      # get a vector of the variable values in the reference data
      tab <- table(my[,i])
      vals <- as.numeric(names(tab))
      for (j in 1:length(vals)) {
         # find variable value that has the largest sum of weights
         # within the nn:s
         jvar <- vals[j]
         # row sums of nn that have variable value jvar
         jvarsum <- (as.numeric(nnvar==jvar) * wei) %*% rep.int(1,k)

         # Even case handling: select randomly which of the even cases
         # are replaced
         eqs <- jvarsum == maxsum
         eqcount[eqs] <- eqcount[eqs] + 1
         res[ eqs & runif(rows) < 1/eqcount ] <- jvar

         # new maxima
         gts <- jvarsum > maxsum
         res[gts] <- jvar
         maxsum[gts] <- jvarsum[gts]
         eqcount[gts] <- 1
      }
      y[,i] <- res
   }

   # back to usual random number generation
   set.seed(NULL)

   if (! is.null(colnames(my)) ) {
      colnames(y) <- colnames(my)
   }
   return(y)
}

fknncv.base <- function(mx, my, k=5, g=1, out="est", zerodist=NULL, onlyzeros=F,
                        itemgroupid=NULL, group.action=NULL, query.incl.ref=F)
  #  mx:  matrix of x variables of reference data
  #       or a list of matricies if used for querying one data set with another
  #       the data sets have to have the same amount of columns (same amount of features)
  #       data set #1 (mx[[1]]) is the matrix to select nearest neighbours from
  #       data set #2 (mx[[2]]) is the matrix to search nearest neighbours for (query data set)
  #  my:  matrix of continuous variables of reference data (y variables)
  #  k:   number of nearest neighbours used in calculation
  #  g:   weighting of the nearest neighbours: larger g gives less weight to
  #       an nn on a larger distance
  #       g=0 : equal weighting of the nn
  #       g=1 : inverse distance weighting of the nn
  #  out:    "nn"   : return numbers and distances of nn
  #          other value : return estimates
  #  zerodist:   value to add in distance to avoid division by zero in weighting,
  #       see fknnestcon()
  #  onlyzeros:  only replace zero distances with the value of zerodist (> 0)
  #          without modifying other distances
  #  itemgroupid: sometimes nns from the same group of items need to be selected or excluded
  #               it can be done by defining a vector with group ids (length of vector=number of rows in mdata)
  #               nns from same/different groups will be kept
  #  group.action: "include" : include nns only from the same group as the query item
  #                "exclude" : exclude nns from the same group as the query item
  #  query.incl.ref: Boolean, is the query data included in the reference data (e.g. ref=Evo+Vj, query=Evo)
  #
  #  Leave-one-out cross validation for knn estimation of continuous variables
  #
  #  Returns:  a matrix of estimates for my, or a list of nn indices and
  #            their distances
  #
  #  7 January 2011 Juho Pitkänen
{
  # number of rows
  # rows <- nrow(mx)
  # if mx a list of matricies is.list(mx)==TRUE
  # if mx is a matrix/data frame is.list(mx)==TRUE
  ifelse(is.null(dim(mx)),rows <- nrow(mx[[1]]),rows <- nrow(mx))
  
  # y may be given as a vector or data frame
  if (! is.matrix(my) ) {
    # a vector or data frame?
    my <- as.matrix(my)
  }
  if (nrow(my) != rows) stop("Number of rows of mx and my differ")
  
  if (!is.null(itemgroupid)) {
    if (!is.null(dim(mx))) {
      # picking nns exclusively from the same group as query item
      if (group.action=="include") {
        # splitting data based on group IDs
        mx.split <- split.data.frame(mx,itemgroupid)
        # getting nns separately for each group
        rann <- lapply(mx.split,function(x) {
          # k+1: query item usually within nns
          rann.tmp <- knn(x, x, k=k+1, eps=0.0)
          # fetching indicies of nns from mx before it was split
          rann.tmp$nn.idx <- apply(rann.tmp$nn.idx,2,function(y) as.integer(row.names(x)[y]))
          # setting row names in rann output to match mx
          rownames(rann.tmp$nn.idx) <- rownames(rann.tmp$nn.dists) <- row.names(x)
          return(rann.tmp)
        })
        # extracting indicies and distances and ordering results
        idx <- lapply(rann,"[[",1); idx <- do.call(rbind,idx); idx <- idx[order(as.integer(rownames(idx))),]
        dists <- lapply(rann,"[[",2); dists <- do.call(rbind,dists); dists <- dists[order(as.integer(rownames(dists))),]
        # coping with query items being within nns
        # all query items selected as their closest nn
        if (all(rownames(idx)==idx[,1])) {
          idx <- idx[,-1]; dists <- dists[,-1]
          # at least one query item not within its nns/not closest nn
        } else {
          dists <- t(sapply(1:nrow(mx),function(x) {
            if (any(rownames(idx)[x]==idx[x,])) return(dists[x,-which(rownames(idx)[x]==idx[x,])]) else return(dists[x,1:k])
          }))
          idx <- t(sapply(1:nrow(mx),function(x) {
            if (any(rownames(idx)[x]==idx[x,])) return(idx[x,-which(rownames(idx)[x]==idx[x,])]) else return(idx[x,1:k])
          }))
        }
        # THIS BIT NEEDS TO BE REVISED, UTTERLY SLOW!!!--->
        # picking nns from any but the same group as query item
      } else {
        # maxskip: the maximum amount of nns needed to be excluded (the group with the most members)
        maxskip <- max(table(itemgroupid))
        # rann <- nn2(mx, mx, k=k+maxskip, eps=0.0)
        rann <- knn(mx, mx, k=k+maxskip, eps=0.0)
        
        ok.idx <- apply(rann$nn.idx,2,function(x) itemgroupid[x])
        # excluding query items from nns happens automatically
        # (query item cannot be itself if nn is not from the same group)
        ok.idx <- ok.idx!=itemgroupid
        idx <- t(sapply(1:nrow(mx[[2]]),function(x) rann$nn.idx[x,ok.idx[x,]][1:k]))
        dists <- t(sapply(1:nrow(mx[[2]]),function(x) rann$nn.dists[x,ok.idx[x,]][1:k]))
      } # <---
    } else {
      if (group.action=="include") {
        # ADDED LATER!
      } else {
        # maxskip: the maximum amount of nns needed to be excluded (the group with the most members)
        maxskip <- max(table(itemgroupid[[1]]))
        rann <- knn(mx[[1]], mx[[2]], k=k+maxskip, eps=0.0)
        ok.idx <- apply(rann$nn.idx,2,function(x) itemgroupid[[1]][x])
        # excluding query items from nns happens automatically
        # (query item cannot be itself if nn is not from the same group)
        ok.idx <- ok.idx!=itemgroupid[[2]]
        idx <- t(sapply(1:nrow(mx[[2]]),function(x) rann$nn.idx[x,ok.idx[x,]][1:k]))
        dists <- t(sapply(1:nrow(mx[[2]]),function(x) rann$nn.dists[x,ok.idx[x,]][1:k]))
      }
    }
  } else {
    # adding option to query one data set with another
    if (is.null(dim(mx))) {
      # the item itself cannot be included in the nns
      if (!query.incl.ref) {
        rann <- knn(mx[[1]], mx[[2]], k=k, eps=0.0)
        idx <- rann$nn.idx; dists <- rann$nn.dists
        # the item itself might be included in the nns (e.g. ref=Evo+Vj, query=Evo)
      } else {
        rann <- knn(mx[[1]], mx[[2]], k=k+1, eps=0.0)
        idx <- rann$nn.idx; dists <- rann$nn.dists
        rownums <- 1:nrow(idx)
        if (all(rownums==idx[,1])) {
          idx <- idx[,-1]; dists <- dists[,-1]
          # at least one query item not within its nns/not closest nn
        } else {
          dists <- t(sapply(rownums,function(x) {
            if (any(x==idx[x,])) return(dists[x,-which(x==idx[x,])]) else return(dists[x,1:k])
          }))
          idx <- t(sapply(rownums,function(x) {
            if (any(x==idx[x,])) return(idx[x,-which(x==idx[x,])]) else return(idx[x,1:k])
          }))
        }
      }
      # this bit is the original one, simple query from one data set
    } else {
      # find k+1 nn:s: each item is included in the search data, so get one extra nn
      # rann <- nn2(mx, mx, k=k+1, eps=0.0)
      rann <- knn(mx, mx, k=k+1, eps=0.0)
      idx <- rann$nn.idx; dists <- rann$nn.dists
      rownums <- 1:nrow(idx)
      # 2016/09/13 Balazs: improving speed (mx: 1806 rows, k=5, speed improved from 443 sec to 307 sec)
      # coping with query items being within nns
      # all query items selected as their closest nn
      if (all(rownums==idx[,1])) {
        idx <- idx[,-1]; dists <- dists[,-1]
        # at least one query item not within its nns/not closest nn
      } else {
        dists <- t(sapply(rownums,function(x) {
          if (any(x==idx[x,])) return(dists[x,-which(x==idx[x,])]) else return(dists[x,1:k])
        }))
        idx <- t(sapply(rownums,function(x) {
          if (any(x==idx[x,])) return(idx[x,-which(x==idx[x,])]) else return(idx[x,1:k])
        }))
      }
    }
    
    
    # # get k nn:s so that the query item itself is not within the nn:s
    # rownums <- 1:nrow(rann$nn.idx)
    # ok.idx <- rann$nn.idx != rownums
    # 
    # if (table(ok.idx)[1] == rows) {
    #   # item is within nn:s in each row, i.e. there is one false and 
    #   # k true values in each row of ok.idx: just drop the nn:s of the false values
    #   idx <- t(rann$nn.idx)[t(ok.idx)]
    #   dim(idx) <- c(k, rows)
    #   idx <- t(idx)
    #   dists <- t(rann$nn.dists)[t(ok.idx)]
    #   dim(dists) <- c(k, rows)
    #   dists <- t(dists)
    # } else {
    #   # item is not within nn:s in some lines: those lines would broke
    #   # the method above
    #   idx <- matrix(nrow=rows, ncol=k)
    #   dists <- matrix(nrow=rows, ncol=k)
    #   for (i in 1:rows) {
    #     idx[i,] <- rann$nn.idx[i, ok.idx[i,] ][1:k]
    #     dists[i,] <- rann$nn.dists[i, ok.idx[i,] ][1:k]
    #   }
    # }
    
  }
  
  if (out == "nn") {
    return(list(idx=idx, dists=dists))
  } else {
    # calculate continuous estimates
    y <- fknnestcon(idx, dists, my, k=k, g=g, zerodist=zerodist, onlyzeros=onlyzeros)
    return(y)
  }
}

fknncv <- function(mx, my, my.query=NULL, sca="x", k=5, g=1, out="est", w=NULL, fs.vars=F,
                   zerodist=NULL, onlyzeros=F, itemgroupid=NULL, group.action=NULL, query.incl.ref=F)
   #  sca:    scale x and/or y variables by standard deviation
   #          "both" : scale x and y variables
   #          "x" : scale x variables
   #          "y" : scale y variables
   #          other value : no scaling
   #  out:    "est"  : return estimates
   #          "rmse" : return rmse and bias vectors
   #          "nn"   : return indices and distances of nn
   #          other value : return estimates
   #  w:      weights for columns of mx, i.e. feature weights. If scaling
   #          is done, weights are applied after scaling.
   #  fs.vars:  if TRUE and out="rmse", return also feature selection variables
   #            needed to run fknnreg() or knnimage.con()
   #  zerodist:   value to add in distance to avoid division by zero in weighting,
   #          see fknnestcon()
   #  onlyzeros:  only replace zero distances with the value of zerodist (> 0),
   #          see fknnestcon()
   #
   #  A wrapper for calling fknncv.base, with a possibility to scale x and/or y
   #  variables, apply weights for columns of mx, and return either estimates or
   #  rmse and bias vectors or nearest neighbours and their distances
   #
   #  Returns:  either a matrix of estimates for my, or a list of rmse and bias
   #  vectors for columns of my, or a list of nn of indices and their distances
   #
   #  11 January 2011 Juho Pitkänen
{
   # for searching nn
   library(RANN); library(nabor)

   if (fs.vars && out != "rmse") {
      message("fknncv(): parameter fs.vars=T is only used if out='rmse'")
   }

   # checking if my.query is not NULL
   if (is.null(dim(mx))&is.null(my.query)) stop("No my.query defined!")
   
   # number of rows in the query matrix
   ifelse(is.null(dim(mx)),rows <- nrow(mx[[1]]),rows <- nrow(mx))

   # y may be given as a vector or data frame
   is.dframe <- F
   if (! is.matrix(my) ) {
      # a vector or data frame
      if (is.data.frame(my)) is.dframe <- T
      my <- as.matrix(my)
   }
   if (nrow(my) != rows) stop("Number of rows of mx and my differ")

   # scale variables
   # if (sca == "x" || sca == "both") {
   #    x.sd <- apply(mx, 2, sd)
   #    mx <- scale(mx, center=F, scale=x.sd)
   # }
   if (sca == "x" || sca == "both") {
      if (is.null(dim(mx))) {
        # checking for columns with sd=0 in both datasets
        x.sd <- apply(mx[[1]], 2, sd)
        if (any(x.sd==0)) stop(paste0("Standard deviation of columns in mx[[1]] ",paste0(names(x.sd)[which(x.sd==0)],collapse=", "),
                                      " is zero. Scaling cannot be done for these columns (and they are meaningless), remove them!"))
        x.sd <- apply(mx[[2]], 2, sd)
        if (any(x.sd==0)) stop(paste0("Standard deviation of columns in mx[[2]] ",paste0(names(x.sd)[which(x.sd==0)],collapse=", "),
                                      " is zero. Scaling cannot be done for these columns (and they are meaningless), remove them!"))
        # scaling is done to the standard deviation of all the data (reference+query)
        x.sd <- apply(rbind(mx[[1]],mx[[2]]),2,sd)
        mx <- lapply(mx,function(mx.x) {
          mx.x <- scale(mx.x, center=F, scale=x.sd); return(mx.x)
        })
      } else {
        x.sd <- apply(mx, 2, sd)
        if (any(x.sd==0)) stop(paste0("Standard deviation of columns ",paste0(names(x.sd)[which(x.sd==0)],collapse=", "),
                                      " is zero. Scaling cannot be done for these columns (and they are meaningless), remove them!"))
        mx <- scale(mx, center=F, scale=x.sd)
      }
   }
   # if (sca == "y" || sca == "both") {
   #   y.sd <- apply(my, 2, sd)
   #   my <- scale(my, center=F, scale=y.sd)
   # }
   if (sca == "y" || sca == "both") {
     if (!is.null(my.query)) {
       # scaling is done to the standard deviation of all the data (reference+query)
       y.sd <- apply(rbind(my,my.query), 2, sd)
       my <- scale(my, center=F, scale=y.sd)
       my.query <- scale(my.query, center=F, scale=y.sd)
     } else {
       y.sd <- apply(my, 2, sd)
       my <- scale(my, center=F, scale=y.sd)
     }
   }

   # weighting of x variables
   if (length(w) > 0) {
      if (sum(w) <= 0) {
         stop("Sum of weights in w is <= 0")
      } else if (ifelse(is.null(dim(mx)),length(w) == ncol(mx[[1]]),length(w) == ncol(mx))) {
      # } else if (!is.lsit(mx) && length(w) == ncol(mx)) {
        # all features, or subset, and their weights
        # feature is not selected if weight is 0
        indices <- w > 0
        wei.sel <- w[indices]
        if (is.null(dim(mx))) {
          mx <- lapply(mx,function(mx.x) {
            mx.x <- mx.x[,indices]
            # apply weights
            mx.x <- t(wei.sel*t(mx.x))
          })
        } else {
          mx <- mx[,indices]
          # apply weights
          mx <- t(wei.sel*t(mx))
        }
      } else {
         stop("Number of weights in w and columns of mx differ")
      }
   } else {
     ifelse(is.null(dim(mx)),indices <- rep(T,ncol(mx[[1]])),indices <- rep(T,ncol(mx)))
   }

   # get continuous estimates, or nn indices and distances
   y <- fknncv.base(mx, my, k=k, g=g, out=out, zerodist=zerodist, onlyzeros=onlyzeros, itemgroupid=itemgroupid, group.action=group.action,
                    query.incl.ref=query.incl.ref)

   if (out == "rmse") {
      # return rmse, bias, rmse-% and bias-%
      # error now so that underestimation gives negative bias
      ifelse(is.null(dim(mx)),res <- rmse(my.query,y,neg.under=T),res <- rmse(my,y,neg.under=T))

      if (fs.vars) {
        # add feature indices and parameters used
        ifelse(is.null(dim(mx)),res$in.use <- colnames(mx[[1]]),res$in.use <- colnames(mx))
        if (is.null(w)) {
          ifelse(is.null(dim(mx)),res$weights <- rep(1,ncol(mx[[1]])),res$weights <- rep(1, ncol(mx)))
          # res$in.use <- colnames(mx)
        } else {
          res$weights <- w
          # res$in.use <- colnames(mx)
          res$in.use.weights <- wei.sel
        }
        res$k <- k
        res$g <- g
        if (sca == "x" || sca == "both") {
          res$x.sd <- x.sd
          res$in.use.sd <- x.sd[indices]
        }
      }
      return(res)
    } else {
      # return estimates, or nn indices and distances in a list
      if (out != "nn" && is.dframe) y <- as.data.frame(y)
      return(y)
    }
}

fknncv.fs <- function(mx, my, my.query=my.query, k=5, g=1, n.drop=0, wrmse=NULL,
                      zerodist=NULL, onlyzeros=F, itemgroupid=NULL, group.action=NULL, query.incl.ref=F)
  #  For feature selection iterations. Like fknncv.base but
  #  1) additionally rmse and bias of the estimates are calculated within this
  #  function
  #  2) no check that my is a matrix
  #
  #  wrmse: only used if n.drop > 0
  #  itemgroupid: sometimes nns from the same group of items need to be selected or excluded
  #               it can be done by defining a vector with group ids (length of vector=number of rows in mdata)
  #               nns from same/different groups will be kept
  #  group.action: "include" : include nns only from the same group as the query item
  #                "exclude" : exclude nns from the same group as the query item
  #  query.incl.ref: Boolean, is the query data included in the reference data (e.g. ref=Evo+Vj, query=Evo)
  #
  #  Returns:  a list of rmse and bias vectors for columns of my
  #
  #  7 January 2011 Juho Pitkänen
{
  # setting itemgroupid to NULL
  # if (is.null(dim(mx))) itemgroupid <- NULL
  
  
  if (!is.null(itemgroupid)) {
    if (!is.null(dim(mx))) {
      # picking nns exclusively from the same group as query item
      if (group.action=="include") {
        # splitting data based on group IDs
        mx.split <- split.data.frame(mx,itemgroupid)
        # getting nns separately for each group
        rann <- lapply(mx.split,function(x) {
          # k+1: query item usually within nns
          rann.tmp <- knn(x, x, k=k+1, eps=0.0)
          # fetching indicies of nns from mx before it was split
          rann.tmp$nn.idx <- apply(rann.tmp$nn.idx,2,function(y) as.integer(row.names(x)[y]))
          # setting row names in rann output to match mx
          rownames(rann.tmp$nn.idx) <- rownames(rann.tmp$nn.dists) <- row.names(x)
          return(rann.tmp)
        })
        # extracting indices and distances and ordering results
        idx <- lapply(rann,"[[",1); idx <- do.call(rbind,idx); idx <- idx[order(as.integer(rownames(idx))),]
        dists <- lapply(rann,"[[",2); dists <- do.call(rbind,dists); dists <- dists[order(as.integer(rownames(dists))),]
        # coping with query items being within nns
        # all query items selected as their closest nn
        if (all(rownames(idx)==idx[,1])) {
          idx <- idx[,-1]; dists <- dists[,-1]
          # at least one query item not within its nns/not closest nn
        } else {
          dists <- t(sapply(1:nrow(mx),function(x) {
            if (any(rownames(idx)[x]==idx[x,])) return(dists[x,-which(rownames(idx)[x]==idx[x,])]) else return(dists[x,1:k])
          }))
          idx <- t(sapply(1:nrow(mx),function(x) {
            if (any(rownames(idx)[x]==idx[x,])) return(idx[x,-which(rownames(idx)[x]==idx[x,])]) else return(idx[x,1:k])
          }))
        }
        # THIS BIT NEEDS TO BE REVISED, UTTERLY SLOW!!!--->
        # picking nns from any but the same group as query item
      } else {
        # maxskip: the maximum amount of nns needed to be excluded (the group with the most members)
        maxskip <- max(table(itemgroupid))
        # rann <- nn2(mx, mx, k=k+maxskip, eps=0.0)
        rann <- knn(mx, mx, k=k+maxskip, eps=0.0)
        
        ok.idx <- apply(rann$nn.idx,2,function(x) itemgroupid[x])
        # excluding query items from nns happens automatically
        # (query item cannot be itself if nn is not from the same group)
        ok.idx <- ok.idx!=itemgroupid
        idx <- t(sapply(1:nrow(mx[[2]]),function(x) rann$nn.idx[x,ok.idx[x,]][1:k]))
        dists <- t(sapply(1:nrow(mx[[2]]),function(x) rann$nn.dists[x,ok.idx[x,]][1:k]))
      } # <---
    } else {
      if (group.action=="include") {
        # ADDED LATER!
      } else {
        # maxskip: the maximum amount of nns needed to be excluded (the group with the most members)
        maxskip <- max(table(itemgroupid[[1]]))
        rann <- knn(mx[[1]], mx[[2]], k=k+maxskip, eps=0.0)
        ok.idx <- apply(rann$nn.idx,2,function(x) itemgroupid[[1]][x])
        # excluding query items from nns happens automatically
        # (query item cannot be itself if nn is not from the same group)
        ok.idx <- ok.idx!=itemgroupid[[2]]
        idx <- t(sapply(1:nrow(mx[[2]]),function(x) rann$nn.idx[x,ok.idx[x,]][1:k]))
        dists <- t(sapply(1:nrow(mx[[2]]),function(x) rann$nn.dists[x,ok.idx[x,]][1:k]))
      }
    }
  } else {
    # adding option to query one data set with another
    if (is.null(dim(mx))) {
      # the item itself cannot be included in the nns
      if (!query.incl.ref) {
        rann <- knn(mx[[1]], mx[[2]], k=k, eps=0.0)
        idx <- rann$nn.idx; dists <- rann$nn.dists
        # the item itself might be included in the nns (e.g. ref=Evo+Vj, query=Evo)
      } else {
        rann <- knn(mx[[1]], mx[[2]], k=k+1, eps=0.0)
        idx <- rann$nn.idx; dists <- rann$nn.dists
        rownums <- 1:nrow(idx)
        if (all(rownums==idx[,1])) {
          idx <- idx[,-1]; dists <- dists[,-1]
          # at least one query item not within its nns/not closest nn
        } else {
          dists <- t(sapply(rownums,function(x) {
            if (any(x==idx[x,])) return(dists[x,-which(x==idx[x,])]) else return(dists[x,1:k])
          }))
          idx <- t(sapply(rownums,function(x) {
            if (any(x==idx[x,])) return(idx[x,-which(x==idx[x,])]) else return(idx[x,1:k])
          }))
        }
      }
      # this bit is the original one, simple query from one data set
    } else {
      # find k+1 nn:s: each item is included in the search data, so get one extra nn
      # rann <- nn2(mx, mx, k=k+1, eps=0.0)
      rann <- knn(mx, mx, k=k+1, eps=0.0)
      idx <- rann$nn.idx; dists <- rann$nn.dists
      rownums <- 1:nrow(idx)
      # 2016/09/13 Balazs: improving speed (mx: 1806 rows, k=5, speed improved from 443 sec to 307 sec)
      # coping with query items being within nns
      # all query items selected as their closest nn
      if (all(rownums==idx[,1])) {
        idx <- idx[,-1]; dists <- dists[,-1]
        # at least one query item not within its nns/not closest nn
      } else {
        dists <- t(sapply(rownums,function(x) {
          if (any(x==idx[x,])) return(dists[x,-which(x==idx[x,])]) else return(dists[x,1:k])
        }))
        idx <- t(sapply(rownums,function(x) {
          if (any(x==idx[x,])) return(idx[x,-which(x==idx[x,])]) else return(idx[x,1:k])
        }))
      }
    }
    
    # # get k nn:s so that the query item itself is not within the nn:s
    # rownums <- 1:nrow(rann$nn.idx)
    # ok.idx <- rann$nn.idx != rownums
    # 
    # if (table(ok.idx)[1] == rows) {
    #   # item is within nn:s in each row, i.e. there is one false and 
    #   # k true values in each row of ok.idx: just drop the nn:s of the false values
    #   idx <- t(rann$nn.idx)[t(ok.idx)]
    #   dim(idx) <- c(k, rows)
    #   idx <- t(idx)
    #   dists <- t(rann$nn.dists)[t(ok.idx)]
    #   dim(dists) <- c(k, rows)
    #   dists <- t(dists)
    # } else {
    #   # item is not within nn:s in some lines: those lines would broke
    #   # the method above
    #   idx <- matrix(nrow=rows, ncol=k)
    #   dists <- matrix(nrow=rows, ncol=k)
    #   for (i in 1:rows) {
    #     idx[i,] <- rann$nn.idx[i, ok.idx[i,] ][1:k]
    #     dists[i,] <- rann$nn.dists[i, ok.idx[i,] ][1:k]
    #   }
    # }
    
  }
  
  # calculate continuous estimates
  y <- fknnestcon(idx, dists, my, k=k, g=g, zerodist=zerodist, onlyzeros=onlyzeros)
  
  # bias and rmse
  #   error now so that underestimation gives negative bias
  # dy <- y - my
  if (is.null(dim(mx))) dy <- y-my.query else dy <- y-my
  
  if (n.drop > 0) {
    # remove largest residuals based on sum of absolute weighted residuals
    #   apply column weights
    dy2 <- dy %*% diag(wrmse)
    # row sums of absolute residuals
    rsum <- apply(dy2, 1, FUN=function(x) sum(abs(x)))
    # find largest ones and remove from dy
    rvec <- order(rsum)
    rows.drop <- rvec[(length(rvec) - n.drop + 1):length(rvec)]
    dy <- dy[-(rows.drop), , drop=F]
  }
  
  # else... added 3.5.2018
  if (is.null(dim(mx))) rows <- nrow(mx[[2]]) else rows <- nrow(mx)
  bias <- apply(dy, 2, mean)
  sum.dy2 <- apply(dy^2, 2, sum)
  rmse <- sqrt(sum.dy2/rows)
  
  res <- list(rmse=rmse, bias=bias)
  return(res)
}

ffeatsel.bin <- function(mx, my, my.query=NULL,sca="x", k=5, g=1, zerodist=NULL, onlyzeros=F,
      maxpen=0, nopen=3, wrmse=1, w.bias=1, popSize=100, iters=30, suggestions=NULL,
      elit.pct=NA, elitism=NA, ltr.pct=100, itemgroupid=NULL, group.action=NULL, query.incl.ref=F, ...)
   #  mx:     matrix of x variables of reference data
   #  my:     matrix of continuos variables of reference data (y variables)
   #  sca:    scale x and/or y variables by standard deviation
   #          "both" : scale x and y variables
   #          "x" : scale x variables
   #          "y" : scale y variables
   #          other value : no scaling
   #  k:      number of nearest neighbours used in calculation
   #  g:      weighting parameter of the nearest neighbours, see fknnestcon()
   #  zerodist:   value to add in distance to avoid division by zero in weighting,
   #          see fknnestcon()
   #  onlyzeros:  only replace zero distances with the value of zerodist (> 0),
   #          see fknnestcon()
   #  maxpen: maximum penalty in fitness value when all features are selected;
   #          As a proportion of the mean standard deviation of the y variables;
   #          If <= 0, selection is not affected by the number of the selected
   #          features
   #  nopen:  number of features that can be selected without any penalty
   #  wrmse:  weighting vector for RMSE:s and absolute biases of my columns.
   #          Default is equal weighting
   #  w.bias: weight for bias vector in calculation of fitness value, given that
   #          weight for rmse vector is 1
   #  elit.pct:  elitism parameter of rbga.bin as a percentage of popSize
   #  elitism:   elitism parameter for rbga.bin directly. This is overridden by
   #             elit.pct, if both have values
   #  ltr.pct:   percentage of least trimmed residuals to be used in evaluation
   #  itemgroupid: sometimes nns from the same group of items need to be selected or excluded
   #               it can be done by defining a vector with group ids (length of vector=number of rows in mdata)
   #               nns from same/different groups will be kept
   #  group.action: "include" : include nns only from the same group as the query item
   #                "exclude" : exclude nns from the same group as the query item
   #  query.incl.ref: Boolean, is the query data included in the reference data (e.g. ref=Evo+Vj, query=Evo)
   #  other parameters: passed to rbga.bin(), see rbga.bin help. Parameters
   #          size and evalFunc are not passed because they are defined here
   #
   #  On/off feature selection for knn estimation of continuos variables.
   #  Selection is based on the sum of (weighted) RMSEs and absolute biases of
   #  variable estimates and additionally to a parametrized penalty of selecting
   #  more features.  If there are several y variables, scaling of them here or
   #  before may be needed.
   #
   #  Returns:  list object returned by rbga.bin() of genalg package, in which
   #            values of parameters k and g and standard deviation of x
   #            variables, if x variables are scaled by standard deviation, have
   #            been added
   #
   #  10 January 2011 Juho Pitkänen
{
   evaluate <- function(indices) {
      # number of selected features
      n.sel <- sum(indices)
      if (n.sel > 1) {
         # at least two features selected
         # subdata <- mx[, indices==1]
         ifelse(is.null(dim(mx)),subdata <- lapply(mx,function(mx.x) mx.x[,indices==1,drop=F]),subdata <- mx[,indices==1,drop=F])
         res <- fknncv.fs(subdata, my, my.query=my.query,k=k, g=g, n.drop=n.drop, wrmse=wrmse,
                          zerodist=zerodist, onlyzeros=onlyzeros, itemgroupid=itemgroupid, group.action=group.action,query.incl.ref=query.incl.ref)
         # weighted means of RMSE:s and biases and, possibly, weighting 
         # of bias compared to rmse and penalty of selecting more features
         result <- as.vector(wrmse %*% res$rmse) +
                   w.bias * as.vector(wrmse %*% abs(res$bias)) + penvec[n.sel]
      } else {
         result <- noresult
      }
      result
   }

   library(RANN)
   library(genalg)

   if (!is.na(elit.pct)) {
      if (elit.pct < 0.1) {
         stop("elit.pct too small")
      } else if (elit.pct > 99) {
         stop("elit.pct too large")
      }
      if (!is.na(elitism)) {
         warning("both elit.pct and elitism given - using elit.pct")
      }
      elitism <- floor(elit.pct/100 * popSize)
   }
   
   # checking if my.query is not NULL
   if (is.null(dim(mx))&is.null(my.query)) stop("No my.query defined!")
   # checking if number of columns (features) in reference and test set are equal
   if (is.null(dim(mx))) {
     if (ncol(mx[[1]])!=ncol(mx[[2]])) stop("Number of features is reference and test data set differ!")
     if (nrow(mx[[2]])!=nrow(my.query)) stop("Number of rows in test data set and my.query differ!")
     # if the query data is included in the reference data, it has to be the first in the combined mx table
     # and my has to have the same order of tables (needed to be able to find the query item itself within nns)
     # e.g.: mx <- list(rbind(evo.feat[-1],vj.feat[-1]),evo.feat[-1]); my <- rbind(evo.plotdata[,attrib],vj.plotdata[,attrib])
     #       my.query <- evo.plotdata[,attrib]
     # in case of exclusion this is irrelevant (the item itself cannot be included in the nns)
     if (!is.null(group.action)&&(group.action=="exclude")) query.incl.ref <- F
     if (query.incl.ref) {
       if (!all(mx[[1]][1:nrow(mx[[2]]),]==mx[[2]]) | !all(my[1:nrow(my.query),]==my.query)) stop("Query data has to be the first in mx[[1]] and my!")
     }
   }
   ifelse(is.null(dim(mx)),n.feat <- ncol(mx[[1]]),n.feat <- ncol(mx))
   ifelse(is.null(dim(mx)),rows <- nrow(mx[[1]]),rows <- nrow(mx))
   
   if (!is.null(itemgroupid)) {
     if (is.null(group.action)) stop("Set group.action parameter!")
     # mx is a list (reference and query different)
     if (is.null(dim(mx))) {
       if (all(mapply(function(x,y) nrow(x)!=length(y),mx,itemgroupid))) stop("Length of itemgroupid and number of rows in mx differ! Exiting...")
       if (is.null(group.action)) stop("Set group.action parameter!")
       # resetting row numbers to be able to identify nns after splitting the data based on groupids
       rownames(mx[[1]]) <- 1:rows
       if (group.action=="include") {
         # checking if k value(s) are suitable
         # if k is bigger than or equals the size of the group, k nns cannot be selected
         # execution is stopped and problematic k values are displayed
         if (length(k)==1) {
           if (k>=min(table(itemgroupid[[1]]))) stop(paste0("k (",k,") equals/is bigger than the amount of items in the smallest group (",
                                                            min(table(itemgroupid[[1]])),")! Exiting..."))
         } else {
           k.test <- seq(k[1],k[2])>=min(table(itemgroupid[[1]]))
           if (any(k.test)) stop(paste0("k=(",paste(seq(k[1],k[2])[k.test],collapse=","),") equals/is bigger than the amount of items in the smallest group (",
                                        min(table(itemgroupid[[1]])),")! Exiting..."))
         }
       } else {
         # checking if k value(s) are suitable
         # if k is bigger than or equals the size of any group combination, k nns cannot be selected
         # execution is stopped and problematic k values are displayed
         if (length(k)==1) {
           # checking if number of items in n-1 groups is less than k
           check.amount <- combn(table(itemgroupid[[1]]),length(table(itemgroupid[[1]]))-1,sum)
           if (any(k>check.amount)) stop(paste0("k (",k,") is less than the amount of items in some group combination! Exiting..."))
         } else {
           # checking if number of items in n-1 groups is less than k
           k.check <- seq(k[1],k[2])
           check.amount <- combn(table(itemgroupid[[1]]),length(table(itemgroupid[[1]]))-1,sum)
           k.test <- sapply(k.check,function(x) any(x>check.amount))
           if (any(k.test)) stop(paste0("k=(",paste(seq(k[1],k[2])[k.test],collapse=","),
                                        ") is less than the amount of items in some group combination! Exiting..."))
         }
       }
       # reference and query same
     } else {
       if (length(itemgroupid)!=rows) stop("Length of itemgroupid and number of rows in mx differ! Exiting...")
       # resetting row numbers to be able to identify nns after splitting the data based on groupids
       rownames(mx) <- 1:rows
       if (group.action=="include") {
         # checking if k value(s) are suitable
         # if k is bigger than or equals the size of the group, k nns cannot be selected
         # execution is stopped and problematic k values are displayed
         if (length(k)==1) {
           if (k>=min(table(itemgroupid))) stop(paste0("k (",k,") equals/is bigger than the amount of items in the smallest group (",
                                                       min(table(itemgroupid)),")! Exiting..."))
         } else {
           k.test <- seq(k[1],k[2])>=min(table(itemgroupid))
           if (any(k.test)) stop(paste0("k=(",paste(seq(k[1],k[2])[k.test],collapse=","),") equals/is bigger than the amount of items in the smallest group (",
                                        min(table(itemgroupid)),")! Exiting..."))
         }
       } else {
         # checking if k value(s) are suitable
         # if k is bigger than or equals the size of any group combination, k nns cannot be selected
         # execution is stopped and problematic k values are displayed
         if (length(k)==1) {
           # checking if number of items in n-1 groups is less than k
           check.amount <- combn(table(itemgroupid),length(table(itemgroupid))-1,sum)
           if (any(k>check.amount)) stop(paste0("k (",k,") is less than the amount of items in some group combination! Exiting..."))
         } else {
           # checking if number of items in n-1 groups is less than k
           k.check <- seq(k[1],k[2])
           check.amount <- combn(table(itemgroupid),length(table(itemgroupid))-1,sum)
           k.test <- sapply(k.check,function(x) any(x>check.amount))
           if (any(k.test)) stop(paste0("k=(",paste(seq(k[1],k[2])[k.test],collapse=","),
                                        ") is less than the amount of items in some group combination! Exiting..."))
         }
       }
     }
   }
   
   # y may be given as a vector or data frame
   if (! is.matrix(my) ) {
      # a vector or data frame?
      my <- as.matrix(my)
   }
   if (nrow(my) != rows) stop("Number of rows of mx and my differ")
   
   # scale variables
   x.sd <- NULL
   # if (sca == "x" || sca == "both") {
   #    x.sd <- apply(mx, 2, sd)
   #    mx <- scale(mx, center=F, scale=x.sd)
   # }
   if (sca == "x" || sca == "both") {
     if (is.null(dim(mx))) {
       # checking for columns with sd=0 in both datasets
       x.sd <- apply(mx[[1]], 2, sd)
       if (any(x.sd==0)) stop(paste0("Standard deviation of columns in mx[[1]] ",paste0(names(x.sd)[which(x.sd==0)],collapse=", "),
                                     " is zero. Scaling cannot be done for these columns (and they are meaningless), remove them!"))
       x.sd <- apply(mx[[2]], 2, sd)
       if (any(x.sd==0)) stop(paste0("Standard deviation of columns in mx[[2]] ",paste0(names(x.sd)[which(x.sd==0)],collapse=", "),
                                     " is zero. Scaling cannot be done for these columns (and they are meaningless), remove them!"))
       # scaling is done to the standard deviation of all the data (reference+query)
       x.sd <- apply(rbind(mx[[1]],mx[[2]]),2,sd)
       mx <- lapply(mx,function(mx.x) {
         mx.x <- scale(mx.x, center=F, scale=x.sd); return(mx.x)
       })
     } else {
       x.sd <- apply(mx, 2, sd)
       if (any(x.sd==0)) stop(paste0("Standard deviation of columns ",paste0(names(x.sd)[which(x.sd==0)],collapse=", "),
                                     " is zero. Scaling cannot be done for these columns (and they are meaningless), remove them!"))
       mx <- scale(mx, center=F, scale=x.sd)
     }
   }
   # if (sca == "y" || sca == "both") {
   #   y.sd <- apply(my, 2, sd)
   #   my <- scale(my, center=F, scale=y.sd)
   # }
   if (sca == "y" || sca == "both") {
     if (!is.null(my.query)) {
       # scaling is done to the standard deviation of all the data (reference+query)
       y.sd <- apply(rbind(my,my.query), 2, sd)
       my <- scale(my, center=F, scale=y.sd)
       my.query <- scale(my.query, center=F, scale=y.sd)
     } else {
       y.sd <- apply(my, 2, sd)
       my <- scale(my, center=F, scale=y.sd)
     }
   }
   
   
   # to get nicer plots, mean sd of y variables is used in
   # result variable initialization in evaluate function
   # y.meansd <- mean(apply(my, 2, sd))
   
   # IS THIS CORRECT? --->
   if (!is.null(my.query)) {
     y.meansd <- mean(apply(rbind(my,my.query), 2, sd))
   } else {
     y.meansd <- mean(apply(my, 2, sd))
   }
   # <---
   
   # just scaled by some multiplier
   noresult <- 1.2*y.meansd

   # penalty of selecting more features
   # max penalty when all features selected
   maxpen <- maxpen * y.meansd
   # max number of features selected without any penalty
   n.nopen <- nopen
   # penalty vector for different numbers of features selected
   penvec <- rep(0, n.feat)
   if (maxpen > 0 && n.feat > n.nopen) {
      # scale linearly after the number of features without penalty
      penvec[(n.nopen+1):n.feat] <- 1:(n.feat-n.nopen)/(n.feat-n.nopen) * maxpen
   }
   # scale wrmse to sum of one so that sum of RMSE:s and penalty stay
   # in the same range
   n.var <- ncol(my)
   if (length(wrmse) > 1) {
      # parameter given
      if (length(wrmse) == n.var && sum(wrmse) > 0) {
         wrmse <- wrmse/sum(wrmse)
      } else {
         if (sum(wrmse) > 0) {
            stop("Number of columns in my and number of values in wrmse are different")
         } else {
            stop("Sum of weights in wrmse is <= 0")
         }
      }
   } else {
      # equal weighting
      wrmse <- rep(1/n.var, n.var)
   }
   
   # count the number of largest residuals to drop, if any
   n.drop <- 0
   if (ltr.pct < 100) {
      n.drop <- round((1 - ltr.pct/100) * rows, 0)
      if (n.drop == 0) n.drop <- 1
   }

   ## feature selection
   rbga.res <- rbga.bin(size=n.feat, popSize=popSize, iters=iters,
                        suggestions=suggestions,elitism=elitism, evalFunc=evaluate, ...)
   # add k and g parameters to the result object
   rbga.res$k <- k
   rbga.res$g <- g
   # add scaling of x, if used
   if (! is.null(x.sd) ) {
      rbga.res$x.sd <- x.sd
   }

   # continuous y variables
   rbga.res$disc.yvar <- F
   # to get variable names given for mx and my in the call
   rbga.res$call <- match.call()

   return(rbga.res)
}

ffeatsel.con <- function(mx, my, my.query=NULL, sca="x", k=5, g=1, zerodist=NULL, onlyzeros=F,
      wmin=0.3, maxpen=0, nopen=3, wrmse=1, w.bias=1, popSize=100, iters=30,
      elit.pct=NA, elitism=NA, ltr.pct=100, sugg.base=NULL, suggestions=NULL, itemgroupid=NULL, group.action=NULL, query.incl.ref=F, ...)
   #  mx:     matrix of x variables of reference data
   #  my:     matrix of continuos variables of reference data (y variables)
   #  sca:    scale x and/or y variables by standard deviation
   #          "both" : scale x and y variables
   #          "x" : scale x variables
   #          "y" : scale y variables
   #          other value : no scaling
   #  k:      number of nearest neighbours used in calculation; scalar or vector
   #          c(kmin, kmax). If kmin and kmax values are given, value of k is
   #          included in the search along with feature weights
   #  g:      weighting parameter of the nearest neighbours; scalar or vector
   #          c(gmin, gmax). If gmin and gmax values are given, value of g is
   #          included in the search along with feature weights
   #  zerodist:   value to add in distance to avoid division by zero in weighting,
   #          see fknnestcon()
   #  onlyzeros:  only replace zero distances with the value of zerodist (> 0),
   #          see fknnestcon()
   #  wmin:   minimum value used as a feature weight. If weight is less, feature
   #          is not selected.
   #  maxpen: maximum penalty in fitness value when all features are selected;
   #          As a proportion of the mean standard deviation of the y variables;
   #          If <= 0, selection is not affected by the number of the selected
   #          features
   #  nopen:  number of features that can be selected without any penalty
   #  wrmse:  weighting vector for RMSE:s and absolute biases of my columns.
   #          Default is equal weighting
   #  w.bias: weight for bias vector in calulation of fitness value, given that
   #          weight for rmse vector is 1
   #  ltr.pct:   percentage of least trimmed residuals to be used in evaluation
   #  elit.pct:  elitism parameter of rbga as a percentage of popSize
   #  elitism:   elitism parameter for rbga directly. This is overridden by
   #             elit.pct, if both have values
   #  sugg.base: result of an earlier run of ffeatsel.bin() or ffeatsel.con()
   #             to start from it's population using suggestions parameter
   #             of rbga()
   #  suggestions: suggestions parameter for rbga directly. This is overridden
   #               by sugg.base, if both have values
   #  itemgroupid: sometimes nns from the same group of items need to be selected or excluded
   #               it can be done by defining a vector with group ids (length of vector=number of rows in mdata)
   #               nns from same/different groups will be kept
   #  group.action: "include" : include nns only from the same group as the query item
   #                "exclude" : exclude nns from the same group as the query item
   #  query.incl.ref: Boolean, is the query data included in the reference data (e.g. ref=Evo+Vj, query=Evo)
   #  other parameters: passed to rbga(), see rbga help. Parameters stringMin,
   #          stringMax and evalFunc are not passed because they are defined here
   #
   #  Feature selection and feature weight search for knn estimation of
   #  continuos variables. Additionally k and g parameter values can be searched
   #  between given limits. Selection is based on the sum of (weighted) RMSEs
   #  and absolute biases of variable estimates and additionally to a
   #  parametrized penalty of selecting more features. In feature weight
   #  search, genetic algorithm is set to produce weights in range [0, 1] but
   #  before rmse evaluation a feature having weight under wmin is considered to
   #  be not selected.
   #
   #  Returns:  list object returned by rbga() of genalg package, in which
   #            values of parameters k, g, wmin and standard deviation of x
   #            variables (mx columns), if x variables are scaled by standard
   #            deviation, have been added
   #
   #  12 January 2011 Juho Pitkänen
{
   eval.con <- function(wei) {
      ## optimize feature weights
      # feature is not selected if weight is less than wmin
      indices <- wei >= wmin
      # number of selected features
      n.sel <- sum(indices)
      if (n.sel > 1) {
        # at least two features selected
        # subdata <- mx[, indices==1]
        # # weights of selected features
        wei.sel <- wei[indices]
        # # apply weights
        # subdata <- t(wei.sel*t(subdata))
        if (is.null(dim(mx))) {
          subdata <- lapply(mx,function(mx.x) {
            mx.x <- mx.x[,indices]
            # apply weights
            mx.x <- t(wei.sel*t(mx.x))
          })
        } else {
          subdata <- mx[,indices]
          # apply weights
          subdata <- t(wei.sel*t(subdata))
        }
        res <- fknncv.fs(subdata, my, my.query=my.query, k=k, g=g, n.drop=n.drop, wrmse=wrmse,
                         zerodist=zerodist, onlyzeros=onlyzeros, itemgroupid=itemgroupid, group.action=group.action,query.incl.ref=query.incl.ref)
        # weighted means of RMSE:s and biases and, possibly, weighting 
        # of bias compared to rmse and penalty of selecting more features
        result <- as.vector(wrmse %*% res$rmse) +
          w.bias * as.vector(wrmse %*% abs(res$bias)) + penvec[n.sel]
      } else {
         result <- noresult
      }
      result
   }

   eval.con.kg <- function(wei) {
      ## optimize feature weights and k and/or g
      # local variables from values in upper function
      k <- k
      g <- g
      # k and/or g are first values in wei vector if they are optimized
      if (k.opt) {
         # k is integer
         k <- round(wei[1])
         wei <- wei[-1]
      }
      if (g.opt) {
         # g in steps of 0.1
         g <- round(wei[1], 1)
         wei <- wei[-1]
      }
      # feature is not selected if weight is less than wmin
      indices <- wei >= wmin
      # number of selected features
      n.sel <- sum(indices)

      if (n.sel > 1) {
         # at least two features selected
         # subdata <- mx[, indices==1]
         # weights of selected features
         wei.sel <- wei[indices]
         # # apply weights
         # subdata <- t(wei.sel*t(subdata))
         if (is.null(dim(mx))) {
           subdata <- lapply(mx,function(mx.x) {
             mx.x <- mx.x[,indices]
             # apply weights
             mx.x <- t(wei.sel*t(mx.x))
           })
         } else {
           subdata <- mx[,indices]
           # apply weights
           subdata <- t(wei.sel*t(subdata))
         }
         res <- fknncv.fs(subdata, my, my.query=my.query,k=k, g=g, n.drop=n.drop, wrmse=wrmse,
                          zerodist=zerodist, onlyzeros=onlyzeros, itemgroupid=itemgroupid, group.action=group.action,query.incl.ref=query.incl.ref)
         # weighted means of RMSE:s and biases and, possibly, weighting 
         # of bias compared to rmse and penalty of selecting more features
         result <- as.vector(wrmse %*% res$rmse) +
                   w.bias * as.vector(wrmse %*% abs(res$bias)) + penvec[n.sel]
      } else {
         result <- noresult
      }
      result
   }

   library(RANN)
   library(genalg)

   if (!is.na(elit.pct)) {
      if (elit.pct < 0.1) {
         stop("elit.pct too small")
      } else if (elit.pct > 99) {
         stop("elit.pct too large")
      }
      if (!is.na(elitism)) {
         warning("both elit.pct and elitism given - using elit.pct")
      }
      elitism <- floor(elit.pct/100 * popSize)
   }

   # checking if my.query is not NULL
   if (is.null(dim(mx))&is.null(my.query)) stop("No my.query defined!")
   # checking if number of columns (features) in reference and test set are equal
   if (is.null(dim(mx))) {
     if (ncol(mx[[1]])!=ncol(mx[[2]])) stop("Number of features is reference and test data set differ!")
     if (nrow(mx[[2]])!=nrow(my.query)) stop("Number of rows in test data set and my.query differ!")
     # if the query data is included in the reference data, it has to be the first in the combined mx table
     # and my has to have the same order of tables (needed to be able to find the query item itself within nns)
     # e.g.: mx <- list(rbind(evo.feat[-1],vj.feat[-1]),evo.feat[-1]); my <- rbind(evo.plotdata[,attrib],vj.plotdata[,attrib])
     #       my.query <- evo.plotdata[,attrib]
     # in case of exclusion this is irrelevant (the item itself cannot be included in the nns)
     if (!is.null(group.action)&&(group.action=="exclude")) query.incl.ref <- F
     if (query.incl.ref) {
       if (!all(mx[[1]][1:nrow(mx[[2]]),]==mx[[2]]) | !all(my[1:nrow(my.query),]==my.query)) stop("Query data has to be the first in mx[[1]] and my!")
     }
   }

   # number of features
   # n.feat <- ncol(mx)
   ifelse(is.null(dim(mx)),n.feat <- ncol(mx[[1]]),n.feat <- ncol(mx))
   # number of rows
   # rows <- nrow(mx)
   ifelse(is.null(dim(mx)),rows <- nrow(mx[[1]]),rows <- nrow(mx))
   
   if (length(k) > 2) {
     warning("k should be a scalar or a vector of two numbers.")
     warning("Using first value in the vector")
     k <- round(k[1])
   }
   if (length(g) > 2) {
     warning("g should be a scalar or a vector of two numbers.")
     warning("Using first value in the vector")
     g <- g[1]
   }
   
   if (!is.null(itemgroupid)) {
     if (is.null(group.action)) stop("Set group.action parameter!")
     # mx is a list (reference and query different)
     if (is.null(dim(mx))) {
       if (all(mapply(function(x,y) nrow(x)!=length(y),mx,itemgroupid))) stop("Length of itemgroupid and number of rows in mx differ! Exiting...")
       # resetting row numbers to be able to identify nns after splitting the data based on groupids
       rownames(mx[[1]]) <- 1:rows
       if (group.action=="include") {
         # checking if k value(s) are suitable
         # if k is bigger than or equals the size of the group, k nns cannot be selected
         # execution is stopped and problematic k values are displayed
         if (length(k)==1) {
           if (k>=min(table(itemgroupid[[1]]))) stop(paste0("k (",k,") equals/is bigger than the amount of items in the smallest group (",
                                                       min(table(itemgroupid[[1]])),")! Exiting..."))
         } else {
           k.test <- seq(k[1],k[2])>=min(table(itemgroupid[[1]]))
           if (any(k.test)) stop(paste0("k=(",paste(seq(k[1],k[2])[k.test],collapse=","),") equals/is bigger than the amount of items in the smallest group (",
                                        min(table(itemgroupid[[1]])),")! Exiting..."))
         }
       } else {
         # checking if k value(s) are suitable
         # if k is bigger than or equals the size of any group combination, k nns cannot be selected
         # execution is stopped and problematic k values are displayed
         if (length(k)==1) {
           # checking if number of items in n-1 groups is less than k
           check.amount <- combn(table(itemgroupid[[1]]),length(table(itemgroupid[[1]]))-1,sum)
           if (any(k>check.amount)) stop(paste0("k (",k,") is less than the amount of items in some group combination! Exiting..."))
         } else {
           # checking if number of items in n-1 groups is less than k
           k.check <- seq(k[1],k[2])
           check.amount <- combn(table(itemgroupid[[1]]),length(table(itemgroupid[[1]]))-1,sum)
           k.test <- sapply(k.check,function(x) any(x>check.amount))
           if (any(k.test)) stop(paste0("k=(",paste(seq(k[1],k[2])[k.test],collapse=","),
                                        ") is less than the amount of items in some group combination! Exiting..."))
         }
       }
     # reference and query same
     } else {
       if (length(itemgroupid)!=rows) stop("Length of itemgroupid and number of rows in mx differ! Exiting...")
       # resetting row numbers to be able to identify nns after splitting the data based on groupids
       rownames(mx) <- 1:rows
       if (group.action=="include") {
         # checking if k value(s) are suitable
         # if k is bigger than or equals the size of the group, k nns cannot be selected
         # execution is stopped and problematic k values are displayed
         if (length(k)==1) {
           if (k>=min(table(itemgroupid))) stop(paste0("k (",k,") equals/is bigger than the amount of items in the smallest group (",
                                                       min(table(itemgroupid)),")! Exiting..."))
         } else {
           k.test <- seq(k[1],k[2])>=min(table(itemgroupid))
           if (any(k.test)) stop(paste0("k=(",paste(seq(k[1],k[2])[k.test],collapse=","),") equals/is bigger than the amount of items in the smallest group (",
                                        min(table(itemgroupid)),")! Exiting..."))
         }
       } else {
         # checking if k value(s) are suitable
         # if k is bigger than or equals the size of any group combination, k nns cannot be selected
         # execution is stopped and problematic k values are displayed
         if (length(k)==1) {
           # checking if number of items in n-1 groups is less than k
           check.amount <- combn(table(itemgroupid),length(table(itemgroupid))-1,sum)
           if (any(k>check.amount)) stop(paste0("k (",k,") is less than the amount of items in some group combination! Exiting..."))
         } else {
           # checking if number of items in n-1 groups is less than k
           k.check <- seq(k[1],k[2])
           check.amount <- combn(table(itemgroupid),length(table(itemgroupid))-1,sum)
           k.test <- sapply(k.check,function(x) any(x>check.amount))
           if (any(k.test)) stop(paste0("k=(",paste(seq(k[1],k[2])[k.test],collapse=","),
                                        ") is less than the amount of items in some group combination! Exiting..."))
         }
       }
     }
   }
   
   # y may be given as a vector or data frame
   if (! is.matrix(my) ) {
      # a vector or data frame?
      my <- as.matrix(my)
   }
   if (nrow(my) != rows) stop("Number of rows of mx and my differ")
   if (is.null(dim(mx))) {
     if (ncol(my)!=ncol(my.query)) stop("Number of columns in my and my.query differ")
   }

   # scale variables
   x.sd <- NULL
   # if (sca == "x" || sca == "both") {
   #    x.sd <- apply(mx, 2, sd)
   #    mx <- scale(mx, center=F, scale=x.sd)
   # }
   if (sca == "x" || sca == "both") {
     if (is.null(dim(mx))) {
       # checking for columns with sd=0 in both datasets
       x.sd <- apply(mx[[1]], 2, sd)
       if (any(x.sd==0)) stop(paste0("Standard deviation of columns in mx[[1]] ",paste0(names(x.sd)[which(x.sd==0)],collapse=", "),
                                     " is zero. Scaling cannot be done for these columns (and they are meaningless), remove them!"))
       x.sd <- apply(mx[[2]], 2, sd)
       if (any(x.sd==0)) stop(paste0("Standard deviation of columns in mx[[2]] ",paste0(names(x.sd)[which(x.sd==0)],collapse=", "),
                                     " is zero. Scaling cannot be done for these columns (and they are meaningless), remove them!"))
       # scaling is done to the standard deviation of all the data (reference+query)
       x.sd <- apply(rbind(mx[[1]],mx[[2]]),2,sd)
       mx <- lapply(mx,function(mx.x) {
         mx.x <- scale(mx.x, center=F, scale=x.sd); return(mx.x)
       })
     } else {
       x.sd <- apply(mx, 2, sd)
       if (any(x.sd==0)) stop(paste0("Standard deviation of columns ",paste0(names(x.sd)[which(x.sd==0)],collapse=", "),
                                     " is zero. Scaling cannot be done for these columns (and they are meaningless), remove them!"))
       mx <- scale(mx, center=F, scale=x.sd)
     }
   }
   # if (sca == "y" || sca == "both") {
   #   y.sd <- apply(my, 2, sd)
   #   my <- scale(my, center=F, scale=y.sd)
   # }
   if (sca == "y" || sca == "both") {
     if (!is.null(my.query)) {
       # scaling is done to the standard deviation of all the data (reference+query)
       y.sd <- apply(rbind(my,my.query), 2, sd)
       my <- scale(my, center=F, scale=y.sd)
       my.query <- scale(my.query, center=F, scale=y.sd)
     } else {
       y.sd <- apply(my, 2, sd)
       my <- scale(my, center=F, scale=y.sd)
     }
   }
   # to get nicer plots, mean sd of y variables is used in
   # result variable initialization in evaluate function
   # y.meansd <- mean(apply(my, 2, sd))
   
   # IS THIS CORRECT? --->
   if (!is.null(my.query)) {
     y.meansd <- mean(apply(rbind(my,my.query), 2, sd))
   } else {
     y.meansd <- mean(apply(my, 2, sd))
   }
   # <---

   # just scaled by some multiplier
   noresult <- 1.2*y.meansd

   # penalty of selecting more features
   # max penalty when all features selected
   maxpen <- maxpen * y.meansd
   # max number of features selected without any penalty
   n.nopen <- nopen
   # penalty vector for different numbers of features selected
   penvec <- rep(0, n.feat)
   if (maxpen > 0 && n.feat > n.nopen) {
      # scale linearly after the number of features without penalty
      penvec[(n.nopen+1):n.feat] <- 1:(n.feat-n.nopen)/(n.feat-n.nopen) * maxpen
   }
   # scale wrmse to sum of one so that sum of RMSE:s and penalty stay
   # in the same range
   n.var <- ncol(my)
   if (length(wrmse) > 1) {
      # parameter given
      if (length(wrmse) == n.var && sum(wrmse) > 0) {
         wrmse <- wrmse/sum(wrmse)
      } else {
         if (sum(wrmse) > 0) {
            stop("Number of columns in my and number of values in wrmse are different")
         } else {
            stop("Sum of weights in wrmse is <= 0")
         }
      }
   } else {
      # equal weighting
      wrmse <- rep(1/n.var, n.var)
   }

   # count the number of largest residuals to drop, if any
   n.drop <- 0
   if (ltr.pct < 100) {
      n.drop <- round((1 - ltr.pct/100) * rows, 0)
      if (n.drop == 0) n.drop <- 1
   }

   # vectors of minimum and maximum weights for rbga(): range [0, 1] but
   # values under wmin are considered as 0
   minvec <- rep(0, n.feat)
   maxvec <- rep(1, n.feat)

   # should k and g be optimized
   k.opt <- length(k)==2
   g.opt <- length(g)==2
   if (g.opt) {
      # g limits to vectors
      # in evaluation, values used now in steps of 0.1
      halfbin <- 0.049
      gmin <- g[1] - halfbin
      gmax <- g[2] + halfbin
      minvec <- c(gmin, minvec)
      maxvec <- c(gmax, maxvec)
   } else {
      # for fsuggest.prevrun
      gmin <- -1
      gmax <- -1
   }
   if (k.opt) {
      # k limits to vectors
      halfbin <- 0.49
      kmin <- k[1] - halfbin
      kmax <- k[2] + halfbin
      minvec <- c(kmin, minvec)
      maxvec <- c(kmax, maxvec)
   } else {
      # for fsuggest.prevrun
      kmin <- -1
      kmax <- -1
   }

   # is there suggestions based on earlier run of rbga.bin or rbga
   if (!is.null(sugg.base)) {
      popul <- fsuggest.prevrun(sugg.base, n.feat, k.opt, g.opt, kmin, kmax,
            gmin, gmax, popSize)
      if (is.null(popul)) {
         return(NULL)
      }
      suggestions <- popul
   }

   ## feature selection
   if (k.opt || g.opt) {
      rbga.res <- rbga(stringMin=minvec, stringMax=maxvec, popSize=popSize,
            iters=iters, elitism=elitism, evalFunc=eval.con.kg, 
            suggestions=suggestions, ...)
   } else {
      rbga.res <- rbga(stringMin=minvec, stringMax=maxvec, popSize=popSize,
            iters=iters, elitism=elitism, evalFunc=eval.con,
            suggestions=suggestions, ...)
   }
   rbga.res$k <- k
   rbga.res$g <- g
   rbga.res$wmin <- wmin
   # add scaling of x, if used
   if (! is.null(x.sd) ) {
      rbga.res$x.sd <- x.sd
   }

   # continuous y variables
   rbga.res$disc.yvar <- F
   # to get variable names given for mx and my in the call
   rbga.res$call <- match.call()

   return(rbga.res)
}

ffeatsel.results <- function(mx, my, my.query=NULL, rbga.res, itemgroupid=NULL, group.action=NULL, query.incl.ref=F, ...)
   #  A wrapper for calling either ffeatsel.bin.results() or
   #  ffeatsel.con.results() according to the type of rbga.res
{
   if (class(rbga.res) != "rbga") {
      stop("rbga.res is not an output of either ffeatsel.bin() or ffeatsel.con()") 
   }

   if (rbga.res$type == "binary chromosome") {
      res <- ffeatsel.bin.results(mx, my, my.query=my.query, rbga.res, itemgroupid=itemgroupid, group.action=group.action, query.incl.ref=query.incl.ref, ...)
   } else if (rbga.res$type == "floats chromosome") {
      res <- ffeatsel.con.results(mx, my, my.query=my.query, rbga.res, itemgroupid=itemgroupid, group.action=group.action, query.incl.ref=query.incl.ref, ...)
   } else {
      stop("rbga.res is not an output of either ffeatsel.bin() or ffeatsel.con()") 
   }
   return(res)
}

ffeatsel.bin.results <- function(mx, my, my.query=NULL, rbga.res, sca="x",sel.best=1L,
      k.new=-1, g.new=-1, out="rmse", itemgroupid=NULL, group.action=NULL, query.incl.ref=F)
   #  rbga.res: result of ffeatsel.bin(), giving the selected features from mx
   #            and k and g paramaters
   #  sca:    scale x and/or y variables by standard deviation. By default,
   #          y variables are not scaled here to get RMSEs in original units
   #  k.new:  by default, use k parameter value of rbga.res, but k can be changed
   #          with k.new
   #  g.new:  by default, use g parameter value of rbga.res, but g can be changed
   #          with g.new
   #  out:    "est" : return estimates
   #          "rmse" : return a list (see Returns below)
   #          other value : like "rmse"
   #  sel.best: select best feature combination by providing the index number of the vector
   #            which(rbga.res$evaluations==min(rbga.res$evaluations))
   #            there might be more than one combination with the best evaulation value
   #
   #  Calculates knn crossvalidation results using the best found feature set
   #  (one of those, if many) in rbga.res
   #
   #  Returns:  if out="est", returns estimates,
   #            otherwise returns a list,
   #            consisting of
   #            - rmse, bias, rmse-% and bias-% vectors for columns of my
   #            - a vector of feature weights (0/1)
   #            - a vector of the column names of the features in use
   #            - values of parameters k and g
   #            - a vector of standard deviations of x variables (i.e. features),
   #              if x variables were scaled here
   #            - a vector of standard deviations of x variables in use, if x
   #              variables were scaled here
{
   ## find out selected features and calculate results
   # Note that there may be another feature combinations that produce the
   # same fitness: rbga.res should be checked also manually

   if (class(rbga.res) != "rbga") {
      stop("rbga.res is not an object returned by ffeatsel.bin()")
   }
   if (rbga.res$type != "binary chromosome") {
      stop("rbga.res is not an object returned by ffeatsel.bin()")
   }

  # checking if my.query is not NULL
  if (is.null(dim(mx))&is.null(my.query)) stop("No my.query defined!")
  # checking if number of columns (features) in reference and test set are equal
  if (is.null(dim(mx))) {
    if (ncol(mx[[1]])!=ncol(mx[[2]])) stop("Number of features is reference and test data set differ!")
    if (nrow(mx[[2]])!=nrow(my.query)) stop("Number of rows in test data set and my.query differ!")
  }
  
  if (!is.null(itemgroupid)) {
    if (is.null(group.action)) stop("Set group.action parameter!")
    if (is.null(dim(mx))) {
      if (all(mapply(function(x,y) nrow(x)!=length(y),mx,itemgroupid))) stop("Length of itemgroupid and number of rows in mx differ! Exiting...")
    } else {
      rows <- nrow(mx)
      if (length(itemgroupid)!=rows) stop("Length of itemgroupid and number of rows in mx differ! Exiting...")
      # # resetting row numbers to be able to identify nns after splitting the data based on groupids
      # rownames(mx) <- 1:rows
    }
  }

   # parameters used in the optimization
   k <- rbga.res$k
   g <- rbga.res$g

   # are there replace values for k and g
   if (k.new > 0) {
      k <- k.new
   }
   if (g.new > -0.9) {
      g <- g.new
   }

   # observations can be different than in feature selection but scaling
   # of the x varibles should be the same
   x.sd <- NULL
   if (sca == "x" || sca == "both") {
     x.sd <- rbga.res$x.sd
     if (is.null(dim(mx))) {
       if (is.null(x.sd)) {
         warning("x1 and x2 was scaled now even it was not scaled in feature selection result")
         x.sd <- apply(rbind(mx[[1]],mx[[2]]),2,sd)
       } else {
         if ( !identical(names(x.sd), colnames(mx[[1]])) ) {
           # a subset of mx1 may have been used in feature selection
           if (all(names(x.sd) %in% colnames(mx[[1]]))) {
             mx[[1]] <- subset(mx[[1]], select=names(x.sd))
           } else {
             stop("x1 variable matrix does not have all feature selection columns")
           }
         }
         if ( !identical(names(x.sd), colnames(mx[[2]])) ) {
           # a subset of mx2 may have been used in feature selection
           if (all(names(x.sd) %in% colnames(mx[[2]]))) {
             mx[[2]] <- subset(mx[[2]], select=names(x.sd))
           } else {
             stop("x2 variable matrix does not have all feature selection columns")
           }
         }
       }
       mx[[1]] <- scale(mx[[1]], center=F, scale=x.sd)
       mx[[2]] <- scale(mx[[2]], center=F, scale=x.sd)
     } else {
       # x.sd <- rbga.res$x.sd
       if (is.null(x.sd)) {
         warning("x was scaled now even it was not scaled in feature selection result")
         x.sd <- apply(mx, 2, sd)
       } else {
         if ( !identical(names(x.sd), colnames(mx)) ) {
           # a subset of mx may have been used in feature selection
           if ( all(names(x.sd) %in% colnames(mx)) ) {
             mx <- subset(mx, select=names(x.sd))
           } else {
             stop("x variable matrix does not have all feature selection columns")
           }
         }
       }
       mx <- scale(mx, center=F, scale=x.sd)
     }
     if (sca == "x") {
       sca <- "none"
     } else {
       sca <- "y"
     }
   }

   indices <- rbga.res$population[which(rbga.res$evaluations==min(rbga.res$evaluations))[sel.best], ]
   # subdata <- mx[, indices==1]
   
   if (is.null(dim(mx))) {
     subdata <- lapply(mx,function(mx.x) {
       mx.x <- mx.x[,indices==1,drop=F]
     })
   } else {
     subdata <- mx[,indices==1,drop=F]
   }
   
   if (out == "est") {
      # return estimates
      res <- fknncv(subdata, my, my.query=my.query,k=k, g=g, sca=sca, out="est", itemgroupid=itemgroupid, group.action=group.action, query.incl.ref=query.incl.ref)
      if (is.data.frame(my)) {
         res <- as.data.frame(res)
      }
      return(res)
   } else {
      # calculate rmse and bias
      res <- fknncv(subdata, my, my.query=my.query, k=k, g=g, sca=sca, out="rmse", itemgroupid=itemgroupid, group.action=group.action, query.incl.ref=query.incl.ref)
      # add feature indices and parameters used
      res$weights <- indices
      # res$in.use <- colnames(mx)[indices==1]
      ifelse(is.null(dim(mx)),res$in.use <- colnames(mx[[1]])[indices==1],res$in.use <- colnames(mx)[indices==1])
      res$k <- k
      res$g <- g
      if (! is.null(x.sd) ) {
        res$x.sd <- x.sd
        res$in.use.sd <- x.sd[indices==1]
      }
      return(res)
   }
}

ffeatsel.con.results <- function(mx, my, my.query=NULL, rbga.res, sca="x",
      k.new=-1, g.new=-1, out="rmse", itemgroupid=NULL, group.action=NULL, query.incl.ref=F)
   #  rbga.res: result of ffeatsel.con(), giving the feature weights for mx
   #            and k, g and wmin paramaters
   #  sca:    scale x and/or y variables by standard deviation. By default,
   #          y variables are not scaled here to get RMSEs in original units
   #  k.new:  by default, use k parameter value of rbga.res, but k can be changed
   #          with k.new
   #  g.new:  by default, use g parameter value of rbga.res, but g can be changed
   #          with g.new
   #  out:    "est" : return estimates
   #          "rmse" : return a list (see Returns below)
   #          other value : like "rmse"
   #
   #  Calculates knn crossvalidation results using the best found feature set
   #  (one of those, if many) in rbga.res
   #
   #  Returns:  if out="est", returns estimates,
   #            otherwise returns a list,
   #            consisting of
   #            - rmse, bias, rmse-% and bias-% vectors for columns of my
   #            - a vector of feature weights
   #            - a vector of the column names of the features in use
   #            - a vector of feature weights for the features in use
   #            - values of parameters k, g and wmin
   #            - logical variables showing were k and g optimized
   #            - a vector of standard deviations of x variables (i.e. features),
   #              if x variables were scaled here
   #            - a vector of standard deviations of x variables in use, if x
   #              variables were scaled here
{
  ## find out selected features and calculate results
  # Note that there may be another feature combinations that produce the
  # same fitness: rbga.res should be checked also manually

  if (class(rbga.res) != "rbga") {
    stop("rbga.res is not an object returned by ffeatsel.con()")
  }
  if (rbga.res$type != "floats chromosome") {
    stop("rbga.res is not an object returned by ffeatsel.con()")
  }

  # checking if my.query is not NULL
  if (is.null(dim(mx))&is.null(my.query)) stop("No my.query defined!")
  # checking if number of columns (features) in reference and test set are equal
  if (is.null(dim(mx))) {
    if (ncol(mx[[1]])!=ncol(mx[[2]])) stop("Number of features is reference and test data set differ!")
    if (nrow(mx[[2]])!=nrow(my.query)) stop("Number of rows in test data set and my.query differ!")
  }
  # # setting itemgroupid to NULL
  # if (is.null(dim(mx))) itemgroupid <- NULL
  
  if (!is.null(itemgroupid)) {
    if (is.null(group.action)) stop("Set group.action parameter!")
    if (is.null(dim(mx))) {
      if (all(mapply(function(x,y) nrow(x)!=length(y),mx,itemgroupid))) stop("Length of itemgroupid and number of rows in mx differ! Exiting...")
    } else {
      rows <- nrow(mx)
      if (length(itemgroupid)!=rows) stop("Length of itemgroupid and number of rows in mx differ! Exiting...")
      # # resetting row numbers to be able to identify nns after splitting the data based on groupids
      # rownames(mx) <- 1:rows
    }
  }
  
   # parameters used in the optimization
   k <- rbga.res$k
   g <- rbga.res$g
   wmin <- rbga.res$wmin
   # were k and g optimized
   k.opt <- length(k)==2
   g.opt <- length(g)==2

   # have to scale x variables before applying weights
   x.sd <- NULL
   if (sca == "x" || sca == "both") {
     x.sd <- rbga.res$x.sd
     if (is.null(dim(mx))) {
       if (is.null(x.sd)) {
         warning("x1 and x2 was scaled now even it was not scaled in feature selection result")
         x.sd <- apply(rbind(mx[[1]],mx[[2]]),2,sd)
       } else {
         if ( !identical(names(x.sd), colnames(mx[[1]])) ) {
           # a subset of mx1 may have been used in feature selection
           if (all(names(x.sd) %in% colnames(mx[[1]]))) {
             mx[[1]] <- subset(mx[[1]], select=names(x.sd))
           } else {
             stop("x1 variable matrix does not have all feature selection columns")
           }
         }
         if ( !identical(names(x.sd), colnames(mx[[2]])) ) {
           # a subset of mx2 may have been used in feature selection
           if ( all(names(x.sd) %in% colnames(mx[[2]])) ) {
             mx[[2]] <- subset(mx[[2]], select=names(x.sd))
           } else {
             stop("x2 variable matrix does not have all feature selection columns")
           }
         }
       }
       mx[[1]] <- scale(mx[[1]], center=F, scale=x.sd)
       mx[[2]] <- scale(mx[[2]], center=F, scale=x.sd)
     } else {
       # x.sd <- rbga.res$x.sd
       if (is.null(x.sd)) {
         warning("x was scaled now even it was not scaled in feature selection result")
         x.sd <- apply(mx, 2, sd)
       } else {
         if ( !identical(names(x.sd), colnames(mx)) ) {
           # a subset of mx may have been used in feature selection
           if ( all(names(x.sd) %in% colnames(mx)) ) {
             mx <- subset(mx, select=names(x.sd))
           } else {
             stop("x variable matrix does not have all feature selection columns")
           }
         }
       }
       mx <- scale(mx, center=F, scale=x.sd)
     }
     if (sca == "x") {
       sca <- "none"
     } else {
       sca <- "y"
     }
   }

   # so it is better to scale y here as well if needed
   # if (sca == "y") {
   #    y.sd <- apply(my, 2, sd)
   #    my <- scale(my, center=F, scale=y.sd)
   # }
   if (sca == "y") {
     if (!is.null(my.query)) {
       # scaling is done to the standard deviation of all the data (reference+query)
       y.sd <- apply(rbind(my,my.query), 2, sd)
       my <- scale(my, center=F, scale=y.sd)
       my.query <- scale(my.query, center=F, scale=y.sd)
     } else {
       y.sd <- apply(my, 2, sd)
       my <- scale(my, center=F, scale=y.sd)
     }
   }
   
   # weights of the first feature set giving the best evaluation
   wei <- rbga.res$population[which(rbga.res$evaluations==min(rbga.res$evaluations))[1], ]
   # k and/or g are first values in wei vector if they were optimized
   if (k.opt) {
      # k is integer
      k <- round(wei[1])
      wei <- wei[-1]
   }
   if (g.opt) {
      # g in steps of 0.1
      g <- round(wei[1], 1)
      wei <- wei[-1]
   }
   # are there replace values for k and g. Also optimized values are replaced.
   if (k.new > 0) {
      k <- k.new
      cat("- k changed by parameter to", k, "\n")
   }
   if (g.new > -0.9) {
      g <- g.new
      cat("- g changed by parameter to", g, "\n")
   }
   if (k.new > 0 || g.new > -0.9) {
      cat("\n")
   }

   # subset of features and their weights
   indices <- wei >= wmin
   # subdata <- mx[, indices==1]
   wei.sel <- wei[indices]
   # apply weights
   # subdata <- t(wei.sel*t(subdata))
   if (is.null(dim(mx))) {
     subdata <- lapply(mx,function(mx.x) {
       mx.x <- mx.x[,indices==1,drop=F]
       # apply weights
       mx.x <- t(wei.sel*t(mx.x))
       })
   } else {
     subdata <- mx[,indices==1,drop=F]
     subdata <- t(wei.sel*t(subdata))
   }
   
   if (out == "est") {
      # return estimates
      res <- fknncv(subdata, my, my.query=my.query, k=k, g=g, sca="none", out="est", itemgroupid=itemgroupid, group.action=group.action, query.incl.ref=query.incl.ref)
      if (is.data.frame(my)) {
         res <- as.data.frame(res)
      }
      return(res)
   } else {
      # calculate rmse and bias
      res <- fknncv(subdata, my, my.query=my.query, k=k, g=g, sca="none", out="rmse", itemgroupid=itemgroupid, group.action=group.action, query.incl.ref=query.incl.ref)
      # add weights and parameters used
      wei[wei < wmin] <- 0
      res$weights <- wei
      # res$in.use <- colnames(mx)[indices==1]
      ifelse(is.null(dim(mx)),res$in.use <- colnames(mx[[1]])[indices==1],res$in.use <- colnames(mx)[indices==1])
      res$in.use.weights <- wei[indices==1]
      res$k <- k
      res$g <- g
      res$wmin <- wmin
      if (k.opt) {
         res$k.opt <- T
      } else {
         res$k.opt <- F
      }
      if (g.opt) {
         res$g.opt <- T
      } else {
         res$g.opt <- F
      }
      if (! is.null(x.sd) ) {
        res$x.sd <- x.sd
        res$in.use.sd <- x.sd[indices==1]
      }
      return(res)
   }
}

fsuggest.prevrun <- function(sugg.base, n.feat, k.opt, g.opt, kmin, kmax,
      gmin, gmax, popSize)
   #  Internal function to create initial population in ffeatsel.con() from
   #  the result of an earlier run of either ffeatsel.bin or ffeatsel.con.
   #  Only unique population lines are kept. If k and/or g were optimized
   #  and are to be optimized now, they are initialized with the existing
   #  optimized values, if the values are in the current ranges for k and g.
   #  Otherwise if k and g are to be optimized now, they are initialized with
   #  average values.
{
   if (class(sugg.base) != "rbga") {
      stop("sugg.base is not an object returned by rbga()")
   }

   popul <- sugg.base$population
   # order population by evaluations and keep unique rows
   popul <- popul[order(sugg.base$evaluations), ]
   popul <- unique(popul)

   # parameters used in the earlier optimization
   s.k <- sugg.base$k
   s.g <- sugg.base$g
   # were k and g optimized: affects number of columns in population
   s.k.opt <- length(s.k)==2
   s.g.opt <- length(s.g)==2
   # remove columns if optimized
   prev.k <- NA
   if (s.k.opt) {
      prev.k <- popul[1,1]
      popul <- as.matrix(popul[,-1])
   }
   prev.g <- NA
   if (s.g.opt) {
      prev.g <- popul[1,1]
      popul <- as.matrix(popul[,-1])
   }
   # check number of columns
   if (ncol(popul) != n.feat) {
      stop("number of columns in mx and population of sugg.base are different -\nsugg.base cannot be used")
   }

   # if suggestion are from ffeatsel.con(), zero column weights under wmin
   if (!is.null(sugg.base$wmin)) {
      popul[popul < sugg.base$wmin] <- 0
   }

   # add columns in population if k and/or g are now optimized
   if (g.opt) {
      if (s.g.opt & prev.g >= gmin & prev.g <= gmax) {
         # initialize with previous result if within current range
         gcol <- rep(prev.g, nrow(popul))
      } else {
         # initialize with average value
         gcol <- rep((gmin+gmax)/2, nrow(popul))
      }
      popul <- cbind(gcol, popul)
   }
   if (k.opt) {
      if (s.k.opt & prev.k >= kmin & prev.k <= kmax) {
         kcol <- rep(prev.k, nrow(popul))
      } else {
         kcol <- rep((kmin+kmax)/2, nrow(popul))
      }
      popul <- cbind(kcol, popul)
   }
   # number of suggestions have to be less than population size
   if (nrow(popul) >= popSize) {
      popul <- popul[1:(popSize-1),]
   }
   return(popul)
}

rmse <- function(y, yhat, neg.under=T)
   #  neg.under:  if TRUE, underestimation gives negative bias, otherwise
   #              positive
   #  25 January 2011 Juho Pitkänen
{
   if (! is.matrix(y)) {
      # a vector or data frame?
      y <- as.matrix(y)
   }
   if (! is.matrix(yhat)) {
      yhat <- as.matrix(yhat)
   }

   # number of rows
   rows <- nrow(y)
   cols <- ncol(y)

   if (nrow(yhat) != rows) {
      stop("Number of rows of y and yhat differ")
   }
   if (ncol(yhat) != cols) {
      stop("Number of columns of y and yhat differ")
   }

   # residuals
   if (neg.under) {
      # underestimation gives negative bias
      dy <- yhat - y
   } else {
      # underestimation gives positive bias
      dy <- y - yhat
   }
   # bias and rmse, also percentages
   bias <- apply(dy, 2, mean)
   sum.dy2 <- apply(dy^2, 2, sum)
   rmse <- sqrt(sum.dy2/rows)
   rmse.pct <- 100*rmse/apply(y, 2, mean)
   bias.pct <- 100*bias/apply(y, 2, mean)

   # rounding
   bias <- round(bias, 3)
   rmse <- round(rmse, 3)
   rmse.pct <- round(rmse.pct, 2)
   bias.pct <- round(bias.pct, 2)

   res <- list(rmse=rmse, bias=bias, rmse.pct=rmse.pct, bias.pct=bias.pct)
   return(res)
}

rmse.by <- function(y, yhat, by.vec=NULL, as.df=T, ...)
   # as.df: if TRUE, return result as data frame, otherwise as list
   # '...': for neg.under option of rmse()
{
   if (is.null(by.vec)) {
      res <- rmse(y, yhat, ...)
   } else {
      # if y is vector, no variable name
      isvec.y <- is.vector(y)
      if (! is.matrix(y)) {
         # a vector or data frame?
         y <- as.matrix(y)
      }
      if (! is.matrix(yhat)) {
         yhat <- as.matrix(yhat)
      }
      res <- tapply(seq(along=by.vec), list(targets=by.vec),
                    function(i, x, z) rmse(x[i,], z[i,], ...), x=y, z=yhat)
      if (as.df) {
         z <- res
         # number of by variable values
         n.by <- length(z)
         # names in first list element: rmse, bias, rmse.pct, bias.pct
         stats <- names(z[[1]])
         # names from yhat
         if (isvec.y) {
            # no variable name
            cn.var <- "variable"
         } else {
            cn.var <- names(z[[1]]$rmse)
         }

         # matrix of rmse
         res <- matrix(unlist(lapply(z, "[[", stats[1])), nrow=length(z), byrow=T)
         for (i in 2:length(stats)) {
            # row bind other statistics
            tmp <- matrix(unlist(lapply(z, "[[", stats[i])), nrow=length(z), byrow=T)
            res <- rbind(res, tmp)
         }

         res <- as.data.frame(res)
         names(res) <- cn.var
         # by value of calculation
         res$byval <- names(z)
         # names of statistics
         res$stat <- rep(stats, each=n.by)
         # order of columns: stat, byval, variables
         nc <- ncol(res)
         res <- res[, c(nc, nc-1, 1:(nc-2))]
      }
   }
   return(res)
}

fslist.catvar <- function(mx, my, my.query=NULL, fs.list, name="rmse",
                          itemgroupid=NULL, group.action=NULL, query.incl.ref=F)
   #  mx:       matrix of x variables of reference data
   #  my:       matrix of continuos variables of reference data (y variables)
   #  fs.list:  one output or a list of outputs from feature selection runs with
   #            either ffeatsel.bin() or ffeatsel.con()
   #  name:     name of result variable from either ffeatsel.bin.results() or
   #            ffeatsel.con.results() to be listed, usually one of "rmse",
   #            "bias", "rmse.pct", "bias.pct", "in.use" or "in.use.weights".
   #
   #  Summary of results for given parameter name from feature selection runs.
   #
   #  Returns:  a data frame, or a list for "in.use" or "in.use.weights", of
   #            results of given variable in feature selection runs
{
   require(genalg)

   if (is.list(fs.list)) {
      if (class(fs.list) == "rbga") {
         # just one result
         fs.list <- list(fs.list)
      } else if (! (class(fs.list[[1]]) == "rbga") ) {
         stop("Input is not a list of rbga-objects")
      }
   } else {
      stop("Input is not a list of rbga-objects")
   }

   first <- fs.list[[1]]
   if (first$type == "binary chromosome") {
      binary <- T
   } else if (first$type == "floats chromosome") {
      binary <- F
   } else {
      stop("Type of input is not binary or floats chromosome")
   }

   if (binary) {
      first.res <- ffeatsel.bin.results(mx, my, my.query=my.query, first,
                                        itemgroupid=itemgroupid, group.action=group.action, query.incl.ref=query.incl.ref)
   } else {
      first.res <- ffeatsel.con.results(mx, my, my.query=my.query, first,
                                        itemgroupid=itemgroupid, group.action=group.action, query.incl.ref=query.incl.ref)
   }
   if (! (name %in% names(first.res)) ) {
      cat("No variable", name, "in feature selection result\n")
      stop("Variable given in parameter 'name' not found")
   }

   # return list for these names because row length may vary, otherwise
   # return data frame
   list.output.names <- c("in.use", "in.use.weights")
   list.output <- name %in% list.output.names

   n.run <- length(fs.list)
   if (list.output) {
      res <- list(first.res[[name]])
   } else {
      res <- as.data.frame( t(first.res[[name]]) )
   }
   if (n.run > 1) {
      for (i in 2:n.run) {
         if (binary) {
            i.res <- ffeatsel.bin.results(mx, my, my.query=my.query, fs.list[[i]],
                                          itemgroupid=itemgroupid, group.action=group.action, query.incl.ref=query.incl.ref)
         } else {
            i.res <- ffeatsel.con.results(mx, my, my.query=my.query, fs.list[[i]],
                                          itemgroupid=itemgroupid, group.action=group.action, query.incl.ref=query.incl.ref)
         }
         if (list.output) {
            res[[i]] <- i.res[[name]]
         } else {
            res <- rbind(res, i.res[[name]])
         }
      }
   }
   if (is.data.frame(res)) {
      if (ncol(res) == 1) {
         colnames(res) <- name
      }
   }

   return(res)
}

fslist.best <- function(fs.list)
   #  fs.list:  one output or a list of outputs from feature selection runs with
   #            either ffeatsel.bin() or ffeatsel.con()
   #
   #  Finds the best evaluation value and it's run number in feature selection
   #  runs, the run number being the first run if there are several runs with
   #  the same best evaluation value.
   #
   #  Returns:  a list of the run number and the best evaluation value
{
   if (is.list(fs.list)) {
      if (class(fs.list) == "rbga") {
         # just one result
         fs.list <- list(fs.list)
      } else if (! (class(fs.list[[1]]) == "rbga") ) {
         stop("Input is not a list of rbga-objects")
      }
   } else {
      stop("Input is not a list of rbga-objects")
   }

   n.run <- length(fs.list)
   n.iter <- length(fs.list[[1]]$best)

   best.val <- fs.list[[1]]$best[n.iter]
   best.i <- 1
   if (n.run > 1) {
      for (i in 2:n.run) {
         if (fs.list[[i]]$best[n.iter] < best.val) {
            best.val <- fs.list[[i]]$best[n.iter]
            best.i <- i
         }
      }
   }
   return(list(nr=best.i, val=best.val)) 
}

roundUp.by <- function(x, width=10, down=F)
{
   if (down) {
      res <- width*floor(x/width)
   } else {
      res <- width*ceiling(x/width)
   }
   return(res)
}


fslist.plot <- function(fs.list)
   #  fs.list:  one output or a list of outputs from feature selection runs with
   #            either ffeatsel.bin() or ffeatsel.con()
   #
   #  Plots best and mean evaluation values of feature selection runs.
{
   require(genalg)

   if (is.list(fs.list)) {
      if (class(fs.list) == "rbga") {
         # just one result
         fs.list <- list(fs.list)
      } else if (! (class(fs.list[[1]]) == "rbga") ) {
         stop("Input is not a list of rbga-objects")
      }
   } else {
      stop("Input is not a list of rbga-objects")
   }

   tmp <- fslist.best(fs.list)
   best.val <- tmp$val
   best.i <- tmp$nr

   n.run <- length(fs.list)
   cat("Best evaluation value:", best.val, "of #", best.i, "\n")
   # find maximum value for ylim
   ymax <- 0
   for (i in 1:n.run) {
      ymax <- max(c(ymax, fs.list[[i]]$mean))
   }
   ymax <- roundUp.by(ymax, width=0.02)
   ymin <- roundUp.by(best.val, width=0.02, down=T)

   for (i in 1:n.run) {
      cat("Plot of #", i, "\n")
      plot(fs.list[[i]]$best, type = "l", main = "Best and mean evaluation value", 
           ylim = c(ymin, ymax), xlab = "generation", ylab = "evaluation value")
      lines(fs.list[[i]]$mean, col = "blue")

      abline(h=best.val, col=2)
      if (i < n.run) {
         readline("> Press Enter to continue ")
      }
   }
}

fxvars.byfeatsel <- function(mx, fs.res=NULL, verbose=T, name.mx=NULL)
   #  mx:       matrix of x variables
   #  fs.res:   feature selection result
   #  verbose:  be verbose about features and their scaling and weighting
   #  name.mx:  descriptive or variable name of mx for informative output
   #
   #  Scale and weight variables in x matrix similarly as in the feature
   #  selection, whose results are in fs.res.
{
   if (is.null(fs.res)) {
      stop("No feature selection result given")
   }

   if (is.null(name.mx)) name.mx <- "mx"
   if (verbose) {
      cat("All features in", name.mx, ":\n")
      cat(" ", colnames(mx), "\n")
      cat("Selected features:\n")
      cat(" ", fs.res$in.use, "\n")
   }
   if ( ! all(fs.res$in.use %in% colnames(mx)) ) {
      stop(name.mx, " does not contain all selected features")
   }
   if (is.data.frame(mx)) {
      is.dframe <- T
   } else {
      is.dframe <- F
   }

   # only selected features
   if ( ! identical(colnames(mx), fs.res$in.use) ) {
      mx <- subset(mx, select=fs.res$in.use)
   }

   # scaling of columns
   if (!is.null(fs.res$in.use.sd)) {
      mx <- scale(mx, center=F, scale=fs.res$in.use.sd)
      if (verbose) {
         cat("Scaling by\n", round(fs.res$in.use.sd, 2), "\n")
      }
   } else if (verbose) {
      cat("No scaling of columns\n\n")
   }

   # weighting of columns
   if ( is.null(fs.res$in.use.weights) ) {
      if (verbose) cat("No weighting of columns\n\n")
   } else {
      mx <- t(fs.res$in.use.weights * t(mx))
      if (verbose) {
         cat("Weighting by\n", round(fs.res$in.use.weights, 2), "\n")
         cat("\n")
      }
   }

   if (is.dframe) mx <- as.data.frame(mx)
   return(mx)
}

frmse.test <- function(mx, my, fs.res=NULL, ntest=100, nruns=1000, k=NULL,
      g=NULL, graphs=T, hist.bin=2, verbose=T, out="none")
   #  mx:     matrix of x variables of reference data
   #  my:     matrix of continuos variables of reference data (y variables)
   #  fs.res: feature selection result
   #  ntest:  number of observations in one test set to be used for accuracy
   #          assessment. Number of observations used for training is n-ntest,
   #          where n is total number of observations
   #  nruns:  number of cycles
   #  k:      number of nearest neighbours used in calculation. As a default,
   #          value is taken from fs.res
   #  g:      weighting of the nearest neighbours. As a default, value is
   #          taken from fs.res
   #  graphs: plot graphs of rmse histogram and test set mean values of ground
   #          truth and estimates (TRUE or FALSE)
   #  hist.bin:  bin width in rmse histogram
   #  verbose:   print accuracy statistics and feature information on console
   #             (TRUE or FALSE)
   #  out:    "stats"   : return summary statistics in a list
   #          "values"  : return individual values of rmse, bias and test set
   #                      mean values of ground truth and estimates in a list 
   #          other value : return nothing
   #
   #  Input data is randomly divided into training and test sets for knn 
   #  regression and rmse and bias statistics are calculated. This is repeated
   #  nruns times and mean statistics of accuracy and rmse histograms are
   #  printed.
{
   if (is.null(fs.res)) {
      stop("No feature selection result given")
   }

   n <- nrow(mx)
   if (ntest >= n) {
      stop("Number of observations for testing >= number of all observations")
   }
   n.train <- n - ntest
   if (n.train >= n) {
      stop("Number of observations for training >= number of all observations")
   }
   if (verbose) {
      cat("\n")
      cat(n.train, "observations for training,", ntest, "for testing\n")
      cat("\n")
   }

   # x features, scaling and weighting from feature selection
   mx <- fxvars.byfeatsel(mx, fs.res=fs.res, verbose=verbose)
   
   # k and g
   if (is.null(k)) {
      k <- fs.res$k
   }
   if (is.null(g)) {
      g <- fs.res$g
   }

   # initialize vector results
   vrmse <- NULL
   vbias <- NULL
   vy.mean <- NULL
   vyhat.mean <- NULL

   for (i in 1:nruns) {
      train.numbers <- sample.int(n, n.train)
      in.train <- 1:n %in% train.numbers

      # training data
      mx.train <- subset(mx, in.train)
      my.train <- subset(my, in.train)
      # evaluation data
      mx.test <- subset(mx, !in.train)
      y <- subset(my, !in.train)

      # find knn
      rann <- nn2(mx.train, mx.test, k=k, eps=0.0)
      # calculate continuous estimates
      yhat <- fknnestcon(rann$nn.idx, rann$nn.dists, my.train, k=k, g=g)

      res <- rmse(y, yhat)
      # rmse and bias of variables
      vrmse <- rbind(vrmse, res$rmse)
      vbias <- rbind(vbias, res$bias)
      # means of measured and estimates
      vy.mean <- rbind(vy.mean, apply(y, 2, mean))
      vyhat.mean <- rbind(vyhat.mean, apply(yhat, 2, mean))
   }

   mean.rmse <- round(apply(vrmse, 2, mean), 2)
   median.rmse <- round(apply(vrmse, 2, median), 2)
   set.stats <- rmse(vy.mean, vyhat.mean)
   set.rmse <- round(set.stats$rmse, 2)
   set.bias <- round(set.stats$bias, 2)
   if (verbose) {
      cat("---", ntest, "random plots in each test set,", nruns, "runs ---\n")
      cat("Variables                :", colnames(my), "\n")
      cat("Mean of plotwise RMSEs   :", mean.rmse, "\n")
      cat("Median of plotwise RMSEs :", median.rmse, "\n")
      cat("\n")
      cat("RMSE of mean estimates for test sets :", set.rmse, "\n")
      cat("bias of mean estimates for test sets :", set.bias, "\n")
      cat("\n")
   }

   if (graphs) {
      graphics.off()
      par(ask=T)
      for (i in 1:ncol(my)) {
         # histogram of plotwise RMSEs
         hmin <- (floor( min(vrmse[,i])/hist.bin ) - 1) * hist.bin
         hmax <- (ceiling( max(vrmse[,i])/hist.bin ) + 1) * hist.bin
         mtitle <- paste("Plotwise RMSEs of", colnames(my)[i], "(mean RMSE", mean.rmse[i],")")
         hist(vrmse[,i], breaks=seq(hmin, hmax, hist.bin), xlab="RMSE", main=mtitle)
         abline(v=mean.rmse[i], col=2)

         mtitle <- paste("Mean values within test sets", "(RMSE of mean est.", set.rmse[i],")")
         xlab <- paste(colnames(my)[i], ", ground truth", sep="")
         plot(vy.mean[,i], vyhat.mean[,i], asp=1, main=mtitle, xlab=xlab, ylab="estimate")
         abline(0,1,col=2)
      }
      par(ask=F)
   }

   if (out == "stats") {
      return(list(mean.rmse=mean.rmse, median.rmse=median.rmse,
                  setmean.rmse=set.rmse, setmean.bias=set.bias))
   } else if (out == "values") {
      return(list(rmse=vrmse, bias=vbias, y.mean=vy.mean, yhat.mean=vyhat.mean))
   } else {
      return(invisible())
   }
}

fknnreg <- function(mx, my, newx, fs.res=NULL, k=NULL, g=NULL, verbose=T, ...)
   #  mx:     matrix of x variables of reference data, containing at least
   #          those variables that are selected according to fs.res
   #  my:     matrix of continuos variables of reference data (y variables)
   #  newx:   matrix of x variables of query data, containing at least
   #          those variables that are selected according to fs.res
   #  fs.res: feature selection result in a list that contains elements
   #          in.use    : vector of names of selected features
   #          in.use.sd : if exists, scaling of features in feature selection
   #          in.use.weights : if exists, weighting of features in feature selection
   #  k:      number of nearest neighbours used in calculation. As a default,
   #          value is taken from fs.res
   #  g:      weighting of the nearest neighbours. As a default, value is
   #          taken from fs.res
   #  ...:    for zerodist and onlyzeros parameters of fknnestcon()
   #
   #  Runs knn regression for query data (newx), based on given feature
   #  selection result and reference data of x and y variables. 
   #  Returns y matrix of estimates for query data.
{
   if (is.null(fs.res)) {
      stop("No feature selection result given")
   }

   # x features, scaling and weighting from feature selection
   mx <- fxvars.byfeatsel(mx, fs.res=fs.res, verbose=verbose)
   newx <- fxvars.byfeatsel(newx, fs.res=fs.res, verbose=verbose, name.mx="newx")
   
   # k and g
   if (is.null(k)) {
      k <- fs.res$k
   }
   if (is.null(g)) {
      g <- fs.res$g
   }

   # find knn
   # Andras improved performance with knn 24.11.2016 --->
   # rann <- nn2(mx, newx, k=k, eps=0.0)
   rann <- knn(mx, newx, k=k, eps=0.0)
   # <----
   # calculate continuous estimates
   newy <- fknnestcon(rann$nn.idx, rann$nn.dists, my, k=k, g=g, ...)

   if (is.data.frame(my)) {
      newy <- as.data.frame(newy)
   }
   return(newy)
}

fknncv0 <- function(mx, my, testx, testy, fs.res=NULL, k=NULL, g=NULL,
      zerodist=NULL, onlyzeros=F, verbose=F, out="rmse")
   #  mx:     matrix of x variables of reference data, containing at least
   #          those variables that are selected according to fs.res
   #  my:     matrix of continuos variables of reference data (y variables)
   #  testx:   matrix of x variables of test data, containing at least
   #          those variables that are selected according to fs.res
   #  testy:   matrix of continuos variables of test data (y variables)
   #  fs.res: feature selection result in a list that contains elements
   #    "in.use"    : vector of names of selected features
   #    "in.use.sd" : if exists, scaling of features in feature selection
   #    "in.use.weights" : if exists, weighting of features in feature selection
   #  k:      number of nearest neighbours used in calculation. As a default,
   #          value is taken from fs.res
   #  g:      weighting of the nearest neighbours. As a default, value is
   #          taken from fs.res
   #  zerodist:   value to add in distance to avoid division by zero in weighting,
   #          see fknnestcon()
   #  onlyzeros:  only replace zero distances with the value of zerodist (> 0),
   #          see fknnestcon()
   #  out:    "rmse" : return a list of rmse, bias, rmse-% and bias-% vectors
   #                   for columns of my
   #          "est" : return estimates
   #           other value : like "rmse"
   #
   #  Runs knn regression for test data set, based on given feature selection
   #  result and reference data of x and y variables. Mainly for a special case
   #  of validation when some of the test observations are same as in the
   #  training data. These are handled with leave-one-out cross-validation.
   #  Function can also be used for accuracy assessment with non-overlapping
   #  test set.
   #
   #  When an observation is both in training and test data, it is dropped from
   #  knn:s within training data when the estimate for the observation is
   #  calculated. Existence of duplicates in training data is not checked so
   #  only first occurence of the same training observation is removed for each
   #  test observation.
{
   if (is.null(fs.res)) {
      stop("No feature selection result given")
   }

   # x features, scaling and weighting from feature selection
   mx <- fxvars.byfeatsel(mx, fs.res=fs.res, verbose=verbose)
   testx <- fxvars.byfeatsel(testx, fs.res=fs.res, verbose=verbose)
   
   # k and g
   if (is.null(k)) {
      k <- fs.res$k
   }
   if (is.null(g)) {
      g <- fs.res$g
   }

   if (is.data.frame(my)) {
      isdf.my <- T
   } else {
      isdf.my <- F
   }
   if (! is.matrix(my)) {
      # a vector or data frame
      my <- as.matrix(my)
   }
   if (! is.matrix(testy)) {
      testy <- as.matrix(testy)
   }

   # find k+1 nn:s: each item may be included in the search data, so get one extra nn
   rann <- nn2(mx, testx, k=k+1, eps=0.0)

   ## Get k nn:s so that item itself is not within the nn:s.
   # Items having zero distance for the 1. nn, noting float inaccuracy
   delta0 <- 0.000000015
   dist0 <- rann$nn.dists[,1] < delta0
   if (any(dist0)) {
      # ids (row numbers) in test set
      idx0.test <- which(dist0)
      # ids (row numbers) in training set
      idx0.train <- rann$nn.idx[idx0.test,1]

      # check y variables for zero distance observation pairs; now assuming
      # that can compare y variables with just '=='
      if ( all(my[idx0.train,] == testy[idx0.test,]) ) {
         # y variables match on all rows, so just drop the 1. nn
         rann$nn.dists[idx0.test,1:k] <- rann$nn.dists[idx0.test,2:(k+1)]
         rann$nn.idx[idx0.test,1:k] <- rann$nn.idx[idx0.test,2:(k+1)]
         if (verbose) cat("Dropped", length(idx0.test), "1st nn:s from training data\n")
      } else {
         # find rows on which y variables match
         idx.match <- which(apply(my[idx0.train,] == testy[idx0.test,], 1, all))
         idx0.test2 <- idx0.test[idx.match]
         if (length(idx0.test2) > 0) {
            # drop the 1. nn
            rann$nn.dists[idx0.test2,1:k] <- rann$nn.dists[idx0.test2,2:(k+1)]
            rann$nn.idx[idx0.test2,1:k] <- rann$nn.idx[idx0.test2,2:(k+1)]
            if (verbose) cat("Dropped", length(idx0.test2), "1st nn:s from training data\n")
         }
         if (k >= 2) {
            # for remaining lines, have to check if other nn:s have zero distance;
            # find rows on which y variables don't match
            idx.nomatch <- which( apply(my[idx0.train,] == testy[idx0.test,], 1, function(x) !all(x) ) )
            idx0.test2 <- idx0.test[idx.nomatch]
            if (length(idx0.test2) > 0) {
               # just using loop now to check other nn:s
               for (i in idx0.test2) {
                  for (ik in 2:k) {
                     if (rann$nn.dists[i,ik] > delta0) break
                     # zero distance, check y variables
                     idx0.train <- rann$nn.idx[i,ik]
                     if ( all(my[idx0.train,] == testy[i,]) ) {
                        # drop the ik:th nn
                        rann$nn.dists[i,1:k] <- rann$nn.dists[i,(-ik)]
                        rann$nn.idx[i,1:k] <- rann$nn.idx[i,(-ik)]
                        if (verbose) cat("Dropped one ", ik, ". nn from training data\n", sep="")
                        break
                     }
                  }  # end for ik
               }  # end for i
            }
         }  # end if k
      }
   }  # end if any 
   rann$nn.idx <- rann$nn.idx[,1:k]
   rann$nn.dists <- rann$nn.dists[,1:k]

   # calculate continuous estimates
   yhat <- fknnestcon(rann$nn.idx, rann$nn.dists, my, k=k, g=g, zerodist=zerodist, onlyzeros=onlyzeros)

   if (out == "est") {
      # return estimates
      if (isdf.my) {
         yhat <- as.data.frame(yhat)
      }
      return(yhat)
   } else {
      # return rmse, bias, rmse-% and bias-%
      res <- rmse(testy, yhat, neg.under=T)
      return(res)
   }
}

knnimage.con <- function(mx, my, fs.res=NULL, inimage=NULL, band.labels=NULL,
                         maskimage=NULL, outimage=NULL, trf=NULL, k=NULL, g=NULL,
                         datatype='FLT4S', NAflag=-1, verbose=F, ...)
   #  mx:     matrix of x variables of reference data, containing at least
   #          those variables that are selected according to fs.res
   #  my:     matrix of continuos variables of reference data (y variables)
   #  fs.res: feature selection result in a list that contains elements
   #          in.use    : vector of names of selected features
   #          in.use.sd : if exists, scaling of features in feature selection
   #          in.use.weights : if exists, weighting of features in feature selection
   #  inimage:  file name of input image, or a list of the file names of the input
   #          images. Input image(s) must have at least those bands that have got
   #          selected in fs.res.
   #  band.labels:  a vector of band labels for only input image, or a list of
   #          bands labels for input images. A vector of labels for an image
   #          can be of length one, in which case bands of multilayer image are
   #          labeled with numbers (e.g. "b1", "b2"), or the vector can have direct
   #          labels for all bands (e.g. "b2", "b3", "dem"). If NULL, the bands are
   #          labeled by the raster package. Band labels of selected bands should
   #          match those in fs.res.
   #  maskimage: file name of mask image or NULL if no mask image is used
   #  outimage:  file name of output image
   #  trf:    transforms to be calculated from input image bands in a text string,
   #          e.g. "ndvi <- (b4-b3)/(b4+b3); b3.div.b1 <- b3/b1"
   #  k and g:   see fknnestcon(). As a default, value is taken from fs.res in fknnreg().
   #  datatype:  data type for output image, e.g. 'INT1U', 'INT2S' or 'FLT4S'. For details,
   #             see help of dataType() in raster package
   #  NAflag:    NoData value for output image. See help of writeRaster() in raster package
   #  verbose:   more output on console
   #  ...:    for zerodist and onlyzeros parameters of fknnestcon()
   #
   #  Knn regression from input image(s) to output image, based on reference data and
   #  feature selection result.

{
   stopifnot(!is.null(fs.res))
   stopifnot(!is.null(inimage))
   stopifnot(!is.null(outimage))

   library(raster)
   library(rgdal)

   if (!is.list(inimage)) {
      if (length(inimage) > 1) {
         cat("Coersing vector of input file names into list\n")
      }
      inimage <- as.list(inimage)
   }
   n.images <- length(inimage)

   # check the number of labels for images
   set.labels <- !is.null(band.labels)
   if (set.labels) {
      if (n.images == 1 & is.list(band.labels)) {
         if (length(band.labels != 1)) {
            stop("There should be only one vector of band labels for the only input image")
         }
      } else if (n.images == 1) {
         # a list also for one image
         band.labels <- list(band.labels)
      }
      if (n.images > 1 & !is.list(band.labels)) {
         stop("band.labels should be a list with the same length as inimage")
      }
      if (n.images > 1 & is.list(band.labels)) {
         if (n.images != length(band.labels)) {
            stop("band.labels should be a list with the same length as inimage")
         }
      }
   }

   # check that extents of input images match and collect a vector of band labels
   for (i in 1:n.images) {
      ima <- brick(inimage[[i]])
      if (set.labels) {
         lbl <- band.labels[[i]]
         if (length(lbl) == nlayers(ima)) {
            ima.labels <- lbl
         } else if (length(lbl) == 1) {
            # add band numbers in labels
            ima.labels <- paste0(lbl, 1:nlayers(ima))
         } else {
            err <- paste("Length of band labels should be 1 or", nlayers(ima), "for input image", inimage[[i]])
            stop(err)
         }
      }
      if (i == 1) {
         # band labels in a vector
         if (set.labels) bnames <- ima.labels
         # for comparing extents
         if (n.images > 1) ima1 <- brick(inimage[[i]])
      } else {
         # compare extents
         if (! compareRaster(ima1, ima)) {
            err <- paste("Extent and/or number of rows or columns of", inimage[[1]],
                         "and", inimage[[i]], "input images does not match")
            stop(err)
         }
         if (set.labels) bnames <- c(bnames, ima.labels)
      }
   }
   # brick if only one image, otherwise stack
   if (n.images > 1) {
      ima <- stack(inimage)
   }

   if (set.labels) {
      # set band labels for input image(s) if not default ones
      names(ima) <- bnames
   } else {
      # get band labels for check
       bnames <- names(ima)
   }

   mxnames <- colnames(mx)
   # names of selected features
   selnames <- fs.res$in.use
   
   if (verbose) {
      cat("Bands in mx are\n", mxnames, "\n")
      cat("Selected bands in feature selection result are\n", selnames, "\n")
      cat("Bands of input image(s) are labeled as\n", bnames, "\n")
      cat("\n")
   }
   # check that selected features are in reference x data
   if ( ! all(selnames %in% mxnames) ) {
      cat("Matrix mx does not contain all selected features:\n")
      cat("  Features missing are", selnames[!selnames %in% mxnames], "\n")
      stop("Stopping")
   }
 
   # check that selected features are in input images
   if ( ! all(selnames %in% bnames) ) {
      cat("Band labeling for input images does not contain all selected features:\n")
      cat("  Features missing are", selnames[!selnames %in% bnames], "\n")
      if (is.null(trf)) {
         stop("Stopping")
      } else {
         # no check of transformation names
         cat("Transformations to be calculated are\n")
         cat(" ", trf, "\n")
         cat("\n")
      }
   } else if ( ! identical(bnames, selnames) ) {
      # input image(s) have also other than selected bands, or the order of bands
      # is different so read only bands needed in right order
      idx <- NULL
      for (sname in selnames) {
         ind <- which(bnames %in% sname)
         idx <- c(idx, ind)
      }
      ima <- subset(ima, idx)
      if (verbose) {
         cat("Bands to be read from input image(s) are\n", names(ima), "\n")
         cat("\n")
      }
   }

   # check mask image
   if (is.null(maskimage)) {
      have.mask <- F
   } else {
      mask.ima <- raster(maskimage)
      if (! compareRaster(ima, mask.ima)) {
         stop("Extent and/or number of rows or columns of mask and input image does not match")
      }
      have.mask <- T
   }

   # write to a new binary file in chunks
   if (ncol(my) == 1) {
      out.ima <- raster(ima)
   } else {
      out.ima <- brick(ima, nl=ncol(my))
   }
   # get chunk sizes
   tr <- blockSize(ima)

   out.ima <- writeStart(out.ima, filename=outimage, datatype=datatype, NAflag=NAflag, format='GTiff', overwrite=TRUE)

   if (interactive()) progbar <- txtProgressBar(min=1, max=tr$n, width=70)

   for (i in 1:tr$n) {
      # newx gets colum names from ima
      newx <- getValuesBlock(ima, row=tr$row[i], nrows=tr$nrows[i])
      # input image can not have NA pixels in knn estimation
      ok.ima <- complete.cases(newx)

      if (have.mask) {
         # drop NA pixels in mask and input image
         mask.val <- getValuesBlock(mask.ima, row=tr$row[i], nrows=tr$nrows[i])
         ok.mask <- complete.cases(mask.val)
         ok <- ok.mask & ok.ima
      } else {
         ok <- ok.ima
      }

      # do we have NAs
      if (all(ok)) {
         have.nas <- F
         some.ok <- T
      } else {
         have.nas <- T
         # is there any pixels to process
         some.ok <- any(ok)
         if (some.ok) {
            # keep only ok pixels
            newx <- newx[ok,]
         }
      }

      if (some.ok) {
         if (!is.null(trf)) {
            # transformations of bands; need a data frame for within()
            newx <- as.data.frame(newx)
            newx <- within(newx, eval(parse(text=trf)) )
         }
         if (i == 1 && verbose) {
            # verbose only once
            est <- fknnreg(mx, my, newx, fs.res=fs.res, k=k, g=g, verbose=T, ...)
         } else {
            est <- fknnreg(mx, my, newx, fs.res=fs.res, k=k, g=g, verbose=F, ...)
         }
         # have to be a matrix for writeValues()
         est <- as.matrix(est)
      }

      if (have.nas) {
         # return NAs in place to have full coverage for writing
         tmp <- matrix(nrow=length(ok.ima), ncol=ncol(my))
         colnames(tmp) <- colnames(my)
         if (some.ok) {
            tmp[ok,] <- est
         }
         est <- tmp
      }
      out.ima <- writeValues(out.ima, est[,], tr$row[i])

      if (interactive()) {
         if (i==1) {
            cat("<", paste0(rep("=",68), collapse=""), ">", "\n", sep="")
         }
         setTxtProgressBar(progbar, i)
      }
   }

   if (interactive()) {
      close(progbar)
      cat("\n")
   }

   out.ima <- writeStop(out.ima)
}

band.arith <- function(x, names1=NULL, names2=NULL, op="/", op.name="div",
                       oneway=T, verbose=T)
   #  x:       input data frame
   #  names1:  column names of first argument in column transforms
   #  names2:  column names of second argument in column transforms. If NULL,
   #           copied from names1
   #  op:      operation to be calculated, e.g. "/", "*", "+" and "-"
   #  op.name: text name of operation for naming result columns that will
   #           be like "b2.div.b1", if op.name is "div" and input columns
   #           are "b1" and "b2"
   #  oneway:  if T, only calculate one transform from two input columns and
   #           not with reverse order of columns
   #
   #  Calculate transformations between two separate columns with operations
   #  like "/", "*", "+" and "-".
{
   stopifnot(!is.null(names1))
   stopifnot(names1 %in% colnames(x))

   if (is.null(names2)) {
      names2 <- names1
   } else {
      stopifnot(names2 %in% colnames(x))
   }

   if (verbose) cat("Creating transforms:\n")
   # have to be a data frame for within()
   x <- as.data.frame(x)
   # names of transformations made
   done <- NULL
   for (i in names1) {
      for (j in names2) {
         if (i == j) {
            next
         }
         cname <- paste0(i, ".", op.name, ".", j)

         if (oneway) {
            # have to check that only one trasform is calculated from two input columns
            revname <- paste0(j, ".", op.name, ".", i)
            if (revname %in% done) {
               next
            }
         }
         trf <- paste(cname, "<-", i, op, j)
         if (verbose) cat("  ", trf, "\n")
         x <- within(x, eval(parse(text=trf)) )
         done <- c(done, cname)
      }
   }

   if (op == "/") {
      if ( any(is.infinite(as.matrix(x))) ) {
         if (is.data.frame(x)) {
            isdf <- T
            x <- as.matrix(x)
         } else {
            isdf <- F
         }
         x[is.infinite(x)] <- 0
         if (isdf) {
            x <- as.data.frame(x)
         }
         if (verbose) {
            cat("Results of division by 0 set to 0\n")
         }
      }
   }
   return(x)
}

band.ndvilike <- function(x, names1=NULL, names2=NULL, op.name="ndvi",
                       oneway=T, verbose=T)
   #  x:       input data frame
   #  names1:  column names of first argument in column transforms
   #  names2:  column names of second argument in column transforms. If NULL,
   #           copied from names1
   #  op.name: text name of operation for naming result columns that will
   #           be like "b2.ndvi.b1", if op.name is "ndvi" and input columns
   #           are "b2" and "b1"
   #  oneway:  if T, only calculate one transform from two input columns and
   #           not with reverse order of columns
   #
   #  Calculate ndvi-like transformations between two separate columns, like
   #  (b3 - b2)/(b3 + b2)
{
   stopifnot(!is.null(names1))
   stopifnot(names1 %in% colnames(x))

   if (is.null(names2)) {
      names2 <- names1
   } else {
      stopifnot(names2 %in% colnames(x))
   }

   if (verbose) cat("Creating transforms:\n")
   # have to be a data frame for within()
   x <- as.data.frame(x)
   # names of transformations made
   done <- NULL
   for (ii in names1) {
      for (jj in names2) {
         if (ii == jj) {
            next
         }
         if (identical(names1, names2)) {
            # assuming that band names are like b1, b2, ... and that
            # order in transformation would be (b2-b1)/(b2+b1)
            i <- jj
            j <- ii
         } else {
            i <- ii
            j <- jj
         }
         cname <- paste0(i, ".", op.name, ".", j)

         if (oneway) {
            # have to check that only one trasform is calculated from two input columns
            revname <- paste0(j, ".", op.name, ".", i)
            if (revname %in% done) {
               next
            }
         }
         trf <- paste(cname, "<- (", i, "-", j, ") / (", i, "+", j, ")")
         if (verbose) cat("  ", trf, "\n")
         x <- within(x, eval(parse(text=trf)) )
         done <- c(done, cname)
      }
   }

   if ( any(is.infinite(as.matrix(x))) ) {
      if (is.data.frame(x)) {
         isdf <- T
         x <- as.matrix(x)
      } else {
         isdf <- F
      }
      x[is.infinite(x)] <- 0
      if (isdf) {
         x <- as.data.frame(x)
      }
      if (verbose) {
         cat("Results of division by 0 set to 0\n")
      }
   }
   return(x)
}

loop.number <- function(obj)
   #  A monitor function for rbga-functions of genalg just to print loop number.
   #  To reset the counter, variable ii has to be removed manually.
{
   if (! exists("ii", where= globalenv())) {
      ii <<- 1
   } else {
      ii <<- ii + 1
   }
   cat("iteration ii", ii, " (remove ii before new run)\r")
}

knnreg.nafill <- function(dfin, cn.notused=NULL, k=1, g=1, n.sample=100000,
                           seed=NULL, sigdigits=6, verbose=T)
   #  dfin:  input data frame
   #  cn.notused:  column names in dfin that are not NA filled or used as
   #               independent variables in knn regression
   #  k:  number of nearest neighbours used in calculation
   #  g:  weighting of the nearest neighbours
   #  n.sample:  for large data sets, number of items used as candidate knn:s
   #  seed:  seed for selecting the sample of candidate knn:s
   #  sigdigits:  number of significant digits to be kept for replaced NA values
   #
   #  Filling of NA values in data frame columns by knn regression in which
   #  columns without NA values are used as independent variables
   #
   #  Returns:  input data frame that has NA values filled
   #
   #  6 May 2014 Juho Pitkänen
{
   stopifnot(is.data.frame(dfin))

   if (!is.null(cn.notused)) {
      mx.all <- subset(dfin, select=!(names(dfin) %in% cn.notused))
   } else {
      mx.all <- dfin
   }

   n <- nrow(dfin)
   # find columns that have and don't have NA values
   nacols <- NULL
   okcols <- NULL
   # find vector of items that have NA values
   navec <- rep(F, n) 
   for (cn in names(mx.all)) {
      cnvec <- is.na(mx.all[[cn]])
      if (any(cnvec)) {
         navec <- navec | cnvec
         nacols <- c(nacols, cn)
      } else {
         okcols <- c(okcols, cn)
      }
   }
   stopifnot(!is.null(okcols))
   stopifnot(!is.null(nacols))
   if (verbose) cat("Columns having NA values:", nacols, "\n\n")

   # get training data
   mx.all <- subset(mx.all, !navec)
   nn <- nrow(mx.all)
   if (nn > n.sample) {
      if (!is.null(seed)) {
         set.seed(seed)
      }
      svec <- sample(nn, n.sample)
      mx.all <- mx.all[svec, ]
   }
   mx <- subset(mx.all, select=okcols)
   my <- subset(mx.all, select=nacols)

   # create feature selection result for fknnreg()
   fake.fs <- list()
   fake.fs$in.use <- okcols
   # sd for scaling of independent columns
   fake.fs$in.use.sd <- apply(mx, 2, sd)

   # get estimates for NA columns for items that have NA values
   newx <- subset(dfin, navec, select=okcols)
   est <- fknnreg(mx, my, newx, fs.res=fake.fs, k=k, g=g, verbose=verbose)
   if (!is.null(sigdigits)) {
      est <- signif(est, sigdigits)
   }

   # replace NA values in columns
   for (cn in names(my)) {
      # there are estimates for all NA columns of data frame in rows that have
      # any NA values but replacing is done only for NA values in the current column
      tmp <- dfin[[cn]]
      cnvec <- is.na(tmp)
      tmp[navec] <- est[[cn]]
      dfin[[cn]][cnvec] <- tmp[cnvec] 
   }
   return(dfin)
}

knnreg.influent <- function(mx, my, fs.res=NULL, in.y=F, count=3, wrmse=1, sca="both",
                       k=NULL, g=1, w=NULL, zerodist=NULL, onlyzeros=F, verbose=F)
   #  mx:     matrix or data frame of x variables of reference data
   #  my:     matrix or data frame of continuos variables of reference data
   #          (y variables)
   #  fs.res: feature selection result. If given, scaling of x and values of k
   #          and g are taken from this
   #  in.y    if T, drop item also from rmse evaluation. Otherwise item is
   #          only excluded from knn.
   #  wrmse:  weighting vector for RMSE:s and absolute biases of my columns.
   #          Default is equal weighting
   #  sca:    scale x and/or y variables by standard deviation
   #          "both" : scale x and y variables
   #          "x" : scale x variables
   #          "y" : scale y variables
   #          other value : no scaling
   #  w:      weights for columns of mx, i.e. feature weights. If scaling
   #          is done, weights are applied after scaling.
   #  zerodist:   value to add in distance to avoid division by zero in weighting,
   #          see fknnestcon()
   #  onlyzeros:  only replace zero distances with the value of zerodist (> 0),
   #          see fknnestcon()
   #
   #  Most influent items in leave-one-out cross validation of knn regression
   #  based on the exclusion of the item from the nn search only, or optionally
   #  from the rmse evaluation as well
   #
   #  Returns:  a list of indices of most influent items, original and without
   #            each returned item evaluation values, rmse:s and biases
   #
   #  10 April 2014 Juho Pitkänen
{
   stopifnot( !is.null(fs.res) | !is.null(k) )

   # for searching nn
   library(RANN)

   # number of rows
   # rows <- nrow(mx)
   ifelse(is.null(dim(mx)),rows <- nrow(mx[[1]]),rows <- nrow(mx))

   # y may be given as a vector or data frame
   if (! is.matrix(my) ) {
      # a vector or data frame?
      my <- as.matrix(my)
   }
   if (nrow(my) != rows) stop("Number of rows of mx and my differ")
   
   if (!is.null(fs.res)) {
      # x features, scaling and weighting of x from feature selection
      mx <- fxvars.byfeatsel(mx, fs.res=fs.res, verbose=verbose)
      k <- fs.res$k
      g <- fs.res$g
   } else {
      # scaling of x variables
      if (sca == "x" || sca == "both") {
         x.sd <- apply(mx, 2, sd)
         if (any(x.sd==0)) stop(paste0("Standard deviation of columns ",paste0(names(x.sd)[which(x.sd==0)],collapse=", "),
                                       " is zero. Scaling cannot be done for these columns (and they are meaningless), remove them!"))
         mx <- scale(mx, center=F, scale=x.sd)
      }
      # weighting of x variables
      if (length(w) > 0) {
         if (sum(w) <= 0) {
            stop("Sum of weights in w is <= 0")
         } else if (length(w) == ncol(mx)) {
            # all features, or subset, and their weights; feature is not
            # selected if weight is 0
            indices <- w > 0
            mx <- mx[, indices]
            wei.sel <- w[indices]
            # apply weights
            mx <- t(wei.sel*t(mx))
         } else {
            stop("Number of weights in w and columns of mx differ")
         }
      }
   }

   # scaling of y variables
   if (sca == "y" || sca == "both") {
      y.sd <- apply(my, 2, sd)
      my <- scale(my, center=F, scale=y.sd)
   }
   # scale wrmse to sum of one
   n.var <- ncol(my)
   if (length(wrmse) > 1) {
      # parameter given
      if (length(wrmse) == n.var && sum(wrmse) > 0) {
         wrmse <- wrmse/sum(wrmse)
      } else {
         if (sum(wrmse) > 0) {
            stop("Number of columns in my and number of values in wrmse are different")
         } else {
            stop("Sum of weights in wrmse is <= 0")
         }
      }
   } else {
      # equal weighting
      wrmse <- rep(1/n.var, n.var)
   }

   # get base results
   y <- fknncv.base(mx, my, k=k, g=g, zerodist=zerodist, onlyzeros=onlyzeros)
   res.all <- rmse(my, y, neg.under=T)
   # evaluation value
   eval.all <- as.vector(wrmse %*% res.all$rmse) + as.vector(wrmse %*% abs(res.all$bias))

   # find k+2 nn:s: each item is included in the search data, so get one extra nn
   # for that and another for one by one exclusion
   rann <- nn2(mx, mx, k=k+2, eps=0.0)

   rownums <- 1:nrow(rann$nn.idx)

   # current minimum evaluation values and corresponding rmse:s, biases and
   # indices of items to drop
   min.inds <- rep(0, count)
   min.evals <- rep(eval.all, count)
   ormse <- res.all$rmse
   obias <- res.all$bias
   for (i in 2:count) {
      ormse <- rbind(ormse, res.all$rmse)
      obias <- rbind(obias, res.all$bias)
   }

   # influence of each item as a knn
   for (ind in 1:rows) {
      # get k nn:s so that item itself is not within the nn:s and leave also
      # i:th item out of x variables
      ok.idx <- rann$nn.idx != rownums & rann$nn.idx != ind
      idx <- matrix(nrow=rows, ncol=k)
      dists <- matrix(nrow=rows, ncol=k)
      for (i in 1:rows) {
         idx[i,] <- rann$nn.idx[i, ok.idx[i,] ][1:k]
         dists[i,] <- rann$nn.dists[i, ok.idx[i,] ][1:k]
      }

      # calculate continuous estimates
      y <- fknnestcon(idx, dists, my, k=k, g=g, zerodist=zerodist, onlyzeros=onlyzeros)

      sset <- rep(T, rows)
      if (in.y) {
         # drop item from evaluation as well
         sset[ind] <- F
      }
      # get evaluation value and compare to the best values
      res <- rmse(my[sset,], y[sset,], neg.under=T)
      eval.ind <- as.vector(wrmse %*% res$rmse) + as.vector(wrmse %*% abs(res$bias))
      if (eval.ind < min.evals[count]) {
         min.evals[count] <- eval.ind
         min.inds[count] <- ind
         ormse[count, ] <- res$rmse
         obias[count, ] <- res$bias
         if (ncol(ormse) == 1) {
            # keep as matrix
            ormse[,1] <- ormse[order(min.evals),1]
            obias[,1] <- obias[order(min.evals),1]
         } else {
            ormse <- ormse[order(min.evals),]
            obias <- obias[order(min.evals),]
         }
         min.inds <- min.inds[order(min.evals)]
         min.evals <- min.evals[order(min.evals)]
      }
   }

   rownames(ormse) <- NULL
   return(list(inds=min.inds, evals=min.evals, eval.ori=eval.all,
               rmse=ormse, rmse.ori=res.all$rmse,
               bias=obias, bias.ori=res.all$bias))
}

xy.idw <- function(xyz, q.xy, cnames.xy=c("x", "y"), k=5, g=1, ...)
   # xyz:  data frame of reference data containing only x and y coordinates and
   #       continuous variable(s) of interest
   # q.xy: data frame of coordinates of query points
   # cnames.xy: column names of x and y coordinates in xyz and q.xy
   # k:    number of nearest neighbours used in calculation
   # g:    weighting of the nearest neighbours: larger g gives less weight to
   #       an nn on a larger distance
   #       g=0 : equal weighting of the nn
   #       g=1 : inverse distance weighting of the nn
   # ...:  for zerodist and onlyzeros parameters of fknnestcon()
   #
   # Idw based predictions from knn for query points.
   # Returns a data frame of x and y coordinates and knn predictions
   #
   # 17 November 2014 Juho Pitkänen
{
   stopifnot(cnames.xy %in% names(xyz))
   stopifnot(cnames.xy %in% names(q.xy))

   if (ncol(q.xy) > 2) {
      # keep only columns of the coordinates
      q.xy <- q.xy[, cnames.xy]
   }

   # separate data frames for knn
   mx <- subset(xyz, select=cnames.xy)
   my <- subset(xyz, select=names(xyz)[!names(xyz) %in% cnames.xy])

   # names of columns for knn to find neighbours
   fs.res <- list(in.use=cnames.xy)

   # knn estimation
   res0 <- fknnreg(mx, my, q.xy, fs.res=fs.res, k=k, g=g, verbose=F, ...)

   # return a df of coordinates and knn predictions
   res <- as.data.frame(cbind(q.xy, res0))
   return(res)
}

xy.idw.radius <- function(xyz, q.xy, cnames.xy=c("x", "y"), radius=NULL, g=1,
                          k.max=50, ...)
   # xyz:  data frame of reference data containing only x and y coordinates and
   #       continuous variable(s) of interest
   # q.xy: data frame of coordinates of query points
   # cnames.xy: column names of x and y coordinates in xyz and q.xy
   # radius: radius for selection of neighbours
   # k.max: maximum number of nearest neighbours to be searched for. Should be
   #        large enough for finding the maximum number of neighbours within the radius.
   # g:    weighting of the nearest neighbours: larger g gives less weight to
   #       an nn on a larger distance
   #       g=0 : equal weighting of the nn
   #       g=1 : inverse distance weighting of the nn
   # ...:  for zerodist and onlyzeros parameters of fknnestcon()
   #
   # Idw based predictions from neighbours within radius for query points.
   # Returns a data frame of x and y coordinates and knn predictions
   #
   # 17 November 2014 Juho Pitkänen
{
   stopifnot(require(RANN))
   stopifnot(cnames.xy %in% names(xyz))
   stopifnot(cnames.xy %in% names(q.xy))
   stopifnot(!is.null(radius))

   k.max <- min(k.max, nrow(xyz))

   if (ncol(q.xy) > 2) {
      # keep only columns of the coordinates
      q.xy <- q.xy[, cnames.xy]
   }

   # separate data frames for knn
   mx <- subset(xyz, select=cnames.xy)
   my <- subset(xyz, select=names(xyz)[!names(xyz) %in% cnames.xy])

   # find nn
   rann <- nn2(mx, q.xy, searchtype="radius", radius=radius, k=k.max)
   # number of nn found for each query point
   kvec <- apply(rann$nn.idx, 1, FUN=function(x) sum(x > 0))

   # result matrix of NAs; NAs are kept when there are no nn within the radius
   res0 <- matrix(nrow=nrow(q.xy), ncol=ncol(my))
   colnames(res0) <- colnames(my)

   for (k in 1:max(kvec)) {
      # calculate for subsets of different k values 
      sset <- kvec == k
      if (any(sset)) {
         res0[sset,] <- fknnestcon(rann$nn.idx[sset,, drop=F], rann$nn.dists[sset,, drop=F], my,
                                   k=k, g=g, ...)
      }
   }

   # return a df of coordinates and knn predictions
   res <- as.data.frame(cbind(q.xy, res0))
   return(res)
}
