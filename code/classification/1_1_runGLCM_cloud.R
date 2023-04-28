#! /usr/bin/Rscript
# Author: Nils Rietze
#  nils.rietze@uzh.ch
# 27.09.2022

require(glcm)
require(raster)
require(pbapply)
require(parallel)

GetGLCM <- function(FILENAME, window_size, statistics, PlotOutput = F){
  # Load raster
  raster <- raster::stack(FILENAME)
  lapply(1:nlayers(raster), function(x) raster[[x]])
  
  # Compute GLCM
  print('Computing GLCM...')
  
  # Assign variable name if the index stack or reflectance stack is computed
  if (grepl(x = FILENAME, pattern = 'indices')){
    vars <- c('ndvi','rcc','gcc','bcc')
  } else if (grepl(x = FILENAME, pattern = 'reflectance_.*')){
    vars <- c('blue','green','red','rededge','nir')
  }
  
  # Loop through all bands of the raster 
  for (i in 1:dim(raster)[3]){
    print(sprintf('...for %s',vars[i]))
    
    textures <- glcm(raster[[i]],
                     window = c(window_size,window_size),
                     shift = list( c(0,1), c(1,1), c(1,0), c(1,-1) ),
                     statistics = statistics
    )
    CRS <- sp::CRS('+init=epsg:32655')    # WGS84 UTM 55N 
    crs(textures) <- CRS    
    year <- 2021
    site <- strsplit(FILENAME, '/')[[1]][1]
    FILENAME_OUT <- sprintf('%s_%s_%s_GLCM%s.tif',site,year,vars[i],window_size)
    print(paste('Writing textures to:', FILENAME_OUT))
    
    writeRaster(textures,paste(dirname(FILENAME),FILENAME_OUT,sep='/'))
  }
    
  # for (stat in statistics){
  #    FILENAME_OUT <- fnAppend(FILENAME,paste0('GLCM',stat))
  #    print(paste('Writing textures to:', FILENAME_OUT))
  #    writeRaster(textures[paste0('glcm',stat)],FILENAME_OUT)
  # }
  
  if (PlotOutput) plot(textures)
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
startTime <- Sys.time()
type <- 'indices'
# cl <- parallel::makeCluster(8)

# PATH <- sprintf('~/data/%s/MSP/%s/',site,type)
PATH <- '~/data/'
setwd(PATH)

# get filenames
if (type == 'reflectances'){
  flist <- list.files(path = '.',pattern = 'reflectance_.*\\.tif$',recursive = T,include.dirs = T)
  window_size <- 5
} else if (type == 'indices'){
  flist <- list.files(path = '.',pattern = '.*indices.*\\.tif$',recursive = T,include.dirs = T)
  window_size <- 5
}

index_exclude <- grepl("GLCM", flist)

flist <- flist[!index_exclude]
print(flist)
stats <- c("mean", "variance", "homogeneity","dissimilarity")

# GetGLCM(flist, window_size = window_size, stats, PlotOutput = F)
pblapply(flist, 
         FUN = GetGLCM,
         cl = 8,
         window_size = window_size, statistics = stats, PlotOutput = F)
  
print(Sys.time() - startTime)