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
  # lapply(1:nlayers(raster), function(x) raster[[x]])
  
  # Compute GLCM
  print('Computing GLCM...')
  
  # Assign variable name if the index stack or reflectance stack is computed
  if (grepl(x = FILENAME, pattern = 'index_stack')){
    vars <- names(raster)
  } else if (grepl(x = FILENAME, pattern = 'msp_.*resampled')){
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
        
    # Get site name and year from filename
    regex_pattern <- "(CBH|TLB|Ridge)"
    match <- regmatches(FILENAME, regexec(regex_pattern, FILENAME))
    if (length(match[[1]]) > 1) {
      site <- match[[1]][1]
    } else {
      ""
    }
    
    regex_pattern <- "(2020|2021)"
    match <- regmatches(FILENAME, regexec(regex_pattern, FILENAME))
    if (length(match[[1]]) > 1) {
      year <- match[[1]][1]
    } else {
      ""
    }
    
    FILENAME_OUT <- sprintf('%s_%s_%s_GLCM%s.tif',site,year,vars[i],window_size)
    print(paste('Writing textures to:', FILENAME_OUT))
    
    writeRaster(textures,paste0('./indices/',FILENAME_OUT))
  }
    
  # for (stat in statistics){
  #    FILENAME_OUT <- fnAppend(FILENAME,paste0('GLCM',stat))
  #    print(paste('Writing textures to:', FILENAME_OUT))
  #    writeRaster(textures[paste0('glcm',stat)],FILENAME_OUT)
  # }
  
}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
startTime <- Sys.time()

PATH <- '../../../data/mosaics/'
setwd(PATH)

for (type in c('reflectances','indices')){
  # get filenames
  if (type == 'reflectances'){
    flist <- list.files(path = './',pattern = 'msp_.*resampled\\.tif$',recursive = F,include.dirs = T)
    window_size <- 5
  } else if (type == 'indices'){
    flist <- list.files(path = './',pattern = ".*index_stack_.*\\.tif$",recursive = T,include.dirs = T)
    window_size <- 5
  }
  
  index_exclude <- grepl("GLCM", flist)
  
  flist <- flist[!index_exclude]
  print(flist)
  stats <- c("mean", "variance", "homogeneity","dissimilarity")
  
  # GetGLCM(flist, window_size = window_size, stats, PlotOutput = F)
  pblapply(flist[c(2,4)], 
           FUN = GetGLCM,
           cl = 8,
           window_size = window_size, statistics = stats, PlotOutput = F)
    
  print(Sys.time() - startTime)
  }