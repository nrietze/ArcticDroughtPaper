# -*- coding: utf-8 -*-
"""
Prepare the multispectral drone imagery for the classification

Author: Nils Rietze - nils.rietze@uzh.ch
Created: 24.04.2023
"""

# imports
import os
from glob import glob

import subprocess
import time
import numpy as np
import random

import pandas as pd

from skimage.io import imread

from osgeo import gdal
import geopandas as gpd
import rasterio # for saving GeoTiff image

import rioxarray as rxr

from modules import FetchImageData, ExportToTif, SetBandDescriptions, ExtractFromMultipolygon,image_minmax, plot_2images

# --------------------------------------

random.seed(10)

# Indicate whether to produce the multispectral index stack (usually only done once at the beginning)
run_stacking = True

# Indicate whether random training & testing data needs to be sampled
run_pointsampling = True

TIF_PATH = "../../../data/mosaics/"
os.chdir(TIF_PATH)

try:
    os.makedirs("./indices/")
except:
    pass

for year in [2020,2021]:
    for site in ['TLB','Ridge','CBH']:
    
        print('Preparing data from {} in {}'.format(site,year), end='\n')
        
        # 1. CONFIGURE PATHS AND LOAD DATA
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # List all thermal files (already resampled)
        FNAME_TIR = f"{site}_thermal_{year}_resampled.tif"
        FNAME_MSP = f"{site}_msp_{year}_resampled.tif"
    
        # 2. COMPUTE MULTISPECTRAL INDICES
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if run_stacking:
            
            # Check if a multispectral image stack already exists, if not interrupt
            if not os.path.exists(FNAME_MSP):
                print(f'Please resample all multispectral files for {site} {year} first!', end='\n')
                break
            
            FNAME_MSP_INDEXSTACK = f"./indices/{site}_msp_index_stack_{year}"
            
            # Check if a multispectral index stack already exists, if yes then skip
            if os.path.exists(FNAME_MSP_INDEXSTACK + ".tif"):
                print('Index stack exists, indices already computed. Skipping...', end = '\n')
            else:
                print('Copmuting multispectral indices...', end='\n')
                
                # Read multispectral data as numpy array
                msp = imread(FNAME_MSP)
                msp = np.ma.masked_less_equal(msp,0.) # Mask no data values (0) 

                # Read multispectral data as rasterio array
                I_msp = rxr.open_rasterio(FNAME_MSP)
                
                # Display RGB and MSP data
                msp3 = image_minmax(msp[:,:,[4,2,1]]) # creating falsecolor image (NIR, red, green)
                rgb3 = image_minmax(msp[:,:,[2,1,0]]) # creating RGB image
                plot_2images(rgb3,msp3, titles = ["RGB image", "MSP image"])
            
                # Extract individual spectral bands
                blue = msp[:,:,0]
                green = msp[:,:,1]
                red = msp[:,:,2]
                rededge = msp[:,:,3]
                nir = msp[:,:,4]
                
                # Compute NDVI    
                ndvi = (nir - red) / ( nir + red )
                
                # Compute BCC, RCC, GCC
                rcc = red / ( red + green + blue )
                gcc = green / ( red + green + blue )
                bcc = blue /  (red + green + blue )
               
                # Export the individual index rasters in GeoTiff format
                out_list = [ndvi,rcc,gcc,bcc] 
                
                for i, var in enumerate(['NDVI','RCC','GCC','BCC']):
                    fname_out = f"./indices/{site}_{var}_{year}.tif" 
                    ExportToTif(FNAME_MSP, fname_out, out_list[i])
                    
                # Stack GLCM rasters
                """
                The GLCM rasters have 4 bands: mean, variance, homogeneity, dissimilarity
                
                """
                if year == 2021:
                    FLIST_GLCM = glob('./indices/*GLCM*.tif')
                
                    try: 
                        glcmfiles_out = [os.path.splitext(fn)[0] + '.vrt' for fn in FLIST_GLCM]
                        vrt_options = gdal.BuildVRTOptions(bandList = [2], # band number 2 for GLCM variances
                                                            outputSRS = 'EPSG:32655') 
                        my_vrt = gdal.BuildVRT('./indices/GLCM.vrt', FLIST_GLCM, options=vrt_options)
                        my_vrt = None
                    except:
                        print("No GLCM files found. Please prepare the GLCM files first!")
                        break
                    
                # Stack the index rasters to multiband raster
                # ----
                print('Creating MSP index stack...', end='\n')
                
                # List all multispectral index files except GLCMS (already resampled)
                indexfiles = list(set(glob(f'./indices/{site}_*_{year}.tif')) - set(glob(f'./indices/{site}_{year}_*GLCM*.tif'))) 
                
                vrt_options = gdal.BuildVRTOptions(separate=True,outputSRS = 'EPSG:32655') 
                my_vrt = gdal.BuildVRT(FNAME_MSP_INDEXSTACK + '.vrt', indexfiles, options=vrt_options)
                my_vrt = None
                time.sleep(10)
                
                # Convert virtual raster to tif with gdal_translate in subprocess
                args = ['gdal_translate', 
                        FNAME_MSP_INDEXSTACK + '.vrt', 
                        FNAME_MSP_INDEXSTACK + '.tif']
                subprocess.Popen(args)
                
                time.sleep(10)
                os.remove(FNAME_MSP_INDEXSTACK + '.vrt')
                
                # Set Band descriptions
                desc = [os.path.basename(fn).split('.')[0].split('_')[1] for fn in indexfiles]
                SetBandDescriptions(FNAME_MSP_INDEXSTACK + '.tif', zip(range(1,len(desc)+1), desc))
                
                time.sleep(10)
                
                f = open(FNAME_MSP_INDEXSTACK + '.tif')
                if not f.closed:
                    print('Closing index stack file.')
                    f.close()
                
        # 3. GENERATE RANDOM SAMPLE POINTS
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if run_pointsampling:
            # Load Raster with sample grid (needs to be the same gridding for all rasters of that site & year)
            src = rasterio.open(FNAME_MSP) 
            
            # Load land cover polygons 
            shapefile = gpd.read_file(f'../shapefiles/{year}_polygons.shp')
            
            # Check if GLCM files are available, if not then generate those in R first
            try: 
                FLIST_GLCM = glob('./indices/*GLCM*.tif')
                # run dummy test to see if any GLCM files are around
                rxr.open_rasterio(FLIST_GLCM[0])
            except:
                print("No GLCM files found. Please prepare the GLCM files first!")
                break
            
            project_area = gpd.read_file(f'../shapefiles/{site.lower()}_project_area.shp')
            
            clipped_shapefile = gpd.clip(shapefile,project_area)
            
            print('Sampling random points...', end='\n')
            # Sample 200 random points within each polygon, based on the grid of `src`
            dfs = list(clipped_shapefile.apply(lambda row: ExtractFromMultipolygon(row, 
                                                                           tif = src, 
                                                                           Dataname = 'blue'),
                                       axis=1)
                       )
            
            gdfOut = pd.concat(dfs).reset_index()
            gdfOut = gdfOut.set_crs(src.crs.data['init'])
            
            gdfOut.rename(columns = {'Classname':'label',
                                     'x':'lon_utm', 
                                     'y':'lat_utm'}, inplace = True)
            gdfOut['region'] = site
            
            I_msp = rxr.open_rasterio(FNAME_MSP)
            
            # Retrieve reflectances from multispectral raster and add to geoDataFrame
            print('Fetching image data from MSP bands...', end = '\n')
            gdfOut.loc[:, ["b","g","r","re","nir"]] = FetchImageData(I_msp, gdfOut.lon_utm, gdfOut.lat_utm)
            
            FNAME_MSP_INDEXSTACK = f"./indices/{site}_msp_index_stack_{year}"
            
            I_msp_ind = rxr.open_rasterio(FNAME_MSP_INDEXSTACK + '.tif')
            
            index_stack_vars = I_msp_ind.long_name # ["bcc","gcc","ndvi","rcc"] + glcmnames
            
            # Retrieve indices from multispectral index raster and add to geoDataFrame
            print('Fetching image data from MSP indexes...', end = '\n')
            gdfOut.loc[:,index_stack_vars] = FetchImageData(I_msp_ind, gdfOut.lon_utm, gdfOut.lat_utm)
            
            # Retrieve GLCM values from GLCM raster and add to geoDataFrame
            if year == 2021:
                FLIST_GLCM = glob(f'./indices/{site}_*GLCM*.tif')
                
                for FNAME_GLCM in FLIST_GLCM:
                    with rxr.open_rasterio(FNAME_GLCM) as I_glcm:
                        # mean, variance, homogeneity, dissimilarity
                        glcm_stats = ["m", "v", "h","d"] 
                        spectral_name = FNAME_GLCM.split('_')[2]
                        glcm_varnames = [f'{a}_{spectral_name}' for a in glcm_stats]
                        
                        print(f'Fetching image data from {spectral_name} GLCMs ...', end = '\n')
                        gdfOut.loc[:,glcm_varnames] = FetchImageData(I_glcm, gdfOut.lon_utm, gdfOut.lat_utm)
            
            # Export geoDataFrame to shapefile (MultiPoint)
            schema = gpd.io.file.infer_schema(gdfOut)
            # schema['properties']['int_column'] = 'int:18'
            
            gdfOut.to_file(f'../shapefiles/{year}_{site}_points.shp',
                           driver='ESRI Shapefile', schema=schema)
            
        print(f'done for {site}.', end='\n')
    
    # Concatenate all trainingpoints into one dataframe:
    FLIST_TRAININGPOINTS = glob(f'../shapefiles/{year}_*_points.shp')
    
    gdfAll = pd.concat(
        map(gpd.read_file,FLIST_TRAININGPOINTS),
        ignore_index=False).reset_index(drop=True)
    
    gdfAll.to_file(f'../shapefiles/{year}_all_points.shp',
                   driver='ESRI Shapefile')
