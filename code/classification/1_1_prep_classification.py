# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:09:36 2022

@author: nils
"""

# imports
import os
from glob import glob

import subprocess

import numpy as np
import random

import pandas as pd

from skimage.io import imread

from osgeo import gdal
import geopandas as gpd
import rasterio # for saving GeoTiff image

import rioxarray as rxr


from elena_functions import image_minmax, plot_2images

from modules import FetchImageData, ExportToTif, SetBandDescriptions, ExtractFromMultipolygon

# --------------------------------------

random.seed(10)

# Indicate whether to produce the multispectral index stack (usually only done once at the beginning)
run_stacking = True

# Indicate whether random training & testing data needs to be sampled
run_pointsampling = True

TIF_PATH = "../../../data/mosaics/"
os.chdir(TIF_PATH)

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
            
            FNAME_MSP_INDEXSTACK = f"{site}_msp_index_stack_{year}.tif"
            
            # Check if a multispectral index stack already exists, if yes then skip
            if os.path.exists(FNAME_MSP_INDEXSTACK):
                print('Index stack exists, indices already computed. Skipping...', end = '\n')
            else:
                print('Copmuting multispectral indices...', end='\n')
                
                # Read multispectral data as numpy array
                msp = imread(FNAME_MSP)
                msp = np.ma.masked_equal(msp,0.) # Mask no data values (0) 

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
                
                # GLCMs
                # each has 4 bands ("mean", "variance", "homogeneity","dissimilarity"), we will only use variance for now
                # glcmfiles_IN = glob('MSP/indices/*GLCM.tif')
                # glcmfiles_OUT = [os.path.splitext(fn)[0] + '.vrt' for fn in glcmfiles_IN]
                # vrt_options = gdal.BuildVRTOptions(bandList = [2], # band number 2 for GLCM variances
                #                                    outputSRS = 'EPSG:32655') 
                # my_vrt = gdal.BuildVRT('MSP/indices/GLCM5.vrt', glcmfiles_IN, options=vrt_options)
                # my_vrt = None
                
                out_list = [ndvi,
                            rcc,gcc,bcc] 
                
                for i, var in enumerate(['NDVI','RCC','GCC','BCC']):
                    fname_out = f"{site}_{var}_{year}.tif" # Saving the index image in GeoTiff format
                    ExportToTif(FNAME_MSP, fname_out, out_list[i])
                
                # Create multiband raster
                indexfiles = list(set(glob('MSP/indices/*.tif')) - set(glob('MSP/indices/*GLCM*.tif'))) # List all multispectral index files except GLCMS (already resampled)
                # indexfiles = glob('MSP/indices/*.tif')
                
                """
                The GLCM rasters have 4 bands: mean, variance, homogeneity, dissimilarity
                
                """
                vrt_options = gdal.BuildVRTOptions(separate=True,
                                                   outputSRS = 'EPSG:32655') 
                my_vrt = gdal.BuildVRT(FNAME_MSP_INDEXSTACK + '.vrt', indexfiles, options=vrt_options)
                my_vrt = None
                
        
                # Run GDALWARP as cmd in subprocess
                print('Exporting MSP index stack...', end='\n')
                args = ['gdal_translate', 
                        FNAME_MSP_INDEXSTACK + '.vrt', 
                        FNAME_MSP_INDEXSTACK + '.tif']
                subprocess.Popen(args)
                    
                import time
                time.sleep(10)
                
                # Set Band descriptions
                desc = ['_'.join(os.path.basename(fn).split('.')[0].split('_')[2:]) for fn in indexfiles]
                SetBandDescriptions(FNAME_MSP_INDEXSTACK + '.tif', zip(range(1,len(desc)+1), desc))
                
                time.sleep(10)
                
                f = open(FNAME_MSP_INDEXSTACK + '.tif')
                if not f.closed:
                    print('Closing index stack file.')
                    f.close()
                
        # 3. GENERATE RANDOM SAMPLE POINTS
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if run_pointsampling:
            src = rasterio.open(FNAME_MSP) # Load Raster with sample grid (needs to be the same gridding for all rasters of that site & year)
            
            if year == 2020:
                shapefile = gpd.read_file('C:/data/0_Kytalyk/0_drone/classification/2020_training_polygons.shp')
            else:
                shapefile = gpd.read_file('C:/data/0_Kytalyk/0_drone/classification/training_polygons.shp')
            
            project_area = gpd.read_file('C:/data/0_Kytalyk/0_drone/project_areas_pix4d/%s_project_area.shp' % site.lower())
            
            clipped_shapefile = gpd.clip(shapefile,project_area)
            
            print('Sampling random points...', end='\n')
            dfs = list(clipped_shapefile.apply(lambda row: ExtractFromMultipolygon(row, 
                                                                           tif = src, 
                                                                           Dataname = 'blue'),
                                       axis=1)
                       )
            
            gdfOut = pd.concat(dfs).reset_index()
            gdfOut = gdfOut.set_crs(src.crs.data['init'])
            
            print('Fetching image data...', end = '\n')
            
            gdfOut.rename(columns = {'Classname':'label',
                                     'x':'lon_utm', 
                                     'y':'lat_utm'}, inplace = True)
            gdfOut['region'] = site
            
            I_msp = rxr.open_rasterio(FNAME_MSP)
            
            gdfOut.loc[:, ["b","g","r","re","nir"]] = FetchImageData(I_msp, gdfOut.lon_utm, gdfOut.lat_utm)
            
            FNAME_MSP_INDEXSTACK = PATH + "MSP/indices/{}_{}_indices_stack".format(site,year)
            I_msp_ind = rxr.open_rasterio(FNAME_MSP_INDEXSTACK + '.tif')
            
            index_stack_vars = I_msp_ind.long_name # ["bcc","gcc","ndvi","rcc"] + glcmnames
            
            gdfOut.loc[:,index_stack_vars] = FetchImageData(I_msp_ind, gdfOut.lon_utm, gdfOut.lat_utm)
            
            if year == 2021:
                FLIST_GLCM = glob('MSP/indices/*GLCM*.tif')
                for FNAME_GLCM in FLIST_GLCM:
                    with rxr.open_rasterio(FNAME_GLCM) as I_glcm:
                        glcm_stats = ["m", "v", "h","d"] # mean, variance, homogeneity, dissimilarity
                        glcm_varnames = ['{}_{}'.format(a, FNAME_GLCM.split('_')[2]) for a in glcm_stats]
                        gdfOut.loc[:,glcm_varnames] = FetchImageData(I_glcm, gdfOut.lon_utm, gdfOut.lat_utm)
            
            
            schema = gpd.io.file.infer_schema(gdfOut)
            # schema['properties']['int_column'] = 'int:18'
            
            gdfOut.to_file('C:/data/0_Kytalyk/0_drone/classification/{}_training_polygons/{}_{}_Trainingpoints.shp'.format(site,year,site),
                           driver='ESRI Shapefile', schema=schema)
            
        print('done.', end='\n')
    
    # Concatenate all trainingpoints into one dataframe:
    FLIST_TRAININGPOINTS = glob(r'C:/data/0_Kytalyk/0_drone/classification/*_training_polygons/{}_*_Trainingpoints.shp'.format(year))
    
    gdfAll = pd.concat(
        map(gpd.read_file,FLIST_TRAININGPOINTS),
        ignore_index=False).reset_index(drop=True)
    
    gdfAll.to_file('C:/data/0_Kytalyk/0_drone/classification/All_Trainingpoints_%s.shp' % year,
                   driver='ESRI Shapefile')
