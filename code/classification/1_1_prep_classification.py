# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:09:36 2022

@author: nils
"""
year = int(input('Which year do you want to process? \n'))


import os
from glob import glob
from tqdm import tqdm
import sys

import subprocess

import numpy as np
import random

import pandas as pd

from skimage.io import imread

from osgeo import gdal
import geopandas as gpd
from shapely.geometry import Point, mapping
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 

import rasterio # for saving GeoTiff image
from rasterio.mask import mask
from rasterio import Affine
import rioxarray as rxr

os.chdir(r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\code\main_scripts')

from elena_functions import image_minmax, plot_2images

# °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
import xml.etree.ElementTree as ET

import rasterio as rio
from rasterio.shutil import copy as riocopy
from rasterio.io import MemoryFile

def stack_vrts(srcs, band=1):
    vrt_bands = []
    for srcnum, src in enumerate(srcs, start=1):
        with rio.open(src) as ras, MemoryFile() as mem:
            riocopy(ras, mem.name, driver='VRT')
            vrt_xml = mem.read().decode('utf-8')
            vrt_dataset = ET.fromstring(vrt_xml)
            for bandnum, vrt_band in enumerate(vrt_dataset.iter('VRTRasterBand'), start=1):
                if bandnum == band:
                    vrt_band.set('band', str(srcnum))
                    vrt_bands.append(vrt_band)
                    vrt_dataset.remove(vrt_band)
    for vrt_band in vrt_bands:
        vrt_dataset.append(vrt_band)

    return ET.tostring(vrt_dataset).decode('UTF-8')

# --------------------------------------

def ExportToTif(fname_ref, fname_out, data):
    """
    Exports 2D raster data to a pre-specified GeoTIFF format

    Parameters
    ----------
    fname_ref : str
        Filepath of a reference raster with the desired CRS.
    fname_out : str
        Filepath of the output raster.
    data : np.array
        The data for the raster with dimensions h x w.

    Returns
    -------
    

    """
    orig = rasterio.open(fname_ref) # reading out the original coordinates
    new_dataset = rasterio.open( 
        fname_out,
        'w',
        driver = 'GTiff',
        height = data.shape[0],
        width = data.shape[1],
        count = 1,
        dtype = data.dtype,
        crs = orig.crs, # coordinates of original image
        transform = orig.transform,
    )
    new_dataset.write(data,indexes=1)
    new_dataset.close()

# --------------------------------------
    
def ClipTrainingPolygons(gdf,mask):
    
    masked_polygon = gpd.clip(gdf, mask)

# --------------------------------------

def ExtractFromMultipolygon(gdfRow, tif, Dataname: str):
    # extract the Classnames & class id
    ClName = gdfRow.Classname 
    ClId = gdfRow.Classcode 
    
    # extract the geometry in GeoJSON format
    geometry = gdfRow.geometry # list of shapely geometries

    # transform to GeJSON format
    geoms = [mapping(geometry)]

    # extract the raster values values within the polygon 
    out_image, out_transform = mask(tif, geoms, nodata = 0, 
                                    all_touched = True,
                                    crop = True)
    
    # no data values of the original raster
    no_data = 0

    # extract the values of the masked array
    data = out_image[0]

    # extract the row, columns of the valid values
    row, col = np.where(data != no_data) 
    vals = np.extract(data != no_data, data)
    
    T1 = out_transform * Affine.translation(0.5, 0.5) # reference the pixel centre
    rc2xy = lambda r, c: T1  * (c, r)
    
    d = gpd.GeoDataFrame({'col':col,'row':row,'Classname':ClName,'Classcode':ClId,'%s'% Dataname:vals} )
    
    d['id'] = gdfRow.name
    
    # coordinate transformation
    d['x'] = d.apply(lambda row: rc2xy(row.row,row.col)[0], axis=1)
    d['y'] = d.apply(lambda row: rc2xy(row.row,row.col)[1], axis=1)

    # geometry
    d['geometry'] = d.apply(lambda row: Point(row['x'], row['y']), axis=1)

    nsamples = min(len(d),200)
    return d.sample(nsamples,random_state = 11)

# --------------------------------------

def FetchImageData(Raster, lons, lats):
    subset = [Raster.sel(x = x,
                        y = y, method = 'nearest').values for x,y in tqdm(zip(lons,lats))]
    return subset

# --------------------------------------

def set_band_descriptions(filepath, bands):
    """
    filepath: path/virtual path/uri to raster
    bands:    ((band, description), (band, description),...)
    """
    ds = gdal.Open(filepath, gdal.GA_Update)
    for band, desc in bands:
        rb = ds.GetRasterBand(band)
        rb.SetDescription(desc)
    del ds


# %% 
random.seed(10)

run_stacking = True
run_pointsampling = True

for site in ['TLB','Ridge','CBH']:

    # 1. COFIGURE PATHS AND LOAD DATA
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    PATH = "C:/data/0_Kytalyk/0_drone/{}/{}/".format(year, site)
    os.chdir(PATH)
    print('Preparing data from {} in {}'.format(site,year), end='\n')
    
    
    tirfiles = glob(PATH+'TIR/resampled/*15*.tif') # List all thermal files (already resampled)

    # 2. COMPUTE MULTISPECTRAL INDICES
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    FNAME_MSP = glob(PATH+"MSP/resampled/reflectance*.tif")[0]
    
    if run_stacking:
        
        # Check if a multispectral image stack already exists, if not interrupt
        if not os.path.exists(FNAME_MSP):
            print('Please resample all multispectral files for {} {}!'.format(site,year), end='\n')
            break
        
        FNAME_MSP_INDEXSTACK = "MSP/indices/{}_{}_indices_stack".format(site,year)
        
        # Check if a multispectral index stack already exists, if yes then skip
        if not os.path.exists(FNAME_MSP_INDEXSTACK + '.tif'):
            print('Copmuting multispectral indices...', end='\n')
            
            msp = imread(FNAME_MSP)
            msp = np.ma.masked_equal(msp,0.) # Mask no data values (0) 
            # msp = np.where(msp==0,np.nan, msp)
            I_msp = rxr.open_rasterio(FNAME_MSP)
            
            if False: # Display RGB and MSP data
                msp3 = image_minmax(msp[:,:,[4,2,1]]) # creating falsecolor image (NIR, red, green)
                rgb3 = image_minmax(msp[:,:,[2,1,0]]) # creating RGB image
                plot_2images(rgb3,msp3, titles = ["RGB image", "MSP image"])
            
            
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
                fname_out = "MSP/indices/{}_{}_{}.tif".format(site,year,var) # Saving the index image in GeoTiff format
                ExportToTif(FNAME_MSP, fname_out, out_list[i])
            
            
            # Create multiband raster
            indexfiles = list(set(glob('MSP/indices/*.tif')) - set(glob('MSP/indices/*GLCM*.tif'))) # List all multispectral index files except GLCMS (already resampled)
            # indexfiles = glob('MSP/indices/*.tif')
            
            # Option GDAL BuildVRT: issue --> only accepts first band of multiband raster
            """
            The GLCM rasters have 4 bands: mean, variance, homogeneity, dissimilarity
            
            """
            vrt_options = gdal.BuildVRTOptions(separate=True,
                                               # bandList = np.array([2]), # band number 2 for GLCM variances
                                               outputSRS = 'EPSG:32655') 
            my_vrt = gdal.BuildVRT(FNAME_MSP_INDEXSTACK + '.vrt', indexfiles, options=vrt_options)
            my_vrt = None
            
            # Option rasterio BuildVRT: accepts other bands of multiband raster
            # src = rio.open(stack_vrts(indexfiles, band =  1))  
            # with open('MSP/indices/example.vrt', 'w') as f:
            #     f.write(stack_vrts(indexfiles[:2], band =  1))
    
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
            set_band_descriptions(FNAME_MSP_INDEXSTACK + '.tif', zip(range(1,len(desc)+1), desc))
            
            time.sleep(10)
            
            f = open(FNAME_MSP_INDEXSTACK + '.tif')
            if not f.closed:
                print('Closing index stack file.')
                f.close()
            
        else:
            print('Index stack exists, indices already computed. Skipping...', end = '\n')
        
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
