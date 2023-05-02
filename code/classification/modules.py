"""
Custom functions used in the preparation and classificaiton.

Author: Nils Rietze - nils.rietze@uzh.ch
Created: 24.04.2023
"""
# imports
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # for plotting
from matplotlib.lines import Line2D # for creating plot legend

import tifffile as tiff # for reading tiff images

from sklearn.ensemble import RandomForestClassifier # for Random Forest classification
from skimage.segmentation import slic # for segmentation to superpixels
from skimage.measure import regionprops # for finding center coordinates of superpixels
from libpysal.weights import KNN # for finding the connectivity matrix between the superpixels
from sklearn.cluster import AgglomerativeClustering # for clustering the superpixels


# --------------------------------------
def fitting_rf_for_region(labdat, featurenames, excl_low_imp, plot_importance = False):
    """
    

    Parameters
    ----------
    labdat : pandas.DataFrame
        DataFrame with the training datat for the RF classifier, is produced in the
        1_1_prep_classification.py and exported to shapefiles. A column named
        'label' needs to be included, which has all the training labels of the data points.
    featurenames : list
        List with the column names of the spectral features (e.g. BCC, NDVI) 
        in labdat that are used in the RF classifier.
    excl_low_imp : bool
        Whether the RF classifier is excluding the features with low importance.
    plot_importance : bool, optional
        Plots the variable importance as bar charts. The default is False.

    Returns
    -------
    dict
        A dictionary with the classifier and featurenames.

    """
    # fitting the Random Forest classifier on the training data (labelled points) from the current region
    xtrain = labdat[featurenames]
    ytrain = labdat["label"]
    clf = RandomForestClassifier(n_estimators=100,random_state = 0, oob_score = True)
    clf.fit(xtrain, ytrain)
    importance = clf.feature_importances_
    importance = [round(importance[i]*100,1) for i in range(len(importance))]
    print("OOB accuracy before feature selection:",round(clf.oob_score_, 2))
    
    if excl_low_imp: # excluding the features with low importance and repeating the classification
        importance = np.array(importance)
        featurenames = np.array(featurenames)
        ind_nonimp = np.where(importance < 100/(2*len(importance)))
        featurenames = np.delete(featurenames,ind_nonimp, 0)
        xtrain = labdat[featurenames]
        clf = RandomForestClassifier(n_estimators=100,random_state = 0, oob_score = True)
        clf.fit(xtrain, ytrain)
        importance = clf.feature_importances_
        importance = [round(importance[i]*100,1) for i in range(len(importance))]
        print("OOB accuracy after feature selection:",round(clf.oob_score_, 2))
        
    if plot_importance:
        # plot feature importance
        forest_importances = pd.Series(importance, index=featurenames)
        std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
        
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        plt.show()
        
    return {'clf': clf, 'featurenames': featurenames}


# --------------------------------------
def plot_2images(img1,img2, figsize=(14,6), titles = ["",""]): # plotting 2 images
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img1)
    plt.axis("off")
    plt.title(titles[0])
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(img2)
    plt.axis("off")
    plt.title(titles[1])
    plt.tight_layout(pad=1)
    plt.show()

# --------------------------------------
def image_minmax(im, imax=None, imin=None): # performs image minmax normalization
    if imax is None:
        imax = im.max()
        imin = im.min()       
    im = (im - imin)/(imax - imin)
    return im

# --------------------------------------
from osgeo import gdal
import rasterio # for saving GeoTiff image

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
from tqdm import tqdm
import rioxarray as rxr

def FetchImageData(Raster, lons, lats):
    """
    Retrieve values in a raster near to given coordinates

    Parameters
    ----------
    Raster : xarray.DataSet or xarray.DataArray
    lons : list, array or pandas.Series
        List of longitudes.
    lats : list, array or pandas.Series
        List of latitudes

    Returns
    -------
    subset : list
        A list of raster values extracted from the coordinates.

    """
    subset = [Raster.sel(x = x,
                         y = y, method = 'nearest').values for x,y in tqdm(zip(lons,lats))]
    return subset

# --------------------------------------
def SetBandDescriptions(filepath, bands):
    """
    filepath: path/virtual path/uri to raster
    bands:    ((band, description), (band, description),...)
    """
    ds = gdal.Open(filepath, gdal.GA_Update)
    for band, desc in bands:
        rb = ds.GetRasterBand(band)
        rb.SetDescription(desc)
    del ds
    
# --------------------------------------
import geopandas as gpd
from shapely.geometry import Point, mapping
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
from rasterio.mask import mask
from rasterio import Affine

def ExtractFromMultipolygon(gdfRow, tif, Dataname: str):
    """
    Randmoly selects 200 pixel locations in a multipolygon feature layer 
    based on the grid of the tif layer.

    Parameters
    ----------
    gdfRow : geopandas.DataFrame
        DESCRIPTION.
    tif : rasterio raster file
        The Raster file that provides the pixel grid on which the random selection takes place.
    Dataname : str
        DESCRIPTION.

    Returns
    -------
    geopandas.DataFrame
        Subset of the input geopandas.DataFrame with random point locations within each polygon.

    """
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
from skimage.io import imread

def GatherGLCM(FNAME_GLCM):
    glcm = imread(FNAME_GLCM)
    glcm_stats = ["m", "v", "h","d"] # mean, variance, homogeneity, dissimilarity
    spectral_name = FNAME_GLCM.split('_')[2]
    glcm_varnames = [f'{a}_{spectral_name}' for a in glcm_stats]
    df = pd.DataFrame(glcm.reshape(glcm.shape[0]*glcm.shape[1], glcm.shape[2]), 
                   columns = glcm_varnames)
    return df