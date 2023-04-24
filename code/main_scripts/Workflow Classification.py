# Nils Rietze
# 09.09.2022
# coding: utf-8

#%%
import os
from glob import glob
import re

import numpy as np
import pandas as pd
import seaborn as sns

import tifffile as tiff # for reading the original TIFF images
from skimage.io import imread

from osgeo import gdal
import geopandas as gpd
from pyproj import Transformer,Proj
import rasterio # for saving GeoTiff image
import xarray as xr
from tqdm import tqdm # for the beautiful progress-bars

from skimage.transform import rescale # for rescaling the images
from sklearn.impute import KNNImputer # for imputting the missing values
from skimage.segmentation import mark_boundaries
import scipy 

from myfunctions import *


#%%
PATH = "C:/data/0_Kytalyk/0_drone/2021/TLB/"
os.chdir(PATH)


#%% 0. Loading data

# listing all MSP images and selecting those for which MSP is not available
indexfiles = glob(PATH+'MSP/indices/*.tif')

mspfiles = glob('MSP/resampled/*15*.tif')

tirfiles = glob('TIR/resampled/*15*.tif')

#%% 1. Preprocessing


#%% 2. Classification

# a) Loading imagery data

# Reading out an image and calculating the indexes
fname_msp = PATH + "MSP/resampled/TLB_2021_MSP_stack.tif"
I_msp = xr.open_rasterio(fname_msp)
# I_msp = I_msp.copy()/I_msp.quantile(0.99, dim=['x','y'])
I_msp.values = image_minmax(I_msp.values)

msp = imread(fname_msp) # reading out preprocessed image

msp3 = image_minmax(msp[:,:,[2,4,1]]) # creating falsecolor image (NIR, red, green)
rgb3 = image_minmax(msp[:,:,[4,1,0]]) # creating RGB image

plot_2images(rgb3,msp3, titles = ["RGB image", "MSP image"])

fname_msp_ind = PATH + "MSP/indices/TLB_2021_indices_stack_cut.tif"
I_msp_ind = xr.open_rasterio(fname_msp_ind)
# I_msp_ind = I_msp_ind.copy()/I_msp_ind.quantile(0.99, dim=['x','y'])
I_msp_ind.values = image_minmax(I_msp_ind.values)

msp_ind = image_minmax(imread(fname_msp_ind))

# dsm = imread(dsmfiles[0])

I_tir = xr.open_rasterio(tirfiles[0])

msp_xx = image_minmax(msp)
dim_msp = pd.DataFrame(msp_xx.reshape(msp_xx.shape[0]*msp_xx.shape[1], msp_xx.shape[2]),
                   columns = ["b","g","nir","re","r"])

dim_ind = pd.DataFrame(msp_ind.reshape(msp_ind.shape[0]*msp_ind.shape[1], msp_ind.shape[2]), 
                   columns = ["bcc","bcc_std","gcc","gcc_std","ndvi","ndvi_std","rcc","rcc_std"])
dim = pd.concat([dim_msp,dim_ind],axis=1)
dim['sumb'] = dim["r"] + dim["g"] + dim["b"]

#%% b) Loading Training data

# Random points sampled in the training polygons:
labdat = gpd.read_file('C:/data/0_Kytalyk/0_drone/classification/TLB_training_polygons/TLB_Trainingpoints.shp')
labdat.rename(columns = {'Classname':'label','x':'lon_utm', 'y':'lat_utm'}, inplace = True)
labdat['region'] = 'thawlakebed'

# Read plot data
plotdat = pd.read_csv(r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\data\classification_test\plot_data.csv',sep = ';')
plotdat['region'] = plotdat.site
plotdat.head()

transformer = Transformer.from_crs("epsg:4326",'epsg:32655' )
p1 = Proj("epsg:4326")

def trans_fct(row):
    x1,y1 = p1(row.lon, row.lat)
    x,y = transformer.transform(y1, x1)
    return x,y

plotdat.loc[:,['lat_utm','lon_utm']] = plotdat.loc[:,['lat','lon']].apply(trans_fct, axis=1, result_type='expand').values

plotdat = plotdat.loc[plotdat.site == 'thawlakebed']


#%% Fetch image data
def FetchImageData(Raster, lons, lats):
    subset = [Raster.sel(x = x,
                        y = y, method = 'nearest').values for x,y in tqdm(zip(lons,lats))]
    return subset

    
labdat.loc[:, ["b","g","nir","re","r"]] = FetchImageData(I_msp, labdat.lon_utm, labdat.lat_utm)

labdat.loc[:, ["bcc","bcc_std","gcc","gcc_std","ndvi","ndvi_std","rcc","rcc_std"]] = FetchImageData(I_msp_ind, labdat.lon_utm, labdat.lat_utm)


#%% Plot class separability
import seaborn as sns

fig,axs = plt.subplots(3,3, figsize = (20,20),sharex = True)

for i,var in enumerate(featurenames):
    ax = axs.ravel()[i]
    sns.violinplot(data=labdat, x="label", y=var,ax = ax)
    ax.set(xlabel = '')
fig.tight_layout()
# plt.savefig(r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\figures_and_maps\classification\separability_violins_v1.png',bbox_inches = 'tight')

#%% Plot class separability
coldict = {"Shrubs": "lime", "Dry": "orange", "OpenWater": "blue", "Wet": "green", "Structures": "gray"}
labdat["col"] = list(map(coldict.get, labdat.label.values))

xvar = 'bcc'
yvar = 'rcc'

fig = plt.figure(figsize = (14,6))
plt.subplot(1, 1, 1)
plt.scatter(labdat[xvar], labdat[yvar], c = labdat.col, s = 10, alpha = .4)
plt.xlabel(xvar)
plt.ylabel(yvar)
plt.title("Training points for one image")
# plt.axis((min(labdat.ndvi),max(labdat.ndvi),min(labdat.sumb),max(labdat.sumb)))



legend_elements = [Line2D([0], [0], marker='o', color='w', label='Shrubs', markerfacecolor='lime', markersize=6),
                   Line2D([0], [0], marker='o', color='w', label='Wet', markerfacecolor='green', markersize=6),
                   Line2D([0], [0], marker='o', color='w', label='Dry', markerfacecolor='orange', markersize=6),
                   Line2D([0], [0], marker='o', color='w', label='OpenWater', markerfacecolor='blue', markersize=6),
                   Line2D([0], [0], marker='o', color='w', label='Structures', markerfacecolor='gray', markersize=6)]
plt.legend(handles = legend_elements, loc = "upper left", title="labels")

# plt.savefig(r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\figures_and_maps\classification\separability_%s_%s.png' % (xvar,yvar),bbox_inches = 'tight')
plt.show()

# %% Run RF classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay, cohen_kappa_score, classification_report

# Splitting the data into training and test data
seed = 0  # so that the result is reproducible
labdat_train, labdat_test = train_test_split(labdat, test_size = .333, random_state=seed,stratify = labdat['label'])

# Train RF classifier
featurenames = ['b','g','bcc', 'gcc', 'gcc_std', 'ndvi', 'ndvi_std', 'rcc']
dct = fitting_rf_for_region(labdat_train, featurenames, excl_low_imp = False, plot_importance = True) 
clf = dct["clf"]
featurenames = dct["featurenames"]

#  Plot confusion matrix
ypred = clf.predict(labdat_test[featurenames])

cfm = confusion_matrix(labdat_test['label'], ypred)

classes = labdat_test['label'].unique()
kappa = cohen_kappa_score(labdat_test['label'], ypred)

df_cfm = pd.DataFrame(cfm, index = classes, columns = classes)
plt.figure(figsize = (10,7))
ax = sns.heatmap(df_cfm, annot=True, cmap = 'Blues')
ax.set(xlabel = 'Predicted', ylabel = 'True', title = 'Cohens Kappa = %.2f' % kappa)
# ax.figure.savefig(r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\figures_and_maps\classification\cfm.png')

print(classification_report(labdat_test['label'], ypred, target_names=classes))

# %% Spatial semivariogram
import skgstat as skg
maxlag = 100
step = 0.15

V = skg.Variogram(labdat_train[['lon_utm','lat_utm']], labdat_train.ndvi,
                  model = 'spherical',n_lags = 50, bin_func = 'even', maxlag = maxlag)
fig = V.plot()
fig.axes[0].set(xlabel ='Distance (m)')
fig.axes[1].set(xticks = np.linspace(0,maxlag, 11),
                xticklabels = np.linspace(0,maxlag, 11)*step)

print(V)

#%%  predict
prlab = clf.predict(dim[featurenames]) # predicting labels for the image
prlab = prlab.reshape((msp.shape[0],msp.shape[1])) # reshaping prediction back to the image shape

plot_clfres(prlab, msp3, 'TLB_2021_classified_v1', 
            r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\data\classification_data\TLB\\', save = False)


prlab_num = prlab.copy()
for label,class_code in zip(labdat['label'].unique(),labdat['Classcode'].unique()):
    print(label +': '+class_code)
    prlab_num[prlab_num == label] = class_code
prlab_num = prlab_num.astype(int)


# Saving the MSP image in GeoTiff format
orig = rasterio.open(fname_msp) # reading out the original coordinates
new_dataset = rasterio.open( 
    r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\data\classification_data\TLB\TLB_classified_v1.tif',
    'w',
    driver='GTiff',
    height=prlab_num.shape[0],
    width=prlab_num.shape[1],
    count=1,
    dtype=prlab_num.dtype,
    crs=orig.crs, # coordinates of original image
    transform=orig.transform,
)
new_dataset.write(prlab_num,indexes=1)
new_dataset.close()



# %%

dA = xr.DataArray(data = [prlab,I_tir[0,:,:]])

df = pd.DataFrame()
for cl,name in enumerate(classes):
    print(name)
    df[name] = dA[1].where(dA[0] == name).values.flatten()
    
df = df.astype(float)
df.head()

ax = sns.violinplot(x="variable", y="value", data=df[df>10].melt())
ax.set(xlabel = '',
       ylabel = 'LST')


