# -*- coding: utf-8 -*-
"""
Classify the multispectral drone imagery

Author: Nils Rietze - nils.rietze@uzh.ch
Created: 24.04.2023
"""
# imports
import os, time
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from osgeo import gdal
import geopandas as gpd
import rioxarray as rxr
import rasterio

from skimage.io import imread

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report
from sklearn.metrics import accuracy_score

from skimage.morphology import rectangle   # for Structuring Elements (e.g. disk, rectangle)
from skimage.filters.rank import modal # for applying the majority filter
from skimage.util import img_as_ubyte

from tqdm.auto import tqdm
from joblib import Parallel, delayed

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


from modules import GatherGLCM, fitting_rf_for_region

# %% Set variables
seed = 15  

predict = False
plot_results = False

TIF_PATH = "../../../data/"
os.chdir(TIF_PATH)

sites = ['CBH', 'Ridge', 'TLB']

# Dictionary of community names for label data
lbl = {'OpenWater':'Open water',
       'Mud': 'Mud',
       'Wet': 'LW1',
       'ledum_moss_cloudberry': 'LW2', 
       'Shrubs': 'HP2', 
       'Dry': 'HP1',
       'TussockSedge': 'TS'}

#%% Classify tundra plant communities
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for year in [2020,2021]:
    
    # Load label data (Random points sampled in the training polygons):
    FNAME_TRAIN_POINTS = f'./shapefiles/{year}_all_points.shp'
    labdat = gpd.read_file(FNAME_TRAIN_POINTS)
    labdat.rename(columns = {'Classname':'label',
                             'x':'lon_utm', 
                             'y':'lat_utm'}, inplace = True)

    labdat = labdat.loc[labdat["label"] !='Structures',:]
    
    # Rename class labels 
    if year == 2020:
        lbl_2020 = {'OpenWater':'Open water',
                    'NotWater': 'Not water'}
        labdat['label'] = labdat['label'].map(lbl_2020)
        label_order = ['Open water','Not water']
    else:
        labdat['label'] = labdat['label'].map(lbl)
        label_order = ['HP1','HP2','LW1','LW2','TS','Open water','Mud']
        labdat['label'] = labdat.label.astype("category").cat.set_categories(label_order, ordered=True)
        
    for j,site in enumerate(sites): # 
        # Define spectral bands and indices used for classification
        if year == 2020:
            featurenames = ['b','g','BCC', 'GCC', 'RCC', 'NDVI']
        else:
            featurenames = ['b','g','BCC', 'GCC', 'RCC', 'NDVI',
                            'v_bcc','m_green','h_red','m_red'] 
    
        print(f'Processing {site} {year}...', end='\n')
        
        # 1. LOAD IMAGE DATA
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        
        # Load Multispectral reflectances:
        FNAME_MSP = f"./mosaics/{site}_msp_{year}_resampled.tif"
        
        # Check if a multispectral image stack already exists, if not interrupt
        if not os.path.exists(FNAME_MSP): 
            print(f'Please resample all multispectral files for {site} {year}!', end='\n')
            break
        
        msp = imread(FNAME_MSP)
        msp[msp==-10000.] = np.nan
        msp = np.ma.masked_less_equal(msp,0.) # Mask no data values (0) 
        I_msp = rxr.open_rasterio(FNAME_MSP)
    
        # Load Multispectral indices:
        FNAME_MSP_INDEXSTACK = f"./mosaics/indices/{site}_msp_index_stack_{year}.tif"
        
        if not os.path.exists(FNAME_MSP_INDEXSTACK): # Check if a multispectral image stack already exists, if not interrupt
            print('Please prepare all multispectral indices for {} {}!'.format(site,year), end='\n')
            break
        
        I_msp_ind = rxr.open_rasterio(FNAME_MSP_INDEXSTACK)
        feature_idx = np.isin(I_msp_ind.long_name,featurenames) # Retrieve bands of msp_ind array that are used here
        msp_ind = imread(FNAME_MSP_INDEXSTACK)[:,:,feature_idx] # Slice the array for the indices needed
    
        # 2. PREPARE DATAFRAMES FOR RF CLASSIFICATION:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 2.a) PLOT SEPARABILITY
        if plot_results:
            
            if not os.path.exists("../figures/classification/"):
                os.makedirs("../figures/classification/")
                
            nfeat = len(featurenames)
            
            fig = plt.figure(nfeat, figsize = (50,20))
            
            cols = 6
            rows = int(np.ceil(nfeat/cols))
            
            for i,var in enumerate(featurenames):
                if i == 0:
                    ax = fig.add_subplot(rows,cols,i+1)
                else:
                    ax = fig.add_subplot(rows,cols,i+1,sharex = ax)
                sns.violinplot(data=labdat.loc[labdat.region == site], x="label", y=var,ax = ax)
                ax.set(xlabel = '')
                ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, horizontalalignment = 'right')
            fig.tight_layout()
            plt.savefig( f'../figures/classification/separability_violins_{site}_{year}.png',
                        bbox_inches = 'tight')
            plt.close()
    
        # 3. RUN RF CLASSIFICATION
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('Running classification...', end = '\n')
    
        # Splitting the data into training and test data (stratify by the vegetation classes and polygons)
        test_size = 0.2
    
        # Shuffle the label dataframe
        labdat = labdat.sample(frac=1, random_state = seed)
        
        # Split the polygon data into training set (80%) and test set (20%) with stratification by polygon ('id')
        labdat_train, labdat_test = train_test_split(labdat.loc[labdat.region == site],
                                                     test_size = test_size, 
                                                     random_state = seed,
                                                     stratify = labdat.loc[labdat.region == site,'id'])
        
        # Train RF classifier using the indices in featurenames
        dct = fitting_rf_for_region(labdat_train,
                                    featurenames,
                                    excl_low_imp = False, 
                                    plot_importance = False) 
        clf = dct["clf"]
        featurenames = dct["featurenames"]
        
        # predict classes for the test set
        ypred = clf.predict(labdat_test[featurenames])
    
        classes = labdat_test['label'].unique()
        
        # generate confusion matrix
        cfm = confusion_matrix(y_true = labdat_test['label'], 
                               y_pred = ypred, 
                               labels = classes)
        
        # compute accuracy scores
        kappa = cohen_kappa_score(labdat_test['label'], ypred)
        acc_pct = accuracy_score(labdat_test['label'],ypred) * 100
        
        # convert confusion matrix into dataframe
        df_cfm = pd.DataFrame(cfm, index = classes, columns = classes) 
        
        # reorder confusion matrix
        new_order = [n for n in label_order if n in classes]
        df_cfm = df_cfm.reindex(index=new_order, columns=new_order)
        
        # compute user and produces accuracies
        df_cfm['total'] = df_cfm.sum(axis=1)
        df_cfm['PA'] = np.diag(df_cfm) / df_cfm['total'] * 100
        df_cfm.loc['total',new_order] = df_cfm.sum(axis=0)
        df_cfm.loc['UA',new_order] = np.diag(df_cfm.loc[:,new_order]) / df_cfm.loc['total',new_order] * 100
        df_cfm.loc['total','total'] = cfm.sum()
        df_cfm.loc['UA','PA'] = np.diag(cfm).sum() / cfm.sum() * 100 # overall accuracy
        
        # save confusion matrices as csv to the results folder
        if year == 2021:
            df_cfm.round(2).to_csv(f'./tables/results/Table_S{j+4}.csv',sep = ';')
            
        print('Overall accuracy:',acc_pct)
        
        # Save the classification report to a text file
        with open(f'./tables/intermediate/classification_report_{site}_{year}.txt', "w") as f:
            print(featurenames, file = f)
            print(classification_report(y_true = labdat_test['label'], 
                                        y_pred = ypred, 
                                        labels = classes),
                  file = f)
            print('Overall accuracy:',acc_pct, file = f)
            print('Kappa score:',kappa, file = f)
    
        # 3.a) PLOT CONFUSION MATRIX
        if plot_results:
            plt.figure(figsize = (10,7))
            ax = sns.heatmap(df_cfm, annot=True, cmap = 'Blues',fmt = '.0f')
            ax.set_xlabel( 'Predicted', fontdict=dict(weight='bold'))
            ax.set_ylabel( 'True', fontdict=dict(weight='bold'))
            ax.set_title('Cohens Kappa = %.2f' % kappa, fontdict=dict(weight='bold'))
            ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, horizontalalignment = 'right')
            ax.figure.savefig(f'../figures/classification/confusion_matrix_{site}_{year}.png',
                              bbox_inches = 'tight')
            plt.close()
    
    
        # 4. PREDICT SCENE
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if predict:
            dim_msp = pd.DataFrame(msp.reshape(msp.shape[0]*msp.shape[1], msp.shape[2]),
                           columns = ["b","g","r","re","nir"])
            dim_ind = pd.DataFrame(msp_ind.reshape(msp_ind.shape[0]*msp_ind.shape[1], msp_ind.shape[2]), 
                           columns = I_msp_ind.long_name)
            if year == 2021:
                FLIST_GLCM = glob(f'./mosaics/indices/{site}_*GLCM*.tif')
                
                dim_glcm = pd.concat(
                    map(GatherGLCM,FLIST_GLCM),
                    axis = 1,
                    ignore_index=False)
                dim = pd.concat([dim_msp,dim_ind,dim_glcm],axis=1)
            else:
                dim = pd.concat([dim_msp,dim_ind],axis=1)
            
            dim_valid = dim.dropna()
            
            print('Predicting entire map...', end = '\n \n')
            
            # Faster implementation using joblib parallel
            ncores = 3
            split_df = np.array_split(dim_valid[featurenames],ncores)
            start = time.time()
            results = ProgressParallel(n_jobs = ncores)(delayed(clf.predict)(split_df[i])
                                for i in range(ncores)
                                )
            end = time.time()
            print('{:.4f} s'.format(end-start))
            prlab = np.hstack(results)
    
            prlab_entire = np.ones((len(dim)),dtype = object)
            prlab_entire[~dim.isnull().any(axis=1)] = prlab
            
            prlab = prlab_entire.reshape((msp.shape[0],msp.shape[1])) # reshaping prediction back to the image shape
    
            prlab_num = prlab.copy()
            for label,class_code in zip(labdat.loc[labdat.region == site,'label'].unique(),
                                        labdat.loc[labdat.region == site,'Classcode'].unique()):
                print(label +': '+class_code, end = '\n')
                prlab_num[prlab_num == label] = class_code
            prlab_num = prlab_num.astype(int)
    
            # Apply majority filter in a square area
            ncells = 5
            prlab_num_filtered = modal(prlab_num,rectangle(ncells,ncells))
    
            # reconvert class values to nan where original image has no-data values
            prlab_out = np.where(np.ma.masked_equal(msp[:,:,0],0.).mask,
                                 255, prlab_num_filtered)
            prlab_out = img_as_ubyte(prlab_out)
    
            # Saving the MSP image in GeoTiff format
            orig = rasterio.open(FNAME_MSP) # reading out the original coordinates
            new_dataset = rasterio.open( 
                f'./landcover/{site}_{year}_classified_filtered_{ncells}.tif',
                'w',    
                driver='GTiff',
                height=prlab_out.shape[0],
                width=prlab_out.shape[1],
                count=1,
                dtype=np.uint8,
                crs=orig.crs, # coordinates of original image
                transform=orig.transform,
            )
            new_dataset.write(prlab_out,indexes=1)
            new_dataset.close()
        
    print('done.', end = '\n')