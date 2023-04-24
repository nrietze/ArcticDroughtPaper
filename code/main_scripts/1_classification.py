# -*- coding: utf-8 -*-
"""Classify the multispectral drone imagery

<Long description>

Author: Nils Rietze - nils.rietze@uzh.ch
Created: 24.04.2023
"""
year = int(input('Which year do you want to process? \n'))

# imports

import os, time
from glob import glob
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from osgeo import gdal
import geopandas as gpd
from pyproj import Transformer,Proj
import rioxarray as rxr
import rasterio

from skimage.io import imread

from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.model_selection import GroupShuffleSplit

from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report
from sklearn.metrics import make_scorer, accuracy_score

from skimage.morphology import rectangle   # for Structuring Elements (e.g. disk, rectangle)
from skimage.filters.rank import modal # for applying the majority filter

os.chdir(r'C:\Users\nils\OneDrive - Universität Zürich UZH\Dokumente 1\1_PhD\5_CHAPTER1\code\main_scripts')

from elena_functions import fitting_rf_for_region


from tqdm.auto import tqdm

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


def GatherGLCM(FNAME_GLCM):
    glcm = imread(FNAME_GLCM)
    glcm_stats = ["m", "v", "h","d"] # mean, variance, homogeneity, dissimilarity
    glcm_varnames = ['{}_{}'.format(a, FNAME_GLCM.split('_')[2]) for a in glcm_stats]
    df = pd.DataFrame(glcm.reshape(glcm.shape[0]*glcm.shape[1], glcm.shape[2]), 
                   columns = glcm_varnames)
    return df

# %%
predict = False
plot_results = True
run_CV = False

# 0. LOAD TRAINING DATA:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Random points sampled in the training polygons:
FNAME_TRAIN_POINTS = 'C:/data/0_Kytalyk/0_drone/classification/All_Trainingpoints_%s.shp' % year
labdat = gpd.read_file(FNAME_TRAIN_POINTS)
labdat.rename(columns = {'Classname':'label',
                         'x':'lon_utm', 
                         'y':'lat_utm'}, inplace = True)

labdat = labdat.loc[labdat["label"] !='Structures',:]

label_order = [ 'Wet','ledum_moss_cloudberry', 'Dry', 'Shrubs', 'TussockSedge','OpenWater', 'Mud']
labdat['label'] = labdat.label.astype("category").cat.set_categories(label_order, ordered=True)

# Read plot data
plotdat = pd.read_csv(r'C:\Users\nils\OneDrive - Universität Zürich UZH\Dokumente 1\1_PhD\5_CHAPTER1\data\classification_test\plot_data.csv',sep = ';')
plotdat['region'] = plotdat.site
plotdat.head()

def trans_fct(row):
    x1,y1 = p1(row.lon, row.lat)
    x,y = transformer.transform(y1, x1)
    return x,y

transformer = Transformer.from_crs("epsg:4326",'epsg:32655' )
p1 = Proj("epsg:4326")

plotdat.loc[:,['lat_utm','lon_utm']] = plotdat.loc[:,['lat','lon']].apply(trans_fct, 
                                                                          axis=1, 
                                                                          result_type='expand').values

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

for site in ['TLB','Ridge','CBH']: # 
    # Create new subfolder for results
    PATH_RES = r'C:\Users\nils\OneDrive - Universität Zürich UZH\Dokumente 1\1_PhD\5_CHAPTER1\data\classification_data\{}\{}'.format(site,year)
    PATH_OUT = PATH_RES + '/V3'
    # try: # if an earlier version exits, create folder with incremented suffix
    #     all_folders = next(os.walk(PATH_RES))[1]
    #     all_folders.sort()
        
    #     latest = all_folders[-1].replace('V', '')
    #     new = int(latest) + 1
    #     PATH_OUT = PATH_RES + '/V'+str(new)
    #     os.makedirs(PATH_OUT)
        
    #     print('New Folder created:', PATH_OUT)
        
    # except IndexError: # Create first new folder
    #     new = 1
    #     PATH_OUT = PATH_RES + '/V'+str(new)
    #     os.makedirs(PATH_OUT)
        
        # print('New Folder created:', PATH_OUT)
    
    if year == 2020:
        featurenames = ['b','g','BCC', 'GCC', 'RCC', 'NDVI']
    else:
        featurenames = ['b','g','BCC', 'GCC', 'RCC', 'NDVI',
                        'v_bcc','m_green','h_red','m_red'] 
        # featurenames = ['r','b','g','BCC', 'GCC', 'RCC', 'NDVI',
        #                 'm_bcc', 'v_bcc', 'h_bcc', 'd_bcc', 'm_blue', 'v_blue', 'h_blue',
        #                 'd_blue', 'm_gcc', 'v_gcc', 'h_gcc', 'd_gcc', 'm_green', 'v_green',
        #                 'h_green', 'd_green', 'm_ndvi', 'v_ndvi', 'h_ndvi', 'd_ndvi', 'm_nir',
        #                 'v_nir', 'h_nir', 'd_nir', 'm_rcc', 'v_rcc', 'h_rcc', 'd_rcc',
        #                 'm_rededge', 'v_rededge', 'h_rededge', 'd_rededge', 'm_red', 'v_red',
        #                 'h_red', 'd_red'] 

    print('Processing {}...'.format(site), end='\n')
    
    # 1. LOAD IMAGE DATA
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('Loading data...', end = '\n')
    
    PATH = "C:/data/0_Kytalyk/0_drone/{}/{}/".format(year, site)
    os.chdir(PATH)
 
    
    # Load Multispectral reflectances:
    FNAME_MSP = glob("MSP/resampled/reflectance*.tif")[0]
    
    if not os.path.exists(FNAME_MSP): # Check if a multispectral image stack already exists, if not interrupt
        print('Please resample all multispectral files for {} {}!'.format(site,year), end='\n')
        break
    
    msp = imread(FNAME_MSP)
    msp[msp==-10000.] = np.nan
    msp = np.ma.masked_equal(msp,0.) # Mask no data values (0) 
    I_msp = rxr.open_rasterio(FNAME_MSP)
    

    # Load Multispectral indices:
    FNAME_MSP_INDEXSTACK = "MSP/indices/{}_{}_indices_stack.tif".format(site,year)
    # FNAME_MSP_INDEXSTACK = "MSP/indices/TLB_2021_indices_stack.tif"
    
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
        # plt.savefig(PATH_OUT + '/separability_violins_{}_{}.png'.format(site,year),
        #             bbox_inches = 'tight')
        plt.close()

    # 3. RUN RF CLASSIFICATION
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('Running classification...', end = '\n')

    # Splitting the data into training and test data (stratify by the vegetation classes and polygons)
    seed = 15  
    test_size = 0.2

    # # Create the GroupShuffleSplit object
    # gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    
    # labdat_subset = labdat.loc[labdat.region == site]
    
    # # keep splitting until each label is contained in both splits
    # while True:
    #     for train_idx, test_idx in gss.split(labdat_subset, groups=labdat_subset['id']):
    #         labdat_train = labdat_subset.iloc[train_idx]
    #         labdat_test = labdat_subset.iloc[test_idx]
        
    #     # check if each label is present in both splits
    #     train_labels = set(labdat_train['label'].unique())
    #     test_labels = set(labdat_test['label'].unique())
    #     if train_labels == test_labels == set(labdat_subset['label'].unique()):
    #         break
        
    # # Shuffle the train and test dataframes
    # labdat_train = labdat_train.sample(frac=1)
    # labdat_test = labdat_test.sample(frac=1)

    labdat_train, labdat_test = train_test_split(labdat.loc[labdat.region == site],
                                                  test_size = test_size, 
                                                  random_state = seed,
                                                  stratify = labdat.loc[labdat.region == site,'id'])
    
    # Train RF classifier
    dct = fitting_rf_for_region(labdat_train,
                                featurenames,
                                excl_low_imp = False, 
                                plot_importance = False) 
    clf = dct["clf"]
    featurenames = dct["featurenames"]
    ypred = clf.predict(labdat_test[featurenames])

    classes = labdat_test['label'].unique()
    
    lbl = {'OpenWater':'Open water',
           'Mud': 'Mud',
           'Wet': 'LW1',
           'ledum_moss_cloudberry': 'LW2', 
           'Shrubs': 'HP2', 
           'Dry': 'HP1',
           'TussockSedge': 'TS'}
    lbl_names = [lbl[c] for c in classes]
    
    cfm = confusion_matrix(y_true = labdat_test['label'], 
                           y_pred = ypred, 
                           labels = classes)

    kappa = cohen_kappa_score(labdat_test['label'], ypred)
    acc_pct = accuracy_score(labdat_test['label'],ypred) * 100
    pr_total = np.mean(np.diag(cfm) / cfm.sum(axis=0))
    
    # confusion matrix into dataframe
    df_cfm = pd.DataFrame(cfm, index = lbl_names, columns = lbl_names) 
    
    # reorder confusion matrix
    label_order = ['HP1','HP2','LW1','LW2','TS','Open water','Mud']
    new_order = [n for n in label_order if n in lbl_names]
    df_cfm = df_cfm.reindex(index=new_order, columns=new_order)
    
    df_cfm['total'] = df_cfm.sum(axis=1)
    df_cfm['PA'] = np.diag(df_cfm) / df_cfm['total'] * 100
    df_cfm.loc['total',new_order] = df_cfm.sum(axis=0)
    df_cfm.loc['UA',new_order] = np.diag(df_cfm.loc[:,new_order]) / df_cfm.loc['total',new_order] * 100
    df_cfm.loc['total','total'] = cfm.sum()
    df_cfm.loc['UA','PA'] = np.diag(cfm).sum() / cfm.sum() * 100 # overall accuracy
    os.chdir(PATH_OUT)
    
    df_cfm.round(2).to_csv(f'./confusion_matrix{site}.csv',sep = ';')
    print('Total precision:',pr_total)
    
    with open("output.txt", "a") as f:
        print(featurenames, file = f)
        print(classification_report(y_true = labdat_test['label'], 
                                    y_pred = ypred, 
                                    labels = lbl_names),
              file = f)
        print('Total precision:',pr_total, file = f)

    # 3.a) PLOT CONFUSION MATRIX
    if plot_results:
        plt.figure(figsize = (10,7))
        ax = sns.heatmap(df_cfm, annot=True, cmap = 'Blues',fmt = '.0f')
        ax.set_xlabel( 'Predicted', fontdict=dict(weight='bold'))
        ax.set_ylabel( 'True', fontdict=dict(weight='bold'))
        # ax.set_title('Cohens Kappa = %.2f' % kappa, fontdict=dict(weight='bold'))
        ax.set_title('Overall accuracy = {:.1f} %'.format(acc_pct), fontdict=dict(weight='bold'))
        ax.set_xticklabels(ax.get_xticklabels(), rotation = 30, horizontalalignment = 'right')
        ax.figure.savefig(r'cfm_{}_{}.png'.format(site,year),
                          bbox_inches = 'tight')
        plt.close()

    # 4. CROSS-VALIDATE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    
    if run_CV:
        print('Cross validating...', end = '\n')

        scoring = {"Accuracy": make_scorer(accuracy_score)}
        logo = LeaveOneGroupOut()
    
        scores = cross_val_score(clf, n_jobs = 3, 
                                X = labdat_test[featurenames], 
                                y = labdat_test["label"],
                                cv= logo, groups = labdat_test.id,
                                scoring = make_scorer(accuracy_score))
        with open("output.txt", "a") as f:
            print("%0.2f accuracy with a standard deviation of %0.2f" % 
                  (scores.mean(), scores.std()), file = f)
            print(scores, file = f)
    
        # Use cross_validate, if estimator, time, etc. is needed
        # scores = cross_validate(clf, n_jobs = 3,
        #                         X = labdat_train[featurenames], 
        #                         y = labdat_train["label"],
        #                         cv= logo, groups = labdat_train.id,
        #                         scoring = make_scorer(accuracy_score))
        # print("%0.2f accuracy with a standard deviation of %0.2f" % 
        #       (scores['test_score'].mean(), scores['test_score'].std()))        


    # 5. PREDICT SCENE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if predict:
        from skimage.util import img_as_ubyte
        os.chdir(PATH)
        
        dim_msp = pd.DataFrame(msp.reshape(msp.shape[0]*msp.shape[1], msp.shape[2]),
                       columns = ["b","g","r","re","nir"])
        dim_ind = pd.DataFrame(msp_ind.reshape(msp_ind.shape[0]*msp_ind.shape[1], msp_ind.shape[2]), 
                       columns = I_msp_ind.long_name)
        if year == 2021:
            FLIST_GLCM = glob('MSP/indices/*GLCM*.tif')
            
            dim_glcm = pd.concat(
                map(GatherGLCM,FLIST_GLCM),
                axis = 1,
                ignore_index=False)
            dim = pd.concat([dim_msp,dim_ind,dim_glcm],axis=1)
        else:
            dim = pd.concat([dim_msp,dim_ind],axis=1)
        dim['sumb'] = dim["r"] + dim["g"] + dim["b"]
        
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

        # start = time.time()
        # prlab = clf.predict(dim[featurenames]) # predicting labels for the image
        # end = time.time()
        # print('{:.4f} s'.format(end-start))
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
            PATH_OUT + '/{}_{}_classified_filtered_new{}.tif'.format(site,year,ncells),
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