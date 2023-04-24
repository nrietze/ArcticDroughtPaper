"""Custom functions

<Long description>

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
from osgeo import gdal
import rasterio # for saving GeoTiff image


def fitting_rf_for_region(labdat, featurenames, excl_low_imp, plot_importance = False):
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


def image_minmax(im, imax=None, imin=None): # performs image minmax normalization
    if imax is None:
        imax = im.max()
        imin = im.min()       
    im = (im - imin)/(imax - imin)
    return im