# -*- coding: utf-8 -*-
"""
Classify the multispectral drone imagery

Author: Nils Rietze - nils.rietze@uzh.ch
Created: 24.04.2023
"""
# imports
import pandas as pd
import numpy as np
import datetime
from glob import glob

import matplotlib.pyplot as plt
from cmcrameri import cm
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sns

from statsmodels.stats.multicomp import MultiComparison
from scipy.stats import ttest_ind
from sklearn.metrics import r2_score

import skgstat as skg
from osgeo import gdal
import rioxarray as rxr
import xarray as xr

import os
os.chdir(r'C:\Users\nrietze\Documents\1_PhD\5_CHAPTER1\paper\github\code\analysis')

from FigFunctions import read_fluxtower,GetMeanTair,PrepRasters,MeltArrays,ScaleMinMax,DifferenceTest,pretty_table,ProgressParallel
from FigFunctions import PlotDensities,PlotBoxWhisker,PlotFcoverVsTemperature,MapFcover_np,PolyFitSensorT,GatherData

# LOAD DATA & PREALLOCATIONS
#----

# configure variables
seed = 15
study_area_sidelength = 400 # in m
dxy = int(study_area_sidelength / .15) # 667 pixels along 100m ==> 1 ha


# Class labels and values for land cover raster
classdict = {0:'HP1',
             1:'LW1',
             2:'HP2',
             3:'water',
             4:'LW2',
             5:'mud',
             7:'TS',
             255:'nodata'}

# Color scheme for plots
colordict = dict(HP1 = '#AAFF00', 
                 HP2 = '#38A800', 
                 LW1 = '#00FFC5',
                 LW2 = '#A90EFF',
                 TS = '#3D3242',
                 mud = '#734C00',
                 water = '#005CE6')

print('Loading data...')

os.chdir('../../../data/')

df_fluxtower = read_fluxtower('./tables/CR3000_Kytalyk_Meteo.dat')

parser = lambda date: pd.to_datetime(date,format='%d.%m.%Y %H:%M').tz_localize('Asia/Srednekolymsk')
df_flighttimes = pd.read_csv('./tables/flight_times.csv', 
                             sep = ';', 
                             parse_dates = [4,5],
                             date_parser=parser)

sites = ['CBH', 'TLB', 'Ridge']

# Create empty dataframe for flight conditions
df_tab1 = pd.DataFrame(columns = ['year','site','mean_Tsurf','mean_Tair','mean_SWin','mean_windspeed','mean_Tsurf_Tair','mean_wdi'])

df_list = list(range(3))

I_cl_list = list()
I_wdi_20_list = list()
I_wdi_21_list = list()

# Create the index levels
years = [2020, 2021] * 3
multi_index = list(zip(years, np.repeat(sites ,2)))
multi_index = pd.MultiIndex.from_tuples(multi_index, names=['Year', 'Site'])

# create output directories if they don't exist yet:
if not os.path.exists('./tables/results/'):
    os.makedirs('./tables/results/')
if not os.path.exists('./tables/intermediate/'):
    os.makedirs('./tables/intermediate/')
 
Plot_Mosaic_Panels = True   
 
if Plot_Mosaic_Panels:
    # source = 'LC' ,'WDI'
    source = 'TIR' 
    
    nrows = 2
    fig_size = (24,17)
    
    if source == 'TIR':
        cmap = cm.lajolla_r
        vmin = 15; vmax = 35
        fname = '../figures/Fig_S7.png' 
    
    elif source == 'WDI':
        cmap = cm.bilbao
        vmin = 0; vmax = .4
        nrows = 1
        fig_size = (24,9)
        fname = '../figures/Fig_S13.png'
        
    else: 
        fname = '../figures/Fig_S8.png'
        
    # Create a custom color map based on the color dictionary
    color_dict = {'0.0': '#AAFF00', # HP1
                  '1.0': '#00FFC5', # LW1
                  '2.0': '#38A800', # HP2
                  '3.0': '#005CE6', # water
                  '4.0': '#A90EFF', # LW2
                  '5.0': '#734C00', # mud
                  '6.0': '#3D3242', # TS (6 for the overview panel plot, colormaps need to have ranges for values)
                  'nan': '#FFFFFF'}
    
    # Figure instance for mosaic overviews
    sns.set_theme(style="ticks",font_scale=3)
    fig,axs = plt.subplots(nrows= nrows,ncols = 3, sharex = False, sharey = False, figsize = fig_size)  

# Iterate through sites to load data
for i,site in enumerate(sites):
    PATH_20_TIR = rf'.\mosaics\{site}_thermal_2020_resampled.tif'
    PATH_21_TIR = rf'.\mosaics\{site}_thermal_2021_resampled.tif'
    
    PATH_MSP_20 = rf'.\mosaics\{site}_msp_2020_resampled.tif'
    PATH_MSP_21 = rf'.\mosaics\{site}_msp_2021_resampled.tif'
        
    PATH_CL = rf'.\landcover\{site}_2021_classified_filtered5.tif'
    PATH_WATERMASK = rf'.\landcover\{site}_2020_classified_filtered5.tif'
        
    # Define corner coordinates of study area extent
    if site == 'Ridge':
        ymin = 200
        xmin = 1200
    else:
        ymin = 300
        xmin = 200
    
    ymax = ymin + dxy
    xmax = xmin + dxy
    
    # Set extent
    extent = {'xmin': xmin,'xmax': xmax,
              'ymin': ymin,'ymax': ymax}
    
    # Retrieve mean air temperature fro flux tower data during drone flights
    T_air_20 = GetMeanTair(df_flighttimes, df_fluxtower, site, 2020)
    T_air_21 = GetMeanTair(df_flighttimes, df_fluxtower, site, 2021)
    
    # Read and subset raster data
    I_cl_s,I_wm_s, I_tir_20_s, I_tir_21_s, I_wdi_20_s, I_wdi_21_s,I_ndvi_20_s, I_ndvi_21_s = PrepRasters(PATH_CL,PATH_WATERMASK,
                                                                                  PATH_20_TIR,T_air_20,
                                                                                  PATH_21_TIR,T_air_21,
                                                                                  PATH_MSP_20,PATH_MSP_21,
                                                                                  extent)
    
    
    names = [classdict[i] for i in np.unique(I_cl_s)]
    
    # Replace no data values (255) with nan
    I_cl_s = np.where(I_cl_s == 255,np.nan, I_cl_s)
    
    I_cl_list.append( I_cl_s)
    I_wdi_20_list.append(I_wdi_20_s)
    I_wdi_21_list.append(I_wdi_21_s)
    
    # write flight meteo conditions to dataframe
    row1 = [2020,site, 
           I_tir_20_s.mean(), # extract mean Tsurf from thermal image
           T_air_20,
           GetMeanTair(df_flighttimes, df_fluxtower, site, 2020,returnSW=True,returnTemp=False,returnWS=False), #SWin
           GetMeanTair(df_flighttimes, df_fluxtower, site, 2020,returnSW=False,returnTemp=False,returnWS=True), #Wind speed
           (I_tir_20_s - T_air_20).mean(),
           I_wdi_20_s.mean()
           ]
    df_tab1.loc[len(df_tab1)] = row1
    row2 = [2021,site, 
           I_tir_21_s.mean(), # extract mean Tsurf from thermal image
           T_air_21,
           GetMeanTair(df_flighttimes, df_fluxtower, site, 2021,returnSW=True,returnTemp=False,returnWS=False), #SWin
           GetMeanTair(df_flighttimes, df_fluxtower, site, 2021,returnSW=False,returnTemp=False,returnWS=True), #Wind speed
           (I_tir_21_s - T_air_21).mean(),
           I_wdi_21_s.mean()
           ]
    df_tab1.loc[len(df_tab1)] = row2
    
    # Plot overview of drone data in panels
    if Plot_Mosaic_Panels:
        # Plot thermal data
        if source == 'TIR':
            I_src_20 = I_tir_20_s
            I_src_21 = I_tir_21_s
            
        # Plot WDI difference maps
        elif source == 'WDI':
            I_src_20 = I_wdi_20_s
            I_src_21 = I_wdi_21_s
        
        # Plot land cover classes
        else:
            I_src_20 = np.where(I_wm_s == 9, np.nan , I_wm_s)
            I_src_21 = np.where(I_cl_s == 7, 6 , I_cl_s)
            
            unique_values = np.unique(I_src_21.flatten())
            
            if site == 'CBH':
                unique_values_cbh = unique_values
            
            colors = [color_dict[str(value)] for value in unique_values]
            cmap = ListedColormap(colors)
            vmin = 0
            vmax = max(unique_values) + 1 if site != 'Ridge' else 3
        
        if source == 'WDI':
            # Plot 2020 drone data
            p = axs[i].imshow(I_src_20 - I_src_21,cmap = cmap, 
                                 aspect = 'equal',
                                 interpolation = 'nearest',
                                 vmin = vmin, vmax = vmax)
    
            axs[i].set_title( site, fontweight = 'bold')
        
        else:
            # Plot 2020 drone data
            p = axs[0][i].imshow(I_src_20,cmap = cmap, 
                                 aspect = 'equal',
                                 interpolation = 'nearest',
                                 vmin = vmin, vmax = vmax)
    
            axs[0][i].set_title( site, fontweight = 'bold')
            
            # Plot 2021 drone data
            axs[1][i].imshow(I_src_21,cmap = cmap, 
                             interpolation = 'nearest',
                             aspect = 'equal',
                             vmin = vmin, vmax = vmax)
    
        for ax in axs.flat:
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
            
            ax.tick_params(axis='both', which='both', 
                            labelleft = False,labelbottom = False,
                            length=0)
        
    # check if a csv of image data already exists:
    try:
        df_m_s = pd.read_csv(f'./tables/intermediate/{site}_data_thermal.csv',sep=';')
        print('Data for this site is available. Loading ...')
        
    except:
        # if it doesn't exist, compile the samples:
        print('Sampling thermal data...')
        
        # Reformatting TIR mosaics to long dataframes
        df_tir = pd.concat(
            map(lambda arr,year: MeltArrays(arr,I_cl_s,names,year,'deltaT'),
                [I_tir_20_s - T_air_20,I_tir_21_s - T_air_21],[2020,2021]), 
            ignore_index=False)
        
        # Reformatting NDVI mosaics to long dataframes
        df_ndvi = pd.concat(
            map(lambda arr,year: MeltArrays(arr,I_cl_s,names,year,'ndvi'),
                [I_ndvi_20_s,I_ndvi_21_s],[2020,2021]), 
            ignore_index=False)
        df_tir['ndvi'] = df_ndvi.ndvi
        
        # Reformatting water deficit index to long dataframes:
        df_wdi = pd.concat(
            map(lambda arr,year: MeltArrays(arr,I_cl_s,names,year,'wdi'),
                [I_wdi_20_s,I_wdi_21_s],[2020,2021]), 
            ignore_index=False)
        df_tir['wdi'] = df_wdi.wdi
        
        df_wdi_diff = MeltArrays(I_wdi_20_s - I_wdi_21_s,
                                 I_cl_s,names,2021,'diff_wdi')
        df_tir['diff_wdi'] = df_wdi_diff.diff_wdi
        
        
        df_ndvi_diff = MeltArrays(I_ndvi_20_s - I_ndvi_21_s,
                                 I_cl_s,names,2021,'diff_ndvi')
        df_tir['diff_ndvi'] = df_ndvi_diff.diff_ndvi
        
        if site == 'CBH':
            from skimage.io import imread

            I_tpi_cbh = imread(r'C:/Users/nrietze/Documents/1_PhD/5_CHAPTER1/data/wetness_index/TPI_CBH.tif')
            I_tpi_cbh_s =  I_tpi_cbh[extent['ymin']:extent['ymax'],extent['xmin']:extent['xmax']]
            
            df_tpi = MeltArrays(I_tpi_cbh_s,I_cl_s,names,2021,'tpi')
            df_tir['tpi'] = df_tpi.tpi
        
        # Generate random sample of size n per class and year:
        n = 20000
        
        df_tir = df_tir.dropna()
        unique_classes = df_tir['variable'].unique()
        
        selected_indices = []
        
        for cls in unique_classes:
            mask1 = np.logical_and(df_tir['variable'] == cls, df_tir['year'] == 2020)
            cls_indices_array1 = np.random.choice(df_tir.loc[mask1].index, 
                                                  size=min(n, len(df_tir.loc[mask1].index)), 
                                                  replace=False)
            # mask2 = np.logical_and(df_tir['variable'] == cls, df_tir['year'] == 2021)
            # cls_indices_array2 = np.random.choice(df_tir.loc[mask2].index,
            #                                       size=min(n, len(df_tir.loc[mask2].index)), 
            #                                       replace=False)
            selected_indices.extend(cls_indices_array1)
            # selected_indices.extend(cls_indices_array2)
        
        df_m_s = df_tir.loc[selected_indices]
        
        # Export to csv
        df_m_s.to_csv(f'./tables/intermediate/{site}_data_thermal.csv',sep=';')
        
        # Generate descripitve statistics and save as csv
        df_m_s_stats = df_m_s.groupby(['year','variable']).describe()
        
        def iqr_func(col):
            """
            Compute interquartile range
            """
            return np.nanquantile(col, q=0.75) - np.nanquantile(col, q=0.25)
        
        for main_col in df_m_s_stats.columns.levels[0]:
            df_m_s_stats[(main_col, 'iqr')] = df_m_s.groupby(['year','variable']).agg(iqr_func)[main_col]
        
        # Export table of descriptive statistics
        df_m_s_stats.to_csv(fr'./tables/results/Table_S_{site}.csv',sep = ';')
    
    # Sort community labels along deltaT gradient in 2021
    label_order = df_m_s.loc[df_m_s.year == 2021].groupby('variable').deltaT.mean().sort_values().index.to_list()
         
    # Set 'variable' column (community labels) to categorical
    df_m_s['variable'] = df_m_s.variable.astype("category").cat.set_categories(label_order, ordered=True)
    
    # Store sampled dataframe to list
    df_list[i] = df_m_s
    
    print('done.')

if Plot_Mosaic_Panels:
    
    if not source == 'WDI':
        # Add year labels on y-axis
        fig.text(0, 0.75, 2020, va='center', rotation='vertical', fontweight = 'bold')
        fig.text(0, 0.25, 2021, va='center', rotation='vertical', fontweight = 'bold')
        
        # Add a scale bar below the bottom center subplot
        # bottom_center_ax = fig.add_subplot(3, 2, 5)  # 3 rows, 2 columns, position 5
        bottom_center_ax = fig.add_axes([0.345, -0.01, 0.315, 0.05])
        bottom_center_ax.axhline(3000, color='gray', linestyle='-',lw = 20)
        bottom_center_ax.set_axis_off()
        
        # Add a text label below the line
        bottom_center_ax.text(0.5, -0.25, '400 m',color='gray',
                              ha='center', va='center', transform=bottom_center_ax.transAxes)

    else:
        # Add a scale bar below the bottom center subplot
        bottom_center_ax = fig.add_axes([0.348, 0.01, 0.305, 0.05])
        bottom_center_ax.axhline(3000, color='gray', linestyle='-',lw = 20)
        bottom_center_ax.set_axis_off()
        
        # Add a text label below the line
        bottom_center_ax.text(0.5, -0.2, '400 m',color='gray',
                              ha='center', va='top', transform=bottom_center_ax.transAxes)
    # despine figure
    # sns.despine()
    
    # Add the colorbar for thermal data
    if source == 'TIR':
        # Create the colorbar axes below the subplots
        cax = fig.add_axes([0.2, -0.1, 0.6, 0.05])
        
        cbar = plt.colorbar(p, cax=cax, 
                            orientation='horizontal',
                            extend = 'both',
                            label = 'Surface temperature (°C)')
        cbar.set_ticks(np.arange(vmin,vmax+.1,5))  # Example tick positions
        
        cbar.outline.set_visible(False)
        
    # Add the colorbar for WDI change
    elif source == 'WDI':
        # Create the colorbar axes below the subplots
        cax = fig.add_axes([0.2, -0.1, 0.6, 0.05])
        
        cbar = plt.colorbar(p, cax=cax, 
                            orientation='horizontal',
                            extend = 'both',
                            label = 'Difference in water deficit index (-)')
        cbar.set_ticks(np.arange(vmin,vmax+.1, .1))  # Example tick positions
        
        cbar.outline.set_visible(False)
        
    # Add legend for land cover map
    else:
        lbl = {'3.0':'Open water',
               '5.0': 'Mud',
               '1.0': 'LW1: Low-centered \nwetland complex (bottom)',
               '4.0': 'LW2: Low-centered \nwetland complex (top)', 
               '2.0': 'HP2: High-centered \npolygons (dwarf birch)', 
               '0.0': 'HP1: High-centered \npolygons (dry sedges & lichen)',
               '6.0': 'TS: Tussock – sedge'}
        
        patches = [mpatches.Patch(color=color_dict[str(value)], label=lbl[str(value)]) for value in unique_values_cbh[:-1]]
        fig.legend(handles=patches,
                   frameon=False, 
                   loc='lower center',
                   ncol = 3,
                   title = 'Land cover',
                   bbox_to_anchor = (0.5, -0.3))
        
    # Adjust spacing between subplots and colorbar
    plt.subplots_adjust(top = 0.8,bottom=0.2,
                        left = 0.1, right = 0.8,
                        hspace=0.1, wspace=0.1)  
    fig.tight_layout()
    
    
    plt.savefig(fname,dpi = 200,bbox_inches = 'tight')

# write flight conditions to csv for Table 1 in main paper
df_tab1.to_csv('../data/tables/results/Table_1.csv',sep = ';')

# # %% 1. DENSITY PLOT IN NORMAL YEAR & DIFFERENCE TESTS
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# group_var = 'variable'
# xvar = 'deltaT'
# xlim = [0,25]

# for i,site in enumerate(sites):
#     print(f'Generating subplot in {site} for Figure 2...')   
    
#     df_m_s = df_list[i]
    
#     label_order = df_m_s.variable.cat.categories.to_list()
    
#     # Perform Tukey HSD
#     # sample 1 % of observations (= 200) per plant community and year
#     df_sample = df_m_s.groupby(['variable','year']).sample(frac = 0.1,
#                                                            random_state = seed)
    
#     mc20,mc21 = df_sample.loc[(df_sample.variable != 'water') & 
#                               (df_sample.variable != 'mud')].groupby(['year']).apply(lambda x: MultiComparison(x[xvar], x[group_var]))
    
#     # Save test results in dataframe (for significance brackets in plot)
#     thsd = pd.DataFrame(data = mc21.tukeyhsd()._results_table.data[1:], 
#                         columns = mc21.tukeyhsd()._results_table.data[0])
    
#     print('2021: \n', mc21.tukeyhsd(), end = '\n')                                                                                       
                                                                                               
#     PATH_SAVE = f'../figures/Fig_2_{site}.png'
    
#     # Plot Densities in 2021
#     ax = PlotDensities(df_m_s, xvar,PATH_SAVE,
#                        xlim = xlim,
#                        colors = colordict,
#                        showSignif = True, data_pairtest = thsd,
#                        order = label_order,
#                        showWater = False,
#                        showBothYears = False,
#                        saveFig = True)
    
#     plt.show()
    
#     # export results form differnce test in 2021 
#     # thsd.to_csv(f'./tables/results/Table_S8_2021_{site}.csv', sep=';', index = False)
    
#     # Plot Densities in 2020 and 2021
#     thsd = pd.DataFrame(data = mc20.tukeyhsd()._results_table.data[1:], 
#                         columns = mc20.tukeyhsd()._results_table.data[0])
    
#     PATH_SAVE = f'../figures/Fig_2_both_{site}.png'
#     ax = PlotDensities(df_sample, xvar,PATH_SAVE,
#                         xlim = [-5,25],
#                         colors = colordict,
#                         showSignif = False, data_pairtest = thsd,
#                         order = label_order,
#                         showWater = True,
#                         showBothYears = True,
#                         saveFig = False)
    
#     plt.show()
    
#     # export results form differnce test in 2020 
#     # thsd.to_csv(f'./tables/results/Table_S8_2020_{site}.csv', sep=';', index = False)
    
#     print('done.')

# %% Figure 2 - Density plots of Ts - Ta in 2021 across all sites
# ----

group_var = 'variable'
xvar = 'deltaT'
xlim = [7,25] if xvar == 'deltaT' else None
PATH_OUT = '../figures/Fig_2.eps' 
lw = 5
showSignif = True

fig, axs = plt.subplots(nrows = 3, figsize=(30,30), dpi = 200,sharex = True) # create subplot instance

# Create the subplots
labs = ['(a)','(b)','(c)']

for i,site in enumerate(sites):
    print(f'Generating subplot in {site} for Figure 1...')   
    
    df_m_s = df_list[i]
    
    label_order = df_m_s.variable.cat.categories.to_list()
    
    # Perform Tukey HSD
    # sample 1 % of observations (= 200) per plant community and year
    df_sample = df_m_s.groupby(['variable','year']).sample(frac = 0.1,
                                                           random_state = seed)
    
    mc20,mc21 = df_sample.loc[(df_sample.variable != 'water') & 
                              (df_sample.variable != 'mud')].groupby(['year']).apply(lambda x: MultiComparison(x[xvar], x[group_var]))
    
    # Save test results in dataframe (for significance brackets in plot)
    thsd = pd.DataFrame(data = mc21.tukeyhsd()._results_table.data[1:], 
                        columns = mc21.tukeyhsd()._results_table.data[0])
    
    print('2021: \n', mc21.tukeyhsd(), end = '\n')                                                                                       
                                                                                               
    # Plot Densities in 2021
    h, l, ax = PlotDensities(df_m_s, xvar,
                             ax = axs[i],
                             xlim = xlim,
                             colors = colordict,
                             showSignif = showSignif, 
                             data_pairtest = thsd,
                             order = label_order,
                             showWater = False,
                             showBothYears = False,
                             PATH_OUT = 'PATH_SAVE',
                             saveFig = False)
        
    if site == 'CBH':
        handles = h
        labels = l
    
    ax.set(ylabel = '')
    
    ax.get_legend().remove()
    
    axs[i].text(-.05, 1.15, labs[i] + ' ' + site, transform=axs[i].transAxes, weight='bold')
 
# Set the y-label for the entire figure
if showSignif:
    fig.text(- 0.01, 0.5, 'Kernel density estimate', va='center', rotation='vertical')
    xy_leg1 = (0.5, -0.02)
else:
    fig.text(0.04, 0.5, 'Kernel density estimate', va='center', rotation='vertical')
    xy_leg1 = (0.5, -0.1)
    
# Plot the legend below the third panel
# legend for the plant communities
leg1 = fig.legend(handles[:6], labels[:6], 
                    loc='upper center', 
                    frameon=False,
                    # mode = 'expand',
                    title = 'Plant community',
                    bbox_to_anchor = xy_leg1, 
                    ncol=2)
# legend for the years
leg2 = fig.legend(handles[7:], labels[7:], 
                    loc='upper center', 
                    frameon=False,
                    # mode = 'expand',
                    bbox_to_anchor=(0.5, 0), 
                    ncol=2)

# Adjust the position of the legend within the figure
fig.add_artist(leg1)
fig.add_artist(leg2)
fig.tight_layout()
fig.savefig(PATH_OUT,bbox_inches = 'tight')

plt.show() 
   

# %% Figure 3 & Figure S12 - Boxplot of WDI in both years
#----

yvar = 'wdi'

# Lists for the test result dataframes
tt_list = list()
thsd_list = list()
wdi_diff_thsd_list = list()
iqr_list = list()

# create subplot instance for the TLB & Ridge plot that go to the suppl. materials
fig, axs = plt.subplots(nrows = 2, figsize=(30,20), dpi = 200) 

# Adjust spacing between subplots and colorbar
plt.subplots_adjust(hspace=.6) 

# Create the subplots
labs = ['(a)','(b)']

# create dummy subplot for CBH first
_, _= plt.subplots() 

for i,site in enumerate(sites):
    print(f'Generating subplot in {site} for Figure 3...')   
    
    df_m_s = df_list[i]
    
    label_order = df_m_s.variable.cat.categories.to_list()
        
    # Generate descripitve statistics of water deficit index and save as csv
    df_wdi_iqr = df_m_s.groupby(['year','variable'])['wdi'].quantile([.25,.75]).unstack()
    df_wdi_iqr['iqr'] = df_wdi_iqr[.75] - df_wdi_iqr[.25]
    
    # Unstack the first level of the MultiIndex column
    df_wdi_iqr = df_wdi_iqr.unstack(level=0)
    
    # Swap the levels of the MultiIndex
    df_wdi_iqr = df_wdi_iqr.swaplevel(0, 1, axis=1)
    
    df_wdi_iqr['site'] = site
    
    # write IQR to list
    iqr_list.append(df_wdi_iqr)
    
    # sample random observations per plant community and year to avoid effect of large sample
    df_sample = df_m_s.groupby(['variable','year']).sample(frac = 0.1)
    
    # Welch's ttest for unequal variances between years
    alpha = .01
    ttest = df_sample.groupby(['variable']).apply(lambda x: ttest_ind(x.loc[x.year==2020,yvar],
                                                                      x.loc[x.year==2021,yvar], 
                                                                      equal_var=False))
    df_ttest = pd.DataFrame([[z,p] for z,p in ttest.values],columns = ['tstat','pval']).set_index(ttest.index)
    df_ttest['reject'] = df_ttest.pval < alpha
    df_ttest.loc[df_ttest.pval < 0.01,'text'] = '< 0.01'
    
    df_ttest['site'] = site
    
    # Mean WDI differences between years for each group
    df_ttest['meandiff'] = df_sample.groupby(['variable']).apply(lambda x: round(x.loc[x.year == 2020,'wdi'].mean() - x.loc[x.year == 2021,'wdi'].mean(),2))
    
    df_ttest['mean_wdi_2021'] = df_sample.groupby(['variable']).apply(lambda x: round(x.loc[x.year == 2021,'wdi'].mean(),2))
    
    # Relative increase in mean WDI from 2021 to 2020 (% increase)
    df_ttest['pct_change'] = round(df_ttest['meandiff'] / df_ttest['mean_wdi_2021'] * 100,2)
    
    # write t-test table to list
    tt_list.append( df_ttest)
    
    PATH_SAVE = '../figures/Fig_3.eps'
    
    if site == 'CBH':
        # Plot CBH boxplot individually for main text
        ax = PlotBoxWhisker(df_m_s, yvar,
                           label_order = label_order,
                           colors = colordict,
                           showSignif=True,
                           data_ttest = df_ttest,
                           showWater = True,
                           PATH_OUT=PATH_SAVE,
                           saveFig = True)
        plt.show()
    else:
        ax = PlotBoxWhisker(df_m_s, yvar,
                            ax = axs[i-1],
                            label_order = label_order,
                            colors = colordict,
                            showSignif=True,
                            data_ttest = df_ttest,
                            showWater = True,
                            PATH_OUT=PATH_SAVE,
                            saveFig = False)
        axs[i-1].text(-0.1, 1.3, labs[i-1] + ' ' + site, 
                      transform=axs[i-1].transAxes, weight='bold')
    
    mc20,mc21 = df_sample.loc[(df_sample.variable != 'water') & 
                              (df_sample.variable != 'mud')].groupby(['year']).apply(lambda x: MultiComparison(x[yvar], x['variable']))

    # Save test results in dataframe (for significance brackets in plot)
    for i,mc_table in enumerate([mc20,mc21]):
        thsd = pd.DataFrame(data = mc_table.tukeyhsd()._results_table.data[1:], 
                            columns = mc_table.tukeyhsd()._results_table.data[0])
        
        print(f'{2020+i}: \n', mc_table.tukeyhsd(), end = '\n')
        
        thsd['site'] = site
        
        # append results to list
        thsd_list.append(thsd)
        
    # Multicomparison of WDI differences per plant community
    mask = np.logical_and(df_m_s.loc[df_m_s.year == 2020].variable != 'water',
                          df_m_s.loc[df_m_s.year == 2020].variable != 'mud'
                          )
    df_sample_2 = df_m_s.loc[df_m_s.year == 2020].loc[mask].sample(frac = 0.01,random_state = seed)
    
    mc_wdi = MultiComparison(df_sample_2['diff_wdi'], df_sample_2['variable'])
    print('Difference test for WDI changes:',mc_wdi.tukeyhsd(), end = '\n')
    
    # Save test results in dataframe (for significance brackets in plot)
    thsd = pd.DataFrame(data = mc_wdi.tukeyhsd()._results_table.data[1:], 
                        columns = mc_wdi.tukeyhsd()._results_table.data[0])
    thsd['site'] = site
    
    # write Tukey HSD test table to list
    wdi_diff_thsd_list.append(thsd)
    
    print('done.')
 
# Show subplot figure for Ridge & TLB
plt.sca(axs[1])
plt.show()

# Export figure
fig.savefig('../figures/Fig_S12.png',bbox_inches = 'tight')

# Export t-test results to csv
df_ttest = pd.concat(tt_list,axis =0).sort_values(['site','variable'])
df_ttest.to_csv('./tables/results/Table_S12.csv', sep = ';')
# 
# Export IQR values to csv
df_wdi_iqr = pd.concat(iqr_list,axis =0).sort_values(['site','variable']).round(2).sort_index(level = 'year',axis=1)
df_wdi_iqr.to_csv('./tables/results/Table_S13.csv', sep = ';')

# Export Tukey HSD table for WDI changes
df_thsd = pd.concat(thsd_list[::2],axis =0).to_csv('./tables/results/Table_S14.csv', sep=';', index = False)
df_thsd = pd.concat(thsd_list[1::2],axis =0).to_csv('./tables/results/Table_S15.csv', sep=';', index = False)

# Export Tukey HSD table for WDI changes
df_wdi_diff_thsd = pd.concat(wdi_diff_thsd_list,axis =0)
df_wdi_diff_thsd.to_csv('./tables/results/Table_S16.csv', sep=';', index = False)

# %% Figure 4 & Figure S14
#----

# give window side length in number of pixels (33 px of 15 cm resolution make 4.95 m)
windowsize = 33

ylabel = 'Grid cell $\Delta WDI_{2020 - 2021}$'

# Set binning variables
bin_var = 'meanT'
val_var = 'meanT' if bin_var == 'fcover' else 'fcover'

bin_interval = 1 if bin_var == 'fcover' else 0.005

windowsize_m = int(round(windowsize*.15,0))

# create figure instance and set theme for the TLB & Ridge plot that go to the suppl. materials
fig, axs = plt.subplots(ncols = 2, figsize=(45,20), dpi = 200,sharey = True) 

# Create the subplots
labs = ['(a)','(b)']

# create dummy subplot for CBH first
_, _= plt.subplots() 

for i,site in enumerate(sites):
    print(f'Mapping fCover in {site} ...')   
    
    if os.path.exists(f'./tables/intermediate/Table_Fig_S14_{site}.csv'):
        df_out = pd.read_csv(f'./tables/intermediate/Table_Fig_S14_{site}.csv',sep = ';')
    else:
        df_out = MapFcover_np(windowsize, I_cl_list[i],(I_wdi_20_list[i] - I_wdi_21_list[i]))
        df_out = df_out.dropna()
        df_out.to_csv(f'./tables/intermediate/Table_Fig_S14_{site}.csv',sep = ';')
    
    df_m_s = df_list[i]
    
    label_order = df_m_s.variable.cat.categories.to_list()
    
    PATH_SAVE = f'../figures/Fig_4.jpg' #jpg because eps has rendering issues
    
    if site == 'CBH':
        _, ax = plt.subplots(figsize=(45,20), dpi = 200) 
        # Plot CBH boxplot individually for main text
        ax,l,h,_ = PlotFcoverVsTemperature(data = df_out, 
                                           ax = ax,
                                           binning_variable = bin_var,
                                           bin_interval = bin_interval,
                                           value_variable = val_var,
                                           ylab = ylabel, 
                                           yvar = 'meanT',
                                           label_order = label_order,
                                           colors = colordict,
                                           model = 'cubic', 
                                           plot_type = 'regplot',
                                           PATH_OUT = PATH_SAVE,
                                           saveFig = True)
        plt.show()
        handles = h
        labels = l
        
    else:
        ax,l,h,_ = PlotFcoverVsTemperature(data = df_out, 
                                     ax = axs[i-1],
                                     binning_variable = bin_var,
                                     bin_interval = bin_interval,
                                     value_variable = val_var,
                                     ylab = ylabel, 
                                     yvar = 'meanT',
                                     label_order = label_order,
                                     colors = colordict,
                                     model = 'cubic', 
                                     plot_type = 'regplot',
                                     PATH_OUT = PATH_SAVE,
                                     saveFig = False)
        
        axs[i-1].text(-0.1, 1.1, labs[i-1] + ' ' + site, 
                      transform=axs[i-1].transAxes, weight='bold')
    
        axs[i-1].get_legend().remove()
        
# Plot the legend below the third panel
# legend for the plant communities
lab_adj = [x for i,x in enumerate(labels) if i!=2]
hand_adj = [x for i,x in enumerate(handles) if i!=2]

leg1 = fig.legend(hand_adj,lab_adj, 
                  loc='upper center', 
                  frameon=False,
                  # mode = 'expand',
                  title = 'Plant community',
                  bbox_to_anchor = (0.5,0), 
                  ncol=2)

# Adjust the position of the legend within the figure
fig.add_artist(leg1)
fig.tight_layout()
fig.savefig('../figures/Fig_S14.png',bbox_inches = 'tight')

# Show subplot figure for Ridge & TLB
plt.sca(axs[1])
plt.show()

# %% Figure S1 -  Precipitation and temperature climatology
# ----

# From Chokurdakh meteorological station:
df_spei = pd.read_csv(r'./tables/spei_monthly.csv',sep = ',')
df_spei['date'] = [datetime.date(year = int(x[1].year), month = int(x[1].month), day=1) for x in df_spei.iterrows()]

summer_mask =  (df_spei.month >= 6) & (df_spei.month <= 8)
df_spei_summer = df_spei[summer_mask]

# resample monthly values to summer averages
df_meteo_summer = df_spei_summer.groupby('year').agg({'precip':'sum', 'temp':'mean'}).reset_index()

# set names of y-axis labels
var_name = ['Total summer \nprecipitation (mm)', 'Mean summer \nair temperature (°C)']

# set names of annotations
stat_names = ['mean','10$^{th}$ percentile','90$^{th}$ percentile']

ylims = [(0,200),(3,12)]

# set y axis ticks
yticks = [np.arange(ylims[0][0],ylims[0][1]+1,50,dtype = int), 
          np.arange(ylims[1][0],ylims[1][1]+1,3,dtype = int)]

colors = ['blue','black']

sns.set_theme(style="ticks",
              font_scale = 3)

fig,axs = plt.subplots(2,1,figsize = (20,15),sharex = True)
sub_label = [ '(a)', '(b)']

for i,var in enumerate(['precip','temp']):
    df_meteo_summer.plot(x = 'year',y = var,
                         ax = axs[i],
                         lw = 2, color = colors[i],
                         legend = False)
    
    axs[i].scatter(x = [2020,2021],
                y = df_meteo_summer[var].iloc[-2:],
                s = 2000,
                facecolors='none', edgecolors=(0.5, 0.5, 0.5, 0.6), linewidths=5, marker='o')
    
    # Plot mean, 10th-percentile, and 90th-percentile as horizontal line
    for jj,q in enumerate([df_meteo_summer[var].mean(), 
                           df_meteo_summer[var].quantile(.1), 
                           df_meteo_summer[var].quantile(.9)]):
            
        axs[i].axhline(q,
                       ls = '--', color = 'gray')
        axs[i].annotate(stat_names[jj],
                        xy = (2024,q), 
                        va = 'center',
                        transform = axs[i].transData,
                        color = 'gray')
    
    axs[i].set(ylabel = var_name[i],
               xlabel = '',
               title = '',
               ylim = ylims[i],
               xlim = [1945,2024],
               yticks = yticks[i]
               )
    
    axs[i].text(-0.1, 1.05,sub_label[i], 
                transform=axs[i].transAxes,
                fontweight='bold', va='top', ha='right')
    
# Adjust the position of the x-axis tick labels by moving them down
axs[1].tick_params(axis='x', pad=20) 
    
sns.despine()
plt.savefig('../figures/Fig_S1.png',bbox_inches = 'tight', facecolor='white')

plt.show()

# Generate statistics table
# df_meteo_summer[['precip','temp']].describe().to_csv('./tables/intermediate/Chok_climatology_stats.csv',sep = ';')

# Climatological mean of precipitation
mean_precip = df_meteo_summer.precip.mean()

print('2020 Summer precipitation was {:.1f} % of climatological average.'.format(df_meteo_summer.iloc[-2].precip/mean_precip * 100))
print('2021 Summer precipitation was {:.1f} % of climatological average.'.format(df_meteo_summer.iloc[-1].precip/mean_precip * 100))

#%% Figure S2 - SPEI3
# ----

# From Chokurdakh meteorological station:
df_spei = pd.read_csv(r'./tables/spei_monthly.csv',sep = ',')
df_spei['date'] = [datetime.date(year = int(x[1].year), month = int(x[1].month), day=1) for x in df_spei.iterrows()]

summer_mask =  (df_spei.month >= 6) & (df_spei.month <= 8)
df_spei_summer = df_spei[summer_mask]

var_name = ['$SPEI_3$', '$SPEI_6$']

# group the data by year
groups = df_spei.groupby(df_spei.year)
    
# create a figure and axis object
sns.set_theme(style="ticks", 
              font_scale = 3)

fig,axs = plt.subplots(2,1,figsize = (15,15),sharex = False)
sub_label = [ '(a)', '(b)']

xlim = [5,9]

# obtain a colormap and normalize the line colors based on the year
cmap = plt.cm.get_cmap('binary')
norm = plt.Normalize(df_spei.year.min(), df_spei.year.max())

for i,spei in enumerate(['spei_3_months','spei_6_months']):

    # plot each group as a separate line on the same figure
    for name, group in groups:
        x = group.month
        y = group[spei]
        c = cmap(norm(name))
        if name == 2020 or name == 2021:
            c = 'brown' if name == 2020 else 'midnightblue'
            axs[i].plot(x, y, 
                        alpha = .6,
                        color=c,
                        label=name, lw=5)
            y_annot = group[spei].iloc[6] # July SPEI
            axs[i].annotate(name, xy=(7, y_annot), xytext=(7.5 + 2021 % name, 2.5), 
                            transform = axs[i].transData,
                            color = c,
                            arrowprops=dict(arrowstyle='->', color=c))
        else:
            axs[i].plot(x, y, 
                        alpha = .3,
                        color='gray', 
                        label=name)
            
    # Fill below -2 for extreme drought
    axs[i].fill_between(x = [4,10], y1 = -1.5, y2 = -1.99, color='orange', alpha = 0.1)
    axs[i].text(9.1, -1.75,'severe drought', 
                transform=axs[i].transData,
                color='orange',
                fontweight='bold', va='center', ha='left')
    
    # Fill below -2 for extreme drought
    axs[i].fill_between(x = [4,10], y1 = -2, y2 = -7, color='red', alpha = 0.1)
    axs[i].text(9.1, -2.75,'extreme drought', 
                transform=axs[i].transData,
                color='red',
                fontweight='bold', va='center', ha='left')
    
    # Plot 10th-percentile as horizontal line
    axs[i].axhline(df_spei_summer[spei].quantile(.1),
                   xmin = (6-xlim[0])/(xlim[1]-xlim[0]), xmax=(8-xlim[0])/(xlim[1]-xlim[0]),
                   ls = '--', lw = 2, color = 'k')
    axs[i].annotate('10$^{th}$ percentile (JJA)',
                    xy = (6.2,df_spei_summer[spei].quantile(.1)), 
                    xytext = (5.1, - 2.6),
                    va = 'top',transform = axs[i].transData,
                    arrowprops=dict(arrowstyle='->', color='black', lw = 2),
                    # bbox = dict(boxstyle='round', facecolor=(1, 1, 1, 0.7), edgecolor=(1, 1, 1, 0.7)),
                    color = 'k')
    
    # set the axis labels and title
    axs[i].set(ylabel = var_name[i],
               xlim = xlim,
               ylim = [-3.5, 3.5],
               yticks = np.arange(-3,4,1),
               yticklabels = np.arange(-3,4,1),
               xticks = np.arange(5,10),
               xticklabels = ['May','Jun','Jul','Aug','Sep'])
    axs[i].text(-0.15, 0.95,sub_label[i], 
                transform=axs[i].transAxes,
                fontweight='bold', va='top', ha='right')
    
    # Thicken the axis spines
    axs[i].spines['bottom'].set_linewidth(2)
    axs[i].spines['left'].set_linewidth(2)
    
sns.despine()


# fig.suptitle('Standardized Precipitation-Evaporation Index (z-values)')
plt.savefig('../figures/Fig_S2.png',bbox_inches = 'tight', facecolor='white')

#%% Figure S3 - Weather conditions
# ----

import matplotlib.dates as md
import datetime

print('Mean Tair between first and last flight, 2020:',
          df_fluxtower.loc['24-07-2020'].between_time('16:20','18:48').Barani_Temp_2_Avg.mean())
print('Mean Tair between first and last flight, 2021:',
      df_fluxtower.loc['19-07-2021'].between_time('13:50','16:45').Barani_Temp_2_Avg.mean())

print('Mean SWin between first and last flight, 2020:',
      df_fluxtower.loc['24-07-2020'].between_time('16:20','18:48').CMP21_IN_Avg.mean())
print('Mean SWin between first and last flight, 2021:',
      df_fluxtower.loc['19-07-2021'].between_time('13:50','16:45').CMP21_IN_Avg.mean())
  
# create figure instance and set theme
sns.set_theme(style="whitegrid",
              font_scale = 2.5,
              rc = {"axes.spines.right": False, "axes.spines.top": False})
fig,axs = plt.subplots(1,3, figsize = (30,10))

df_fluxtower['time'] = [datetime.datetime.combine(datetime.date.today(),t.time()) for t  in df_fluxtower.index]

df_fluxtower_2020 = df_fluxtower.loc['24-07-2020'].between_time('11:00','20:00')
df_fluxtower_2021 = df_fluxtower.loc['19-07-2021'].between_time('11:00','20:00')

dfl = [df_fluxtower_2020,df_fluxtower_2021]

def XDate(df_flighttimes,site,year,start):
    if start:
        t = df_flighttimes[np.logical_and(df_flighttimes.site == site,
                                          df_flighttimes.year == year)].start_time_local.dt.time.values[0]
        return datetime.datetime.combine(datetime.date.today(),t)
    else:
        t = df_flighttimes[np.logical_and(df_flighttimes.site == site,
                                          df_flighttimes.year == year)].end_time_local.dt.time.values[0]
        return datetime.datetime.combine(datetime.date.today(),t)

ylabs = ['Air temperature (°C)','Incoming shortwave radiation (W m$^{-2}$)','Wind speed (m s$^{-1}$)']
offsets = [1,30,.2]
labs = ['(a)','(b)','(c)']

conds = np.full((3, 3), False) # Boolean array for GetMeanTair function
np.fill_diagonal(conds,True)

for i,var in enumerate(['Barani_Temp_2_Avg','CMP21_IN_Avg','Cup_2M_Avg']):
    
    p = [ axs[i].plot(df['time'],
                      df[var],
                      color = cc,
                      marker = 'o',linewidth = 3,
                      label = df.index.year[0]) for df,cc in zip(dfl,['brown','midnightblue'])]
    axs[i].text(-0.1, 1.1, labs[i], transform=axs[i].transAxes, weight='bold')
    
    for site in ['TLB','CBH','Ridge']:
        for year in [2020,2021]:
            color = 'brown' if year == 2020 else 'midnightblue'
            ybar = GetMeanTair(df_flighttimes, df_fluxtower, site, year,
                               returnTemp = conds[i][0],
                               returnSW = conds[i][1],
                               returnWS = conds[i][2]) # get y value for horizontal bars

            # Plot droneflights as horizontal bars
            axs[i].hlines(y = ybar,
                        xmin = XDate(df_flighttimes,site,year,start= True),
                        xmax = XDate(df_flighttimes,site,year,start=False),
                        lw = 5, color = color,alpha = .8)
            axs[i].annotate(site,
                            xy = (XDate(df_flighttimes,site,year,start= True),
                                  ybar + offsets[i]), 
                            color = color, transform = axs[i].transData,
                            clip_on=False, annotation_clip=False,
                            verticalalignment = 'bottom',
                            horizontalalignment='left',
                            rotation = 0)
            
    axs[i].set(ylabel = ylabs[i],
               xlabel = 'Local time')
    axs[i].xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
    axs[i].grid(False)
    
    # Thicken the axis spines
    axs[i].spines['bottom'].set_linewidth(5)
    axs[i].spines['left'].set_linewidth(5)

fig.autofmt_xdate()
fig.legend(p,labels = [2020,2021],loc = 'lower center',frameon = False,
           ncol = 2, bbox_to_anchor=(0.5, -0.05) )

PATH_OUT = r'..\figures\Fig_S3.png'
plt.savefig(PATH_OUT,bbox_inches = 'tight',dpi=300)


#%% Figure S4 - Boxplot of NDVI differences
# ----

yvar = 'ndvi'

# Lists for the test result dataframes
tt_list = list()
thsd_list = list()

# create subplots
fig, axs = plt.subplots(nrows = 3, figsize=(30,30), dpi = 200) 

# Adjust spacing between subplots and colorbar
plt.subplots_adjust(hspace=.6) 

# Create the subplots
labs = ['(a)','(b)','(c)']

for i,site in enumerate(sites):
    print(f'Generating NDVI difference plot in {site}...')   
    
    df_m_s = df_list[i]
    
    label_order = df_m_s.variable.cat.categories.to_list()
        
    # sample random observations per plant community and year to avoid effect of large sample
    df_sample = df_m_s.groupby(['variable','year']).sample(frac = 0.1)
    
    # Welch's ttest for unequal variances between years
    alpha = .01
    ttest = df_sample.groupby(['variable']).apply(lambda x: ttest_ind(x.loc[x.year==2020,yvar],
                                                                      x.loc[x.year==2021,yvar], 
                                                                      equal_var=False))
    df_ttest = pd.DataFrame([[z,p] for z,p in ttest.values],columns = ['tstat','pval']).set_index(ttest.index)
    df_ttest['reject'] = df_ttest.pval < alpha
    df_ttest.loc[df_ttest.pval < 0.01,'text'] = '< 0.01'
    
    df_ttest['site'] = site
    
    # Mean WDI differences between years for each group
    df_ttest['meandiff'] = df_sample.groupby(['variable']).apply(
        lambda x: round(x.loc[x.year == 2020,yvar].mean() -
                        x.loc[x.year == 2021,yvar].mean(),2))
    
    # write t-test table to list
    tt_list.append( df_ttest)
    
    ax = PlotBoxWhisker(df_m_s, yvar,
                        ax = axs[i-1],
                        label_order = label_order,
                        colors = colordict,
                        showSignif=True,
                        data_ttest = df_ttest,
                        showWater = False,
                        PATH_OUT=PATH_SAVE,
                        saveFig = False)
    
    axs[i-1].text(-0.1, 1.3, labs[i-1] + ' ' + site, 
                  transform=axs[i-1].transAxes, weight='bold')
    
plt.savefig('../figures/Fig_S4.png',bbox_inches = 'tight', dpi = 200)
plt.show()


# %% Figure S5 - Plot Sensor temperature over time
# ----

fl = glob('./tables/metadata*.csv')

unc_instr = .1

df = pd.concat(map(lambda fn:GatherData(fn, unc_instr, FitOnUnstable = False),fl), 
               ignore_index=False)
sites = df.site.unique()

sns.set_theme(style="ticks",
              # rc={"figure.figsize":(20, 10)},
              font_scale = 2.75)
base_colors  = {'color': ["brown", "midnightblue"]}

fig, axs = plt.subplots(nrows = 2, ncols = 3, sharex = True, sharey = True, figsize = (20,10))
legend_handles = []  # Store legend handles for the common legend

for i, ax in enumerate(axs.flat):
    year = 2020 + i // 3
    label = '(' + chr(97 + i) + ')'
    site = sites[i//2]

    mask = np.logical_and(df['year'] == year, df['site'] == site)
    
    # Line plot
    sns.lineplot(data = df[mask], x="flighttime_min", y="T_sensor", 
                 color='gray', alpha = .7,ax=ax)

    # Scatter plot with color mapping
    ncolors = 3
    if year == 2020:
        hue_palette = sns.light_palette("brown", n_colors=ncolors)[-2:]  # Brown tones for 2020
    else:
        hue_palette = sns.light_palette("midnightblue", n_colors=ncolors)[-2:]  # Blue tones for 2021

    scatterplot = sns.scatterplot(data=df[mask], x="flighttime_min", y="T_sensor", hue='isStable',
                    palette=hue_palette, hue_order=[False, True], marker='+', s = 200,
                    ax=ax)

    # Set x and y limits and ticks
    ax.set(xlim=(0, 45), ylim=(25, 45), 
           ylabel = '', xlabel = '',
           xticks=np.arange(0, 50, 5), yticks=np.arange(25, 50, 5))

    # Add label
    ax.text(0.15, 1.1, label, transform=ax.transAxes, fontweight='bold', va='top', ha='right')

    # Set title
    if i//3 == 0:
        ax.set_title(f'{sites[i]}', pad=10, fontweight='bold')
    
    # Remove the subplot legend
    handles, labels = scatterplot.get_legend_handles_labels()
    legend_handles.extend(handles)  # Collect handles for common legend

fig.text(1, 0.75, '2020', va='center',ha = 'center', rotation='vertical',fontweight = 'bold')    
fig.text(1, 0.25, '2021', va='center',ha = 'center', rotation='vertical',fontweight = 'bold')    

# Set common x and y labels
fig.text(-0.02, 0.5, 'Sensor temperature (°C)', va='center', rotation='vertical')
fig.text(0.5, 0, 'Flight time (minutes)', va='center', rotation='horizontal', ha='center')

# Create a common legend without duplicate labels
legend_text = f'Is the sensor temperature stable, i.e., within {unc_instr:.1f} °C of the minimum?'
unique_labels = labels * 2

for handle in legend_handles:
    handle.set_sizes([200,200])

common_legend = fig.legend(legend_handles[4:8], unique_labels, title=legend_text,
                           frameon = False,
                           ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.25))


# Remove individual subplot legends
for ax in axs.flat:
    ax.spines['top'].set_alpha(0)
    ax.spines['right'].set_alpha(0)
    ax.get_legend().remove()

plt.tight_layout()
plt.savefig('../figures/Fig_S5.png',bbox_inches = 'tight', facecolor='white')

plt.show()

# %% Figure S6 - Plot drift fitting curve
# ----

plot_fit = 'fit_2'
cutoff = 13

r_sq = df.groupby(['site','year']).apply(lambda x: r2_score(x['LST_deviation'], x[plot_fit]))

sns.set_theme(style="ticks",
              # rc={"figure.figsize":(20, 10)},
              font_scale = 2.75)
base_colors  = {'color': ["brown", "midnightblue"]}

fig, axs = plt.subplots(nrows = 2, ncols = 3, sharex = True, sharey = True, figsize = (20,10))
legend_handles = []  # Store legend handles for the common legend

for i, ax in enumerate(axs.flat):
    year = 2020 + i // 3
    label = '(' + chr(97 + i) + ')'
    site = sites[i % 3]
    
    mask = np.logical_and(df['year'] == year, df['site'] == site)
    
    # Line plot
    if site == 'CBH' or year == 2021:
        sns.lineplot(data = df[mask], x="T_sensor", y=plot_fit, 
                     color='black', ax=ax)
    
    elif site == 'TLB' and year == 2020 or site == 'Ridge' and year == 2020:
        for line in ax.lines[:2]:
            line.set_alpha(0)
           
        sns.lineplot(data = df[mask][df[mask].flighttime_min <= cutoff],
                     x = "T_sensor" ,y = plot_fit,
                     color = 'black', ax = ax)
        sns.lineplot(data = df[mask][df[mask].flighttime_min > cutoff],
                     x = "T_sensor" ,y = plot_fit,
                     color = 'black', ax = ax)
    
    ax.axhline(0,c = 'gray',alpha = .2,ls = '--')

    # Scatter plot with color mapping
    ncolors = 3
    if year == 2020:
        hue_palette = sns.light_palette("brown", n_colors=ncolors)[-2:]  # Brown tones for 2020
    else:
        hue_palette = sns.light_palette("midnightblue", n_colors=ncolors)[-2:]  # Blue tones for 2021

    scatterplot = sns.scatterplot(data=df[mask], x="T_sensor", y="LST_deviation", hue='isStable',
                    palette=hue_palette, hue_order=[False, True], s = 200,
                    alpha = .3,
                    ax=ax)

    # Set x and y limits and ticks
    ax.set(xlim=(25, 45), ylim=(-15, 5), 
           ylabel = '', xlabel = '',
           xticks=np.arange(25,50,5), yticks=np.arange(-15,6,5))

    # Add label
    ax.text(0.15, 1.1, label, transform=ax.transAxes, fontweight='bold', va='top', ha='right')

    # Set title
    if i//3 == 0:
        ax.set_title(f'{sites[i]}', pad=10, fontweight='bold')
    
    # Remove the subplot legend
    handles, labels = scatterplot.get_legend_handles_labels()
    legend_handles.extend(handles)  # Collect handles for common legend
    
    # Annotate a value into each bar in each plot
    ax.annotate("R$^2$ = {:.2f}".format(r_sq.iloc[i]) , 
                (30 + (10 * (i//3)),-12), # Placement
                ha='center', va='center', color='black', rotation=0, xytext=(0, 20),
                textcoords='offset points')
        
fig.text(1, 0.75, '2020', va='center',ha = 'center', rotation='vertical',fontweight = 'bold')    
fig.text(1, 0.25, '2021', va='center',ha = 'center', rotation='vertical',fontweight = 'bold')    

# Set common x and y labels
fig.text(-0.02, 0.5, 'Correction temperature (°C)', va='center', rotation='vertical')
fig.text(0.5, 0, 'Sensor temperature (°C)', va='center', rotation='horizontal', ha='center')

# Create a common legend without duplicate labels
legend_text = f'Is the sensor temperature stable, i.e., within {unc_instr:.1f} °C of the minimum?'
unique_labels = labels * 2

for handle in legend_handles:
    handle.set_sizes([200,200])
    
common_legend = fig.legend(legend_handles[4:8], unique_labels, title=legend_text,
                           frameon = False,
                           ncol=2, loc='lower center', bbox_to_anchor=(0.5, -0.25))


# Remove individual subplot legends
for ax in axs.flat:
    ax.spines['top'].set_alpha(0)
    ax.spines['right'].set_alpha(0)
    ax.get_legend().remove()

plt.tight_layout()

plt.savefig('../figures/Fig_S6.png',dpi = 200,bbox_inches = 'tight', facecolor='white')
plt.show()


# %% Figure S9 - Plot semivariograms
# ----

np.random.seed(seed)

variogram_list = list()

labs = ['(a)','(b)','(c)']

sns.set_theme(style="whitegrid",
              font_scale = 2,
              rc = {"axes.spines.right": False, "axes.spines.top": False})

fig,axs = plt.subplots(3,1,figsize = (15,10),sharex = True,sharey = True)

for i, site in enumerate(['CBH','Ridge','TLB']):
    PATH_21_TIR = rf'.\mosaics\{site}_thermal_2021_resampled.tif'
    
    T_air_21 = GetMeanTair(df_flighttimes, df_fluxtower, site, 2021)
    
    dA_tir21 = rxr.open_rasterio(PATH_21_TIR)
    # Define corner coordinates of study area extent
    if site == 'Ridge':
        ymin = 200
        xmin = 1200
    else:
        ymin = 300
        xmin = 200

    ymax = ymin + dxy
    xmax = xmin + dxy
    
    dA_tir21_s = dA_tir21.isel(x = slice(xmin,xmax), y = slice(ymin,ymax)) - T_air_21
    
    # Select n**2 Random locations in the raster
    n = 200
    x_rand = np.random.choice(dA_tir21_s.x.values.ravel(),n)
    y_rand = np.random.choice(dA_tir21_s.y.values.ravel(),n)
    coords = np.c_[y_rand, x_rand]
    
    R = dA_tir21_s.sel(x = x_rand,
                       y = y_rand, 
                       method = 'nearest')
    
    df = R.to_series().reset_index()
    
    maxlag = 50 # compute semivariogram until 50m distance pairs
    step = 0.15
    nlags = int((maxlag / step) // 10)
    
    V = skg.Variogram(coordinates = df[['x','y']],
                      values = df.iloc[:,-1],
                      model = 'matern', 
                      bin_func = 'even',
                      n_lags = maxlag, # determines number of bins, if equal to maxlag, then bins have size of 1m
                      maxlag = maxlag)
    
    variogram_list.append(V)
    
sns.set_theme(style="ticks",
              font_scale = 2,
              rc = {"axes.spines.right": False, "axes.spines.top": False})

fig,axs = plt.subplots(3,1,figsize = (15,10),sharex = True,sharey = True)

for i, site in enumerate(['CBH','Ridge','TLB']):
    V = variogram_list[i]
    
    ax = axs[i]
    V.plot(axes = ax,grid = False)
    ax.axhline(V.parameters[1], ls = '--', color = 'gray') # horizontal for sill
    ax.axvline(V.parameters[0], ls = '--', color = 'gray') # vertical for range
    
    ax.set(xlabel ='',ylabel ='',
           yticks = np.arange(0,6.5,2), yticklabels = np.arange(0,6.5,2))
    ax.set_title(site, fontweight='bold')

    ax.annotate('Range = {:.1f} m'.format(V.parameters[0]),
                (V.parameters[0] + 2, np.mean(ax.get_ylim())), 
                color = 'k', transform = ax.transData,
                clip_on=False, annotation_clip=False,
                verticalalignment = 'center_baseline', horizontalalignment='left')
    ax.text(0.02, 1.1, labs[i], transform=ax.transAxes, weight='bold')
    
    ax.grid(False)
    ax.spines['left'].set(linewidth = 2, edgecolor = 'black')  # Thicken the axis spines
    ax.spines['bottom'].set(linewidth = 2, edgecolor = 'black')   # Thicken the axis spines
    ax.tick_params(axis='x', width=2, length=5)
    
ax.set(xlabel ='Distance (m)')
fig.text(-0.01, 0.5, 'Semivariance (matheron)', va='center', rotation='vertical')

fig.tight_layout()
fig.savefig('../figures/Fig_S9.png',dpi = 200,bbox_inches = 'tight', facecolor='white')

# %% Figure S10 - Density plots of Ts - Ta in 2020 & 2021 across all sites
# ----

group_var = 'variable'
xvar = 'deltaT'
xlim = [-5,23] if xvar == 'deltaT' else None
lw = 5

fig, axs = plt.subplots(nrows = 3, figsize=(30,30), dpi = 200,sharex = True) # create subplot instance

# Adjust spacing between subplots and colorbar
plt.subplots_adjust(hspace=.35) 

# Create the subplots
labs = ['a)','b)','c)']

for i,site in enumerate(sites):
    print(f'Generating subplot in {site} for Figure 1...')   
    
    df_m_s = df_list[i]
    
    label_order = df_m_s.variable.cat.categories.to_list()
    
    # Perform Tukey HSD
    # sample 1 % of observations (= 200) per plant community and year
    df_sample = df_m_s.groupby(['variable','year']).sample(frac = 0.1,
                                                           random_state = seed)
    
    mc20,mc21 = df_sample.loc[(df_sample.variable != 'water') & 
                              (df_sample.variable != 'mud')].groupby(['year']).apply(lambda x: MultiComparison(x[xvar], x[group_var]))
    
    # Save test results in dataframe (for significance brackets in plot)
    thsd = pd.DataFrame(data = mc21.tukeyhsd()._results_table.data[1:], 
                        columns = mc21.tukeyhsd()._results_table.data[0])
    
    print('2021: \n', mc21.tukeyhsd(), end = '\n')                                                                                       
                                                                                               
    # Plot Densities 
    h, l, ax = PlotDensities(df_sample, xvar,
                       ax = axs[i],
                       xlim = xlim,
                       colors = colordict,
                       showSignif = False, 
                       data_pairtest = thsd,
                       order = label_order,
                       showWater = True,
                       showBothYears = True,
                       PATH_OUT = 'PATH_SAVE',
                       saveFig = False)
    
    if site == 'CBH':
        handles = h
        labels = l
    
    ax.set(ylabel = '')
    
    ax.get_legend().remove()
    
    axs[i].text(-.05, 1.15, labs[i] + ' ' + site, transform=axs[i].transAxes, weight='bold')
 
# Set the y-label for the entire figure
fig.text(0.04, 0.5, 'Kernel density estimate', va='center', rotation='vertical')

# Plot the legend below the third panel
# legend for the plant communities
leg1 = fig.legend(handles[:6], labels[:6], 
                    loc='upper center', 
                    frameon=False,
                    # mode = 'expand',
                    title = 'Plant community',
                    bbox_to_anchor=(0.5, -0.1), 
                    ncol=2)
# legend for the years
leg2 = fig.legend(handles[7:], labels[7:], 
                    loc='upper center', 
                    frameon=False,
                    # mode = 'expand',
                    title  = 'Year',
                    bbox_to_anchor=(0.5, 0), 
                    ncol=2)

# Adjust the position of the legend within the figure
fig.add_artist(leg1)
fig.add_artist(leg2)

PATH_OUT = '../figures/Fig_S10.png' 
fig.savefig(PATH_OUT,bbox_inches = 'tight')

plt.show()    

# %% Figure S11 - TOMST TMS-4 differences
# ----

print('Loading TMS-4 data...')

df_tmst_delta = pd.read_csv('./tables/data_allobs_alldelta.csv',
                  header = [0],index_col = [0],
                  sep=';',parse_dates = True)
df_tmst_delta.index = df_tmst_delta.index.tz_convert('Asia/Srednekolymsk')
    
# Aggregate data by Logger_SN and calculate mean
def aggregate_func(df):
    return pd.merge(
        df.groupby('Logger_SN').mean().reset_index(),
        # df.groupby('Logger_SN').resample('D').mean().reset_index(drop = True),
        df.groupby('Logger_SN').agg(lambda x: x[0])[['custom_class','site']].reset_index(),
        # df.groupby('Logger_SN').resample('D').agg(lambda x: x[0])[['custom_class','site']].reset_index(),
        on='Logger_SN')

lbl = {'water':'Open water',
       'mud': 'Mud',
       'wet': 'LW1',
       'ledum_moss_cloudberry': 'LW2', 
       'shrubs': 'HP2', 
       'dry': 'HP1',
       'tussocksedge': 'TS'}

# yvar = ["dT2T3","dT1Tair",  "dT2Tair",  "dT3Tair" ]
# col_titles = ['$T_2 - T_3$', '$T_1 - T_{air}$', '$T_2 - T_{air}$', '$T_3 - T_{air}$']

col_titles = '$T_2 - T_{air}$' if yvar == ["dT2Tair"] else 'Soil moisture'

order = ['LW1','HP1','HP2','TS']
palette = [colordict[x] for x in ['HP1','HP2','TS','LW1']]

sns.set_theme(style="ticks", 
              rc = {"axes.spines.right": False, "axes.spines.top": False},
              font_scale = 2.5)

fig,axs = plt.subplots(nrows = 3, ncols = 2, figsize = (15,20))
col_titles = ['TLB', 'Ridge']

for i,ax in enumerate(axs.flat):
    site = col_titles[i % 2]
    label = '(' + chr(97 + i) + ')'
    
    if i // 2 == 0:
        # Select data on flightday in 2021
        df_tmst_delta_flightday = df_tmst_delta.loc["2021-07-19"].between_time('13:50','16:30')
        data_agg = aggregate_func(df_tmst_delta_flightday)
        text = 'flightday'
        ylim = [0,10]
    elif i // 2 == 1:
        # Select data between 12:00 and 13:30
        df_tmst_delta_noon = df_tmst_delta.between_time('11:00','13:30')
        data_agg = aggregate_func(df_tmst_delta_noon)
        text = 'seasonal'
        ylim = [-2,6]
        
    if i // 2 < 2:
        yvar = ["dT2Tair"]
        ylab = '$T_2 - T_{air}$ (°C),' + '\n' + text

    else:
        yvar = ["soilmoisture"]
        ylab = 'Raw soil moisture count'
        ylim = [0, 4000]
        
    # melt data frame to long format for facet grid
    df_melted = pd.melt(data_agg.loc[:,yvar+['site','custom_class']], id_vars=['site', 'custom_class'],  value_name='T')
    df_melted[['site','variable','custom_class']] = df_melted[['site','variable','custom_class']].astype("category")

    # rename custom class to official labels
    df_melted['custom_class'] = df_melted['custom_class'].map(lbl)
    df_melted['site'] = df_melted['site'].map({'lakebed':'TLB','ridge':'Ridge'})

    # Map colors as new column for boxplot
    df_melted['color'] = df_melted['custom_class'].map(colordict)
    
    # boxplot
    sns.boxplot(df_melted[df_melted.site == site], x =  'custom_class', y = 'T', 
                ax = ax,
                hue = 'custom_class', palette = palette,
                whis=[1, 99],
                linewidth=.7, width=1, boxprops=dict(alpha=.7),
                order = order, dodge = False,
                fliersize=0.8)
    
    # add horizontal grey line at 0
    ax.axhline(color = 'gray',ls = '--',alpha = .5)
    
    # set figure aesthetics
    ax.set(ylabel = ylab,ylim = ylim,xlabel = '')
    ax.label_outer()
    ax.get_legend().remove()

    # Add label
    ax.text(0.15, 1.1, label, transform=ax.transAxes, fontweight='bold', va='top', ha='right')

fig.text(0.32, .91, 'TLB', fontweight='bold', va='center', rotation='horizontal', ha='center')
fig.text(0.72, .91, 'Ridge', fontweight='bold', va='center', rotation='horizontal', ha='center')
fig.align_ylabels(axs[:, 0])

plt.savefig('../figures/Fig_S11.png',dpi = 200,bbox_inches = 'tight', facecolor='white')

# save difference in deltaT's between groups TOMST vs. drone:
from itertools import combinations

df_out = pd.DataFrame(columns = ['site','name','rel_diff_drone','rel_diff_tomst'], index = range(6),dtype = float)

i = 0
for s in ['TLB','Ridge']:
    # read drone data
    df_m_s = pd.read_csv(f'./tables/intermediate/{s}_data_thermal.csv',sep=';')
    
    # compute mean deltaT per plant community
    df1 = df_m_s.loc[df_m_s.year == 2021].groupby(['variable'])['deltaT'].mean()
    
    # exclude water and mus
    exclude = ['water', 'mud']
    df1 = df1.loc[~df1.index.isin(exclude)]
    
    for x, y in combinations(df1.index.unique(), 2):
        df_out.loc[i,'site'] = s
        df_out.loc[i,'name'] = f'{y}-{x}'
        df_out.loc[i,'rel_diff_drone'] = df1[y] - df1[x]
        
        # extract only TOMST values from the current site s
        mask1 = np.logical_and(df_melted.custom_class == x, df_melted.site == s, df_melted.variable == 'dT2Tair')
        mask2 = np.logical_and(df_melted.custom_class == y, df_melted.site == s, df_melted.variable == 'dT2Tair')
        
        # compute relative deltaT differences between classes
        df_out.loc[i,'rel_diff_tomst'] = df_melted.loc[mask2,'T'].mean() - df_melted.loc[mask1,'T'].mean()
        
        i += 1
# df_out.round(2).to_csv(f'./results/statistics/Suppl_Table_TOMST_{text}_Drone_reldiffs.csv',sep = ';')
print(df_out.round(2))

# %% Figure S15 - Plot TPI in the meanT vs. fcover relationship
# ----

from skimage.io import imread
from matplotlib import colors

site = 'CBH'
i = 0

df_m_s = pd.read_csv(f'./tables/intermediate/{site}_data_thermal.csv',sep=';')

# Load and crop TPI raster
I_tpi_cbh = imread(r'./mosaics/TPI_CBH.tif')
I_tpi_cbh_s =  I_tpi_cbh[extent['ymin']:extent['ymax'],extent['xmin']:extent['xmax']]

# Compute grid cell TPI
windowsize = 33
df_tpi = MapFcover_np(windowsize, I_cl_list[i],I_tpi_cbh_s)
df_dwdi = MapFcover_np(windowsize, I_cl_list[i],(I_wdi_20_list[i] - I_wdi_21_list[i]))

# If this error appears: 
#     BrokenProcessPool: A task has failed to un-serialize. Please ensure that the arguments of the function are all picklable.
# re-run the FigFunctions script to redefine the modules

# Rename columns
df_tpi.columns = ['classes', 'count', 'fcover', 'tpi', 'sigma_tpi']
df_dwdi.columns = ['classes', 'count', 'fcover', 'dwdi', 'sigma_dwdi']

# merge dataframes
df_out = pd.concat([df_tpi, df_dwdi],axis = 1)
df_out = df_out.loc[:, ~df_out.columns.duplicated()]

# Remove empty rows and water & mud from dataframe
df_out = df_out.dropna()    
mask = np.logical_and(df_out.classes != 'water',df_out.classes != 'mud')
data_masked = df_out.loc[mask,:]

# Define binning variables
bin_var = 'dwdi'
val_var = 'dwdi' if bin_var == 'fcover' else 'fcover'
bin_interval = 1 if bin_var == 'fcover' else 0.005

# aggregate val_var into the bins given by bin_var
bins = np.arange(0,data_masked[bin_var].max(),bin_interval)
group = data_masked.groupby(['classes',
                          pd.cut(data_masked[bin_var], bins)
                          ])

# prepare dataframe for plotting
classnames = list(data_masked['classes'].unique())
n_classes = len(classnames)

df_grouped = pd.DataFrame({'classes': np.repeat(classnames,len(bins)-1),
                      bin_var: list(bins[1:]) *n_classes,
                      val_var: group[val_var].apply(np.nanmean).reindex(classnames,level=0).values, # have to reindex to insert values at right class
                      'tpi': group['tpi'].apply(np.nanmean).reindex(classnames,level=0).values, # have to reindex to insert values at right class
                      val_var + '_std':group[val_var].std().reindex(classnames,level=0).values})

cl = 'HP1'

s = 200

df_p = df_grouped.loc[df_grouped['classes'] == cl,:].dropna()

viz_max = df_p.tpi.quantile(.99)
viz_min = -viz_max

norm = colors.Normalize(vmin=viz_min, vmax=viz_max)

sns.set_theme(style="ticks",
              font_scale = 2,
              rc = {"axes.spines.right": False, "axes.spines.top": False})


f,ax = plt.subplots(figsize = (20,10))
scatter = ax.scatter(x = df_p.fcover,y = df_p.dwdi,
                     c = df_p.tpi,cmap = cm.bam,
                     norm = norm,
                     s = s)
cbar = f.colorbar(scatter,
           extend = 'both',
           label = 'Topographic position index')
cbar.ax.locator_params(nbins=5)

ax.set(xlabel = 'Grid cell fCover (%)',
       ylabel = 'Mean grid cell $\Delta WDI_{2020-2021}$')
plt.savefig('../figures/Fig_S15.png'  ,bbox_inches = 'tight',dpi=150)

