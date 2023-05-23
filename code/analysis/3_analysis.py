# -*- coding: utf-8 -*-
"""
Classify the multispectral drone imagery

Author: Nils Rietze - nils.rietze@uzh.ch
Created: 24.04.2023
"""
# imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from cmcrameri import cm
import seaborn as sns

from statsmodels.stats.multicomp import MultiComparison
from scipy.stats import ttest_ind, ttest_rel

import sys, os

from FigFunctions import read_fluxtower,GetMeanTair,PrepRasters,MeltArrays,ScaleMinMax,DifferenceTest,pretty_table,ProgressParallel,PlotDensities,PlotBoxWhisker,PlotFcoverVsTemperature,PolyFitSensorT,GatherData

# %% 0. LOAD DATA & PREALLOCATIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

df_list = list(range(3))

I_cl_list = list()
I_wdi_20_list = list()
I_wdi_21_list = list()

# Create the index levels
years = [2020, 2021] * 3
multi_index = list(zip(years, np.repeat(sites ,2)))
multi_index = pd.MultiIndex.from_tuples(multi_index, names=['Year', 'Site'])

df_flight_means = pd.DataFrame(index = multi_index, columns = ['mean_Tsurf','mean_Tsurf_Tair','mean_wdi'])

# create output directories if they don't exist yet:
if not os.path.exists('./tables/results/'):
    os.makedirs('./tables/results/')
if not os.path.exists('./tables/intermediate/'):
    os.makedirs('./tables/intermediate/')
    
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
    elif site == 'CBH':
        ymin = 300
        xmin = 700
    elif site == 'TLB':
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
    
    # save mean statistics of flight data
    df_flight_means.loc[(2020,site)] = (I_tir_20_s.mean(),(I_tir_20_s - T_air_20).mean(),I_wdi_20_s.mean())
    df_flight_means.loc[(2021,site)] = (I_tir_21_s.mean(),(I_tir_21_s - T_air_21).mean(),I_wdi_21_s.mean())
    
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

# %% 1. DENSITY PLOT IN NORMAL YEAR & DIFFERENCE TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
group_var = 'variable'
xvar = 'deltaT'
xlim = [0,25]

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
                                                                                               
    PATH_SAVE = f'../figures/Fig_1_{site}.png'
    
    # Plot Densities in 2021
    ax = PlotDensities(df_m_s, xvar,PATH_SAVE,
                       xlim = xlim,
                       colors = colordict,
                       showSignif = True, data_pairtest = thsd,
                       order = label_order,
                       showWater = False,
                       showBothYears = False,
                       saveFig = False)
    
    plt.show()
    
    # export results form differnce test in 2021 
    # thsd.to_csv(f'./tables/results/Table_S8_{site}.csv', sep=';', index = False)
    
    # Plot Densities in 2020 and 2021
    thsd = pd.DataFrame(data = mc20.tukeyhsd()._results_table.data[1:], 
                        columns = mc20.tukeyhsd()._results_table.data[0])
    
    PATH_SAVE = f'../figures/Fig_1_both_{site}.png'
    ax = PlotDensities(df_sample, xvar,PATH_SAVE,
                        xlim = [-5,25],
                        colors = colordict,
                        showSignif = False, data_pairtest = thsd,
                        order = label_order,
                        showWater = True,
                        showBothYears = True,
                        saveFig = False)
    
    plt.show()
    
    # export results form differnce test in 2020 
    # thsd.to_csv(f'./tables/results/Table_S8b_{site}.csv', sep=';', index = False)
    
    print('done.')
    
# %% 1a. DENSITY PLOT IN NORMAL YEAR ALL SITES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from matplotlib.lines import Line2D

group_var = 'variable'
xvar = 'wdi'
xlim = [7.5,25]
lw = 5

fig, axs = plt.subplots(nrows = 3, figsize=(30,20), dpi = 200,sharex = True) # create subplot instance

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
                                                                                               
    # Plot Densities in 2021
    h, l, ax = PlotDensities(df_sample, xvar,
                       ax = axs[i],
                       # xlim = xlim,
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
    
    axs[i].text(.75, .9, labs[i] + ' ' + site, transform=axs[i].transAxes, weight='bold')
 
# Set the y-label for the entire figure
fig.text(0.06, 0.5, 'Kernel density estimate', va='center', rotation='vertical')

# Plot the legend below the third panel
# legend for the plant communities
leg1 = fig.legend(handles[:6], labels[:6], 
                    loc='upper center', 
                    frameon=False,
                    # mode = 'expand',
                    title = '$\\bf{Plant \; community}$',
                    bbox_to_anchor=(0.5, -0.1), 
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

# plt.savefig('../../figures_and_maps/wdi/wdi_all_sites_both_years.png')

plt.show()    


# %% 2. BOXPLOT IN BOTH YEARS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
yvar = 'wdi'

for i,site in enumerate(sites):
    print(f'Generating subplot in {site} for Figure 1...')   
    
    df_m_s = df_list[i]
    
    label_order = df_m_s.variable.cat.categories.to_list()
        
    # Generate descripitve statistics of water deficit index and save as csv
    df_wdi_iqr = df_m_s.groupby(['year','variable'])['wdi'].quantile([.25,.75]).unstack()
    df_wdi_iqr['iqr'] = df_wdi_iqr[.75] - df_wdi_iqr[.25]
    # df_wdi_iqr.to_csv(f'./tables/results/Table_S_{site}.csv', sep = ';')
    
    # sample random observations per plant community and year to avoid effect of large sample
    df_sample = df_m_s.groupby(['variable','year']).sample(frac = 0.1)
    
    # Welch's ttest for unequal variances between years
    alpha = .05
    ttest = df_sample.groupby(['variable']).apply(lambda x: ttest_ind(x.loc[x.year==2020,yvar],
                                                                      x.loc[x.year==2021,yvar], 
                                                                      equal_var=False))
    df_ttest = pd.DataFrame([[z,p] for z,p in ttest.values],columns = ['tstat','pval']).set_index(ttest.index)
    df_ttest['reject'] = df_ttest.pval < alpha
    df_ttest.loc[df_ttest.pval < 0.01,'text'] = '< 0.01'
    
    # Export t-test results to csv
    # df_ttest.to_csv(f'./tables/results/Table_S_{site}.csv', sep = ';')
    
    # Mean WDI differences between years for each group
    df_ttest['meandiff'] = df_sample.groupby(['variable']).apply(lambda x: round(x.loc[x.year == 2020,'wdi'].mean() - x.loc[x.year == 2021,'wdi'].mean(),2))
    
    PATH_SAVE = f'../figures/Fig_2_{site}.png'
    
    ax = PlotBoxWhisker(df_m_s, yvar,
                       label_order = label_order,
                       colors = colordict,
                       showSignif=True,
                       data_ttest = df_ttest,
                       showWater = True,
                       PATH_OUT=PATH_SAVE,
                       saveFig = False)
    plt.show()
    
    # Multicomparison of WDI differences per plant community
    mask = np.logical_and(df_m_s.loc[df_m_s.year == 2020].variable != 'water',
                          df_m_s.loc[df_m_s.year == 2020].variable != 'mud'
                          )
    df_sample_2 = df_m_s.loc[df_m_s.year == 2020].loc[mask].sample(frac = 0.01,random_state = seed)
    
    mc_wdi = MultiComparison(df_sample_2['diff_wdi'], df_sample_2['variable'])
    print(mc_wdi.tukeyhsd())
    
    # Save test results in dataframe (for significance brackets in plot)
    thsd = pd.DataFrame(data = mc_wdi.tukeyhsd()._results_table.data[1:], 
                        columns = mc_wdi.tukeyhsd()._results_table.data[0])
    
    
    # Exporttesttable_wdi_diff_{site}
    # thsd.to_csv(f'./tables/results/Table_S_{site}.csv', sep=';', index = False)
    print('done.')

# %% 3. COMPUTE & PLOTFCOVER IN 5 x 5 m GRID CELLS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from osgeo import gdal
import rioxarray as rxr
import xarray as xr

from FigFunctions import MapFcover_np

# give window side length in number of pixels (33 px of 15 cm resolution make 4.95 m)
windowsize = 33

ylabel = 'Grid cell $\Delta WDI_{2020 - 2021}$(-)'

# Set binning variables
bin_var = 'meanT'
val_var = 'meanT' if bin_var == 'fcover' else 'fcover'

bin_interval = 1 if bin_var == 'fcover' else 0.005

windowsize_m = int(round(windowsize*.15,0))

for i,site in enumerate(sites):
    print(f'Mapping fCover in {site} ...')   
    
    df_m_s = df_list[i]
    
    label_order = df_m_s.variable.cat.categories.to_list()
    
    df_out = MapFcover_np(windowsize, I_cl_list[i],(I_wdi_20_list[i] - I_wdi_21_list[i]))
    
    df_out = df_out.dropna()
    
    PATH_SAVE = f'../figures/Fig_3_{site}.png'
    
    ax = PlotFcoverVsTemperature(data = df_out, 
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

# %% 6. SUPPLEMENTARY FIGURES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# %% 6 a. Plot histograms of fCover per class
g = sns.FacetGrid(df_out, col = 'classes',
                  sharey = True,col_wrap=2,
                  height = 6, hue = 'classes',
                  palette = colordict)
g.map_dataframe(sns.histplot, x ='fcover')
g.set_titles(col_template="{col_name}")
g.set(xlim=(0, 100),
      xlabel = 'fCover (%)')

plt.savefig(f'./figures_and_maps/classification/fCover_hist_{site}.png',
            bbox_inches = 'tight', facecolor='white')

#%% 6 b. Weather conditions:
# --------
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
    
sns.set_theme(style="whitegrid",
              font_scale = 2,
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

ylabs = ['Air temperature (°C)','Incoming shortwave radiation (W m$^{-2}$)','Wind speed (m$^{-s}$)']
offsets = [1,30,.2]
labs = ['a)','b)','c)']

conds = np.full((3, 3), False) # Boolean array for GetMeanTair function
np.fill_diagonal(conds,True)

for i,var in enumerate(['Barani_Temp_2_Avg','CMP21_IN_Avg','Cup_2M_Avg']):
    
    
    p = [ axs[i].plot(df['time'],
                      df[var],
                      color = cc,
                      marker = 'o',label = df.index.year[0]) for df,cc in zip(dfl,['brown','midnightblue'])]
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
                        verticalalignment = 'center_baseline',
                        horizontalalignment='left')
            
    axs[i].set(ylabel = ylabs[i],
               xlabel = 'Local time')
    axs[i].xaxis.set_major_formatter(md.DateFormatter('%H:%M'))


fig.autofmt_xdate()
fig.legend(p,labels = [2020,2021],loc = 'upper center',ncol = 2)

PATH_OUT = r'..\figures\Fig_S2.png'
plt.savefig(PATH_OUT,bbox_inches = 'tight',dpi=300)

#%% 6 c. SPEI3:
# --------
import datetime

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
sub_label = [ 'a)', 'b)']

# obtain a colormap and normalize the line colors based on the year
cmap = plt.cm.get_cmap(cm.vik)
norm = plt.Normalize(df_spei.year.min(), df_spei.year.max())

for i,spei in enumerate(['spei_3_months','spei_6_months']):

    # plot each group as a separate line on the same figure
    for name, group in groups:
        x = group.month
        y = group[spei]
        c = cmap(norm(name))
        if name == 2020 or name == 2021:
            axs[i].plot(x, y, 
                        alpha = .6,
                        color=c,
                        label=name, lw=5)
            y_annot = group[spei].iloc[6] # July SPEI
            axs[i].annotate(name, xy=(7, y_annot), xytext=(7.5 + 2021 % name, 2.5), 
                            transform = axs[i].transData,
                            arrowprops=dict(arrowstyle='->', color='black'))
        else:
            axs[i].plot(x, y, 
                        alpha = .3,
                        color=cmap(norm(name)), 
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
                   ls = '--', lw = 2, color = 'k')
    axs[i].annotate('10$^{th}$ percentile',
                    xy = (5.2,df_spei_summer[spei].quantile(.1)), 
                    xytext = (5.1, - 2.4),
                    va = 'top',transform = axs[i].transData,
                    arrowprops=dict(arrowstyle='->', color='black', lw = 2),
                    # bbox = dict(boxstyle='round', facecolor=(1, 1, 1, 0.7), edgecolor=(1, 1, 1, 0.7)),
                    color = 'k')
    
    # set the axis labels and title
    axs[i].set(ylabel = var_name[i],
               xlim = [5,9],
               ylim = [-3.5, 3.5],
               yticks = np.arange(-3,4,1),
               yticklabels = np.arange(-3,4,1),
               xticks = np.arange(5,10),
               xticklabels = ['May','Jun','Jul','Aug','Sep'])
    axs[i].text(-0.15, 0.95,sub_label[i], 
                transform=axs[i].transAxes,
                fontweight='bold', va='top', ha='right')


fig.suptitle('Standardized Precipitation-Evaporation Index (z-values)')
plt.savefig('../figures/Fig_S1.png',bbox_inches = 'tight', facecolor='white')

# %% 6 d. Plot Sensor temperature over time
sys.path.append(r".\code\main_scripts")

from glob import glob
from sklearn.metrics import r2_score

fl = glob(r'C:\data\0_Kytalyk\0_drone\internal_data\metadata*.csv')

unc_instr = .1
cutoff = 13

df = pd.concat(map(lambda fn:GatherData(fn, unc_instr, FitOnUnstable = False),fl), 
               ignore_index=False)

sns.set_theme(style="ticks",
              # rc={"figure.figsize":(20, 10)},
              font_scale = 2.5)

g = sns.FacetGrid(df, col="site",  row="year", sharex=True, margin_titles=True,
                  height = 4, aspect = 1.5)

g.map(sns.lineplot, "flighttime_min", "T_sensor", color = 'gray')
g.map(sns.scatterplot, "flighttime_min", "T_sensor",'isStable',palette = "Paired",marker = '+')

g.set(xlim=(0, 45), ylim=(25, 45), xticks=np.arange(0,50,5), yticks=np.arange(25,50,5))

leg_text = 'Is the sensor temperature stable? \n i.e. within %.1f °C of the minimum?' % unc_instr
g.add_legend(title = leg_text) # , loc = "lower center", bbox_to_anchor=(.35, -.3)

g.set_titles(col_template="{col_name}")

for ax, label in zip(g.axes.flat, ['a)', 'b)','c)','d)','e)','f)']):
    ax.text(0.1, 1,label, 
            transform=ax.transAxes,
            fontweight='bold', va='top', ha='right')

for year,margin_title in zip([2020,2021],g._margin_titles_texts):
    margin_title.set_text(year)
    margin_title.set_rotation(0)
    
g.set_axis_labels(y_var='', x_var='')
g.fig.text(0.02, 0.5, 'Sensor temperature (°C)', va='center', rotation='vertical')
g.fig.text(0.35,0.02, 'Flight time (minutes)', va='center', rotation='horizontal')

# plt.savefig('./figures_and_maps/thermal_mosaics/Fig_S_sensordrift.png',bbox_inches = 'tight', facecolor='white')

plt.show()

# %% 6 e. PLOT SENSOR DRIFT
plot_fit = 'fit_2'

r_sq = df.groupby(['site','year']).apply(lambda x: r2_score(x['LST_deviation'], x[plot_fit]))

sns.set_theme(style="ticks",
              # rc={"figure.figsize":(20, 10)},
              font_scale = 2.5)

g = sns.FacetGrid(df, col="site",  row="year", sharex=True, margin_titles=True,
                  height = 4, aspect = 1.5)
g.map(sns.lineplot, "T_sensor", plot_fit,color = 'orange')

for i,site in enumerate(['Ridge','TLB']):
    data = df.loc[(df.site==site) & (df.year == 2020)]
    
    for line in g.axes[0][i+1].lines[:2]:
        line.set_alpha(0)
        
    sns.lineplot(data = data[data.flighttime_min <= cutoff],
                 x = "T_sensor" ,y = plot_fit,
                 color = 'orange', ax = g.axes[0][i+1])
    sns.lineplot(data = data[data.flighttime_min > cutoff],
                 x = "T_sensor" ,y = plot_fit,
                 color = 'orange', ax = g.axes[0][i+1])

g.map(lambda y, **kw: plt.axhline(0,c = 'gray',alpha = .2,ls = '--'),'LST_deviation')
g.map(sns.scatterplot, "T_sensor", "LST_deviation", 'isStable',palette = 'Paired',alpha = .3)

g.set_axis_labels(y_var='', x_var='')

# Set common y-axis label
g.fig.text(0.02, 0.5, 'Correction temperature (°C)', va='center', rotation='vertical')
# Set common x-axis label
g.fig.text(0.35,0.02, 'Sensor temperature (°C)', va='center', rotation='horizontal')

g.set(xlim=(25, 45), ylim=(-15, 5), xticks=np.arange(25,50,5), yticks=np.arange(-15,6,5))

leg_text = 'Is the sensor temperature stable? \n i.e. within %.1f °C of the minimum?' % unc_instr
g.add_legend(title = leg_text)

g.set_titles(col_template="{col_name}")

# Annotate a value into each bar in each plot
for i,p in enumerate(g.axes.flat):
    p.annotate("R$^2$ = {:.2f}".format(r_sq.iloc[i]) , 
                (40,-12), # Placement
                ha='center', va='center', color='black', rotation=0, xytext=(0, 20),
                textcoords='offset points')


for year,margin_title in zip([2020,2021],g._margin_titles_texts):
    margin_title.set_text(year)
    margin_title.set_rotation(0)

for ax, label in zip(g.axes.flat, ['a)', 'b)','c)','d)','e)','f)']):
    ax.text(0.1, 1,label, 
            transform=ax.transAxes,
            fontweight='bold', va='top', ha='right')
    
g.fig.subplots_adjust(top=0.9)

# plt.savefig('./figures_and_maps/thermal_mosaics/Fig_S_TcorrvsTsens.png',
#             dpi = 200,bbox_inches = 'tight', facecolor='white')
plt.show()


# %% 6 f. Plot semivariogram
import skgstat as skg
from osgeo import gdal
import rioxarray as rxr
import xarray as xr

np.random.seed(seed)

labs = ['a)','b)','c)']

sns.set_theme(style="whitegrid",
              font_scale = 2,
              rc = {"axes.spines.right": False, "axes.spines.top": False})

fig,axs = plt.subplots(3,1,figsize = (15,10),sharex = True,sharey = True)

for i, site in enumerate(['CBH','Ridge','TLB']):
    PATH_21_TIR = rf'.\paper\data\mosaics\{site}_thermal_2021_resampled.tif'
    
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
    
    ax = axs[i]
    V.plot(axes = ax,grid = False)
    ax.axhline(V.parameters[1], ls = '--', color = 'gray') # horizontal for sill
    ax.axvline(V.parameters[0], ls = '--', color = 'gray') # vertical for range
    
    ax.set(xlabel ='',
            ylabel ='',title = site)

    ax.annotate('Range = {:.1f} m'.format(V.parameters[0]),
                (V.parameters[0] + 2, np.mean(ax.get_ylim())), 
                color = 'k', transform = ax.transData,
                clip_on=False, annotation_clip=False,
                verticalalignment = 'center_baseline', horizontalalignment='left')
    ax.text(0.02, 1.1, labs[i], transform=ax.transAxes, weight='bold')

ax.set(xlabel ='Distance (m)')
fig.text(-0.01, 0.5, 'Semivariance (matheron)', va='center', rotation='vertical')

fig.tight_layout()
fig.savefig('./figures_and_maps/thermal_mosaics/Fig_S_semivariogram.png',dpi = 200,bbox_inches = 'tight', facecolor='white')

# %% 6 g. TOMST TMS-4 differences
try:
    any(df_tmst_delta)
except:
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

# plot seasonal averages, if False only look at flightday data
seasonal = False

if seasonal:
    # Select data between 12:00 and 13:30
    df_tmst_delta_noon = df_tmst_delta.between_time('11:00','13:30')
    data_agg = aggregate_func(df_tmst_delta_noon)
    text = 'seasonal'
else:
    # Select data on flightday in 2021
    df_tmst_delta_flightday = df_tmst_delta.loc["2021-07-19"].between_time('13:50','16:30')
    data_agg = aggregate_func(df_tmst_delta_flightday)
    text = 'flightday'

lbl = {'water':'Open water',
       'mud': 'Mud',
       'wet': 'LW1',
       'ledum_moss_cloudberry': 'LW2', 
       'shrubs': 'HP2', 
       'dry': 'HP1',
       'tussocksedge': 'TS'}

# dT_air = ["dT2T3","dT1Tair",  "dT2Tair",  "dT3Tair" ]
# col_titles = ['$T_2 - T_3$', '$T_1 - T_{air}$', '$T_2 - T_{air}$', '$T_3 - T_{air}$']

dT_air = ["dT2Tair"]
col_titles = ['$T_2 - T_{air}$']

# melt data frame to long format for facet grid
df_melted = pd.melt(data_agg.loc[:,dT_air+['site','custom_class']], id_vars=['site', 'custom_class'],  value_name='T')
df_melted[['site','variable','custom_class']] = df_melted[['site','variable','custom_class']].astype("category")

# rename custom class to official labels
df_melted['custom_class'] = df_melted['custom_class'].map(lbl)
df_melted['site'] = df_melted['site'].map({'lakebed':'TLB','ridge':'Ridge'})

# Map colors as new column for boxplot
df_melted['color'] = df_melted['custom_class'].map(colordict)

order = ['LW1','HP1','HP2','TS']
palette = [colordict[x] for x in ['HP1','HP2','TS','LW1']]

g = sns.FacetGrid(df_melted, row='site', col='variable', 
                  margin_titles=True,height = 4, aspect = 1)
g.map_dataframe(sns.boxplot, 'custom_class', 'T', 
                hue = 'custom_class', palette = palette,
                linewidth=.7, width=1, boxprops=dict(alpha=.7),
                order = order, dodge = False,
                fliersize=0.8)
g.map(plt.axhline, color = 'gray',ls = '--',alpha = .5)

row_titles = ['TLB', 'Ridge']

# set the axis labels for the rows and columns
g.set_axis_labels("", "Temperature difference (°C)")
g.set_titles(row_template="{row_name}")

for ax, title in zip(g.axes[0], col_titles):
    ax.set_title(title)
    
# plt.savefig(f'./figures_and_maps/tomst/Fig_S_Tdifferences_{text}.png',dpi = 200,
#             bbox_inches = 'tight', facecolor='white')

# save difference in deltaT's between groups TOMST vs. drone:
from itertools import combinations

df_out = pd.DataFrame(columns = ['site','name','rel_diff_drone','rel_diff_tomst'], index = range(6),dtype = float)

i = 0
for s in ['TLB','Ridge']:
    # read drone data
    df_m_s = pd.read_csv(fr'.\data\thermal_data/{s}_data_thermal.csv',sep=';')
    
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
print(df_out)