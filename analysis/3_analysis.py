import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison

import matplotlib.pyplot as plt
from cmcrameri import cm
import seaborn as sns

from importlib import reload  

from skimage.io import imread

import sys, os
sys.path.append(r"C:\Users\nils\OneDrive - Universität Zürich UZH\Dokumente 1\5_code")

MAIN_PATH = r'C:/Users/nils/OneDrive - Universität Zürich UZH/Dokumente 1/1_PhD/5_CHAPTER1'
sys.path.append(r".\code\main_scripts")

os.chdir(MAIN_PATH + '/code/main_scripts')

from FigFunctions import PlotDensities, PlotBoxWhisker, PlotFcoverVsTemperature,Linear_Reg_Diagnostic


import myfunctions as myf

def GetMeanTair(df_flighttimes, df_fluxtower, site, year,
                returnTemp=True,returnSW = False,returnWS = False):
    """
    Calculate the air temperature during the drone flights from the flux tower 2m level. 
    Looks for values +/- 15 min from flight and computes average.

    Parameters
    ----------
    df_flighttimes : pd.DataFrame
        Dataframe containing the datetime of the flighttimes in local time (tz aware).
    df_fluxtower : pd.DataFrame
        Dataframe containing the air temperature (T107_A_2_Avg) with datetime index in local time (tz aware).
    site : str
        site of the drone flight, needed for the flight times.
    year : int
        year of the drone flight.

    Returns
    -------
    mean_Tair : float64
        Average air temperature during the drone flight.

    """
    row = df_flighttimes[np.logical_and(df_flighttimes.site == site,df_flighttimes.year == year)]
    start = row.start_time_local.item()
    end = row.end_time_local.item()
    
    df_sliced = df_fluxtower[(df_fluxtower.index >= start - pd.Timedelta(15,'min')) &
                             (df_fluxtower.index <= end + pd.Timedelta(15,'min'))]
    if returnTemp:
        return df_sliced.Barani_Temp_2_Avg.mean()
    
    if returnSW:
        return df_sliced.CMP21_IN_Avg.mean()
    
    if returnWS:
        return df_sliced.Cup_2M_Avg.mean()

def ndvi(NIR, RED):
    return ((NIR - RED) / (NIR + RED))
    # L = .5
    # return ((NIR - RED) / (NIR + RED + L) * (1 - L))

def PrepRasters(PATH_CL,PATH_WATERMASK,
                PATH_20_TIR,T_air_20,
                PATH_21_TIR,T_air_21,
                PATH_MSP_20,PATH_MSP_21,
                extent):
    
    I_cl = imread(PATH_CL)
    I_wm = imread(PATH_WATERMASK)
    
    I_cl = np.where(np.logical_and(I_cl == 3, I_wm != 3),255,I_cl)
    
    I_tir_20 = imread(PATH_20_TIR) -  T_air_20
    I_tir_21 = imread(PATH_21_TIR) - T_air_21 
    
    I_tir_20 = np.where(I_tir_20 < -100 ,np.nan, I_tir_20)
    I_tir_21 = np.where(I_tir_21 < -100 ,np.nan, I_tir_21)
    
    I_cl_s = I_cl[extent['ymin']:extent['ymax'],extent['xmin']:extent['xmax']]
    I_wm_s = I_wm[extent['ymin']:extent['ymax'],extent['xmin']:extent['xmax']]
    I_tir_20_s = I_tir_20[extent['ymin']:extent['ymax'],extent['xmin']:extent['xmax']]
    I_tir_21_s = I_tir_21[extent['ymin']:extent['ymax'],extent['xmin']:extent['xmax']]
    
    I_msp_20 = imread(PATH_MSP_20)
    I_msp_21 = imread(PATH_MSP_21)
    I_ndvi_20 = ndvi(I_msp_20[:,:,4],I_msp_20[:,:,2])
    I_ndvi_21 = ndvi(I_msp_21[:,:,4],I_msp_21[:,:,2])
    I_ndvi_20_s = I_ndvi_20[extent['ymin']:extent['ymax'],extent['xmin']:extent['xmax']]
    I_ndvi_21_s = I_ndvi_21[extent['ymin']:extent['ymax'],extent['xmin']:extent['xmax']]

    return I_cl_s,I_wm_s, I_tir_20_s, I_tir_21_s, I_ndvi_20_s, I_ndvi_21_s
    


def MeltArrays(arr, arr_cl,names, year):
    """
    Locates the temperature readings per land cover class and melts the results to a pd.DataFrame with:
        - the class labels = "variable"
        - the temperature values = "value"
        - a column including the year

    Parameters
    ----------
    arr : np.array
        Thermal mosaic as numpy array.
    arr_cl : np.array
        Classified mosaic of the site as numpy array.
    names : list
        List of the land cover classes that are found in the site.
    year : int
        Year of flight.

    Returns
    -------
    df_m : pd.DataFrame
        DESCRIPTION.

    """
    data = [arr[arr_cl == cl] for cl in np.unique(arr_cl)]
    df = pd.DataFrame.from_dict(dict(zip(names, 
                                         map(
                                             pd.Series, data)
                                         )
                                     )
                                )

    df_m = df.melt(value_name = 'Tdiff')
    df_m['year'] = year
    return df_m


def ScaleMinMax(x):
    """
    Normalizes an array by its range (min & max). 
    """
    # return (x - np.nanquantile(x,.0001)) / (np.nanquantile(x,.9999) - np.nanquantile(x,.0001))
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


from scipy.stats import ttest_ind

def DifferenceTest(df: pd.DataFrame, 
                   value_var:str, group_var: str,
                   alpha = 0.05):

    veg_types = df[group_var].unique()
    ttests = []
    for i,vty in enumerate(veg_types):
        for i2,vty2 in enumerate(veg_types):
            if i2 > 0:
                group1 = df.loc[df[group_var] == vty,value_var]
                group2 = df.loc[df[group_var] == vty2,value_var]
                
                t,p = ttest_ind(group1,group2)
                
                ttests.append([f'{vty} - {vty2}:',t.round(4), p.round(4)])
    
    threshold = alpha / len(ttests)
    print(f'\n Significant t-Tests below {threshold}')
    
    for t in ttests:
        if t[2] <= threshold:
            print(t)
            
    return ttests
    
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

def pretty_table(rows, column_count, column_spacing=4):
    aligned_columns = []
    for column in range(column_count):
        column_data = list(map(lambda row: row[column], rows))
        aligned_columns.append((max(map(len, column_data)) + column_spacing, column_data))

    for row in range(len(rows)):
        aligned_row = map(lambda x: (x[0], x[1][row]), aligned_columns)
        yield ''.join(map(lambda x: x[1] + ' ' * (x[0] - len(x[1])), aligned_row))

# %% 0. LOAD DATA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
os.chdir(MAIN_PATH)

# configure variables
seed = 15
study_area_sidelength = 400 # in m

colordict = dict(HP1 = '#AAFF00', 
                 HP2 = '#38A800', 
                 LW1 = '#00FFC5',
                 LW2 = '#A90EFF',
                 TS = '#3D3242',
                 mud = '#734C00',
                 water = '#005CE6')

classdict = {0:'HP1',
             1:'LW1',
             2:'HP2',
             3:'water',
             4:'LW2',
             5:'mud',
             7:'TS',
             255:'nodata'}

print('Loading data...')

df_fluxtower = myf.read_fluxtower('C:/data/0_Kytalyk/7_flux_tower/CR3000_Kytalyk_Meteo.dat')

parser = lambda date: pd.to_datetime(date,format='%d.%m.%Y %H:%M').tz_localize('Asia/Srednekolymsk')

df_flighttimes = pd.read_csv(r'C:\data\0_Kytalyk\0_drone\flight_times.csv', 
                             sep = ';', 
                             parse_dates = [4,5],
                             date_parser=parser)

sites = ['CBH', 'TLB', 'Ridge']
df_list = list(range(3))

for i,site in enumerate(sites):
    
    print(f'...for {site}')
    
    PATH_20_TIR = rf'.\paper\data\mosaics\{site}_thermal_2020_resampled.tif'
    PATH_21_TIR = rf'.\paper\data\mosaics\{site}_thermal_2021_resampled.tif'
    
    PATH_MSP_20 = rf'.\paper\data\mosaics\{site}_msp_2020_resampled.tif'
    PATH_MSP_21 = rf'.\paper\data\mosaics\{site}_msp_2021_resampled.tif'
        
    PATH_CL = rf'.\paper\data\landcover\{site}_2021_classified_filtered5.tif'
    PATH_WATERMASK = rf'.\paper\data\landcover\{site}_2020_classified_filtered5.tif'
        
    # Set number of pixels in study area
    dxy = int(study_area_sidelength / .15) # 667 pixels along 100m ==> 1 ha
    
    if site == 'Ridge':
        ymin = 200
        xmin = 1200
    else:
        ymin = 300
        xmin = 200
    
    ymax = ymin + dxy
    xmax = xmin + dxy
    
    # Study area extent
    extent = {'xmin': xmin,'xmax': xmax,
              'ymin': ymin, 'ymax':ymax}
    
    # Mean Tair during droneflight
    T_air_20 = GetMeanTair(df_flighttimes, df_fluxtower, site, 2020)
    T_air_21 = GetMeanTair(df_flighttimes, df_fluxtower, site, 2021)
    
    # Read raster data
    I_cl_s,I_wm_s, I_tir_20_s, I_tir_21_s, I_ndvi_20_s, I_ndvi_21_s = PrepRasters(PATH_CL,PATH_WATERMASK,
                                                                                  PATH_20_TIR,T_air_20,
                                                                                  PATH_21_TIR,T_air_21,
                                                                                  PATH_MSP_20,PATH_MSP_21,
                                                                                  extent)
    
    names = [classdict[i] for i in np.unique(I_cl_s)]
    
    I_cl_s = np.where(I_cl_s == 255,np.nan, I_cl_s) # replace no data (255) with nan
    
    try: # check if a csv of image data already exists
        df_m_s = pd.read_csv(fr'.\data\thermal_data/{site}_data_thermal.csv',sep=';')
        print('Data for this site is available. Loading ...')
        # df_m_s_stats = pd.read_csv(fr'.\results\statistics\map_stats\{site}_statistics.csv',sep = ';')
    except:
        print('No thermal data found. Sampling...')
        
        df_m = pd.concat(
            map(lambda arr,year: MeltArrays(arr,I_cl_s,names,year),
                [I_tir_20_s,I_tir_21_s],[2020,2021]), 
            ignore_index=False).reset_index(drop=True)
        
        d = pd.concat(
            map(lambda arr,year: MeltArrays(arr,I_cl_s,names,year),
                [I_ndvi_20_s,I_ndvi_21_s],[2020,2021]), 
            ignore_index=False).reset_index(drop=True)
        df_m['NDVI'] = d.Tdiff
        
        # Compute standardize temperatures:
        df_m['wdi'] = df_m.groupby('year').Tdiff.apply(ScaleMinMax) # Standardize MinMax
        
        # Generate random sample of size n per class and year:
        n = 20000
        df_m_s = df_m.sample(frac = 1, random_state = seed) # shuffle data
        df_m_s = pd.concat(
            [g[g.Tdiff.notna()][:n] for _, g in df_m_s.groupby(['year', 'variable'], sort=False, as_index=False)],
            # ignore_index=True 
        )  
        
        df_m_s.to_csv(fr'.\data\thermal_data/{site}_data_thermal.csv',sep=';')
        
        df_m_s_stats = df_m_s.groupby(['year','variable']).describe()
        df_m_s_stats.to_csv(fr'.\results\statistics\map_stats\{site}_statistics.csv',sep = ';')
    
    # Sort data along moisture gradient
    if site == 'TLB':
        label_order = ['water','LW1', 'HP1', 'HP2']
    elif site == 'CBH':
        label_order = ['water','mud','LW1','LW2', 'HP1', 'HP2','TS']
    elif site == 'Ridge':
        label_order = ['LW1','HP2','TS']
         
    df_m_s['variable'] = df_m_s.variable.astype("category").cat.set_categories(label_order, ordered=True)
    df_list[i] = df_m_s

print('done.')

# %% 2. DENSITY PLOT IN NORMAL YEAR & DIFFERENCE TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
group_var = 'variable'
xvar = 'Tdiff'

from scipy.stats import normaltest
from statsmodels.stats.multicomp import MultiComparison

alpha = 1e-3

fig, axs = plt.subplots(nrows = 3, figsize=(30,20), dpi = 200,sharex = True) # create subplot instance
labs = ['a)','b)','c)']

PATH_SAVE = r'.\figures_and_maps\thermoregulation\Figure_1.png'

for i,site in enumerate(sites):
    print(f'Analysing {site}...')
    
    normtest = df_list[i].groupby(['year','variable']).apply(lambda x: normaltest(x[xvar]))
    df_normtest = pd.DataFrame([[z,p] for z,p in normtest.values],columns = ['zscore','pval']).set_index(normtest.index)
    df_normtest['reject'] = df_normtest.pval < alpha
    
    print('Normality tests: \n',df_normtest)
    
    # ANOVA
    # import statsmodels.api as sm
    # from statsmodels.formula.api import ols
    # cw_lm=ols('Tdiff ~ C(variable)', data=df_list[i].loc[df_list[i].year==2021]).fit() #Specify C for Categorical
    # print(sm.stats.anova_lm(cw_lm, typ=2))
    
    # Tukey HSD
    # ---------
    # sample 200 observations per plant community and year to avoid effect of large sample
    
    mc20,mc21 = df_list[i].loc[(df_list[i].variable != 'water') & 
                           (df_list[i].variable != 'mud')].groupby(['variable','year']).sample(frac = 0.01,
                                                                                           random_state = seed).groupby(
                               ['year']).apply(lambda x: MultiComparison(x[xvar], x[group_var]))
    
    # print('2020: \n',mc20.tukeyhsd(), end = '\n')
    print('2021: \n', mc21.tukeyhsd(), end = '\n')                                                                                           
                                                                                               
    # pairwise t-test with Bonferroni correction
    # print('2020: \n',mc20.allpairtest(ttest_ind, method = 'bonf')[0], end = '\n')
    # print('2021: \n', mc21.allpairtest(ttest_ind, method = 'bonf')[0], end = '\n')
    
    thsd = pd.DataFrame(data=mc21.tukeyhsd()._results_table.data[1:], 
                        columns=mc21.tukeyhsd()._results_table.data[0])
    
    PlotDensities(df_list[i], xvar,
                  PATH_SAVE,
                   ax = axs[i],
                  colors = colordict,
                  showSignif = True, data_pairtest = thsd,
                  order = label_order,
                  showTair = False, showWater = False,
                  showBothYears = False,
                  saveFig = False)
    
    axs[i].text(0.05, .9, labs[i], transform=axs[i].transAxes, weight='bold')
    
    # with open(f'./results/statistics/difference_tests/testtable_2021_{site}_{xvar}.csv', 'w') as output:
    #     print(mc21.tukeyhsd()._results_table.as_csv(),file = output)
        # print(mc21.tukeyhsd(), file = output)

# [ax.set(xlim = [5,25]) for ax in axs]

plt.savefig(PATH_SAVE,bbox_inches = 'tight',dpi=300)
plt.show()


# %% deltaT along transect for two years
from matplotlib.lines import Line2D

transect_y = 700

plt.plot(ScaleMinMax(I_tir_20_s[1200:1700,transect_y]),
         label = '2020')
plt.scatter(np.arange(0,500),ScaleMinMax(I_tir_20_s[1200:1700,transect_y]),
            c = I_cl_s[1200:1700,transect_y], cmap = 'tab10')
points = plt.plot(ScaleMinMax(I_tir_21_s[1200:1700,transect_y]),
                  label = '2021',ls = '--')
scatter = plt.scatter(np.arange(0,500),ScaleMinMax(I_tir_21_s[1200:1700,transect_y]),
            c = I_cl_s[1200:1700,transect_y],  cmap = 'tab10')

plt.legend(handles=scatter.legend_elements()[0] + [Line2D([0], [0], color='k', ls='-', lw = 1),
                                  Line2D([0], [0], color='k', ls='--', lw = 1)],
           labels = ['dry','wet','shrubs','ledum','tusscksedge','2020','2021'])
plt.show()

# %%
from matplotlib.colors import from_levels_and_colors
# sns.set_theme()

cmap, norm = from_levels_and_colors(list(np.arange(-.5,8)),
                                    ['#AAFF00','#00FFC5',
                                     '#38A800','#005CE6',
                                     '#FF7F7F','#734C00',
                                     '#FFFFFF','#FFFF73'])
# cmap = ListedColormap(colordict)
# cmap = LinearSegmentedColormap.from_list("name", colordict)

fig, axs = plt.subplot_mosaic([['a)', 'b)'], ['c)', 'c)']],
                              layout='constrained')

p1 = axs['a)'].imshow(I_tir_21_s,cmap = cm.lajolla)
p2 = axs['b)'].imshow(I_cl_s, cmap = cmap,norm = norm)
p3 = PlotDensities(df_m_s, xvar,PATH_SAVE,
              showSignif = True, data_pairtest = thsd,
              order = label_order,
              showTair = False, showWater = False,
              showBothYears = False,
              saveFig = True)

fig.colorbar(p1, ax=axs['a)'], label = '$T_{surf} - T_{air}$')
# # Make some room for the colorbar
# fig.subplots_adjust(bottom=0.87)

# # Add the colorbar outside...
# box = ax2.get_position()
# pad, width = 0.02, 0.02
# cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
# fig.colorbar(p,orientation='horizontal', cax=cax)

# %% 3. BOXPLOT IN BOTH YEARS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
from scipy.stats import ttest_ind, ttest_rel

yvar = 'wdi'

fig, axs = plt.subplots(nrows = 3, figsize=(30,40), dpi = 200,sharey = True) # create subplot instance

labs = ['a)','b)','c)']

PATH_SAVE = r'.\figures_and_maps\thermoregulation\Figure_3.png'

for i,site in enumerate(sites):
    print(f'Analysing {site}...')
    
    # Sort data along moisture gradient
    if site == 'TLB':
        label_order = ['water','LW1', 'HP1', 'HP2']
    elif site == 'CBH':
        label_order = ['water','mud','LW1','LW2', 'HP1', 'HP2','TS']
    elif site == 'Ridge':
        label_order = ['LW1','HP2','TS']
    
    # sample random observations per plant community and year to avoid effect of large sample
    df_sample = df_list[i].groupby(['variable','year']).sample(frac = 0.02,
                                                               random_state = seed)
    
    # Welch's ttest for unequal variances
    alpha = .05
    ttest = df_sample.groupby(['variable']).apply(lambda x: ttest_ind(x.loc[x.year==2020,yvar],
                                                                      x.loc[x.year==2021,yvar], 
                                                                      equal_var=False))
    df_ttest = pd.DataFrame([[z,p] for z,p in ttest.values],columns = ['tstat','pval']).set_index(ttest.index)
    df_ttest['reject'] = df_ttest.pval < alpha
    
    # ttest for two dependent samples
    df_ttest_rel = df_sample.groupby(['variable']).apply(lambda x: ttest_rel(x.loc[x.year==2020,yvar],
                                                                             x.loc[x.year==2021,yvar]))
    # Print differences between years for each group
    print(f'Difference in {yvar} between years: \n',
          df_sample.loc[df_sample.year == 2020].groupby('variable')[yvar].mean() - df_sample.loc[df_sample.year == 2021].groupby('variable')[yvar].mean())
    
    # label_order = ['water','wet','ledum_moss_cloudberry', 'shrubs', 'dry','tussocksedge']
    
    PlotBoxWhisker(df_list[i], yvar,
                   ax = axs[i],
                   label_order = label_order,
                   colors = colordict,
                   showSignif=True, data_ttest = df_ttest,
                   showWater = False,
                   PATH_OUT=PATH_SAVE,
                   saveFig = True)
    
    axs[i].text(-0.1, 1.05, labs[i],
                transform=axs[i].transAxes, 
                weight='bold')
    
# [ax.set(xlim = [5,25]) for ax in axs]

plt.savefig(PATH_SAVE,bbox_inches = 'tight',dpi=300)
plt.show()
    
# %% 4. COMPUTE FCOVER IN 1.5 x 1.5 m QUADRATS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from osgeo import gdal
import rioxarray as rxr
import xarray as xr
import rasterio
from scipy.spatial import distance_matrix

import time

def GetFcover(dA_cl, dA_tir,idx):
    df = pd.DataFrame(columns = ['classes','count','fcover','meanT','sigmaT'])
    df['classes'], df['count'] = np.unique(dA_cl,return_counts = True)
    
    df['fcover'] = df['count'] / df['count'].sum() *100
    
    for i,cl in enumerate(df['classes']):
        df.loc[i,'meanT'] = np.nanmean(xr.where(dA_cl == cl, dA_tir,np.nan))
        df.loc[i,'sigmaT'] = np.nanstd(xr.where(dA_cl == cl, dA_tir,np.nan))
    
    df['idx'] = idx
    return df


def MapFcover(windowsize, dA_classes, dA_thermal):
    df_out = pd.DataFrame(columns = ['classes','count','fcover','meanT','sigmaT'])
    window_radius = int(windowsize/2)
    
    centers = np.arange(window_radius,dA_classes.shape[1] - window_radius, window_radius)
    
    ycoords,xcoords = np.meshgrid(centers,centers) # build grid with window centers
    
    # plt.imshow(dA_classes[0],vmax = 8,cmap = cm.bamako);
    # dxy = window_radius/2
    # plt.imshow(I_cl_s,vmax = 8,cmap = cm.bamako);
    # plt.plot(ycoords,xcoords,'.',c = 'w');
    # plt.plot(xcoords-dxy,ycoords-dxy,'-',c = 'w');
    # plt.plot(ycoords -dxy ,xcoords-dxy,'-',c = 'w');
    # plt.plot(ycoords +dxy ,xcoords+dxy,'-',c = 'w');
    # plt.plot(xcoords + dxy,ycoords+dxy,'-',c = 'w');
    # plt.savefig(r'C:/Users/nils/Documents/1_PhD/5_CHAPTER1/figures_and_maps/thermal_mosaics/FcoverGridExample.png',bbox_inches='tight')
    
    start = time.time()
    
    # Using numpy reshape & parallel:
    # ====
    a = dA_classes.values # get numpy array from xarray
    c = a[a.shape[0] % windowsize:, a.shape[0] % windowsize:]
    
    nchunkrows = int(c.shape[0] / windowsize) # get n (nr. of chunkrows/columns), i.e. 8 x 8 = 64 chunks
    L = np.array_split(c,nchunkrows) # select nxn subarrays
    c = np.vstack([np.array_split(ai,nchunkrows,axis=1) for ai in L])
    
    b = dA_thermal.values # get numpy array from xarray
    t = b[b.shape[0] % windowsize:, b.shape[0] % windowsize:]
    
    L = np.array_split(t,nchunkrows) # select nxn subarrays
    t = np.vstack([np.array_split(ai,nchunkrows,axis=1) for ai in L]) # select nxn sub-array that fits to window operation
    
    # df_out = pd.concat([GetFcover(cl,tir,0) for cl,tir in tqdm(zip(c,t), total = c.shape[0] ) ], axis=0)
    
    ncores = 3
    df_out = pd.concat( ProgressParallel(n_jobs = ncores)(delayed(GetFcover)(cl,tir,i) for i,(cl,tir) in enumerate( zip(c,t) ) ),
                       axis = 0)
    
    # ii = 0
    # for yv,xv in tqdm(zip(ycoords.flatten(),xcoords.flatten()),total = len(centers)**2):
    #     df_out = pd.concat([df_out,
    #                         GetFcover(dA_classes.isel(
    #                             x = slice(xv-window_radius, xv+window_radius),
    #                             y = slice(yv-window_radius, yv+window_radius)),
    #                                   dA_thermal.isel(
    #                                       x = slice(xv-window_radius, xv+window_radius),
    #                                       y = slice(yv-window_radius, yv+window_radius)
    #                                       ),ii)],
    #                         axis = 0
    #                         )
    #     ii += 1
    print(time.time() - start, 's')    
    
    classdict = {0:'dry', 1:'wet',2:'shrubs',3:'water',4:'ledum_moss_cloudberry',5:'mud',7:'tussocksedge'}
    df_out['classes'] = df_out['classes'].map(classdict)
    df_out['meanT'] = df_out.meanT.astype(float)
    
    return df_out

dA_tir20 = rxr.open_rasterio(PATH_20_TIR)
dA_tir21 = rxr.open_rasterio(PATH_21_TIR)
 
dA_cl = rxr.open_rasterio(PATH_CL)
da_wm = rxr.open_rasterio(PATH_WATERMASK)

T_air = GetMeanTair(df_flighttimes, df_fluxtower, site, 2020)

dA_tir20_s = dA_tir20.isel(x = slice(xmin,xmax), y = slice(ymin,ymax)) - T_air
dA_tvdi20_s = ScaleMinMax(dA_tir20_s[0])

T_air = GetMeanTair(df_flighttimes, df_fluxtower, site, 2021)

dA_tir21_s = dA_tir21.isel(x = slice(xmin,xmax), y = slice(ymin,ymax)) - T_air - 9.29
dA_tvdi21_s = ScaleMinMax(dA_tir21_s[0])

dA_cl_s = dA_cl.isel(x = slice(xmin,xmax), y = slice(ymin,ymax))

windowsize = 33 # give window side length in number of pixels
# df_out = MapFcover(windowsize, dA_cl_s[0],dA_tir21_s[0]); xlabel = 'Average quadrat $T_{surf}$ - $T_{air}$ (°C)'

df_out = MapFcover(windowsize, dA_cl_s[0],(dA_tvdi20_s - dA_tvdi21_s )); xlabel = 'Grid cell $\Delta WDI_{2020 - 2021}$(-)'
# df_out = MapFcover(windowsize, dA_cl_s[0],(dA_tvdi20_s )); xlabel = 'Average quadrat $TVDI_{2020}$ (-)'
# df_out = MapFcover(windowsize, dA_cl_s[0],(dA_tvdi21_s )); xlabel = 'Average quadrat $TVDI_{2021}$ (-)'
# df_out = MapFcover(windowsize, dA_cl_s[0],(dA_tir20_s - dA_tir21_s)[0]); xlabel = 'Average quadrat $\Delta T_{2020}$ - $\Delta T_{2021}$ (-)'

# %% Plot histograms of fCover per class
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

# %% Plot TPI in the meanT vs. fcover relationship
if site == 'CBH':
    dA_tpi = rxr.open_rasterio(r'.\data\wetness_index\TPI_CBH.tif')
    dA_tpi_s = dA_tpi.isel(x = slice(xmin,xmax), y = slice(ymin,ymax))
    
    try:
        any(df_out['tpi'])
    except:
        df_out_tpi = MapFcover(windowsize, dA_cl_s[0],dA_tpi_s[0])
        df_out['tpi'] = df_out_tpi.meanT
        
        mask = np.logical_and(df_out.classes != 'water',df_out.classes != 'mud')
        data_masked = df_out.loc[mask,:]
        
        bin_var = 'meanT'
        val_var = 'meanT' if bin_var == 'fcover' else 'fcover'
        bin_interval = 1 if bin_var == 'fcover' else 0.005
    
        bins = np.arange(0,data_masked[bin_var].max(),bin_interval)
        group = data_masked.groupby(['classes',
                                  pd.cut(data_masked[bin_var], bins)
                                  ])
        
        classnames = list(data_masked['classes'].unique())
        n_classes = len(classnames)
        
        df_p = pd.DataFrame({'classes': np.repeat(classnames,len(bins)-1),
                             bin_var: list(bins[1:]) *n_classes,
                             val_var: group[val_var].apply(np.nanmean).reindex(classnames,level=0).values, # have to reindex to insert values at right class
                             'tpi': group['tpi'].apply(np.nanmean).reindex(classnames,level=0).values, # have to reindex to insert values at right class
                             val_var + '_std':group[val_var].std().reindex(classnames,level=0).values})
    
    cl = 'LW2'
    
    s = 200
    f,ax = plt.subplots(figsize = (20,10))
    scatter = sns.scatterplot(
                data=df_p.loc[df_p['classes'] == cl,:].dropna(),
                norm = plt.Normalize(vmin=-.06, vmax=.06),
                x='meanT', y='fcover',hue = 'tpi',palette = cm.vik_r,
                s = s,ax = ax);
    for lh in scatter.legend_.legendHandles: 
        lh._sizes = [s] 
    # handles, labels = scatter.legend_elements(prop="colors")
    # labels = ["low", "high"]
    # legend2 = ax.legend(handles, labels, loc="upper left",title = 'TPI (m)')
    # ax.add_artist(legend2)
    ax.set(xlabel = 'Mean grid cell $\Delta$WDI (-)',
           ylabel = 'Grid cell fCover (%)')
    # plt.savefig( fr'.\figures_and_maps\thermoregulation\{site}_TPI_fcover_{cl}.png'   ,bbox_inches = 'tight',dpi=150)

    
# %% 5. PLOT FCOVER VS.TEMPERATURES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bin_var = 'meanT'
val_var = 'meanT' if bin_var == 'fcover' else 'fcover'
# val_var = 'sigmaT' if bin_var == 'fcover' else 'fcover'; xlabel = 'Std. deviation of quadrat $\Delta T_{std,2020}$ - $\Delta T_{std,2021}$ (-)'

bin_interval = 1 if bin_var == 'fcover' else 0.005

windowsize_m = int(round(windowsize*.15,0))

PATH_OUT = rf'.\figures_and_maps\thermoregulation\yeardiff_{site}_Fcover_vs_mean_deltaT_{windowsize_m}m.png'

ax,df_p = PlotFcoverVsTemperature(data = df_out, 
                                  binning_variable = bin_var,
                                  bin_interval = bin_interval,
                                  value_variable = val_var,
                                  label_order = label_order,
                                  colors = colordict,
                                  model = 'cubic', plot_type = 'regplot',
                                  PATH_OUT = PATH_OUT,
                                  xlab = xlabel,xvar = 'meanT',
                                  saveFig = True)

# %% Does the combination of classes have an effect?
xvar = 'meanT'

print('Reshaping data...')
df_out_sample = df_out.groupby('idx').sample(frac=.6)

# exclude "quadrats" with water and mud inside
mask = np.logical_and(df_out_sample.classes != 'water',
                      df_out_sample.classes != 'mud')

# Raandomly sample 60% and mask out the water and mud quadrats
dff2 = df_out_sample.loc[mask,:].pivot(index = 'idx',
                                       columns = 'classes',
                                       values=xvar)
dff_fcov = df_out_sample.loc[mask,:].pivot(index = 'idx',
                                      columns = 'classes',
                                      values='fcover')

# generate number for unique combinations of land cover types
combination_number = dff2.notna().astype(int).apply(
    lambda row: int(''.join(str(x) for x in row)), axis=1)

dff2['column_names'] = dff2.notna().apply(
    lambda row: '_'.join([col for col in dff2.columns if row[col]]), axis=1)

# compute mean temperature for that community combination
dff2[xvar] = dff2.iloc[:,0:-1].mean(axis=1)
dff2['fcover'] = dff_fcov.mean(axis=1)

order = dff2.groupby('column_names')[xvar].mean().sort_values(ascending=True)

f,ax = plt.subplots(figsize = (60,10))
ax = sns.violinplot(dff2,x = 'column_names',y = xvar,
                    inner="quart",order = order.index)
ax.set_xticklabels(ax.get_xticklabels(),rotation=30,
                   horizontalalignment='right')
ax.set(xlabel = '',
       ylabel = 'mean $\Delta$WDI for combo (-)')

PATH_OUT = rf'.\figures_and_maps\thermoregulation\DeltaWDI_{site}_cover_combos.png'
plt.savefig(PATH_OUT,bbox_inches = 'tight',dpi=300)

# ANOVA
print('Running ANOVA...')
wdi_aov = ols('meanT ~ C(column_names)', data = dff2).fit() #Specify C for Categorical
print(sm.stats.anova_lm(wdi_aov, typ=2))

# Tukey HSD
print('Running Tukey HSD test...')
mc_wdi =  MultiComparison(dff2[xvar], dff2['column_names'])
result = mc_wdi.tukeyhsd()
# print(result)

with open(f'./results/statistics/difference_tests/DeltaWDI_{site}_{xvar}.csv', 'w') as output:
    print(mc_wdi.tukeyhsd()._results_table.as_csv(),file = output)

# %%
"""
how does fcover of community Y influence the WDI change of community X?
Run OLS on all combinations of communities and report coefficients.
"""
import scipy
# # example for 1 pair

# vegs = ['dry','wet']
# df_out_sel = df_out.loc[df_out.fcover != 100,:]

# df_out_sel_pairs = pd.merge(df_out_sel.loc[df_out_sel.classes ==vegs[0],:],
#                             df_out_sel.loc[df_out_sel.classes ==vegs[1],:],
#                             on = 'idx')
# ax = sns.scatterplot(df_out_sel_pairs,
#                      y = 'meanT_x',x = 'fcover_y',
#                      hue = 'fcover_x',palette = plt.cm.Greens);
# sns.regplot(df_out_sel_pairs,
#             y = 'meanT_x',x = 'fcover_y',
#             scatter=False, color=".1");
# ax.set(xlabel = f'Fcover {vegs[1]} (%)',ylabel = f'$\Delta$ WDI {vegs[0]} (-)')
# ax.legend_.set_title(f'Fcover {vegs[0]} (%)')

# # with OLS 
# lm1 = ols('meanT_x ~ fcover_y', data = df_out_sel_pairs).fit()
# print(lm1.summary())

# cls = Linear_Reg_Diagnostic(lm1);
# cls()

df_out_sel = df_out.loc[df_out.fcover != 100,:]

# create a list for the model objects
models = []

# Create a list of unique categories in column 1
categories = df_out_sel['classes'].unique()

# create an empty dataframe to store the results
results_df = pd.DataFrame()

# Loop through all possible category pairs
for i, category1 in enumerate(categories):
    for category2 in categories:
        
        print(f"Category pair: {category1} vs {category2}")
        
        # Filter the DataFrame to include only data from the current category pair
        data = pd.merge(df_out_sel[df_out_sel.classes == category1],
                        df_out_sel[df_out_sel.classes == category2],
                        on = 'idx')
        
        # Create a design matrix with the independent variable and an intercept
        X = sm.add_constant(data['fcover_y'])
        
        # Fit an OLS model with the dependent variable
        model = sm.OLS(data['meanT_x'], X).fit()
        models.append(model)
        
        # extract the regression coefficients and p-values
        coef = model.params[1]
        pval = model.pvalues[1]
        
        # Pearson CC
        r = scipy.stats.pearsonr(data.fcover_y,data.meanT_x)
        
        # add the results to the results dataframe
        results_df = results_df.append({
            'category1': category1,
            'category2': category2,
            'coef': coef,
            'pval': pval,
            'pcc' : r[0],
            'pcc_pval' : r[1]
            }, ignore_index=True)
        
        f,ax = plt.subplots(figsize = (10,10))
        sns.scatterplot(data,ax = ax,
                        y = 'meanT_x',x = 'fcover_y',
                        hue = 'fcover_x',palette = plt.cm.Greens);
        sns.regplot(data,
                    y = 'meanT_x',x = 'fcover_y',
                    scatter=False, color=".1");
        ax.set(xlabel = f'Fcover {category2} (%)',ylabel = f'$\Delta$ WDI {category1} (-)')
        ax.legend_.set_title(f'Fcover {category1} (%)')
        plt.show()  


df_triu = pd.DataFrame(columns = results_df.category1.unique(),
                       index = results_df.category2.unique(),dtype = float)
df_triu2 = pd.DataFrame(columns = results_df.category1.unique(),
                       index = results_df.category2.unique(),dtype = float)
for i in df_triu.index:
    for j in df_triu.columns:
        try:
            df_triu.loc[i,j] = results_df.loc[(results_df.category2 == i) &
                                              (results_df.category1 == j),'pcc'].item()
            df_triu2.loc[i,j] = results_df.loc[(results_df.category2 == i) &
                                               (results_df.category1 == j),'pcc_pval'].item()
        except:
            df_triu.loc[i,j] = np.nan
            df_triu2.loc[i,j] = np.nan

ax = sns.heatmap(df_triu,annot = df_triu[df_triu2 < 0.05])            
ax.set(title = 'Pearson correlation coefficient between fcover of row and $\Delta WDI$ of column',
       ylabel = 'community fCover',
       xlabel = 'community $\Delta WDI$')
            
# g = sns.relplot(
#     data=df_triu.stack().reset_index(name="correlation"),
#     x="level_0", y="level_1", hue="correlation", size="correlation",
#     palette="vlag_r", hue_norm=(-.002, .002), edgecolor=".7",
#     height=10, sizes=(50, 250), size_norm=(-.2, .8),
# )
# %% Plot fcover class x vs. fcover class y
vegs = ['shrubs','dry']

plt.scatter(x = df_p.loc[df_p.classes ==vegs[0],'fcover'],
            y = df_p.loc[df_p.classes ==vegs[1],'fcover'])
plt.xlabel(f'Fcover {vegs[0]} (%)')
plt.ylabel(f'Fcover {vegs[1]} (%)')


# %% 6. SUPPLEMENTARY FIGURES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import matplotlib.dates as md
from matplotlib import dates as mPlotDATEs
import datetime

#%% Weather conditions:
# --------
print('Mean Tair between first and last flight, 2020:',
          df_fluxtower.loc['24-07-2020'].between_time('11:20','13:45').Barani_Temp_2_Avg.mean())
print('Mean Tair between first and last flight, 2021:',
      df_fluxtower.loc['19-07-2021'].between_time('13:50','16345').Barani_Temp_2_Avg.mean())

print('Mean SWin between first and last flight, 2020:',
      df_fluxtower.loc['24-07-2020'].between_time('11:20','13:45').CMP21_IN_Avg.mean())
print('Mean SWin between first and last flight, 2021:',
      df_fluxtower.loc['19-07-2021'].between_time('13:50','16345').CMP21_IN_Avg.mean())
    
sns.set_theme(style="whitegrid",
              font_scale = 2,
              rc = {"axes.spines.right": False, "axes.spines.top": False})
fig,axs = plt.subplots(1,3, figsize = (30,10))

df_fluxtower['time'] = [datetime.datetime.combine(datetime.date.today(),t.time()) for t  in df_fluxtower.index]

df_fluxtower_2020 = df_fluxtower.loc['24-07-2020'].between_time('11:00','17:00')
df_fluxtower_2021 = df_fluxtower.loc['19-07-2021'].between_time('11:00','17:00')

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

ylabs = ['Temperature (°C)','Radiation (W m$^{-2}$)','Wind speed (m$^{-s}$)']
offsets = [1,20,.1]
labs = ['a)','b)','c)']

conds = np.full((3, 3), False) # Boolean array for GetMeanTair function
np.fill_diagonal(conds,True)

for i,var in enumerate(['Barani_Temp_2_Avg','CMP21_IN_Avg','Cup_2M_Avg']):
    
    
    p = [ axs[i].plot(df['time'],
                      df[var],
                      marker = 'o',label = df.index.year[0]) for df in dfl]
    axs[i].text(-0.1, 1.1, labs[i], transform=axs[i].transAxes, weight='bold')
    
    for site in ['TLB','CBH','Ridge']:
        for year in [2020,2021]:
            color = 'C0' if year == 2020 else 'C1'
            
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
                        horizontalalignment='center')

    axs[i].set(ylabel = ylabs[i],
               xlabel = 'Local time')
    axs[i].xaxis.set_major_formatter(md.DateFormatter('%H:%M'))


fig.autofmt_xdate()
fig.legend(p,labels = [2020,2021],loc = 'upper center')

# PATH_OUT = r'.\figures_and_maps\thermal_mosaics\Fig_S2.png'
plt.savefig(PATH_OUT,bbox_inches = 'tight',dpi=300)

#%% SPEI3:
# --------
# From Chokurdakh meteorological station:
df_spei = pd.read_csv(r'C:\data\0_Kytalyk\2_chokurdakh_meteodata\spei_monthly.csv',sep = ',')
df_spei['date'] = [datetime.date(year = int(x[1].year), month = int(x[1].month), day=1) for x in df_spei.iterrows()]

summer_mask =  (df_spei.month >= 6) & (df_spei.month <= 8)
df_spei_summer = df_spei[summer_mask]

var_name = ['$SPEI_3$', '$SPEI_6$']

sns.set_theme(style="ticks", 
              font_scale = 3)

fig,axs = plt.subplots(2,1,figsize = (25,10),sharex = True)

for i,spei in enumerate(['spei_3_months','spei_6_months']):
    df_spei_summer.plot(x = 'date',y = spei,
                        ax = axs[i],
                        lw = 2, color = f'C{i}',
                        legend = False)
    
    # Plot 10th-percentile as horizontal line
    axs[i].axhline(df_spei_summer[spei].quantile(.1),
                   ls = '--', color = f'C{i}')
    axs[i].annotate('10$^{th}$ percentile',
                    xy = ('1980-11-01',df_spei_summer[spei].quantile(.1) - .2), 
                    va = 'top',
                    transform = axs[i].transData,
                    color = f'C{i}')
    
    # fill the area between June to August 2020
    for year in [2020,2021]:
        start = datetime.date(year, 6, 1)
        end = datetime.date(year, 8, 31)
        axs[i].fill_betweenx([-5,5],start, end, color='blue', alpha=0.2)

    
    axs[i].set(ylabel = var_name[i],
               xlabel = '',
               title = '',
               ylim = [-3,3],
               xlim = ['1980-01-01','2022-04-01'])
    
    axs[i].grid('on')
    
# axs[i].xaxis.set_major_formatter(md.DateFormatter('%b %Y'))
fig.suptitle( 'Standardized Precipitation-Evaporation Index (z-values)')

plt.show()

#  -----------------
# Or as multiannual chart with months on x-axis:
show = True

if show:
    # group the data by year
    groups = df_spei.groupby(df_spei.year)
        
    # create a figure and axis object
    fig,axs = plt.subplots(2,1,figsize = (15,15),sharex = False)
    sub_label = [ 'a)', 'b)']
    
    # obtain a colormap and normalize the line colors based on the year
    cmap = plt.cm.get_cmap(cm.turku_r)
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
                            alpha = .5,
                            color='gray', 
                            label=name)
                
        # Fill below -2 for extreme drought
        axs[i].fill_between(x = [4,10], y1 = -2, y2 = -7, color='orange', alpha = 0.1)
        axs[i].text(1, 0.15,'extreme drought', 
                    transform=axs[i].transAxes,
                    color='orange',
                    fontweight='bold', va='top', ha='right')
        
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
    plt.savefig('./figures_and_maps/spei/summer_spei_multiannual.png',bbox_inches = 'tight', facecolor='white')

#  -----------------
# from Global SPEI database:
show = False

if show:
    spei3 = xr.open_dataset('C:/data/0_Kytalyk/spei03_thornthwaite.nc')
    spei6 = xr.open_dataset('C:/data/0_Kytalyk/spei06_thornthwaite.nc')
    
    
    fig,ax = plt.subplots(figsize = (20,10))
    
    for i,spei in enumerate([spei3,spei6]):
        print(name[i],'in 2020 for the nearest gridcell: ',
              spei.sel(lat = 70.83,lon = 147.49,time = '2020-07-24',method = 'nearest').spei.item())
        print(name[i],'in 2021 for the nearest gridcell: ',
              spei.sel(lat = 70.83,lon = 147.49,time = '2021-07-19',method = 'nearest').spei.item())
        
        spei.sel(
            lat = 70.83,lon = 147.49,method = 'nearest').sel(
                time = slice( '1980-06-01', '2021-08-31')).spei.plot(ax = ax,label = name[i])
                
    ax.xaxis.set_major_formatter(md.DateFormatter('%b %Y'))
    ax.set(ylabel = 'Standardized Precipitation-Evaporation Index \n (z-values)',
           xlabel = '',
           title = '')
    ax.legend()
    plt.show()
        
# %% PLOT SENSOR DRIFT
# os.chdir(r".\code\main_scripts")

from FigFunctions import GatherData,PolyFitSensorT
from glob import glob
from sklearn.metrics import r2_score

fl = glob(r'C:\data\0_Kytalyk\0_drone\internal_data\metadata*.csv')

unc_instr = .1
cutoff = 13

df = pd.concat(map(lambda fn:GatherData(fn, unc_instr, FitOnUnstable = False),fl), 
               ignore_index=False)

# Plot Sensor temperature over time
g = sns.FacetGrid(df, col="site",  row="year", sharex=True, margin_titles=True)

g.map(sns.lineplot, "flighttime_min", "T_sensor", color = 'gray')
g.map(sns.scatterplot, "flighttime_min", "T_sensor",'isStable',palette = "Paired",marker = '+')

g.add_legend(title = 'Is the sensor temperature stable? \n i.e. within %.1f °C of the minimum?' % unc_instr)

g.set_titles(col_template="{col_name}",row_template="{row_name}")
# for year,margin_title in zip([2020,2021],g._margin_titles_texts):
#     margin_title.set_text(year)
    
g.set_axis_labels(y_var='Sensor temperature (°C)', x_var='Flight time (minutes)', clear_inner=True)

# plt.savefig('../../figures_and_maps/thermal_mosaics/sensorT_time_2.png',bbox_inches = 'tight', facecolor='white')

plt.show()

# Plot relationship sensor vs. target Temperature
plot_fit = 'fit_2'

r_sq = df.groupby(['site','year']).apply(lambda x: r2_score(x['LST_deviation'], x[plot_fit]))


sns.set_theme(style="whitegrid",
              rc={"figure.figsize":(10, 10)})

g = sns.FacetGrid(df, col="site",  row="year", sharex=True, margin_titles=True)
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

g.set_axis_labels(x_var='Sensor temperature (°C)', y_var='Correction temperature (°C)', clear_inner=True)
g.add_legend(title = 'Is the sensor temperature stable? \n i.e. within %.1f °C of the minimum?' % unc_instr,
             bbox_to_anchor=(.4, -.15), loc = 'lower center', borderaxespad=0)

g.set_titles(col_template="{col_name}")

# Annotate a value into each bar in each plot
for i,p in enumerate(g.axes.flat):
    p.annotate("R$^2$ = {:.2f}".format(r_sq.iloc[i]) , 
                (31,0), # Placement
                ha='center', va='center', fontsize=12, color='black', rotation=0, xytext=(0, 20),
                textcoords='offset points')


for year,margin_title in zip([2020,2021],g._margin_titles_texts):
    margin_title.set_text(year)

g.fig.subplots_adjust(top=0.9)

# plt.savefig('../../figures_and_maps/thermal_mosaics/sensorT_correctionT_4.png',bbox_inches = 'tight', facecolor='white')
plt.show()


# %% Plot semivariogram
import skgstat as skg
np.random.seed(123)

# Select n**2 Random locations in the raster
n = 200

Variograms = [0,0,0]
for i,site in enumerate(sites):
    print(f'Computing semivariogram for {site}...')
    
    PATH_21_TIR = rf'.\paper\data\mosaics\{site}_thermal_2021_resampled.tif'
    
    T_air = GetMeanTair(df_flighttimes, df_fluxtower, site, 2021)
    
    dA_tir21_s = rxr.open_rasterio(PATH_21_TIR).isel(x = slice(xmin,xmax), y = slice(ymin,ymax)) - T_air 
    
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
    
    Variograms[i] = skg.Variogram(coordinates = df[['x','y']],
                      values = df.iloc[:,-1],
                      model = 'spherical', 
                      bin_func = 'even',
                      n_lags = maxlag, # determines number of bins, if equal to maxlag, then bins have size of 1m
                      maxlag = maxlag)

    print(Variograms[i])
# %%
fig,axs = plt.subplots(nrows = 6,figsize = (10,30),gridspec_kw={'height_ratios':[3,1,3,1,3,1] },sharex = True)

sns.set_theme(style="whitegrid",
              font_scale = 2,
              rc = {"axes.spines.right": False, "axes.spines.top": False})

labs = ['a)','b)','c)']
j = 0
for i,V in enumerate(Variograms):

    ax1 = axs[j]
    j += 1
    ax2 = axs[j] 
    j += 1
    
    V.plot(axes=[ax1,ax2],grid = False)
    ax1.axhline(V.parameters[1], ls = '--', color = 'gray') # horizontal for sill
    ax1.axvline(V.parameters[0], ls = '--', color = 'gray') # vertical for range
    
    ax1.set(xlabel ='')
    ax2.set(xlabel ='')
    ax1.annotate('Range = {:.1f} m'.format(V.parameters[0]),
                    (V.parameters[0] + 2, np.mean(ax1.get_ylim())), 
                     color = 'k', transform = ax.transData,
                     clip_on=False, annotation_clip=False,
                     verticalalignment = 'center_baseline', horizontalalignment='left')
    
    ax1.text(0.05, .9, labs[i], transform=ax1.transAxes, weight='bold')

ax2.set(xlabel ='Distance (m)')

fig.tight_layout()
fig.savefig(f'./figures_and_maps/thermal_mosaics/semivariogram_spherical.png',dpi = 200,bbox_inches = 'tight', facecolor='white')
plt.show()

# %% Get TOMST vs. LST data plots
from pyproj import Transformer,Proj

import sys
sys.path.append(r"C:/Users/nils/OneDrive - Universität Zürich UZH/Dokumente 1/1_PhD/5_CHAPTER1/code/")

import importer
df_hobo_tlb,df_hobo_ridge, df_fluxtower, df_tmst, df_tmst_delta, df_tmst_h, df_tmst_d, loggers = importer.load_all()
df_hobo = df_hobo_tlb

def trans_fct(row):
    x1,y1 = p1(row.Lon_dd, row.Lat_dd)
    x,y = transformer.transform(y1, x1)
    return x,y

transformer = Transformer.from_crs("epsg:4326",'epsg:32655' )
p1 = Proj("epsg:4326")

loggers.loc[:,['lon_utm','lat_utm']] = loggers.loc[:,['Lat_dd','Lon_dd']].apply(trans_fct, 
                                                                          axis=1, 
                                                                          result_type='expand').values

# %% 
def FetchImageData(Raster, lons, lats):
    buffer = 5 # nr of pixels around the coordinate to account for: 5 --> 5*15cm = 75 cm on each side
    res = .15 * buffer
    subset = [Raster.sel(x = slice(x-res,x+res),
                        y = slice(y+res,y-res),
                        # method = 'nearest'
                        ).mean().values for x,y in tqdm(zip(lons,lats))]
    return subset

if site == 'TLB':
    idx_name = 'lakebed'
    
    data_droneday = df_tmst.loc['19-7-2021']
    data_droneday.index = data_droneday.index.tz_convert('Asia/Srednekolymsk').rename('local_time')

    data_flights = data_droneday.between_time('13:30','14:30')
    data_flights.loc[data_flights.site == 'lakebed'].groupby(['Logger_SN','site']).mean().sort_index(level=1)


    TIR_21_TOMSTloc = pd.DataFrame({'TIR':np.hstack(FetchImageData(dA_tir21, 
                                                         loggers.loc[loggers.site == 'lakebed','lon_utm'],
                                                         loggers.loc[loggers.site == 'lakebed','lat_utm'])),
                                    'Logger_SN' : loggers.loc[loggers.site == 'lakebed','TOMST_serial_number'].astype(str)})
    TLB_LoggerSN = loggers.loc[loggers.site == 'lakebed','TOMST_serial_number'].astype(str)

    TOMST_flightmeans = data_flights.loc[np.isin( data_flights.Logger_SN.values,TLB_LoggerSN.values),:].groupby(['Logger_SN']).mean().sort_index(level=1)
    df = pd.merge(TOMST_flightmeans,TIR_21_TOMSTloc,left_on='Logger_SN', right_on='Logger_SN')
    
elif site == 'Ridge':
    idx_name = 'ridge'
    
    data_droneday = df_tmst.loc['19-7-2021']
    data_droneday.index = data_droneday.index.tz_convert('Asia/Srednekolymsk').rename('local_time')

    data_flights = data_droneday.between_time('13:30','14:30')
    data_flights.loc[data_flights.site == idx_name].groupby(['Logger_SN','site']).mean().sort_index(level=1)


    TIR_21_TOMSTloc = pd.DataFrame({'TIR':np.hstack(FetchImageData(dA_tir21, 
                                                         loggers.loc[loggers.site == idx_name,'lon_utm'],
                                                         loggers.loc[loggers.site == idx_name,'lat_utm'])),
                                    'Logger_SN' : loggers.loc[loggers.site == idx_name,'TOMST_serial_number'].astype(str)})
    TLB_LoggerSN = loggers.loc[loggers.site == idx_name,'TOMST_serial_number'].astype(str)

    TOMST_flightmeans = data_flights.loc[np.isin( data_flights.Logger_SN.values,TLB_LoggerSN.values),:].groupby(['Logger_SN']).mean().sort_index(level=1)
    df = pd.merge(TOMST_flightmeans,TIR_21_TOMSTloc,left_on='Logger_SN', right_on='Logger_SN')


elif site == 'CBH':
    print('No TOMST data in cloudberry hills')

# %% Plot TOMST temperatures vs. LST
import scipy as sp

sns.set_theme(style="ticks", 
              rc={"figure.figsize":(10, 10)},
              font_scale = 2)

df_p = df.loc[:,['T1','T2','T3']].melt()
df_p['LST'] = np.hstack([df.TIR.values] * 3) #[df.TIR.values] * 3
ax = sns.lmplot(df_p, x = 'LST',y = 'value', hue = 'variable',height=6, aspect=1.5)
ax.set(
       # xlim = [25,35],
       # ylim = [2,35],
       xlabel = 'Land surface temperature (°C)',
       ylabel = 'TMS-4 temperature (°C)')
# ax.legend.set_title('TMS-4 level')
sns.move_legend(ax, "upper left", bbox_to_anchor=(.85, .65), title='TMS-4 level')

r, p = sp.stats.pearsonr(df.TIR, df.T2)
print('T2 vs. TIR r =',r)
r, p = sp.stats.pearsonr(df.TIR, df.T3)
print('T3 vs. TIR r =',r)
# plt.savefig(r'.\figures_and_maps\thermoregulation\TLB_TIR_vs_TOMST.png',
#             bbox_inches = 'tight')



# %%
tomst_vars = ['T1','T2','T3','soilmoisture']
hobo_vars  = ['temperature_15cm', 'rh_15cm', 'temperature_60cm', 'rh_60cm','precipitation', 'light_intensity_lux']
T_levels = ['T1 - T3','T1 - T2','T2 - T3','T3 - Tair','T2 - Tair','T1 - Tair']
all_vars = tomst_vars + T_levels + hobo_vars

df_daytime_delta = df_tmst_delta.between_time('07:01','20:00')
df_nighttime_delta = df_tmst_delta.between_time('20:01','07:00')


mean_hourly_deltaT = df_daytime_delta.groupby('Logger_SN',as_index=True)[T_levels].resample('.5H').mean()
mean_hourly_deltaT = mean_hourly_deltaT.reset_index(level=0,drop=False)
mean_hourly_illum = df_hobo.light_intensity_lux.resample('.5H').mean()

mean_hourly_deltaT.loc[:,'CAVM_class'] = codes_out.ids.replace(to_replace=np.unique(codes), value=classes)

df_p = pd.concat([mean_hourly_illum.loc[mean_hourly_deltaT.index],
              mean_hourly_deltaT],axis=1)

illum_cat = pd.qcut(df_p.light_intensity_lux,[0,.8,1],['cloudy','sunny'])
df_pp = pd.concat([df_p,illum_cat],axis=1)

df_pp.columns = ['light_intensity_lux', 'Logger_SN', 'T1 - T3', 'T1 - T2', 'T2 - T3',
       'T3 - Tair', 'T1 - Tair', 'soilmoisture', 'CAVM_class',
       'light_conditions']

fig,axs = plt.subplots(5,1,figsize = (10,18),sharex=True)
for ax,T_level in zip(axs,['T1 - T3','T1 - T2','T2 - T3','T3 - Tair','T1 - Tair']):
    sns.boxplot(x="CAVM_class", y=T_level,
            hue="light_conditions", palette=["gray", "yellow"],
            data=df_pp,
               ax=ax)
    ax.set(xlabel = '',
          ylim = [-20,10])
    ax.get_legend().remove()
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -.5))




