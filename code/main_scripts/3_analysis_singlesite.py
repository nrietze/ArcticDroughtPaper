# %% 0. LOAD DATA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from cmcrameri import cm
import seaborn as sns

import sys, os
sys.path.append(r"C:\Users\nils\OneDrive - Universität Zürich UZH\Dokumente 1\5_code")

MAIN_PATH = r'C:/Users/nils/OneDrive - Universität Zürich UZH/Dokumente 1/1_PhD/5_CHAPTER1'
sys.path.append(r".\code\main_scripts")

os.chdir(MAIN_PATH + '/code/main_scripts')

from FigFunctions import *

import myfunctions as myf

os.chdir(MAIN_PATH)

site = input('Type in the site: \n')
year = 2020
seed = 15

PATH_20_TIR = rf'.\paper\data\mosaics\{site}_thermal_2020_resampled.tif'
PATH_21_TIR = rf'.\paper\data\mosaics\{site}_thermal_2021_resampled.tif'

PATH_MSP_20 = rf'.\paper\data\mosaics\{site}_msp_2020_resampled.tif'
PATH_MSP_21 = rf'.\paper\data\mosaics\{site}_msp_2021_resampled.tif'
    
PATH_CL = rf'.\paper\data\landcover\{site}_2021_classified_filtered5.tif'
PATH_WATERMASK = rf'.\paper\data\landcover\{site}_2020_classified_filtered5.tif'
    
print('Loading data...')

df_fluxtower = myf.read_fluxtower('C:/data/0_Kytalyk/7_flux_tower/CR3000_Kytalyk_Meteo.dat')

parser = lambda date: pd.to_datetime(date,format='%d.%m.%Y %H:%M').tz_localize('Asia/Srednekolymsk')

df_flighttimes = pd.read_csv(r'C:\data\0_Kytalyk\0_drone\flight_times.csv', 
                             sep = ';', 
                             parse_dates = [4,5],
                             date_parser=parser)

# Set side length for study area (in m)
sl = 400 

dxy = int(sl / .15) # 667 pixels along 100m ==> 1 ha

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
I_cl_s,I_wm_s, I_tir_20_s, I_tir_21_s, I_ndvi_20_s, I_ndvi_21_s = PrepRasters(PATH_CL,PATH_WATERMASK,
                                                                              PATH_20_TIR,T_air_20,
                                                                              PATH_21_TIR,T_air_21,
                                                                              PATH_MSP_20,PATH_MSP_21,
                                                                              extent)

# Class labels and values for land cover raster
classdict = {0:'HP1',
             1:'LW1',
             2:'HP2',
             3:'water',
             4:'LW2',
             5:'mud',
             7:'TS',
             255:'nodata'}

names = [classdict[i] for i in np.unique(I_cl_s)]

# Color scheme for plots
colordict = dict(HP1 = '#AAFF00', 
                 HP2 = '#38A800', 
                 LW1 = '#00FFC5',
                 LW2 = '#A90EFF',
                 TS = '#3D3242',
                 mud = '#734C00',
                 water = '#005CE6')

# Replace no data values (255) with nan
I_cl_s = np.where(I_cl_s == 255,np.nan, I_cl_s)

# check if a csv of image data already exists:
try:
    df_m_s = pd.read_csv(fr'.\data\thermal_data/{site}_data_thermal.csv',sep=';')
    print('Data for this site is available. Loading ...')
    # df_m_s_stats = pd.read_csv(fr'.\results\statistics\map_stats\{site}_statistics.csv',sep = ';')
    
except:
    # if it doesn't exist, compile the samples:
    print('No thermal data found. Sampling...')
    
    df_tir = pd.concat(
        map(lambda arr,year: MeltArrays(arr,I_cl_s,names,year,'deltaT'),
            [I_tir_20_s,I_tir_21_s],[2020,2021]), 
        ignore_index=False).reset_index(drop=True)
    
    df_ndvi = pd.concat(
        map(lambda arr,year: MeltArrays(arr,I_cl_s,names,year,'ndvi'),
            [I_ndvi_20_s,I_ndvi_21_s],[2020,2021]), 
        ignore_index=False).reset_index(drop=True)
    df_tir['ndvi'] = df_ndvi.ndvi
    
    # Compute water deficit index:
    df_wdi = pd.concat(
        map(lambda arr,year: MeltArrays(arr,I_cl_s,names,year,'wdi'),
            [ScaleMinMax(I_tir_20_s),ScaleMinMax(I_tir_21_s)],[2020,2021]), 
        ignore_index=False).reset_index(drop=True)
    df_tir['wdi'] = df_wdi.wdi
    
    df_wdi_diff = MeltArrays(ScaleMinMax(I_tir_20_s) - ScaleMinMax(I_tir_21_s),
                             I_cl_s,names,2021,'diff_wdi').reset_index(drop=True)
    df_tir['diff_wdi'] = df_wdi_diff.diff_wdi
    
    # Generate random sample of size n per class and year:
    n = 20000
    df_m_s = df_tir.sample(frac = 1, random_state = seed) # shuffle data
    df_m_s = pd.concat(
        [g[g.deltaT.notna()][:n] for _, g in df_m_s.groupby(['year', 'variable'], sort=False, as_index=False)],
        # ignore_index=True 
    )  
    # Export to csv
    df_m_s.to_csv(fr'.\data\thermal_data/{site}_data_thermal.csv',sep=';')
    
    # Generate descripitve statistics and save as csv
    def iqr_func(col):
        return np.nanquantile(col, q=0.75) - np.nanquantile(col, q=0.25)
    
    df_m_s_stats = df_m_s.groupby(['year','variable']).describe()
    for main_col in df_m_s_stats.columns.levels[0]:
        df_m_s_stats[(main_col, 'iqr')] = df_m_s.groupby(['year','variable']).agg(iqr_func)[main_col]
    df_m_s_stats.to_csv(fr'.\results\statistics\map_stats\{site}_statistics.csv',sep = ';')

# Sort community labels along moisture gradient
if site == 'TLB':
    label_order = ['water','LW1', 'HP1', 'HP2']
    xlim = [5,25]
elif site == 'CBH':
    label_order = ['water','mud','LW1','LW2', 'HP1', 'HP2','TS']
    xlim = [5,25]
elif site == 'Ridge':
    label_order = ['LW1','HP2','TS']
    xlim = [5,20]
     
df_m_s['variable'] = df_m_s.variable.astype("category").cat.set_categories(label_order, ordered=True)

print('done.')

# %% 1. DENSITY PLOT IN NORMAL YEAR & DIFFERENCE TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
group_var = 'variable'
xvar = 'deltaT'

saveresults = False

from scipy.stats import normaltest
alpha = 1e-3
normtest = df_m_s.groupby(['year','variable']).apply(lambda x: normaltest(x[xvar]))
df_normtest = pd.DataFrame([[z,p] for z,p in normtest.values],columns = ['zscore','pval']).set_index(normtest.index)
df_normtest['reject'] = df_normtest.pval < alpha

print('Normality tests: \n',df_normtest)

import statsmodels.api as sm
from statsmodels.formula.api import ols

cw_lm = ols('deltaT ~ C(variable)', data=df_m_s.loc[df_m_s.year==2021]).fit() #Specify C for Categorical
print(sm.stats.anova_lm(cw_lm, typ=2))

# Tukey HSD
# sample 1 % of observations (= 200) per plant community and year
df_sample = df_m_s.groupby(['variable','year']).sample(frac = 0.01,
                                                       random_state = seed)

from statsmodels.stats.multicomp import MultiComparison
mc20,mc21 = df_sample.loc[(df_sample.variable != 'water') & 
                          (df_sample.variable != 'mud')].groupby(['year']).apply(lambda x: MultiComparison(x[xvar], x[group_var]))

# Save test results in dataframe (for significance brackets in plot)
thsd = pd.DataFrame(data = mc21.tukeyhsd()._results_table.data[1:], 
                    columns = mc21.tukeyhsd()._results_table.data[0])

print('2021: \n', mc21.tukeyhsd(), end = '\n')                                                                                       
                                                                                           
# Bonferroni adjusted pair tests
# print('2021: \n', mc21.allpairtest(ttest_ind, method = 'bonf')[0], end = '\n')

PATH_SAVE = fr'.\figures_and_maps\thermoregulation\densityplot_2021_{site}_{xvar}.png'

ax = PlotDensities(df_m_s, xvar,PATH_SAVE,
                   xlim = xlim,
                   colors = colordict,
                   showSignif = True, data_pairtest = thsd,
                   order = label_order,
                   showTair = False, showWater = False,
                   showBothYears = False,
                   saveFig = False)

if saveresults:
    thsd.to_csv(f'./results/statistics/difference_tests/testtable_2021_{site}_{xvar}.csv', sep=';', index = False)

# %% 1 b. DENSITY PLOT IN BOTH YEARS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
xvar = 'wdi'
PATH_SAVE = fr'.\figures_and_maps\thermoregulation\densityplots_bothyears_{site}_{xvar}.png'

PlotDensities(df_m_s, xvar,PATH_SAVE,
              colors = colordict,
              showSignif = False,
              showTair = False, showWater = True,
              showBothYears = True,
              saveFig = False)

# %% 3. BOXPLOT IN BOTH YEARS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
yvar = 'wdi'

saveresults = True

# Generate descripitve statistics and save as csv
df_wdi_iqr = df_m_s.groupby(['year','variable'])['wdi'].quantile([.25,.75]).unstack()
df_wdi_iqr['iqr'] = df_wdi_iqr[.75] - df_wdi_iqr[.25]

# sample random observations per plant community and year to avoid effect of large sample
df_sample = df_m_s.groupby(['variable','year']).sample(frac = 0.01,
                                                       random_state = seed)

# t-test between years
from scipy.stats import ttest_ind, ttest_rel
alpha = .05

# Welch's ttest for unequal variances
ttest = df_sample.groupby(['variable']).apply(lambda x: ttest_ind(x.loc[x.year==2020,yvar],
                                                                  x.loc[x.year==2021,yvar], 
                                                                  equal_var=False))
df_ttest = pd.DataFrame([[z,p] for z,p in ttest.values],columns = ['tstat','pval']).set_index(ttest.index)
df_ttest['reject'] = df_ttest.pval < alpha
df_ttest.loc[df_ttest.pval < 0.01,'text'] = '< 0.01'

# Mean WDI differences between years for each group
df_ttest['meandiff'] = df_sample.groupby(['variable']).apply(lambda x: round(x.loc[x.year == 2020,'wdi'].mean() - x.loc[x.year == 2021,'wdi'].mean(),2))

# ttest for two dependent samples
ttest_rel = df_sample.groupby(['variable']).apply(lambda x: ttest_rel(x.loc[x.year==2020,yvar],
                                                                         x.loc[x.year==2021,yvar]))
df_ttest_rel = pd.DataFrame([[z,p] for z,p in ttest_rel.values],columns = ['tstat','pval']).set_index(ttest_rel.index)
df_ttest_rel['reject'] = df_ttest_rel.pval < alpha


PATH_SAVE = fr'..\..\figures_and_maps\thermoregulation\{site}_{yvar}_boxplots.png'

ax = PlotBoxWhisker(df_m_s, yvar,
                   label_order = label_order,
                   colors = colordict,
                   showSignif=True,data_ttest = df_ttest,
                   showWater = False,
                   PATH_OUT=PATH_SAVE,
                   saveFig = False)

# Multicomparison of WDI differences per plant community
mask = np.logical_and(df_m_s.variable != 'water',
                      df_m_s.variable != 'mud'
                      )
df_sample_2 = df_m_s.loc[df_m_s.year == 2020].loc[mask].sample(frac = 0.01,random_state = seed)

mc_wdi = MultiComparison(df_sample_2['diff_wdi'], df_sample_2['variable'])
print(mc_wdi.tukeyhsd())

# Save test results in dataframe (for significance brackets in plot)
thsd = pd.DataFrame(data = mc_wdi.tukeyhsd()._results_table.data[1:], 
                    columns = mc_wdi.tukeyhsd()._results_table.data[0])

if saveresults:
    df_wdi_iqr.to_csv(f'./results/statistics/map_stats/{site}_{yvar}_quartiles.csv', sep = ';')
    df_ttest.to_csv(f'./results/statistics/difference_tests/{site}_{yvar}_testtable.csv', sep = ';')
    thsd.to_csv(f'./results/statistics/difference_tests/testtable_wdi_diff_{site}.csv', sep=';', index = False)

# %% deltaT along transect for two years
if False:
    from matplotlib.lines import Line2D
    transect_y = 702
    
    def map_func(val):
        return classdict[val]
    named_array = np.vectorize(map_func)(I_cl_s[1200:1700,transect_y])
    
    plt.plot(ScaleMinMax(I_tir_20_s)[1200:1700,transect_y],
             label = '2020',c = 'k')
    plt.scatter(np.arange(0,500),
                ScaleMinMax(I_tir_20_s)[1200:1700,transect_y],
                color = [colordict[x] for x in named_array])
    points = plt.plot(ScaleMinMax(I_tir_21_s)[1200:1700,transect_y],
                      label = '2021',ls = '--',c = 'k')
    scatter = plt.scatter(np.arange(0,500),
                          ScaleMinMax(I_tir_21_s)[1200:1700,transect_y],
                          c = [colordict[x] for x in named_array])
    
    # Define custom legend handles and labels
    handles = []
    labels = []
    
    for category in np.unique(named_array):
        marker_handle = Line2D([0], [0], linestyle='none', marker='o', color=colordict[category])
        handles.append(marker_handle)
        labels.append(category)
        
    for year, linestyle in zip(['2020', '2021'], ['-', '--']):
        line_handle = Line2D([0], [0], color='k', ls=linestyle, lw=1)
        handles.append(line_handle)
        labels.append(year)
    
    # Add custom legend to the plot
    plt.legend(handles=handles, labels=labels, bbox_to_anchor=(1.05, 1.03), loc='upper left')
    
    # Add inset axes for the image
    ax = plt.axes([.7, 0.1, 0.35, 0.35]) # [left, bottom, width, height]
    ax.imshow(ScaleMinMax(I_tir_21_s), cmap = cm.lajolla)
    ax.axhline(transect_y,c = 'blue')
    ax.axis('off')
    
    plt.suptitle('$\Delta T_{surf - air}$ along transect for two years')
    plt.tight_layout() # adjust spacing to make room for the legend
    plt.show()

# %% 4. COMPUTE FCOVER IN 1.5 x 1.5 m QUADRATS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from osgeo import gdal
import rioxarray as rxr
import xarray as xr
import rasterio
from scipy.spatial import distance_matrix

import time

def GetFcover(dA_cl, dA_tir):
    df = pd.DataFrame(columns = ['classes','count','fcover','meanT','sigmaT'])
    df['classes'], df['count'] = np.unique(dA_cl,return_counts = True)
    
    df['fcover'] = df['count'] / df['count'].sum() *100
    
    for i,cl in enumerate(df['classes']):
        df.loc[i,'meanT'] = np.nanmean(xr.where(dA_cl == cl, dA_tir,np.nan))
        df.loc[i,'sigmaT'] = np.nanstd(xr.where(dA_cl == cl, dA_tir,np.nan))
    
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
    df_out = pd.concat( ProgressParallel(n_jobs = ncores)(delayed(GetFcover)(cl,tir) for cl,tir in zip(c,t)),
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
    
    classdict = {0:'HP1', 1:'LW1',2:'HP2',3:'water',4:'LW2',5:'mud',7:'TS'}
    df_out['classes'] = df_out['classes'].map(classdict)
    df_out['meanT'] = df_out.meanT.astype(float)
    
    return df_out

dA_tir20 = rxr.open_rasterio(PATH_20_TIR)
dA_tir21 = rxr.open_rasterio(PATH_21_TIR)
 
dA_cl = rxr.open_rasterio(PATH_CL)
da_wm = rxr.open_rasterio(PATH_WATERMASK)

dA_tir20_s = dA_tir20.isel(x = slice(xmin,xmax), y = slice(ymin,ymax)) - T_air_20
dA_wdi20_s = ScaleMinMax(dA_tir20_s[0])

dA_tir21_s = dA_tir21.isel(x = slice(xmin,xmax), y = slice(ymin,ymax)) - T_air_21
dA_wdi21_s = ScaleMinMax(dA_tir21_s[0])

dA_cl_s = dA_cl.isel(x = slice(xmin,xmax), y = slice(ymin,ymax))

windowsize = 33 # give window side length in number of pixels
# df_out = MapFcover(windowsize, dA_cl_s[0],dA_tir21_s[0]); xlabel = 'Average quadrat $T_{surf}$ - $T_{air}$ (°C)'

df_out = MapFcover(windowsize, dA_cl_s[0],(dA_wdi20_s - dA_wdi21_s )); xlabel = 'Grid cell $\Delta WDI_{2020 - 2021}$(-)'
# df_out = MapFcover(windowsize, dA_cl_s[0],(dA_wdi20_s )); xlabel = 'Average quadrat $wdi_{2020}$ (-)'
# df_out = MapFcover(windowsize, dA_cl_s[0],(dA_wdi21_s )); xlabel = 'Average quadrat $wdi_{2021}$ (-)'
# df_out = MapFcover(windowsize, dA_cl_s[0],(dA_tir20_s - dA_tir21_s)[0]); xlabel = 'Average quadrat $\Delta T_{2020}$ - $\Delta T_{2021}$ (-)'

# %% continuous fCover mapping:
if False:
    from scipy import ndimage
    import multiprocessing as mp
    
    # Define the input array
    arr = np.random.randint(1, 6, (2666, 2666))
    
    n = 33
    
    # Function that calculates the percentage of each category in a window
    def category_percentage(window, category):
        return np.sum(window == category) / (n*n) * 100
    
    # Create a list of functions, one for each category
    category_functions = [lambda x: category_percentage(x, category) for category in np.unique(I_cl_s)]
    
    # Define a function that processes a chunk of the input array in parallel
    def process_chunk(chunk):
        return np.array([ndimage.generic_filter(chunk, func, size=(n, n)) for func in category_functions])
    
    # Split the input array into chunks
    chunk_size = 100
    chunks = [I_cl_s[i:i+chunk_size, j:j+chunk_size] for i in range(0, I_cl_s.shape[0], chunk_size) for j in range(0, I_cl_s.shape[1], chunk_size)]
    
    # Process each chunk in parallel using multiprocessing
    pool = mp.Pool(processes=mp.cpu_count())
    results = pool.map(process_chunk, chunks)
    pool.close()
    
    # Concatenate the results into a single output array
    result = np.concatenate(results)

# %% 5. PLOT FCOVER VS. TEMPERATURES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
bin_var = 'meanT'
val_var = 'meanT' if bin_var == 'fcover' else 'fcover'

bin_interval = 1 if bin_var == 'fcover' else 0.005

windowsize_m = int(round(windowsize*.15,0))

PATH_OUT = rf'.\figures_and_maps\thermoregulation\yeardiff_{site}_Fcover_vs_mean_deltaT_{windowsize_m}m.png'

ax = PlotFcoverVsTemperature(data = df_out, 
                             binning_variable = bin_var,
                             bin_interval = bin_interval,
                             value_variable = val_var,
                             label_order = label_order,
                             colors = colordict,
                             model = 'cubic', plot_type = 'regplot',
                             PATH_OUT = PATH_OUT,
                             xlab = xlabel,xvar = 'meanT',
                             saveFig = True)

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

PATH_OUT = r'.\figures_and_maps\thermal_mosaics\Fig_S2.png'
plt.savefig(PATH_OUT,bbox_inches = 'tight',dpi=300)

#%% 6 c. SPEI3:
# --------
spei3 = xr.open_dataset('C:/data/0_Kytalyk/spei03_thornthwaite.nc')
spei6 = xr.open_dataset('C:/data/0_Kytalyk/spei06_thornthwaite.nc')

name = ['SPEI3', 'SPEI6']

fig,ax = plt.subplots(figsize = (20,10))

for i,spei in enumerate([spei3,spei6]):
    print(name[i],'in 2020 for the nearest gridcell: ',
          spei.sel(lat = 70.83,lon = 147.49,time = '2020-07-24',method = 'nearest').spei.item())
    print(name[i],'in 2021 for the nearest gridcell: ',
          spei.sel(lat = 70.83,lon = 147.49,time = '2021-07-19',method = 'nearest').spei.item())
    
    spei.sel(
        lat = 70.83,lon = 147.49,method = 'nearest').sel(
            time = slice( '2019-06-01', '2021-08-31')).spei.plot(ax = ax,label = name[i])
            
ax.xaxis.set_major_formatter(md.DateFormatter('%b %Y'))
ax.set(ylabel = 'Standardized Precipitation-Evaporation Index \n (z-values)',
       xlabel = '',
       title = '')
ax.legend()
    
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

plt.savefig('./figures_and_maps/thermal_mosaics/Fig_S_sensordrift.png',bbox_inches = 'tight', facecolor='white')

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

plt.savefig('./figures_and_maps/thermal_mosaics/Fig_S_TcorrvsTsens.png',
            dpi = 200,bbox_inches = 'tight', facecolor='white')
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

# %% 6 g. TOMST differences
try:
    any(df_tmst_delta)
except:
    print('Loading data...')
    sys.path.append(r"C:/Users/nils/OneDrive - Universität Zürich UZH/Dokumente 1/1_PhD/5_CHAPTER1/code/")
    
    import importer
    df_hobo_tlb,df_hobo_ridge, df_fluxtower, df_tmst, df_tmst_delta, df_tmst_h, df_tmst_d, loggers = importer.load_all()

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
col_titles = ['$T_2 - T_3$', '$T_1 - T_{air}$', '$T_2 - T_{air}$', '$T_3 - T_{air}$']

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

# %% 6 h. Correlations soil moisture dT2Tair

from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

df = df_tmst_delta.between_time('11:00','13:30')
# df = df_tmst_delta.loc["2021-07-19"].between_time('13:50','16:30').dropna()

# Scale the soilmoisture column
scaler = StandardScaler()
df['soilmoisture_scaled'] = scaler.fit_transform(df[['soilmoisture']])

def aggregate_func(df):
    df1 = df.groupby('Logger_SN').resample('D').mean().reset_index(drop = True)
    df1['Logger_SN'] = df1.Logger_SN.astype(np.int64)
    df2 = df.groupby('Logger_SN').resample('D').agg(lambda x: x[0])[['custom_class','site']].reset_index()
    df1 = df1.sort_values('Logger_SN')
    df2 = df2.sort_values('Logger_SN')
    df1[['custom_class','site']] = df2[['custom_class','site']]
    return df1

df_agg = aggregate_func(df)
df_agg = df_agg.dropna()

# Calculate the Pearson correlation coefficient and p-value between temperature and soilmoisture
grouped = df_agg.groupby(['site','custom_class'])
corr, pval = grouped['dT2Tair', 'soilmoisture_scaled'].apply(lambda x: pearsonr(x.dT2Tair, x.soilmoisture_scaled)).apply(pd.Series).values.T

# Extract the correlation coefficient and p-value for each class
result = pd.DataFrame({'site': grouped.groups.keys(),
                       'class': grouped.groups.keys(),
                       'corr': corr,
                       'pval': pval})

g = sns.FacetGrid(df_agg, row='custom_class', col='site',
                  aspect = 1,height = 5)

# Add a scatterplot to the FacetGrid
g.map(plt.scatter, 'dT2Tair', 'soilmoisture_scaled')
g.set_titles(col_template="{col_name}")
g.set_titles(row_template="{row_name}")

# Display the plot
plt.show()

print(result)

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
from tqdm import tqdm
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




