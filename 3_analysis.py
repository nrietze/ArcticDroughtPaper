import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from cmcrameri import cm
import seaborn as sns

from skimage.io import imread

import sys
sys.path.append(r"C:\Users\nils\Documents\5_code")
import myfunctions as myf

def GetMeanTair(df_flighttimes, df_fluxtower, site, year):
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
    mean_Tair = df_sliced.T107_A_2_Avg.mean()
    return mean_Tair

def PrepRasters(PATH_TLB_CL,PATH_TLB_WATERMASK,
                PATH_20_TIR,T_air_20,
                PATH_21_TIR,T_air_21,
                extent):
    
    I_cl = imread(PATH_TLB_CL)
    I_wm = imread(PATH_TLB_WATERMASK)
    
    I_tir_20 = imread(PATH_20_TIR) -  T_air_20
    I_tir_21 = imread(PATH_21_TIR) - T_air_21
    
    I_cl_s = I_cl[extent['xmin']:extent['xmax'],extent['ymin']:extent['ymax']]
    I_wm_s = I_wm[extent['xmin']:extent['xmax'],extent['ymin']:extent['ymax']]
    I_tir_20_s = I_tir_20[extent['xmin']:extent['xmax'],extent['ymin']:extent['ymax']]
    I_tir_21_s = I_tir_21[extent['xmin']:extent['xmax'],extent['ymin']:extent['ymax']]
    
    return I_cl_s,I_wm_s, I_tir_20_s, I_tir_21_s
    
    # I_ndvi_20 = imread(PATH_NDVI_20)
    # I_ndvi_21 = imread(PATH_NDVI_21)
    # I_ndvi_20_s = I_ndvi_20[extent]
    # I_ndvi_21_s = I_ndvi_21[extent]

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

    df_m = df.melt()
    df_m['year'] = year
    return df_m


def ScaleMinMax(x):
    """
    Normalizes an array by its range (min & max). 
    """
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))

# %% 0. LOAD DATA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

site = 'CBH'
year = 2020

savefigs = False

if site == 'CBH':
    PATH_20_TIR = r'C:\data\0_Kytalyk\0_drone\2020\CBH\TIR\resampled\CBH_thermal_merged_2020_mosaic_thermal ir_average_0.15_resampled.tif'
    PATH_21_TIR = r'C:\data\0_Kytalyk\0_drone\2021\CBH\TIR\resampled\CBH_thermal_merged_2021_mosaic_thermal ir_average_0.15_resampled.tif'
    
    PATH_CL = r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\data\classification_data\CBH\2021\CBH_2021_classified_filtered5.tif'
    PATH_WATERMASK = r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\data\classification_data\CBH\2020\V4\CBH_2020_classified_filtered5.tif'
    
elif site == 'Ridge':
    PATH_20_TIR = r'C:\data\0_Kytalyk\0_drone\2020\CBH\TIR\resampled\CBH_thermal_merged_2020_mosaic_thermal ir_average_0.15_resampled.tif'
    PATH_21_TIR = r'C:\data\0_Kytalyk\0_drone\2021\CBH\TIR\resampled\CBH_thermal_merged_2021_mosaic_thermal ir_average_0.15_resampled.tif'
    
    PATH_CL = r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\data\classification_data\CBH\2021\CBH_2021_classified_filtered5.tif'
    PATH_WATERMASK = r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\data\classification_data\CBH\2020\V4\CBH_2020_classified_filtered5.tif'

elif site == 'TLB':
    PATH_20_TIR = r'C:\data\0_Kytalyk\0_drone\2020\TLB\TIR\resampled\TLB_thermal_2020_mosaic_thermal ir_average_0.15_resampled.tif'
    PATH_21_TIR = r'C:\data\0_Kytalyk\0_drone\2021\TLB\TIR\resampled\TLB_thermal_merged_2021_mosaic_thermal ir_average_0.15_resampled.tif'

    PATH_NDVI_20 = r'C:\data\0_Kytalyk\0_drone\2020\TLB\MSP\indices\TLB_2020_NDVI.tif'
    PATH_NDVI_21 = r'C:\data\0_Kytalyk\0_drone\2021\TLB\MSP\indices\TLB_2021_NDVI.tif'

    PATH_CL = r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\data\classification_data\TLB\2021\V2\TLB_2021_classified_filtered5.tif'
    PATH_WATERMASK = r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\data\classification_data\TLB\2020\V1\TLB_2020_classified_filtered5.tif'

df_fluxtower = myf.read_fluxtower('C:/data/0_Kytalyk/7_flux_tower/CR3000_Kytalyk_Meteo.dat')

parser = lambda date: pd.to_datetime(date,format='%d-%m-%Y %H:%M').tz_localize('Asia/Srednekolymsk')

df_flighttimes = pd.read_csv(r'C:\data\0_Kytalyk\0_drone\flight_times.csv', sep = ';', parse_dates = [4,5],
                             date_parser=parser)

dxy = 3000 # 667 pixels along 100m ==> 1 ha
ymin = 300
ymax = ymin + dxy
xmin = 100
xmax = xmin + dxy
extent = {'xmin': xmin,'xmax': xmax,
          'ymin': ymin, 'ymax':ymax}

T_air_20 = GetMeanTair(df_flighttimes, df_fluxtower, site, 2020)
T_air_21 = GetMeanTair(df_flighttimes, df_fluxtower, site, 2021)

I_cl_s,I_wm_s, I_tir_20_s, I_tir_21_s = PrepRasters(PATH_CL,PATH_WATERMASK,
                                                    PATH_20_TIR,T_air_20,
                                                    PATH_21_TIR,T_air_21,
                                                    extent)

# %% 1. REARRANGE DATA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

classdict = {0:'dry', 1:'wet',2:'shrubs',3:'water',4:'ledum_moss_cloudberry',5:'mud',7:'tussocksedge'}
names = [classdict[i] for i in np.unique(I_cl_s)]

if year == 2020:
    I_tir_20_s[np.logical_and(I_cl_s == 3, I_wm_s != 3)] = np.nan

df_m = pd.concat(
    map(lambda arr,year: MeltArrays(arr,I_cl_s,names,year),
        [I_tir_20_s,I_tir_21_s],[2020,2021]), 
    ignore_index=False).reset_index(drop=True)

# d = pd.concat(
#     map(lambda arr,year: MeltArrays(arr,I_cl_s,names,year),
#         [I_ndvi_20_s,I_ndvi_21_s],[2020,2021]), 
#     ignore_index=False).reset_index(drop=True)
# df_m['NDVI'] = d.value

# Compute standardize temperatures:
df_m['stdT'] = df_m.groupby('year').value.apply(ScaleMinMax) # Standardize MinMax
# df_m['std_value'] = df_m.groupby('year').value.apply(lambda x: x / np.nanquantile(x,.99))  # Standardize by 99th percentile
# df_m['std_value'] = df_m.groupby('year').value.apply(lambda x: x / np.mean(x)) # Standardize by Mean

# Generate random sample of size n per class and year:
n = 100000
df_m_s = df_m.sample(frac = 1, random_state = 10) # shuffle data
df_m_s = pd.concat(
    [g[g.value.notna()][:n] for _, g in df_m_s.groupby(['year', 'variable'], sort=False, as_index=False)],
    ignore_index=True 
)  

# Sort data along moisture gradient
if site == 'TLB':
    label_order = ['water', 'wet', 'shrubs', 'dry']
elif site == 'CBH':
    label_order = ['water', 'mud', 'wet','ledum_moss_cloudberry', 'shrubs', 'dry']
elif site == 'Ridge':
    label_order = ['wet','shrubs','tussocksedge']

df_m_s['variable'] = df_m_s.variable.astype("category").cat.set_categories(label_order, ordered=True)
# df_m_s.to_csv(r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\data\thermal_data/TLB_2021_deltaT.csv',sep=';')

# %% 2.a) DENSITY PLOT IN BOTH YEARS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sys.path.append(r"C:\Users\nils\Documents\1_PhD\5_CHAPTER1\code\main_scripts")

from FigFunctions import PlotDensities

xvar = 'value'
PATH_SAVE = r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\figures_and_maps\thermoregulation\densityplots_bothyears_%s.png'%xvar

PlotDensities(df_m_s, xvar,PATH_SAVE,
              saveFig = False, 
              showSignif = True,
              showTair = False,
              showBothYears = True )

# %% 2.b) DENSITY PLOT IN NORMAL YEAR
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PATH_SAVE = r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\figures_and_maps\thermoregulation\densityplot_2021_%s.png'%xvar

PlotDensities(df_m_s, xvar,PATH_SAVE,
              saveFig = False, 
              showSignif = False,
              showTair = False,
              showBothYears = False )

# %% 3. BOXPLOT IN BOTH YEARS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
from FigFunctions import PlotBoxWhisker

yvar ='stdT'
PATH_SAVE = r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\figures_and_maps\thermoregulation\%s_%s_boxplots.png' % (site,yvar)

PlotBoxWhisker(df_m_s, yvar,
               label_order[2:],
               PATH_SAVE,
               saveFig = False)

# %% 4. COMPUTE FCOVER IN 1.5 x 1.5 m QUADRATS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import xarray as xr
import rasterio.plot
import rioxarray as rxr
from tqdm import tqdm

def GetFcover(dA_cl, dA_tir):
    df = pd.DataFrame(columns = ['classes','count','fcover','meanT'])
    df['classes'], df['count'] = np.unique(dA_cl,return_counts=True)
    
    df['fcover'] = df['count'] / df['count'].sum() *100
    
    for i,cl in enumerate(df['classes']):
        df.loc[i,'meanT'] = np.nanmean(xr.where(dA_cl == cl, dA_tir,np.nan))
        
    return df

dA_tir20 = rxr.open_rasterio(PATH_20_TIR)
dA_tir21 = rxr.open_rasterio(PATH_21_TIR)
 
dA_cl = rxr.open_rasterio(PATH_CL)
da_wm = rxr.open_rasterio(PATH_WATERMASK)

T_air = GetMeanTair(df_flighttimes, df_fluxtower, site, 2020)

dA_tir20_s = dA_tir20.isel(x = slice(xmin,xmax), y = slice(ymin,ymax)) - T_air

T_air = GetMeanTair(df_flighttimes, df_fluxtower, site, 2021)

dA_tir21_s = dA_tir21.isel(x = slice(xmin,xmax), y = slice(ymin,ymax)) - T_air


dA_cl_s = dA_cl.isel(x = slice(xmin,xmax), y = slice(ymin,ymax))

df_out = pd.DataFrame(columns = ['classes','count','fcover','meanT'])
quadrat_sidelength = 10
qu_radius = int(quadrat_sidelength/2)
for it in tqdm(np.arange(qu_radius,dA_cl_s.shape[2],qu_radius)):
    df_out = pd.concat([df_out,
                        GetFcover(dA_cl_s[0].isel(
                            x = slice(it-qu_radius, it+qu_radius),
                            y = slice(it-qu_radius, it+qu_radius)),
                                  dA_tir21_s[0].isel(
                                      x = slice(it-qu_radius, it+qu_radius),
                                      y = slice(it-qu_radius, it+qu_radius)
                                      ))],
                       axis = 0
                       )
# %% 4.a) PLOT FCOVER VS. MEAN TEMPERATURE IN QUADRATS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from FigFunctions import PlotFcoverVsTemperature
PATH_OUT = r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\figures_and_maps\thermoregulation\%s_Fcover_vs_mean_deltaT_%s.png' % (site,year)

binning_var = 'fcover' #variable to bin by
value_var = df_out.columns[np.isin(df_out.columns, binning_var,invert=True)][-1]

bin_interval = 5

PlotFcoverVsTemperature(df_out, 
                        binning_var,
                        bin_interval,
                        value_var,
                        label_order[2:],
                        PATH_OUT,
                        saveFig = False)


# %%
from pyproj import Transformer,Proj

import sys
sys.path.append(r"C:/Users/nils/Documents/1_PhD/5_CHAPTER1/code/")

import importer
df_hobo_tlb,df_hobo_ridge, df_fluxtower, df_tmst, df_tmst_delta, df_tmst_h, df_tmst_d, loggers = importer.load_all()
df_hobo = df_hobo_tlb

def trans_fct(row):
    x1,y1 = p1(row.Lon_dd, row.Lat_dd)
    x,y = transformer.transform(y1, x1)
    return x,y

transformer = Transformer.from_crs("epsg:4326",'epsg:32655' )
p1 = Proj("epsg:4326")

loggers.loc[:,['lat_utm','lon_utm']] = loggers.loc[:,['Lat_dd','Lon_dd']].apply(trans_fct, 
                                                                          axis=1, 
                                                                          result_type='expand').values

# %% 
def FetchImageData(Raster, lons, lats):
    res = .15
    subset = [Raster.sel(x = slice(x-res,x+res),
                        y = slice(y+res,y-res),
                        # method = 'nearest'
                        ).mean().values for y,x in tqdm(zip(lons,lats))]
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
    
elif site == 'CBH':
    print('No TOMST data in cloudberry hills')
    
    

# %%
sns.set_theme(style="ticks", 
              rc={"figure.figsize":(10, 10)},
              font_scale = 2)

df_p = df.loc[:,['T1','T2','T3']].melt()
df_p['LST'] = np.hstack([df.TIR.values] * 3) #[df.TIR.values] * 3
ax = sns.scatterplot(df_p, x = 'LST',y = 'value', hue = 'variable',s = 100)
ax.set(xlim=[25,35],
        ylim=[2,35])

# plt.savefig(r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\figures_and_maps\thermoregulation\TLB_TIR_vs_TOMST.png',
#             bbox_inches = 'tight')
# %% PLOT SENSOR DRIFT
sys.path.append(r"C:\Users\nils\Documents\1_PhD\5_CHAPTER1\code\main_scripts")

from FigFunctions import GatherData
from glob import glob

fl = glob(r'C:\data\0_Kytalyk\0_drone\internal_data\metadata*.csv')

unc_instr = .1
cutoff = 13

df = pd.concat(map(lambda fn:GatherData(fn, unc_instr, FitOnUnstable = False),fl), 
               ignore_index=False)

plot_fit = 'fit_3'

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

for year,margin_title in zip([2020,2021],g._margin_titles_texts):
    margin_title.set_text(year)

g.fig.subplots_adjust(top=0.9)

# plt.savefig('../../figures_and_maps/thermal_mosaics/sensorT_correctionT_4.png',bbox_inches = 'tight', facecolor='white')
plt.show()
# %%
g = sns.FacetGrid(df, col="site",  row="year", sharex=True)

g.map(sns.lineplot, "flighttime_min", "T_sensor", color = 'gray')
g.map(sns.scatterplot, "flighttime_min", "T_sensor",'isStable',palette = "Paired",marker = '+')

g.add_legend(title = 'Is the sensor temperature stable? \n i.e. within %.1f °C of the minimum?' % unc_instr)
# plt.savefig('../../figures_and_maps/thermal_mosaics/sensorT_time_2.png',bbox_inches = 'tight', facecolor='white')

plt.show()

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




