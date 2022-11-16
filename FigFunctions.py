import os
import pandas as pd
import numpy as np

from skimage import io
import pyexiv2

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from cmcrameri import cm

from glob import glob


os.chdir(r'C:\Users\nils\Documents\1_PhD\5_CHAPTER1\code\thermal_drift_correction')
from modules import *


# %% DENSITY PLOT FUNCTION
# ========================
from itertools import combinations

def PlotDensities(data: pd.DataFrame, xvar: str,
                   PATH_OUT: str,
                   saveFig: bool, 
                   showSignif = True,
                   showTair = False,T_air = None,
                   showBothYears = False ):
    
    # Set seaborn theme:
    sns.set_theme(style="ticks",
                  rc={"figure.figsize":(15, 10)},
                  font_scale = 2)
    
    
    colordict = dict(dry = '#AAFF00', 
                     water = '#005CE6',
                     wet = '#00FFC5', 
                     shrubs = '#38A800',
                     ledum_moss_cloudberry = '#FF7F7F',
                     mud = '#734C00',
                     tussocksedge = '#FFFF73')
    lw = 5
    
    xlab = 'Standardized $T_{surf}$ - $T_{air}$ (-)' if xvar == 'stdT' else '$T_{surf}$ - $T_{air}$ (°C)'
    
    if showBothYears:
        ax = sns.kdeplot(data = data[data.year==2020],
                         x=xvar, hue="variable",
                         fill=False, 
                         common_norm=False, 
                         palette=colordict,
                         alpha=1, linewidth=lw,
                         legend = False)
        [L.set_linestyle('--') for L in ax.get_lines()]

        ax2 = sns.kdeplot(data = data[data.year==2021],
                    x=xvar, hue="variable",
                    fill=False, 
                    common_norm=False,
                    palette=colordict,
                    alpha=1, linewidth=lw,
                    ax = ax,
                    legend = True)
        ax.set_xlabel(xlab)

        # legend_lines = ax.lines[4:] 
        legend = ax.get_legend()
        handles = legend.legendHandles + [Line2D([0], [0], color='k',alpha = 0, ls='--', lw = lw),
                                          Line2D([0], [0], color='k', ls='--', lw = lw),
                                          Line2D([0], [0], color='k', ls='-', lw = lw)]
        
        
        labels_adj = [legend.texts[0].get_text()] + [t.get_text() + ' veg.' for t in legend.texts[1:]]
        
        leg = ax.legend(handles = handles, 
                  labels = labels_adj + ['',2020,2021],
                  ncol = 2, loc = 'center right'
                  )
    else:    
        ax = sns.kdeplot(data = data[data.year==2021],
                         x = xvar, hue="variable",
                         fill=False, 
                         common_norm=False,
                         palette = colordict, 
                         alpha=1, linewidth = lw,
                         legend = True)
    
    if showSignif:
        # Add grey bars over the different vegetation classes where differences are significant (t-test)
        yax_max = round(ax.get_ylim()[1],2)
        xs = data[data.year==2021].groupby('variable')[xvar].median()
        xs = xs[~xs.index.isin(['water'])]
        
        xpos = pd.DataFrame({'x1': [x1 for x1,x2 in combinations(xs,2)],
                             'x2': [x2 for x1,x2 in combinations(xs,2)],
                             'diffs': [x1 - x2 for x1,x2 in combinations(xs,2)]})
        xpos = xpos.sort_values('diffs',ascending = False)
        
        i = 1
        for _,r in xpos.sort_values('diffs').iterrows():
            ypos = yax_max - yax_max/10 * i
            ax.hlines(y= ypos, xmin=r.x1, xmax=r.x2,
                      colors='gray', lw=10, alpha=0.5)
            ax.annotate('*',((r.x1 + r.x2)/2 ,ypos - ypos/100), 
                        color = 'k',
                        verticalalignment = 'center_baseline', horizontalalignment='center',
                        fontsize = 30)
            i += 1
    

    if showTair:
        ax.axvline(0, c = 'lightgray', ls = '--', lw = lw)
        ax.annotate('$T_{air}$ = %.1f °C' % T_air ,
                    (0 + 2 ,yax_max/2), 
                    color = 'lightgray',
                    verticalalignment='center_baseline', horizontalalignment='center',
                    rotation = 90,
                    fontsize = 30)
        
    x_min = round(data[xvar].min(),1)
    x_max = round(data[xvar].max(),1)
        
    ax.set(xlim = [np.floor(x_min),np.ceil(x_max)+1],
           xlabel = xlab)
    
    sns.despine()
    ax.grid(False)
    
    # Change witdth of legend lines
    legend = ax.get_legend()
    plt.setp(legend.get_lines(), linewidth=lw)
        
    if not showBothYears:
        # Set Order of legend items manually: (unused, because variable is now categorical)
        handles = legend.legendHandles
        labels_adj = [legend.texts[0].get_text()] + [t.get_text() + ' veg.' for t in legend.texts[1:]]
        
        ax.legend(handles,labels_adj,loc=1)
        
    ax.legend_.set_title('$\\bf{Land \; cover \; type}$')
    ax.legend_._legend_box.align = "left"
    
    if saveFig:
        plt.savefig(PATH_OUT,bbox_inches = 'tight',dpi=600)
    
    plt.show()

# %% BOXPLOT FUNCTION
# ========================
import matplotlib
import colorsys

def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

def PlotBoxWhisker(data: pd.DataFrame, yvar: str,
                   label_order: list,
                   PATH_OUT: str,
                   saveFig: bool):
    
    sns.set_theme(style="ticks", 
                  rc={"figure.figsize":(15, 10)},
                  font_scale = 2)
    
    mask = np.logical_and(data.variable != 'water',data.variable != 'mud')
    
    ax = sns.boxplot(data = data[mask], 
                     x = 'variable', y = yvar,
                     hue = 'year', order = label_order,
                     # dodge = True,
                     fliersize=0.1)
    
    ylab = 'Standardized $T_{surf}$ - $T_{air}$ (-)' if yvar == 'stdT' else '$T_{surf}$ - $T_{air}$ (°C)'
    
    ax.set_ylabel(ylab)
    ax.set_xlabel('')
    ax.set_xticklabels([t + ' veg.' for t in label_order])
    ax.tick_params(axis = 'both', top=True, labeltop=True,
                   bottom=False, labelbottom=False,
                   length = 4)
    ax.legend_.remove()
    
    
    N_cats = len(np.unique(data[mask].variable))
    N_hues = len(np.unique(data[mask].year))

    colordict = dict(dry = '#AAFF00', 
                     water = '#005CE6',
                     wet = '#00FFC5', 
                     shrubs = '#38A800',
                     ledum_moss_cloudberry = '#FF7F7F',
                     mud = '#734C00',
                     tussocksedge = '#FFFF73')
    
    colors = [colordict[cl] for cl in label_order]
    
    colors = np.repeat(colors,N_hues,axis=0)
    
    boxes = [p for p in ax.patches if type(p) == matplotlib.patches.PathPatch]
    
    i = 0
    for artist,color in zip(boxes,colors):
        if (i % 2 ==0):
            a = .7 
            artist.set_alpha(a)
            artist.set_hatch('//')
            rgb = matplotlib.colors.ColorConverter.to_rgb(color)
            artist.set_edgecolor(scale_lightness(rgb,0.7))
        else:
            artist.set_edgecolor(color)
        artist.set_facecolor(color)
        i+=1
    
    sns.despine(trim = True)
    ax.grid(False)
    ax.spines['bottom'].set_alpha(0)
    
    [ax.axvline(x+.5, color='lightgray') for x in ax.get_xticks()]
    
    axB = ax.secondary_xaxis('bottom')
    axB.tick_params(top=False, labeltop=False, 
                    bottom=False, labelbottom=True, 
                    length = 0)
    
    axB.set_xticks(np.arange(-.25,.25 * 3 * N_cats +.5,.5))
    axB.set_xticklabels([' 2020 \n drought','  2021 \n normal'] * N_cats, ha = 'center')
    [t.set_color(i) for (i,t) in zip(['brown','midnightblue'] * N_cats, axB.xaxis.get_ticklabels())]
    axB.spines['bottom'].set_alpha(0)
    
    if saveFig:
        plt.savefig(PATH_OUT, bbox_inches = 'tight')
    plt.show()


# %%
def PlotFcoverVsTemperature(data: pd.DataFrame, 
                            binning_variable: str,
                            bin_interval: float,
                            value_variable: str,
                            label_order: list,
                            PATH_OUT: str,
                            saveFig: bool):
    
    sns.set_theme(style="ticks", 
                  rc={"figure.figsize":(15, 10)},
                  font_scale = 2)
    
    classdict = {0:'dry', 1:'wet',2:'shrubs',4:'ledum_moss_cloudberry',7:'tussocksedge'}

    mask = np.logical_and(data.classes != 3,data.classes != 5)
    data = data.loc[mask,:]
    
    df_meanT = data.groupby(['classes',value_variable])[binning_variable].mean().reset_index()
    
    bins = np.arange(0,df_meanT[binning_variable].max(),bin_interval)
    group = df_meanT.groupby(['classes',
                              pd.cut(df_meanT[binning_variable], bins)
                              ])
    
    n_classes = len(df_meanT['classes'].unique())
    
    df_p = pd.DataFrame({'classes': np.repeat(df_meanT['classes'].unique(),len(bins)-1),
                         binning_variable: list(bins[1:]) *n_classes,
                         value_variable: group[value_variable].apply(np.nanmean).values,
                         value_variable + '_std':group[value_variable].std().values})
    df_p['classes'] = df_p['classes'].map(classdict)
    
    
    colordict = dict(dry = '#AAFF00', 
                     water = '#005CE6',
                     wet = '#00FFC5', 
                     shrubs = '#38A800',
                     ledum_moss_cloudberry = '#FF7F7F',
                     mud = '#734C00',
                     tussocksedge = '#FFFF73')
    
    ax = sns.scatterplot(
        data=df_p, 
        x='meanT', y='fcover', 
        hue="classes",palette = colordict
    )
    
    for cl in df_p['classes'].unique():
        sns.regplot(
            data=df_p.loc[df_p['classes'] == cl,:], x='meanT', y='fcover',
            scatter=False, truncate=False, 
            # order=3, 
            lowess = True,
            color=colordict[cl],ax = ax
        )
    
    ax.set(ylim = [0,100],
           xlabel = 'Average quadrat $T_{surf}$ - $T_{air}$ (°C)',
           ylabel = 'Quadrat Fcover (%)')
    
    sns.despine()
    ax.grid(False)
    
    # Legend:
    legend = ax.get_legend()
    handles = legend.legendHandles
    labels_adj =[t.get_text() + ' veg.' for t in legend.texts]
    
    ax.legend(handles,labels_adj,loc=1)
    ax.legend_.set_title('$\\bf{Land \; cover \; type}$')
    ax.legend_._legend_box.align = "left"
    
    if saveFig:
        plt.savefig(PATH_OUT, bbox_inches = 'tight')
    plt.show()
        
# %%
def GatherData(fname,unc_instr, FitOnUnstable = True):
    df = pd.read_csv(fname,sep = ';', parse_dates = [1], index_col = 0)
    df['site'] = df.filename.apply(lambda s: s.replace('\\','/').split('/')[4].split('_')[0])
    df['year'] = df.gps_time.dt.year
    df['flighttime_min'] = df.flighttime_sec.div(60)
    
    # Instrument T uncertainty
    stable_sensorT = df.T_sensor.min()
    df['isStable'] = np.logical_and(df.T_sensor <= stable_sensorT + unc_instr,
                                    df.T_sensor >= stable_sensorT - unc_instr)

    # Compute LST deviation from when sensor is stable
    stable_LST = df.mean_tile_LST[df['isStable']].mean()
    df['LST_deviation'] = df.mean_tile_LST.values - stable_LST
    
    for deg in range(1,4):
        
        if FitOnUnstable:
#             Fits the correction model only on the unstable data
            z = PolyFitSensorT(df[~df['isStable']],xvar = 'T_sensor', yvar = 'LST_deviation', degree = deg)
            df['fit_%i'% deg] = z(df.T_sensor)
        
        else:
#             Fits the model on all tiles
            z = PolyFitSensorT(df,xvar = 'T_sensor', yvar = 'LST_deviation', degree = deg)
            df['fit_%i'% deg] = z(df.T_sensor)
        
        if (df.site.unique() == 'TLB') or (df.site.unique() == 'Ridge') & (df.year.unique() == 2020):
            cutoff = 13
            # Mae 2 fits for 2020 TLB
            cond1 = df.flighttime_min <= cutoff
            z1 = PolyFitSensorT(df[cond1],
                                xvar = 'T_sensor', 
                                yvar = 'LST_deviation', 
                                degree = deg)
            df.loc[cond1,'fit_%i'% deg] = z1(df[cond1].T_sensor)
            
            cond2 = df.flighttime_min > cutoff
            z2 = PolyFitSensorT(df[cond2],
                                xvar = 'T_sensor', 
                                yvar = 'LST_deviation', 
                                degree = deg)
            df.loc[cond2,'fit_%i'% deg] = z2(df[cond2].T_sensor)
            
    return df

# %%
