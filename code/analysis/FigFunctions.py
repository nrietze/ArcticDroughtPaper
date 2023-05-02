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

MAIN_PATH = r'C:/Users/nils/OneDrive - Universität Zürich UZH/Dokumente 1/1_PhD/5_CHAPTER1'

# %% DENSITY PLOT FUNCTION
# ========================
from itertools import combinations

def PlotDensities(data: pd.DataFrame, xvar: str,
                  PATH_OUT: str,
                  saveFig: bool,
                  colors: dict,
                  ax = None, xlim = None,
                  showSignif = True, data_pairtest = None,
                  order = None,
                  showTair = False,T_air = None,
                  showWater = True,
                  showBothYears = False ):
    
    # Set seaborn theme:
    sns.set_theme(style="ticks",
                  rc={"figure.figsize":(20, 10)},
                  font_scale = 5)
    
    # Mask out Mud & water if not wanted
    if not showWater:
        mask = np.logical_and(data.variable != 'water',data.variable != 'mud')
        data = data[mask]
        order = [x for x in order if x not in ['water','mud']]
        data.loc[:,'variable'] = data.variable.cat.set_categories(order, ordered=True)
        
        colors = {x: colors[x] for x in colors if x not in ['water','mud']}

    lw = 5
    
    xlab = 'Standardized $T_{surf}$ - $T_{air}$ (-)' if xvar == 'stdT' else '$T_{surf}$ - $T_{air}$ (°C)'
    
    lbl = {'water':'Open water',
           'mud': 'Mud',
           'LW1': 'LW1',
           'LW2': 'LW2', 
           'HP2': 'HP2', 
           'HP1': 'HP1',
           'TS': 'TS'}
    
    if showBothYears:
        if ax == None:
            ax = sns.kdeplot(data = data[data.year==2020],
                             x=xvar, hue="variable",
                             fill=False, 
                             common_norm=False, 
                             palette=colors,
                             alpha=1, linewidth=lw,
                             legend = False)
        else: 
            sns.kdeplot(data = data[data.year==2020],
                        ax = ax,
                        x=xvar, hue="variable",
                        fill=False, 
                        common_norm=False, 
                        palette=colors,
                        alpha=1, linewidth=lw,
                        legend = False)
        [L.set_linestyle('--') for L in ax.get_lines()]

        ax2 = sns.kdeplot(data = data[data.year==2021],
                    x=xvar, hue="variable",
                    fill=False, 
                    common_norm=False,
                    palette=colors,
                    alpha=1, linewidth=lw,
                    ax = ax,
                    legend = True)
        ax.set( xlabel = xlab,
               ylabel = 'Kernel density estimate')

        # legend_lines = ax.lines[4:] 
        legend = ax.get_legend()
        handles = legend.legendHandles + [Line2D([0], [0], color='k',alpha = 0, ls='--', lw = lw),
                                          Line2D([0], [0], color='k', ls='--', lw = lw),
                                          Line2D([0], [0], color='k', ls='-', lw = lw)]
        
        labels_adj = [lbl[t.get_text()] for t in legend.texts]
        
        leg = ax.legend(handles = handles, 
                  labels = labels_adj + ['',2020,2021],
                  ncol = 2, loc = 'center right',
                  fontsize=20
                  )
    else:   
        if ax == None:
            ax = sns.kdeplot(data = data[data.year==2021],
                             x = xvar, hue="variable",
                             fill=False, 
                             common_norm=False,
                             palette = colors, 
                             alpha=1, linewidth = lw,
                             legend = True)
        else:
            sns.kdeplot(data = data[data.year==2021],
                        ax = ax,
                        x = xvar, hue="variable",
                        fill=False, 
                        common_norm=False,
                        palette = colors, 
                        alpha=1, linewidth = lw,
                        legend = True)
        ax.set(ylabel = 'Kernel density estimate')
    
    if showSignif:
        # Add significance bars over the different vegetation classes where differences are significant (t-test)
        ylim = ax.get_ylim()
        yrange = ylim[1] - ylim[0]
        h = .5 * (ylim[1] - ylim[0])
        ypos0 = round(ylim[1],2) + h
        
        # get x-coords for significance bar corners
        xs = data[data.year==2021].groupby('variable')[xvar].mean()
        
        import statistics
        # xs = data[data.year==2021].groupby('variable')[xvar].apply(statistics.mode)
        xs = xs[~xs.index.isin(['water'])]
        
        # add vertical lines on centers:
        for cen in xs.iteritems():
            ax.axvline(cen[1], c = colors[cen[0]], ls = '--', lw = lw, alpha = .5)
        
        xpos = pd.DataFrame({
            'group1': [x1 for x1,x2 in combinations(xs.index,2)],
            'group2': [x2 for x1,x2 in combinations(xs.index,2)],
            'x1': [x1 for x1,x2 in combinations(xs,2)],
            'x2': [x2 for x1,x2 in combinations(xs,2)],
            'diffs': [x1 - x2 for x1,x2 in combinations(xs,2)]})
        xpos = xpos.sort_values('diffs',ascending = False)
        
        # Merge with pairwise difference test table to excxlude non-significant pairs
        xpos_merged = pd.merge(left = xpos, 
                               right = data_pairtest.loc[:,['group1','group2','reject']], 
                               how = 'outer',
                               on = ['group1','group2'])
        
        # get rows where matching not successful, then swap group columns and rematch & merge
        missed_idx = xpos_merged.x1.isna()
        pairtest_missed = xpos_merged[['group1','group2','reject']][missed_idx]
        pairtest_missed.columns = ['group2','group1','reject']
        
        xpos_merged = pd.merge(left = xpos_merged,
                               right = pairtest_missed, 
                               how = 'outer',
                               on = ['group1','group2'])
        xpos_merged['reject'] = xpos_merged[['reject_x','reject_y']].any(axis = 'columns')
        
        xpos_merged = xpos_merged[~missed_idx].loc[xpos_merged.reject] # sort out non-significant ones
        
        i = 1
        
        for _,r in xpos_merged.iterrows():
            ypos_top =  ylim[1] + ylim[1]/20 * i
            ypos_bottom = ypos_top - yrange / 50
            # ax.hlines(y = ypos, xmin = r.x1 / yrange, xmax = r.x2 / yrange,
            #           colors='gray', lw=10, alpha=0.5, transform=plt.gcf().transFigure)
            
            line_x, line_y = [r.x1, r.x1, r.x2, r.x2], [ypos_bottom, ypos_top, ypos_top, ypos_bottom]
            
            line = Line2D(line_x, line_y, lw=2, c='k', transform=ax.transData)
            line.set_clip_on(False)
            ax.add_line(line)
            
            ax.annotate('*',((r.x1 + r.x2)/2, ypos_top + yrange / 100), 
                        color = 'k', transform = ax.transData,
                        clip_on=False, annotation_clip=False,
                        verticalalignment = 'center_baseline', horizontalalignment='center',
                        fontsize = 40)
            
            i += 1
            
    if showTair:
        ax.axvline(0, c = 'lightgray', ls = '--', lw = lw)
        ax.annotate('$T_{air}$ = %.1f °C' % T_air ,
                    (0 + 2 ,yax_max/2), 
                    color = 'lightgray',
                    verticalalignment='center_baseline', horizontalalignment='center',
                    rotation = 90,
                    fontsize = 30)
        
    x_min = np.floor( round(np.quantile(data[xvar],.001),1) )
    x_max = np.ceil( round(np.quantile(data[xvar],.999),1) ) + 1 if xvar == 'Tdiff' else 1
        
    if xlim == None:
        ax.set(xlim = [x_min,x_max])
    else:
        ax.set(xlim = xlim)
    
    ax.set(xlabel = xlab)
    
    sns.despine()
    ax.grid(False)
    
    # Change witdth of legend lines
    legend = ax.get_legend()
    plt.setp(legend.get_lines(), linewidth=lw)
        
    if not showBothYears:
        # Set Order of legend items manually: (unused, because variable is now categorical)
        handles = legend.legendHandles
        labels_adj = [lbl[t.get_text()] for t in legend.texts]
        ax.legend(handles,labels_adj)
        
    ax.legend_.set_title('$\\bf{Plant \; community}$')
    ax.legend_._legend_box.align = "left"
    
    # Master legend switch
    ax.get_legend().remove()
    
    if saveFig:
        plt.savefig(PATH_OUT,bbox_inches = 'tight',dpi=300)
    
    return ax
    plt.show()

# %% BOXPLOT FUNCTION
# ========================
import matplotlib
import colorsys
from itertools import combinations

def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

def PlotBoxWhisker(data: pd.DataFrame, yvar: str,
                   label_order: list,
                   colors: dict,
                   showSignif: bool,
                   showWater: bool,
                   PATH_OUT: str,
                   saveFig: bool,
                   ax = None,
                   data_ttest = None):
    
    sns.set_theme(style="ticks", 
                   rc={"figure.figsize":(20, 10)},
                  font_scale = 3)
    
    lbl = {'water':'Open water',
           'mud': 'Mud',
           'LW1': 'LW1',
           'LW2': 'LW2', 
           'HP2': 'HP2', 
           'HP1': 'HP1',
           'TS': 'TS'}
    
    xticklabs = [lbl[t] for t in label_order]
    
    # Mask out Mud & water if not wanted
    if not showWater:
        label_order = [x for x in label_order if x not in ['water','mud']]
        
        mask = np.logical_and(data.variable != 'water',data.variable != 'mud')
        data = data[mask]
        data.loc[:,'variable'] = data.variable.cat.set_categories(list(data.variable.unique()), ordered=True)
        
        colors = {x: colors[x] for x in colors if x not in ['water','mud']}
        
        xticklabs = [lbl[t] for t in label_order]
    
    if ax == None:
        ax = sns.boxplot(data = data, 
                         x = 'variable', y = yvar,
                         hue = 'year', order = label_order,
                         # dodge = True,
                         fliersize=0.1)
    else: 
        sns.boxplot(data = data, 
                    x = 'variable', y = yvar,
                    hue = 'year', order = label_order,
                    ax = ax,
                    fliersize=0.1)
    ylab = 'Water deficit index (-)' if yvar == 'stdT' else '$T_{surf}$ - $T_{air}$ (°C)' # 'Standardized $T_{surf}$ - $T_{air}$ (-)'
    
    ax.set(ylabel = ylab,
           xlabel = '',
           xticklabels = xticklabs)
    
    if yvar == 'stdT':
        ax.set_ylim([0,1]) 
    
    ax.tick_params(axis = 'both', top=True, labeltop=True,
                   bottom=False, labelbottom=False,
                   length = 4)
    ax.legend_.remove()
    
    N_cats = len(data.variable.unique())
    N_hues = len(data.year.unique())
    
    colors = [colors[cl] for cl in label_order]
    
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
    
    if showSignif:
        # get positions of significant pairs that need to be annotated (where ttest is rejected)
        x_toannotate = data_ttest.loc[label_order,'reject'].reset_index().index
        
        for x_annot in x_toannotate.to_numpy(): 
            y_annot = .9 if yvar == 'stdT' else np.quantile(ax.get_yticks(),.8)
            
            ax.annotate('*',(x_annot ,y_annot), 
                        color = 'k', transform = ax.transData,
                        clip_on=False, annotation_clip=False,
                        verticalalignment = 'center_baseline', horizontalalignment='center')
        
    axB = ax.secondary_xaxis('bottom')
    axB.tick_params(top=False, labeltop=False, 
                    bottom=False, labelbottom=True, 
                    length = 0)
    
    axB.set_xticks([-.25 + .5 * i for i in range(N_cats*2)])
    axB.set_xticklabels([' 2020 \n dry','  2021 \n wetter'] * N_cats, ha = 'center')
    [t.set_color(i) for (i,t) in zip(['brown','midnightblue'] * N_cats, axB.xaxis.get_ticklabels())]
    axB.spines['bottom'].set_alpha(0)
    
    if saveFig:
        plt.savefig(PATH_OUT, bbox_inches = 'tight')
        
    return ax
    plt.show()


# %% FCOVER FUNCTIONS
from matplotlib.lines import Line2D

def PlotFcoverVsTemperature(data: pd.DataFrame, 
                            binning_variable: str,
                            bin_interval: float,
                            value_variable: str,
                            colors: dict,
                            label_order: list,
                            model: str,plot_type:str,
                            PATH_OUT: str,
                            xlab: str,xvar: str,
                            saveFig: bool):
    
    sns.set_theme(style="ticks", 
                  rc={"figure.figsize":(15, 10)},
                  font_scale = 3)
    
    # classdict = {0:'dry', 1:'wet',2:'shrubs',4:'ledum_moss_cloudberry',7:'tussocksedge'}

    mask = np.logical_and(data.classes != 'water',data.classes != 'mud')
    data_masked = data.loc[mask,:]
    label_order = [x for x in label_order if x not in ['water','mud']]
    
    # data_masked = data.groupby(['classes',value_variable])[binning_variable].mean().reset_index()
    
    bins = np.arange(0,data_masked[binning_variable].max(),bin_interval)
    group = data_masked.groupby(['classes',
                              pd.cut(data_masked[binning_variable], bins)
                              ])
    
    classnames = list(data_masked['classes'].unique())
    n_classes = len(classnames)
    
    df_p = pd.DataFrame({'classes': np.repeat(classnames,len(bins)-1),
                         binning_variable: list(bins[1:]) *n_classes,
                         value_variable: group[value_variable].apply(np.nanmean).reindex(classnames,level=0).values, # have to reindex to insert values at right class
                         value_variable + '_std':group[value_variable].std().reindex(classnames,level=0).values})
    # df_p['classes'] = df_p['classes'].map(classdict)
    
    lbl = {'water':'Open water',
           'mud': 'Mud',
           'LW1': 'LW1',
           'LW2': 'LW2', 
           'HP2': 'HP2', 
           'HP1': 'HP1',
           'TS': 'TS'}
    
    if model == 'linear':
        poly_order = 1
        lowess = False
    elif model == 'quadratic':
        poly_order = 2
        lowess = False
    elif model == 'cubic':
        poly_order = 3
        lowess = False
    elif model == 'lowess':
        poly_order = 1
        lowess = True
        
    
    if plot_type == 'regplot':
        ax = sns.scatterplot(
            data = df_p, 
            x=xvar, y='fcover', 
            hue="classes",
            hue_order = label_order,
            palette = colors,
            s = 15
        )
        
        for cl in df_p['classes'].unique():
            
            sns.regplot(
                data=df_p.loc[df_p['classes'] == cl,:].dropna(),
                x=xvar, y='fcover',
                scatter=False,
                truncate=True, 
                order=poly_order, 
                lowess = lowess,
                color=colors[cl],ax = ax
            )
        # ax.set_xlim(0,0.8)
           
        # Legend:
        legend = ax.get_legend()
        handles = legend.legendHandles
        labels_adj = [lbl[t.get_text()] for t in legend.texts]
        
        ax.legend(handles,labels_adj,loc= 'best')
        ax.legend_.set_title('$\\bf{Plant \; community}$')
        ax.legend_._legend_box.align = "left"
            
    elif plot_type == 'jointplot':  
        hue_col = 'classes'
        handles = []
        labels = []
        
        g = sns.JointGrid(data=df_p.dropna(), x=xvar, y='fcover',
                          hue = hue_col,
                          hue_order = label_order,
                          ylim = (0, 100),
                          xlim = (0,0.8),
                          # xlim = (df_p[df_p.fcover.notna()].meanT.min(),df_p[df_p.fcover.notna()].meanT.max()),
                          marginal_ticks=False,
                          )
        
        for i,gr in df_p.groupby(hue_col):
            sns.regplot(x=xvar, y='fcover', data=gr,
                        scatter =True, ax=g.ax_joint,
                        order = order,
                        truncate  = True,
                        lowess = lowess,
                        color = colors[i])
            
            lw=1
            handles.append(Line2D([0], [0], color=colors[i], ls='-', lw = lw) )
            labels.append(i)
            
        g.plot_marginals(sns.kdeplot, 
                         fill = False,
                         palette = colors)
        # g.ax_marg_x.set_xlabel('density')
        # g.ax_marg_y.set_ylabel('density')
        
        g.fig.set_size_inches((17, 10))
        ax = g.ax_joint
        
        legend = g.ax_joint.legend(handles, labels,loc='best')
        g.ax_joint.legend_.set_title('$\\bf{Plant \; community}$')
        g.ax_joint.legend_._legend_box.align = "left"
    
    ax.set(ylim = [0,100],
           xlabel = xlab,
           ylabel = 'Grid cell fCover (%)')
    
    sns.despine()
    ax.grid(False)
    
    if saveFig:
        plt.savefig(PATH_OUT, bbox_inches = 'tight')
    plt.show()
    return ax, df_p
        
# %% SENSOR DRIFT FUNCTIONS

# os.chdir(MAIN_PATH + '/code/thermal_drift_correction')
# from modules import *
def PolyFitSensorT(df: pd.DataFrame, xvar:str, yvar:str, degree: int):
    """
    df: Dataframe that contains the dependent and explanatory variable (e.g. sensor Temperature and average tile LST)
    
    returns a numpy polynomial
    """
    x = df[xvar]
    y = df[yvar]
    
    return np.poly1d(np.polyfit(x,y,degree))

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
        
        if (df.site.unique() == 'TLB') & (df.year.unique() == 2020) or (df.site.unique() == 'Ridge') & (df.year.unique() == 2020):
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
            if deg == 2:
                print(f'Polynomial coefficients for {df.site.unique()} in {df.year.unique()}: Fit 1 {z1}')
                print(f'Polynomial coefficients for {df.site.unique()} in {df.year.unique()}: Fit 2 {z2}')
        
        if deg == 2:
            print(f'Polynomial coefficients for {df.site.unique()} in {df.year.unique()}: {z}')
            
    return df

# %% DATA PREPARATION FUNCTIONS

def read_fluxtower(fname):
    """
    Reads a .dat logger file from the flux tower and converts it into a Pandas.DataFrame.
    
    Requires:
    - a filename (incl. path) of the data
    
    Returns:
    - a DataFrame with a timezone-aware datetime index.
    """
    
    df = pd.read_csv(fname,
                     sep='\t',
                     skiprows=[0,2,3],
                     parse_dates = True,
                     index_col = 0,
                     low_memory=False)
    df.index = df.index.tz_localize('Asia/Srednekolymsk')
    
    return df

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


from skimage.io import imread

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
    


def MeltArrays(arr, arr_cl,names, year, val_name):
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
    val_name : str
        Name of the variable that was reshaped.

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

    df_m = df.melt(value_name = val_name)
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

# %% Statsmodels diagnostic plots for OLS
# base code
import statsmodels
from statsmodels.tools.tools import maybe_unwrap_results
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Type

style_talk = 'seaborn-talk'    #refer to plt.style.available

class Linear_Reg_Diagnostic():
    """
    Diagnostic plots to identify potential problems in a linear regression fit.
    Mainly,
        a. non-linearity of data
        b. Correlation of error terms
        c. non-constant variance
        d. outliers
        e. high-leverage points
        f. collinearity

    Author:
        Prajwal Kafle (p33ajkafle@gmail.com, where 3 = r)
        Does not come with any sort of warranty.
        Please test the code one your end before using.
    """

    def __init__(self,
                 results: Type[statsmodels.regression.linear_model.RegressionResultsWrapper]) -> None:
        """
        For a linear regression model, generates following diagnostic plots:

        a. residual
        b. qq
        c. scale location and
        d. leverage

        and a table

        e. vif

        Args:
            results (Type[statsmodels.regression.linear_model.RegressionResultsWrapper]):
                must be instance of statsmodels.regression.linear_model object

        Raises:
            TypeError: if instance does not belong to above object

        Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import statsmodels.formula.api as smf
        >>> x = np.linspace(-np.pi, np.pi, 100)
        >>> y = 3*x + 8 + np.random.normal(0,1, 100)
        >>> df = pd.DataFrame({'x':x, 'y':y})
        >>> res = smf.ols(formula= "y ~ x", data=df).fit()
        >>> cls = Linear_Reg_Diagnostic(res)
        >>> cls(plot_context="seaborn-paper")

        In case you do not need all plots you can also independently make an individual plot/table
        in following ways

        >>> cls = Linear_Reg_Diagnostic(res)
        >>> cls.residual_plot()
        >>> cls.qq_plot()
        >>> cls.scale_location_plot()
        >>> cls.leverage_plot()
        >>> cls.vif_table()
        """

        if isinstance(results, statsmodels.regression.linear_model.RegressionResultsWrapper) is False:
            raise TypeError("result must be instance of statsmodels.regression.linear_model.RegressionResultsWrapper object")

        self.results = maybe_unwrap_results(results)

        self.y_true = self.results.model.endog
        self.y_predict = self.results.fittedvalues
        self.xvar = self.results.model.exog
        self.xvar_names = self.results.model.exog_names

        self.residual = np.array(self.results.resid)
        influence = self.results.get_influence()
        self.residual_norm = influence.resid_studentized_internal
        self.leverage = influence.hat_matrix_diag
        self.cooks_distance = influence.cooks_distance[0]
        self.nparams = len(self.results.params)

    def __call__(self, plot_context='seaborn-paper'):
        # print(plt.style.available)
        with plt.style.context(plot_context):
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
            self.residual_plot(ax=ax[0,0])
            self.qq_plot(ax=ax[0,1])
            self.scale_location_plot(ax=ax[1,0])
            self.leverage_plot(ax=ax[1,1])
            plt.show()

        self.vif_table()
        return fig, ax


    def residual_plot(self, ax=None):
        """
        Residual vs Fitted Plot

        Graphical tool to identify non-linearity.
        (Roughly) Horizontal red line is an indicator that the residual has a linear pattern
        """
        if ax is None:
            fig, ax = plt.subplots()

        sns.residplot(
            x=self.y_predict,
            y=self.residual,
            lowess=True,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        residual_abs = np.abs(self.residual)
        abs_resid = np.flip(np.sort(residual_abs))
        abs_resid_top_3 = abs_resid[:3]
        for i, _ in enumerate(abs_resid_top_3):
            ax.annotate(
                i,
                xy=(self.y_predict[i], self.residual[i]),
                color='C3')

        ax.set_title('Residuals vs Fitted', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        return ax

    def qq_plot(self, ax=None):
        """
        Standarized Residual vs Theoretical Quantile plot

        Used to visually check if residuals are normally distributed.
        Points spread along the diagonal line will suggest so.
        """
        if ax is None:
            fig, ax = plt.subplots()

        QQ = ProbPlot(self.residual_norm)
        QQ.qqplot(line='45', alpha=0.5, lw=1, ax=ax)

        # annotations
        abs_norm_resid = np.flip(np.argsort(np.abs(self.residual_norm)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        for r, i in enumerate(abs_norm_resid_top_3):
            ax.annotate(
                i,
                xy=(np.flip(QQ.theoretical_quantiles, 0)[r], self.residual_norm[i]),
                ha='right', color='C3')

        ax.set_title('Normal Q-Q', fontweight="bold")
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Standardized Residuals')
        return ax

    def scale_location_plot(self, ax=None):
        """
        Sqrt(Standarized Residual) vs Fitted values plot

        Used to check homoscedasticity of the residuals.
        Horizontal line will suggest so.
        """
        if ax is None:
            fig, ax = plt.subplots()

        residual_norm_abs_sqrt = np.sqrt(np.abs(self.residual_norm))

        ax.scatter(self.y_predict, residual_norm_abs_sqrt, alpha=0.5);
        sns.regplot(
            x=self.y_predict,
            y=residual_norm_abs_sqrt,
            scatter=False, ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(residual_norm_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        for i in abs_sq_norm_resid_top_3:
            ax.annotate(
                i,
                xy=(self.y_predict[i], residual_norm_abs_sqrt[i]),
                color='C3')
        ax.set_title('Scale-Location', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$');
        return ax

    def leverage_plot(self, ax=None):
        """
        Residual vs Leverage plot

        Points falling outside Cook's distance curves are considered observation that can sway the fit
        aka are influential.
        Good to have none outside the curves.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(
            self.leverage,
            self.residual_norm,
            alpha=0.5);

        sns.regplot(
            x=self.leverage,
            y=self.residual_norm,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        leverage_top_3 = np.flip(np.argsort(self.cooks_distance), 0)[:3]
        for i in leverage_top_3:
            ax.annotate(
                i,
                xy=(self.leverage[i], self.residual_norm[i]),
                color = 'C3')

        xtemp, ytemp = self.__cooks_dist_line(0.5) # 0.5 line
        ax.plot(xtemp, ytemp, label="Cook's distance", lw=1, ls='--', color='red')
        xtemp, ytemp = self.__cooks_dist_line(1) # 1 line
        ax.plot(xtemp, ytemp, lw=1, ls='--', color='red')

        ax.set_xlim(0, max(self.leverage)+0.01)
        ax.set_title('Residuals vs Leverage', fontweight="bold")
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residuals')
        ax.legend(loc='upper right')
        return ax

    def vif_table(self):
        """
        VIF table

        VIF, the variance inflation factor, is a measure of multicollinearity.
        VIF > 5 for a variable indicates that it is highly collinear with the
        other input variables.
        """
        vif_df = pd.DataFrame()
        vif_df["Features"] = self.xvar_names
        vif_df["VIF Factor"] = [variance_inflation_factor(self.xvar, i) for i in range(self.xvar.shape[1])]

        print(vif_df
                .sort_values("VIF Factor")
                .round(2))


    def __cooks_dist_line(self, factor):
        """
        Helper function for plotting Cook's distance curves
        """
        p = self.nparams
        formula = lambda x: np.sqrt((factor * p * (1 - x)) / x)
        x = np.linspace(0.001, max(self.leverage), 50)
        y = formula(x)
        return x, y