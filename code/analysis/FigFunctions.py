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
import matplotlib.ticker as ticker

def PlotDensities(data: pd.DataFrame, xvar: str,
                  PATH_OUT: str,
                  saveFig: bool,
                  colors: dict,
                  ax = None, 
                  xlim = None,
                  showSignif = True,
                  data_pairtest = None,
                  order = None,
                  showWater = True,
                  showBothYears = False ):
    
    # Set seaborn theme:
    custom_params = {"axes.spines.right": False, "axes.spines.top": False,
                     "figure.figsize":(20, 10)}
    
    sns.set_theme(style="ticks",
                  rc=custom_params,
                  font_scale = 5)
    
    # Mask out Mud & water if not wanted
    if not showWater:
        mask = np.logical_and(data.variable != 'water',data.variable != 'mud')
        data = data[mask]
        order = [x for x in order if x not in ['water','mud']]
        data.loc[:,'variable'] = data.variable.cat.set_categories(order, ordered=True)
        
        colors = {x: colors[x] for x in colors if x not in ['water','mud']}
    else:
        mask = data.variable != 'mud'
        data = data[mask]
        order = [x for x in order if x not in ['mud']]
        data.loc[:,'variable'] = data.variable.cat.set_categories(order, ordered=True)
        
        colors = {x: colors[x] for x in colors if x not in ['mud']}
        
    lw = 8
    
    xlab = 'Water deficit index (-)' if xvar == 'wdi' else '$\Delta T_{surf-air}$ (°C)'
    
    lbl = {'water':'Open water',
           'mud': 'Mud',
           'LW1': 'LW1: Low-centered \nwetland complex (bottom)',
           'LW2': 'LW2: Low-centered \nwetland complex (top)', 
           'HP2': 'HP2: High-centered \npolygons (dwarf birch)', 
           'HP1': 'HP1: High-centered \npolygons (dry sedges & lichen)',
           'TS': 'TS: Tussock–sedge'}
    
    if showBothYears:
        if ax == None: # create new axis instance
            # Plot densities for 2020
            ax = sns.kdeplot(data = data[data.year==2020],
                             x=xvar, hue="variable",
                             fill=False, 
                             common_norm=False, 
                             palette=colors,
                             alpha=1, linewidth=lw,
                             legend = False)
        else:  # plot on existing axis instance (for multipanel figure)
            # Plot densities for 2020
            sns.kdeplot(data = data[data.year==2020],
                        ax = ax,
                        x=xvar, hue="variable",
                        fill=False, 
                        common_norm=False, 
                        palette=colors,
                        alpha=1, linewidth=lw,
                        legend = False)
        
        # set axis labels
        ax.set(xlabel = xlab,
               ylabel = 'Kernel density estimate')
            
        # Make 2020 densities dashed
        [L.set_linestyle('--') for L in ax.get_lines()]

        # Plot densities for 2021
        ax2 = sns.kdeplot(data = data[data.year==2021],
                    x=xvar, hue="variable",
                    fill=False, 
                    common_norm=False,
                    palette=colors,
                    alpha=1, linewidth=lw,
                    ax = ax,
                    legend = True)

        # set legend icons and add line indicator
        legend = ax.get_legend()
        handles = legend.legendHandles + [Line2D([0], [0], color='k',alpha = 0, ls='--', lw = lw),
                                          Line2D([0], [0], color='k', ls='--', lw = lw),
                                          Line2D([0], [0], color='k', ls='-', lw = lw)]
        
        # set adjusted labels for legend
        labels_adj = [lbl[t.get_text()] for t in legend.texts] + ['',2020,2021]
        
        leg = ax.legend(handles = handles, 
                  labels = labels_adj ,
                  loc='center left', bbox_to_anchor=(1.05, 0.5),
                  fancybox=True, shadow=True, ncol=2
                  )
    else: # if only 2021 is displayed
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
    
    # Set y-axis limits to nicely formatted decimals
    n_decimals = 1
    ymax = ax.get_ylim()[1]
    
    if xvar == 'wdi':
        ymax = np.ceil(ymax) + 1
        dty = int
    else:
        ymax = 0.22 if ymax <  0.22 else np.ceil(ymax * 10**n_decimals) / 10**n_decimals
        dty = float
        
    ax.set(xlabel = xlab,
           ylim = [0,ymax],
           yticks = np.linspace(0, ymax, 3,dtype = dty))
    
    if showSignif:
        # Add significance bars over the different vegetation classes where differences are significant (t-test)
        ylim = ax.get_ylim()
        yrange = ylim[1] - ylim[0]
        h = .5 * (ylim[1] - ylim[0])
        ypos0 = round(ylim[1],2) + h
        
        # get x-coords for significance bar corners
        xs = data[data.year==2021].groupby('variable')[xvar].mean()
        
        # xs = data[data.year==2021].groupby('variable')[xvar].apply(statistics.mode)
        xs = xs[~xs.index.isin(['water'])]
        
        # add vertical lines on centers:
        for cen in xs.iteritems():
            ax.axvline(cen[1],c = colors[cen[0]], ls = '--', lw = lw, alpha = .5)
        
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
            ypos_top =  ylim[1] * .9 + ylim[1]/10 * i
            ypos_bottom = ypos_top - yrange / 30
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
            
    x_min = np.floor( round(np.quantile(data[xvar],.001),1) )
    x_max = np.ceil( round(np.quantile(data[xvar],.999),1) ) + 1 if xvar == 'Tdiff' else 1
        
    if xlim == None:
        ax.set(xlim = [x_min,x_max])
    else:
        ax.set(xlim = xlim)
    
    # Create a custom formatting function
    def custom_format(value, pos):
        if value == 0:
            return 0
        elif abs(value) % round(abs(value),1) == 0:
            return '{:0.1f}'.format(value)
        elif abs(value) % round(abs(value),2) == 0 and abs(value) % round(abs(value),1) != 0:
            return '{:0.2f}'.format(value)
        else:
            return '{:0.3f}'.format(value)
    
    # Set the y-axis tick formatter
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_format))
    
    # Adjust the position of the x-axis tick labels by moving them down
    ax.tick_params(axis='x', pad=20) 
    
    sns.despine()
    
    tickwidth = 5
    ticklength = 10
    
    # Thicken the axis spines
    ax.spines['bottom'].set_linewidth(tickwidth)
    ax.spines['left'].set_linewidth(tickwidth)
    
    # Thicken the axis ticks
    ax.tick_params(axis='x', width=tickwidth, length=ticklength)
    ax.tick_params(axis='y', width=tickwidth, length=ticklength)
 
    ax.grid(False)
    
    # Change witdth of legend lines
    legend = ax.get_legend()
    plt.setp(legend.get_lines(), linewidth=lw)
        
    if not showBothYears:
        # Set Order of legend items manually: (unused, because variable is now categorical)
        handles = legend.legendHandles
        labels_adj = [lbl[t.get_text()] for t in legend.texts]
        ax.legend(handles,labels_adj,
                  loc='center left', bbox_to_anchor=(1.05, 0.5),
                  fancybox=True, shadow=True, ncol=1)
        
    ax.legend_.set_title('$\\bf{Plant \; community}$')
    ax.legend_._legend_box.align = "left"

    # Master legend switch
    # ax.get_legend().remove()
    
    if saveFig:
        plt.savefig(PATH_OUT,bbox_inches = 'tight',dpi=300)
    
    return handles, labels_adj, ax

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
                   rc={"figure.figsize":(30, 10)},
                  font_scale = 3)
    
    lbl = {'water':'Open water',
           'mud': 'Mud',
           'LW1': '$\\bf{LW1}$\n Low-centered \nwetland complex \n(bottom)',
           'LW2': '$\\bf{LW2}$\n Low-centered \nwetland complex \n(top)', 
           'HP2': '$\\bf{HP2}$\n High-centered \npolygons \n(dwarf birch)', 
           'HP1': '$\\bf{HP1}$\n High-centered \npolygons \n(dry sedges & lichen)',
           'TS': '$\\bf{TS}$\n Tussock–sedge'}
    
    xticklabs = [lbl[t] for t in label_order]
    
    # Mask out Mud & water if not wanted
    if not showWater:
        label_order = [x for x in label_order if x not in ['water','mud']]
        
        mask = np.logical_and(data.variable != 'water',data.variable != 'mud')
        data = data[mask]
        data.loc[:,'variable'] = data.variable.cat.set_categories(list(data.variable.unique()), ordered=True)
        
        colors = {x: colors[x] for x in colors if x not in ['water','mud']}
        
        xticklabs = [lbl[t] for t in label_order]
        
    # Mask out mud
    else:
        label_order = [x for x in label_order if x not in ['mud']]
        
        mask = data.variable != 'mud'
        data = data[mask]
        data.loc[:,'variable'] = data.variable.cat.set_categories(list(data.variable.unique()), ordered=True)
        
        colors = {x: colors[x] for x in colors if x not in ['mud']}
        
        xticklabs = [lbl[t] for t in label_order]
        
    if ax == None:
        ax = sns.boxplot(data = data, 
                         x = 'variable', y = yvar,
                         hue = 'year', order = label_order,
                         whis=[1, 99],
                         # dodge = True,
                         fliersize=0.1)
    else: 
        sns.boxplot(data = data, 
                    x = 'variable', y = yvar,
                    hue = 'year', order = label_order,
                    whis=[1, 99],
                    ax = ax,
                    fliersize=0.1)
    ylab = 'Water deficit index' if yvar == 'wdi' else '$T_{surf}$ - $T_{air}$ (°C)' # 'Standardized $T_{surf}$ - $T_{air}$ (-)'
    
    ax.set(ylabel = ylab,
           xlabel = '')
    
    # Set the x tick labels on top to the plant communities and align them
    ax.set_xticklabels(xticklabs,
                       horizontalalignment='center', 
                       verticalalignment='top', 
                       fontsize=30)
    
    # set the x tick label positioning
    ax.xaxis.set_tick_params(pad=150)
    
    if yvar == 'wdi':
        ax.set_ylim([0,1]) 
    
    ax.tick_params(axis = 'both', top=True, labeltop=True,
                   bottom=False, labelbottom=False,
                   length = 4)
    
    tickwidth = 2
    ticklength = 10

    # Thicken the axis ticks
    ax.tick_params(axis='y', width=tickwidth, length=ticklength)
    
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
    ax.spines['top'].set_alpha(0)
    ax.spines['right'].set_alpha(0)
    ax.spines['left'].set_linewidth(tickwidth)  # Thicken the axis spines
    
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
    axB.set_xticklabels([' 2020','  2021'] * N_cats, ha = 'center')
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
                            ylab: str,yvar: str,
                            saveFig: bool,
                            ax = None):
    
    # classdict = {0:'dry', 1:'wet',2:'shrubs',4:'ledum_moss_cloudberry',7:'tussocksedge'}
    
    sns.set_theme(style="ticks",
                  font_scale = 5,
                  rc = {"axes.spines.right": False, "axes.spines.top": False})
    
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
           'LW1': 'LW1: Low-centered \nwetland complex (bottom)',
           'LW2': 'LW2: Low-centered \nwetland complex (top)', 
           'HP2': 'HP2: High-centered \npolygons (dwarf birch)', 
           'HP1': 'HP1: High-centered \npolygons (dry sedges & lichen)',
           'TS': 'TS: Tussock–sedge'}
    
    # Degree of the polynomial
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
        point_size = 90
        
        darker_colors = {'HP1': '#93de00',
                         'HP2': '#38A800',
                         'LW1': '#00FFC5',
                         'LW2': '#A90EFF',
                         'TS': '#3D3242',
                         'mud': '#734C00',
                         'water': '#005CE6'}
        
        if ax == None:
            sns.set_theme(font_scale = 2)
            
            ax = sns.scatterplot(
                data = df_p, 
                x = 'fcover', y = yvar, 
                hue="classes",
                hue_order = label_order,
                palette = darker_colors,
                s = point_size
            )
            
        else: 
            sns.scatterplot(
                data = df_p, 
                x = 'fcover', y = yvar, 
                hue="classes",
                hue_order = label_order,
                palette = darker_colors,
                s = point_size,
                ax = ax
            )
        
        for cl in df_p['classes'].unique():
            df_p_cl = df_p.loc[df_p['classes'] == cl,:].dropna()
            x = df_p_cl['fcover'].values
            y = df_p_cl[yvar].values
            
            # Fit the polynomial
            poly_coefficients = np.polyfit(y, x, poly_order)
            poly_fit = np.polyval(poly_coefficients, y)
            
            sns.lineplot(x = poly_fit, y = y,
                         color=colors[cl],
                         sort = False,
                         lw = 10,
                         ax = ax)
            
            # Calculate the confidence interval using bootstrapping
            # Number of bootstrap samples
            num_samples = 100
            
            # Perform bootstrapping
            bootstrapped_fits = []
            for _ in range(num_samples):
                # Resample with replacement
                indices = np.random.choice(range(len(x)), size=len(x), replace=True)
                x_sampled = x[indices]
                y_sampled = y[indices]
                
                # Fit the polynomial
                poly_coefficients = np.polyfit(y_sampled, x_sampled, poly_order)
                poly_fit = np.polyval(poly_coefficients, y)
                bootstrapped_fits.append(poly_fit)
            
            # Calculate the mean and standard deviation of bootstrapped fits
            bootstrapped_fits = np.array(bootstrapped_fits)
            mean_fit = np.mean(bootstrapped_fits, axis=0)
            std_fit = np.std(bootstrapped_fits, axis=0)
            
            # Calculate the confidence interval (95% by default)
            confidence_interval = np.percentile(bootstrapped_fits, [2.5, 97.5], axis=0)
            lower_bound = confidence_interval[0]
            upper_bound = confidence_interval[1]
            
            # Plot the confidence interval
            ax.fill_betweenx(y, lower_bound, upper_bound, color=colors[cl], alpha=0.2)
            
            # sns.regplot(
            #     data = df_p_cl,
            #     y ='fcover', x=yvar, 
            #     scatter=False,
            #     truncate=True, 
            #     order=poly_order, 
            #     lowess = lowess,
            #     color=colors[cl],
            #     ax = ax
            # )
        # ax.set_xlim(0,0.8)
           
        # Legend:
        legend = ax.get_legend()
        handles = legend.legendHandles
        
        line_handles = [plt.Line2D([], [], color=h.get_facecolor(), linestyle='-', linewidth=10) for h in handles]
        
        labels_adj = [lbl[t.get_text()] for t in legend.texts]
        ax.legend(line_handles,labels_adj,
                  loc='upper center', 
                  title = 'Plant community',
                  frameon=False,
                  bbox_to_anchor = (0.5, -0.15), 
                  ncol=2)
        # ax.legend_.set_title('$\\bf{Plant \; community}$')
        # ax.legend_._legend_box.align = "left"
            
    elif plot_type == 'jointplot':  
        hue_col = 'classes'
        handles = []
        labels = []
        
        g = sns.JointGrid(data=df_p.dropna(),
                          x='fcover',y=yvar, 
                          hue = hue_col,
                          hue_order = label_order,
                          xlim = (0, 100),
                          ylim = (0,0.8),
                          # xlim = (df_p[df_p.fcover.notna()].meanT.min(),df_p[df_p.fcover.notna()].meanT.max()),
                          marginal_ticks=False,
                          )
        
        for i,gr in df_p.groupby(hue_col):
            sns.regplot(data=gr,
                        x='fcover', y=yvar,  
                        scatter =True, ax=g.ax_joint,
                        order = poly_order,
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
    
    ax.set(xlim = [0,100],
           ylim = [0,.5],
           ylabel = ylab,
           xlabel = 'Grid cell fCover (%)')
    
    sns.despine()
    
    tickwidth = 5
    ticklength = 10
    
    # Thicken the axis spines
    ax.spines['bottom'].set_linewidth(tickwidth)
    ax.spines['left'].set_linewidth(tickwidth)
    
    # Thicken the axis ticks
    ax.tick_params(axis='x', width=tickwidth, length=ticklength)
    ax.tick_params(axis='y', width=tickwidth, length=ticklength)
    
    ax.grid(False)
    
    if saveFig:
        plt.savefig(PATH_OUT, bbox_inches = 'tight')
    plt.show()
    return ax, labels_adj, line_handles, df_p
        
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

# --------------------------------------
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

# --------------------------------------
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

# --------------------------------------
def ndvi(NIR, RED):
    return ((NIR - RED) / (NIR + RED))


# --------------------------------------
from skimage.io import imread

def PrepRasters(PATH_CL,PATH_WATERMASK,
                PATH_20_TIR,T_air_20,
                PATH_21_TIR,T_air_21,
                PATH_MSP_20,PATH_MSP_21,
                extent):
    
    I_cl = imread(PATH_CL)
    I_wm = imread(PATH_WATERMASK)
    
    # Mask out areas , where we have no water in 2020 but in 2021
    mask1 = (I_wm != 3) & (I_cl == 3)
    
    # Mask out areas , where we have no water in 2021 but have in 2020
    mask2 = (I_wm == 3) & (I_cl != 3)
    
    # Combine the masks to create the final mask
    final_mask = (mask1 | mask2)
    
    # Apply the mask to exclude values in the class map
    I_cl = np.where(final_mask,255,I_cl)
    
    I_tir_20 = np.ma.masked_less_equal(imread(PATH_20_TIR),0.1) 
    I_tir_21 = np.ma.masked_less_equal(imread(PATH_21_TIR),0.1) 
    
    I_cl_s = I_cl[extent['ymin']:extent['ymax'],extent['xmin']:extent['xmax']]
    I_wm_s = I_wm[extent['ymin']:extent['ymax'],extent['xmin']:extent['xmax']]
    
    I_tir_20_s = I_tir_20[extent['ymin']:extent['ymax'],extent['xmin']:extent['xmax']]
    I_tir_21_s = I_tir_21[extent['ymin']:extent['ymax'],extent['xmin']:extent['xmax']]
    
    # get minimum temperature above water
    if any(I_cl.ravel() == 3):
        Tmin_20 = np.nanmin(I_tir_20_s[I_cl_s == 3] - T_air_20 ) 
        Tmin_21 = np.nanmin(I_tir_21_s[I_cl_s == 3] - T_air_21 ) 
    else:
        Tmin_20 = np.nanmin(I_tir_20_s - T_air_20) 
        Tmin_21 = np.nanmin(I_tir_21_s - T_air_21 ) 

    I_wdi_20_s = ScaleMinMax(I_tir_20_s - T_air_20,Tmin_20)
    I_wdi_21_s = ScaleMinMax(I_tir_21_s - T_air_21 ,Tmin_21)
    
    # I_wdi_20_s = I_wdi_20[extent['ymin']:extent['ymax'],extent['xmin']:extent['xmax']]
    # I_wdi_21_s = I_wdi_21[extent['ymin']:extent['ymax'],extent['xmin']:extent['xmax']]
    
    I_msp_20 = imread(PATH_MSP_20)
    I_msp_21 = imread(PATH_MSP_21)
    
    I_ndvi_20 = ndvi(I_msp_20[:,:,4],I_msp_20[:,:,2])
    I_ndvi_21 = ndvi(I_msp_21[:,:,4],I_msp_21[:,:,2])
    
    I_ndvi_20_s = I_ndvi_20[extent['ymin']:extent['ymax'],extent['xmin']:extent['xmax']]
    I_ndvi_21_s = I_ndvi_21[extent['ymin']:extent['ymax'],extent['xmin']:extent['xmax']]
    
    return I_cl_s,I_wm_s, I_tir_20_s, I_tir_21_s, I_wdi_20_s, I_wdi_21_s,I_ndvi_20_s, I_ndvi_21_s
    
# --------------------------------------
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


# --------------------------------------
def ScaleMinMax(x, Tmin):
    """
    Normalizes an array by its range (min & max). 
    """
    # return (x - np.nanquantile(x,.0001)) / (np.nanquantile(x,.9999) - np.nanquantile(x,.0001))
    # return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
    
    # get maximum temperature in scene
    Tmax = np.nanmax(x)
    
    return (x - Tmin) / (Tmax - Tmin)


# --------------------------------------
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

# --------------------------------------
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

# --------------------------------------
def pretty_table(rows, column_count, column_spacing=4):
    aligned_columns = []
    for column in range(column_count):
        column_data = list(map(lambda row: row[column], rows))
        aligned_columns.append((max(map(len, column_data)) + column_spacing, column_data))

    for row in range(len(rows)):
        aligned_row = map(lambda x: (x[0], x[1][row]), aligned_columns)
        yield ''.join(map(lambda x: x[1] + ' ' * (x[0] - len(x[1])), aligned_row))


# --------------------------------------
# from osgeo import gdal
# import xarray as xr

def GetFcover(dA_cl, dA_tir):
    df = pd.DataFrame(columns = ['classes','count','fcover','meanT','sigmaT'])
    df['classes'], df['count'] = np.unique(dA_cl,return_counts = True)
    
    df['fcover'] = df['count'] / df['count'].sum() *100
    
    for i,cl in enumerate(df['classes']):
        df.loc[i,'meanT'] = np.nanmean(np.where(dA_cl == cl, dA_tir,np.nan))
        df.loc[i,'sigmaT'] = np.nanstd(np.where(dA_cl == cl, dA_tir,np.nan))
    
    return df

# --------------------------------------
import time

def MapFcover_np(windowsize, arr_classes, arr_thermal):
    df_out = pd.DataFrame(columns = ['classes','count','fcover','meanT','sigmaT'])
    window_radius = int(windowsize/2)
    
    centers = np.arange(window_radius,arr_classes.shape[1] - window_radius, window_radius)
    
    ycoords,xcoords = np.meshgrid(centers,centers) # build grid with window centers
    
    start = time.time()
    
    # Using numpy reshape & parallel:
    # ====
    a = arr_classes # get numpy array from xarray
    c = a[a.shape[0] % windowsize:, a.shape[0] % windowsize:]
    
    nchunkrows = int(c.shape[0] / windowsize) # get n (nr. of chunkrows/columns), i.e. 8 x 8 = 64 chunks
    L = np.array_split(c,nchunkrows) # select nxn subarrays
    c = np.vstack([np.array_split(ai,nchunkrows,axis=1) for ai in L])
    
    b = arr_thermal # get numpy array from xarray
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
    
    classdict = {0:'HP1',
                 1:'LW1',
                 2:'HP2',
                 3:'water',
                 4:'LW2',
                 5:'mud',
                 7:'TS',
                 np.nan: 'nodata'}
    df_out['classes'] = df_out['classes'].map(classdict)
    df_out['meanT'] = df_out.meanT.astype(float)
    
    return df_out

# --------------------------------------
import time
from osgeo import gdal
import xarray as xr

def MapFcover_xr(windowsize, dA_classes, dA_thermal):
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