import pandas as pd
import numpy as np
import pathlib
from tqdm import tqdm
import ninetysix as ns
import holoviews as hv
import colorcet as cc
hv.extension('bokeh')

import panel as pn
pn.extension('tabulator')
import hvplot.pandas

from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from .functions import *

def scatter_w_linreg(
    df,
    x_var,
    y_var,
    xlim, 
    ylim,
    title=None
):   
    
    # scatter plot
    plot_scatter = ns.viz.hv.Points(
        data=df,
        kdims=[x_var, y_var],
    ).opts(
        width=400,
        height=400,
        size=8,
        xlim=xlim,
        ylim=ylim,
        tools=['hover'],
    )

    # Get the regression line
    x = df[x_var].values.reshape(-1,1)
    y = df[y_var].values
    reg = LinearRegression().fit(x, y)
    fit_x = np.arange(xlim[0], xlim[1]+.5, .5)
    fit_y = reg.coef_[0]*fit_x + reg.intercept_

    # set the title
    if title==None:
        title='R-square: '+str(round(reg.score(x,y), 3))
    else:
        title = title+', R-square: '+str(round(reg.score(x,y), 3))

    # linear regression plot
    plot_linreg = ns.viz.hv.Curve(
        data=(fit_x, fit_y)
    ).opts(
        title=title,
        xlim=xlim,
        ylim=ylim
    )
        
    return plot_scatter*plot_linreg

### Functions for looking at replicates ###
def compare_reps(
    data_df,
    parent_dict,
    root_title,
    axis_lims=(-3,3)
):
    points = hv.Points(
                data_df, 
                kdims=['Fitness_1', 'Fitness_2']
            ).opts(
                tools=['hover'],
                width=450,
                height=400,
                size=6,
                alpha=0.5,
            )

    # Get the regression line
    no_NA_inf = data_df.copy().replace([np.inf, -np.inf], np.nan).dropna()
    x = no_NA_inf['Fitness_1'].values.reshape(-1,1)
    y = no_NA_inf['Fitness_2'].values.reshape(-1,1)
    reg = LinearRegression().fit(x, y)
    fit_x = np.logspace(axis_lims[0],axis_lims[1], 2000)
    fit_y = reg.coef_[0]*fit_x + reg.intercept_
    r_square = round(reg.score(x,y), 3)

    # linear regression plot
    lin_reg = hv.Curve(
        data=(fit_x, fit_y)
    ).opts(
        title=f'{root_title} replicate correlation, R-square: {r_square}',
    )

    return(points*lin_reg)

def reps_over_time(
        data_df,
        parent_dict,
        root_title,
        axis_lims=(-3,3)
    ):
    reps_over_time = {}

    for time in data_df['Time (h)'].unique():
        
        temp = data_df[data_df['Time (h)'] == time].copy()
        reps_over_time[time] = compare_reps(temp, parent_dict, root_title, axis_lims)

    return hv.HoloMap(reps_over_time).opts(shared_axes=False)

def fitness_hists(
    data_df,
    value_col
):
    # plotting
    hists = []

    times = list(data_df['Time (h)'].unique())
    times.sort(reverse=True)
    for time in times:

        temp = data_df[data_df['Time (h)'] == time].copy()

        frequencies, edges = np.histogram(temp[value_col].values, bins=np.logspace(np.log10(0.01),np.log10(100), 150))
        hist = hv.Histogram(
            (edges, frequencies), 
            label=str(time)
            ).opts(
                width=800, 
                logx=True, 
                height=500, 
                ylim=(0,2000),
                xlabel=value_col, 
                color=cc.kbc[time*7],
                fontscale=2
                )
        hists.append(hist)

    return(hv.Overlay(hists))

# plot the heatmap and histogram for a given library
def plot_hm(df, value_col, root_title=None, seq_type='AA'):
    if seq_type == 'AA':
        def hook(plot, element):
            plot.handles['y_range'].factors = [AA1 for AA1 in 'ACDEFGHIKLMNPQRSTVWY*']
            plot.handles['x_range'].factors = [AA1+AA2 for AA1 in 'ACDEFGHIKLMNPQRSTVWY*' for AA2 in 'ACDEFGHIKLMNPQRSTVWY*']

        data_df = df.copy()

        data_df['AA1_AA2'] = data_df['AA1']+ data_df['AA2']

        heatmap = hv.HeatMap(
            data_df.sort_values(['AA1_AA2', 'AA3']),
            kdims=['AA1_AA2', 'AA3'],
            vdims=[value_col]
            ).opts(
                height=300,
                width=1000,
                xrotation=90,
                fontsize={'xticks': 5},
                colorbar=True,
                tools=['hover'],
                hooks=[hook],
                title=root_title,
                clabel=value_col
            )
        
        return heatmap

    elif seq_type == 'DNA':

        data_df = df.copy()

        data_df['codon1_codon2'] = data_df['codon1']+ data_df['codon2']

        heatmap = hv.HeatMap(
            data_df.sort_values(['codon1_codon2', 'codon3']),
            kdims=['codon1_codon2', 'codon3'],
            vdims=[value_col]
            ).opts(
                height=300,
                width=1000,
                xrotation=90,
                fontsize={'xticks': 5},
                colorbar=True,
                tools=['hover'],
                title=root_title,
                clabel=value_col
            )
        
        return heatmap

    else:
        raise NotImplementedError

def plot_and_fit_loglinear(
    temp: pd.DataFrame, 
    x_var: str, 
    y_var: str, 
    k_input: float, 
    max_time=16
    ):
    """_summary_

    Args:
        temp (pd.DataFrame): _description_
        x_var (str): _description_
        y_var (str): _description_
        k_input (float): _description_
        max_time (int, optional): _description_. Defaults to 16.

    Returns:
        _type_: _description_
    """
    t0_OD = temp['InputFreq'].unique()[0]*0.05

    # Use all of the data for the scatterplot
    x_data = np.insert(list(temp[x_var].values), 0, 0)
    y_data = np.insert(list(temp[y_var].values), 0, t0_OD)

    scatterplot = hv.Points(
        (x_data, np.log(y_data))
    ).opts(
        tools=['hover'],
        size=6,
    )
    
    # Only use timepoints up to the max time for the fit
    temp = temp[temp['Time (h)'] <= max_time].copy()
    x_data = np.insert(list(temp[x_var].values), 0, 0)
    y_data = np.insert(list(temp[y_var].values), 0, t0_OD)

    popt = fit_loglinear(x_data, y_data, k_input)

    # Curve plot
    x = np.arange(0,36,1)
    y = loglinear(x, *popt)
    fitplot = hv.Curve((x,y)).opts(color='orange')

    return (fitplot*scatterplot, popt)

def plot_var_distribution(df, variables, x = list('ACDEFGHIKLMNPQRSTVWY*')):
    """_summary_

    Args:
        df (_type_): _description_
        variables (_type_): _description_
        x (_type_, optional): _description_. Defaults to list('ACDEFGHIKLMNPQRSTVWY*').

    Returns:
        _type_: _description_
    """
    df = df.dropna()

    var_values = []
    for variable in variables:
        var_values += list(df[variable].values)
    var_values = np.array(var_values)
    
    var_mean = var_values.mean()
    var_std = var_values.std()
    alpha=0.3

    plots = []

    for i in [3,2,1]:

        plot = hv.Area(
            (x,var_mean+i*var_std,var_mean-i*var_std),
            vdims=['y1','y2']
        ).opts(
            alpha=alpha,
            line_color=None,
            color=cc.kgy[50*i],
            ylabel=f'{",".join(variables)}'
        )
        plots.append(plot)
    
    mean_line = hv.HLine(var_mean).opts(
            color='black',
            ylabel=f'{", ".join(variables)}'
        )

    plots.append(mean_line)

    return hv.Overlay(plots)

def plot_SSM_scatter(df, variables, varied_AA, AAs, parent_positions):
    """_summary_

    Args:
        df (_type_): _description_
        variables (_type_): _description_
        varied_AA (_type_): _description_

    Returns:
        _type_: _description_
    """
    fixed_AAs = [i for i in [1,2,3] if i != varied_AA]

    inds = (df[f'AA{fixed_AAs[0]}'] == AAs[fixed_AAs[0]]) & (df[f'AA{fixed_AAs[1]}'] == AAs[fixed_AAs[1]])

    SSM_scatters = []

    for variable in variables:

        SSM_scatter = hv.Scatter(
            df[inds],
            kdims=[f'AA{varied_AA}'],
            vdims=[variable, 'Time (h)','AA1','AA2','AA3'],
        ).opts(
            xlabel=f'{AAs[varied_AA]}{parent_positions[varied_AA]}X',
            ylabel=f'{", ".join(variables)}',
            jitter=0.2,
            size=4,
            width=350,
            height=350,
            line_color='black',
            line_width=0.4,
            tools=['hover']
        )

        SSM_scatters.append(SSM_scatter)

    return hv.Overlay(SSM_scatters)

def plot_3d_scatters(df, variables, AAs, parent_positions, AA_list=list('ACDEFGHIKLMNPQRSTVWY*')):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        variables (list of (str)): _description_

    Returns:
        dict: _description_
    """

    plots_3d = {'AA1':{}, 'AA2':{}, 'AA3':{}}

    temp = df.copy()

    # Create the SSM_fit_dict
    SSM_fit_dict = {
        f'AA{i}': {
            f'{variable}': {} for variable in variables
        } for i in [1,2,3]
    }

    # Populate the SSM_fit_dict with single substitution fitnesses
    for varied_AA in [1,2,3]:

        fixed_AAs = [i for i in [1,2,3] if i != varied_AA]

        inds = (temp[f'AA{fixed_AAs[0]}'] == AAs[fixed_AAs[0]]) & (temp[f'AA{fixed_AAs[1]}'] == AAs[fixed_AAs[1]])
        
        _df = temp[inds].copy().reset_index(drop=True)

        for AA in AA_list:
            for variable in variables:
                SSM_fit_dict[f'AA{varied_AA}'][variable][AA] = _df[_df[f'AA{varied_AA}'] == AA][variable].values[0]

    # Add the SSM values into the temp dataframe as new columns
    for i in [1,2,3]:
        for variable in variables:
            temp[f'AA{i}_{variable}'] = temp[f'AA{i}'].apply(lambda AA: SSM_fit_dict[f'AA{i}'][variable][AA])

    # For each mutated position, loop through holding it constant while 3D 
    # scatter-plotting the other two 
    for varied_AA in [1,2,3]:
        
        fixed_AAs = [i for i in [1,2,3] if i != varied_AA]

        for AA in AA_list:
            
            plots = []
            
            for variable in variables:
                plot = hv.Scatter3D(
                    temp[temp[f'AA{varied_AA}'] == AA], 
                    kdims=[f'AA{fixed_AAs[0]}_{variable}', f'AA{fixed_AAs[1]}_{variable}'], 
                    vdims=[f'{variable}','AA1','AA2','AA3']
                ).opts(
                    title=f'AA{varied_AA}: {AAs[varied_AA]}{parent_positions[varied_AA]}X = {AA}',
                    alpha=0.5,
                    size=4,
                    width=500,
                    height=500,
                )
                plots.append(plot)

            plots_3d[f'AA{varied_AA}'][AA] = hv.Overlay(plots)

    return plots_3d

### Functions to plot up a dashboard
def create_data_dashboard(
    data_df, 
    parent_dict,
    root_title
    ):
    """
    Creates a plot layout from a data df, AA_df, and a 
    dictionary of the parent amino acids and codons.
    """
    
    return(reps_over_time(data_df,parent_dict,root_title))

### Potentially write a class for creating the dashboard? ###
class Data_Dashboard():
    def __init__(
        self,
        data=None,
        parent_dict=None,
        root_title=None
    ):
        self._data = data
        self._parent_dict = parent_dict
        self._root_title = root_title