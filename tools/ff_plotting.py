#!/usr/bin/env python3
import os
import re
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys
from matplotlib import animation, rc
from IPython.display import HTML, Image
import seaborn
import pickle
import pandas as pd
from numpy import copy
from scipy import stats
import itertools

from sklearn.metrics import r2_score

seaborn.set_theme(style="whitegrid")

seaborn.set_theme()
seaborn.set_context("paper")
seaborn.set_style('white')
#seaborn.set_style("ticks")
zesty = ['#F5793A', '#A95AA1', '#85C0F9', '#0F2080', '#528D6A']
zesty2 = ['#F65300', '#6e3a69', '#6793bc', '#081142', '#39634a'] #c86431
zesty_palette = seaborn.color_palette(palette=zesty)
zesty2_palette = seaborn.color_palette(palette=zesty2)
edges = itertools.cycle(zesty2_palette)
palette = itertools.cycle(zesty_palette)

#src_dir = os.path.abspath("/home/mmfarrugia/repos/q2mm/q2mm")
sys.path.append("/home/mfarrugi/repos/q2mm/rh-hybrid/schrodinger.ve/lib/python3.8/site-packages/q2mm-0.0.0-py3.8.egg")

#from hybrid_optimizer import PSO_GA
import q2mm.hybrid_optimizer as hybrid_optimizer
from q2mm.hybrid_optimizer import PSO_DE
from tools.plotters import plot_cost_history, plot_contour, plot_surface, plot_summary, Mesher, Designer

bond_cols = ["param_type", "atom1", "atom2", "Equilibrium Value", "Force Constant", "Dipole Moment", "FF"]
angle_cols = ["param_type", "atom1", "atom2", "atom3", "Equilibrium Value", "Force Constant", "FF"]

def plot_history(base_direc:str, directories:list, title:str, cycle_iter_length:int, starting_score:float) -> list :
    """_summary_

    Args:
        base_direc (str): _description_
        directories (list): _description_
        title (str): _description_
        cycle_iter_length (int): _description_
        starting_score (float): _description_

    Returns:
        list: _description_
    """    
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    fig.suptitle(title)
    ax[0].set_title('Score Diversity Throughout Parameterization')
    ax[1].set_title('Loss, aka Best Score Throughout Parameterization')

    ax[0].axhline(y=starting_score, color='gray')
    ax[1].axhline(y=starting_score, color='gray')

    final_scores = []

    for directory in directories:

        swarm_history_file = open(os.path.join(base_direc, directory, 'hybrid_opt_history.bin'), 'rb')
        swarm_history = pickle.load(swarm_history_file)
        swarm_history_file.close()
        num_iters = len(swarm_history['Y'])
        num_ffs = len(swarm_history['Y'][0])
        color = next(ax[0]._get_lines.prop_cycler)['color']
        Y_history = pd.DataFrame(np.array(swarm_history['Y']).reshape((num_iters, num_ffs)))
        ax[0].plot(Y_history.index, Y_history.values, '.', color=color)
        loss = Y_history.min(axis=1).cummin()
        loss.plot(kind='line', ax=ax[1], color=color, label='Final Score: '+'{0:.3f}'.format(loss.iloc[-1]))
        final_scores.append(loss.iloc[-1])

    ax[1].legend()

    ax[0].xaxis.set_ticks(np.arange(1, ax[0].get_xlim()[1], cycle_iter_length))
    ax[1].xaxis.set_ticks(np.arange(1, ax[0].get_xlim()[1], cycle_iter_length))

    plt.show()
    return final_scores

def plot_param_history(base_direc:str, directories:list, param_index:int, title:str, cycle_iter_length:int, starting_score:float) -> list :
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    fig.suptitle(title)
    ax[0].set_title('Param '+str(param_index)+' Throughout Parameterization')
    ax[1].set_title('Loss, aka Best Score Throughout Parameterization')

    ax[0].axhline(y=starting_score, color='gray')
    ax[1].axhline(y=starting_score, color='gray')

    final_scores = []

    for directory in directories:

        swarm_history_file = open(os.path.join(base_direc, directory, 'hybrid_opt_history.bin'), 'rb')
        swarm_history = pickle.load(swarm_history_file)
        swarm_history_file.close()
        num_iters = len(swarm_history['Y'])
        num_ffs = len(swarm_history['Y'][0])
        color = next(ax[0]._get_lines.prop_cycler)['color']
        X_history = np.array(swarm_history['X'])
        Y_history = pd.DataFrame(np.array(swarm_history['Y']).reshape((num_iters, num_ffs)))
        param_history = pd.DataFrame(X_history[:,:,param_index])
        ax[0].plot(param_history.index, param_history.values, '.', color=color)
        loss = Y_history.min(axis=1).cummin()
        loss.plot(kind='line', ax=ax[1], color=color, label='Final Score: '+'{0:.3f}'.format(loss.iloc[-1]))
        #final_scores.append(loss.iloc[-1])

    ax[1].legend()

    ax[0].xaxis.set_ticks(np.arange(1, ax[0].get_xlim()[1], cycle_iter_length))
    ax[1].xaxis.set_ticks(np.arange(1, ax[0].get_xlim()[1], cycle_iter_length))

    plt.show()
    return final_scores

def plot_param_history_y(base_direc:str, directories:list, param_index:int, title:str, cycle_iter_length:int, starting_score:float) -> list :
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    fig.suptitle(title)
    ax[0].set_title('Param '+str(param_index)+' Throughout Parameterization')
    ax[1].set_title('Loss, aka Best Score Throughout Parameterization')

    ax[0].axhline(y=starting_score, color='gray')
    ax[1].axhline(y=starting_score, color='gray')

    final_scores = []

    for directory in directories:

        swarm_history_file = open(os.path.join(base_direc, directory, 'hybrid_opt_history.bin'), 'rb')
        swarm_history = pickle.load(swarm_history_file)
        swarm_history_file.close()
        num_iters = len(swarm_history['Y'])
        num_ffs = len(swarm_history['Y'][0])
        color = next(ax[0]._get_lines.prop_cycler)['color']
        X_history = np.array(swarm_history['X'])
        Y_history = pd.DataFrame(np.array(swarm_history['Y']).reshape((num_iters, num_ffs)))
        param_history = pd.DataFrame(X_history[:,:,param_index])
        ax[0].plot(param_history.index, param_history.values, '.', color=color)
        ax[1].plot(param_history.index, Y_history, '.', color=color)
        #loss = Y_history.min(axis=1).cummin()
        #loss.plot(kind='line', ax=ax[1], color=color, label='Final Score: '+'{0:.3f}'.format(loss.iloc[-1]))
        #final_scores.append(loss.iloc[-1])

    ax[1].legend()

    ax[0].xaxis.set_ticks(np.arange(1, ax[0].get_xlim()[1], cycle_iter_length))
    ax[1].xaxis.set_ticks(np.arange(1, ax[0].get_xlim()[1], cycle_iter_length))

    plt.show()
    return final_scores

def plot_param_y(base_direc:str, directories:list, param_index:int, title:str, cycle_iter_length:int, starting_score:float) -> list :
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    fig.suptitle(title)
    ax[0].set_title('Param '+str(param_index)+' Throughout Parameterization')
    ax[1].set_title('Loss, aka Best Score Throughout Parameterization')

    ax[0].axhline(y=starting_score, color='gray')
    ax[1].axhline(y=starting_score, color='gray')

    final_scores = []

    for directory in directories:

        swarm_history_file = open(os.path.join(base_direc, directory, 'hybrid_opt_history.bin'), 'rb')
        swarm_history = pickle.load(swarm_history_file)
        swarm_history_file.close()
        num_iters = len(swarm_history['Y'])
        num_ffs = len(swarm_history['Y'][0])
        color = next(ax[0]._get_lines.prop_cycler)['color']
        X_history = np.array(swarm_history['X'])
        Y_history = np.array(swarm_history['Y']).reshape((num_iters, num_ffs))
        param_history = X_history[:,:,param_index]
        #ax[0].plot(param_history.index, param_history.values, '.', color=color)
        ax[1].plot(param_history, Y_history, '.')
        #loss = Y_history.min(axis=1).cummin()
        #loss.plot(kind='line', ax=ax[1], color=color, label='Final Score: '+'{0:.3f}'.format(loss.iloc[-1]))
        #final_scores.append(loss.iloc[-1])

    ax[1].legend()

    #ax[0].xaxis.set_ticks(np.arange(1, ax[0].get_xlim()[1], cycle_iter_length))
    #ax[1].xaxis.set_ticks(np.arange(1, ax[0].get_xlim()[1], cycle_iter_length))

    plt.show()
    return final_scores

def plot_param_late_y(base_direc:str, directories:list, param_index:int, title:str, cycle_iter_length:int, starting_score:float) -> list :
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    fig.suptitle(title)
    ax[0].set_title('Param '+str(param_index)+' Throughout Parameterization')
    ax[1].set_title('Loss, aka Best Score Throughout Parameterization')

    ax[0].axhline(y=starting_score, color='gray')
    ax[1].axhline(y=starting_score, color='gray')

    final_scores = []

    for directory in directories:

        swarm_history_file = open(os.path.join(base_direc, directory, 'hybrid_opt_history.bin'), 'rb')
        swarm_history = pickle.load(swarm_history_file)
        swarm_history_file.close()
        num_iters = len(swarm_history['Y'])
        num_ffs = len(swarm_history['Y'][0])
        color = next(ax[0]._get_lines.prop_cycler)['color']
        X_history = np.array(swarm_history['X'])
        Y_history = np.array(swarm_history['Y']).reshape((num_iters, num_ffs))
        param_history = X_history[:,:,param_index]
        #ax[0].plot(param_history.index, param_history.values, '.', color=color)
        ax[1].plot(param_history[:-100], Y_history[:-100], '.')
        #loss = Y_history.min(axis=1).cummin()
        #loss.plot(kind='line', ax=ax[1], color=color, label='Final Score: '+'{0:.3f}'.format(loss.iloc[-1]))
        #final_scores.append(loss.iloc[-1])

    ax[1].legend()

    #ax[0].xaxis.set_ticks(np.arange(1, ax[0].get_xlim()[1], cycle_iter_length))
    #ax[1].xaxis.set_ticks(np.arange(1, ax[0].get_xlim()[1], cycle_iter_length))

    plt.show()
    return final_scores

def plot_param_history_histogram(base_direc:str, directories:list, param_index:int, title:str, cycle_iter_length:int, starting_score:float) -> list :
    fig, ax = plt.subplots(1, len(directories), figsize=(24, 8))
    fig.suptitle(title)
    ax[0].set_title('Param '+str(param_index)+' Throughout Parameterization')
    ax[1].set_title('Loss, aka Best Score Throughout Parameterization')

    ax[0].axhline(y=starting_score, color='gray')
    ax[1].axhline(y=starting_score, color='gray')

    final_scores = []

    for i, directory in enumerate(directories):

        swarm_history_file = open(os.path.join(base_direc, directory, 'hybrid_opt_history.bin'), 'rb')
        swarm_history = pickle.load(swarm_history_file)
        swarm_history_file.close()
        num_iters = len(swarm_history['Y'])
        num_ffs = len(swarm_history['Y'][0])
        color = next(ax[0]._get_lines.prop_cycler)['color']
        X_history = np.array(swarm_history['X'])
        Y_history = pd.DataFrame(np.array(swarm_history['Y']).reshape((num_iters, num_ffs)))
        param_history = pd.DataFrame(X_history[:,:,param_index])
        aggregated_param_history = pd.DataFrame(X_history[:,:,param_index].flatten())
        seaborn.kdeplot(data=param_history,  color=color, ax=ax[i])
        seaborn.kdeplot(data=aggregated_param_history,  color=color, ax=ax[i])
        #param_history.plot.kde(ax=ax[i])


    ax[0].legend()

    plt.show()
    return

def plot_param_history_penalty(base_direc:str, directories:list, param_index:int, title:str, cycle_iter_length:int, starting_score:float) -> list :
    fig, ax = plt.subplots(1, len(directories), figsize=(24, 8))
    fig.suptitle(title)
    ax[0].set_title('Param '+str(param_index)+' Throughout Parameterization')
    ax[1].set_title('Loss, aka Best Score Throughout Parameterization')

    ax[0].axhline(y=starting_score, color='gray')
    ax[1].axhline(y=starting_score, color='gray')

    final_scores = []

    for i, directory in enumerate(directories):

        swarm_history_file = open(os.path.join(base_direc, directory, 'hybrid_opt_history.bin'), 'rb')
        swarm_history = pickle.load(swarm_history_file)
        swarm_history_file.close()
        num_iters = len(swarm_history['Y'])
        num_ffs = len(swarm_history['Y'][0])
        color = next(ax[0]._get_lines.prop_cycler)['color']
        X_history = np.array(swarm_history['X'])
        #Y_history = pd.DataFrame(np.array(swarm_history['Y']).reshape((num_iters, num_ffs)))
        #param_history = pd.DataFrame(X_history[:,:,param_index])
        x_vals = X_history[:,3,param_index]
        y_vals = np.reshape(swarm_history['Y'], (num_iters, num_ffs))[:,3]
        #print(y_vals)
        seaborn.scatterplot(x=x_vals, y=y_vals, ax=ax[i])
        seaborn.rugplot(x=x_vals, y=y_vals, ax=ax[i])
        ax[i].set_ylim(top=3)
        #aggregated_param_history = pd.DataFrame(X_history[:,:,param_index].flatten())
        #seaborn.kdeplot(data=param_history,  color=color, ax=ax[i])
        #seaborn.kdeplot(data=aggregated_param_history,  color=color, ax=ax[i])
        #param_history.plot.kde(ax=ax[i])


    ax[0].legend()

    plt.show()
    return

def plot_param_history_histogram3d(base_direc:str, directories:list, param_indices:list, title:str, cycle_iter_length:int, starting_score:float) -> list :
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    fig.suptitle(title)
    ax[0].set_title('Param '+str(param_indices)+' Throughout Parameterization')
    ax[1].set_title('Loss, aka Best Score Throughout Parameterization')

    ax[0].axhline(y=starting_score, color='gray')
    ax[1].axhline(y=starting_score, color='gray')

    final_scores = []

    for directory in directories:

        swarm_history_file = open(os.path.join(base_direc, directory, 'hybrid_opt_history.bin'), 'rb')
        swarm_history = pickle.load(swarm_history_file)
        swarm_history_file.close()
        num_iters = len(swarm_history['Y'])
        num_ffs = len(swarm_history['Y'][0])
        color = next(ax[0]._get_lines.prop_cycler)['color']
        X_history = np.array(swarm_history['X'])
        Y_history = pd.DataFrame(np.array(swarm_history['Y']).reshape((num_iters, num_ffs)))
        param_history = pd.DataFrame(X_history[:,:,param_indices])
        print(param_history)
        seaborn.histplot(x=param_history[0].values, y=param_history[1].values, color=color, ax=ax[0])
        loss = Y_history.min(axis=1).cummin()
        loss.plot(kind='line', ax=ax[1], color=color, label='Final Score: '+'{0:.3f}'.format(loss.iloc[-1]))
        #final_scores.append(loss.iloc[-1])

    ax[1].legend()

    ax[0].xaxis.set_ticks(np.arange(1, ax[0].get_xlim()[1], cycle_iter_length))
    ax[1].xaxis.set_ticks(np.arange(1, ax[0].get_xlim()[1], cycle_iter_length))

    plt.show()
    return final_scores

def plot_scores(starting_scores:pd.DataFrame, starting_score:float, scored_runs:list, final_scores:list, title:str):
    fig, ax = plt.subplots(1, len(scored_runs)+1, figsize=(8*(len(scored_runs)+1), 8))
    fig.suptitle(title)
    palette = itertools.cycle(zesty_palette)
    
    seaborn.regplot(data=starting_scores, x='Reference', y='Calculated', label='FF', fit_reg=True, ax=ax[0], color='gray')
    ax[0].set_ylim(top=5000)
    ax[0].set_xlim(xmax = 5000)
    ax[0].set_title('INITIAL - '+'{0:.3f}'.format(starting_score))
    
    for i, run in enumerate(scored_runs):
        color = next(palette)
        seaborn.regplot(data=run, x='Reference', y='Calculated', label='FF', fit_reg=True, ax=ax[i+1], color=color)
    
        ax[i+1].set_ylim(top=5000)
        ax[i+1].set_xlim(xmax = 5000)
        ax[i+1].set_title('Score: '+'{0:.3f}'.format(final_scores[i]))
    
    plt.show()

def plot_off_diag_scatter(score_matrices:list, total_scores:list, title:str=''):
    fig, ax = plt.subplots(1, len(score_matrices), figsize=(8*(len(score_matrices)), 10))
    fig.suptitle('Off-Diagonal Eigenmatrix terms'+title)
    palette = itertools.cycle(zesty_palette)
    max_y = 0.

    for i, run in enumerate(score_matrices):
        off_diag = run.loc[run['Reference'] == 0.0000]
        off_diag = off_diag.loc[off_diag['Weight'] != 0.0000]
        off_diag = off_diag.sort_values(by='Calculated', ignore_index=True)

        color = next(palette)
        seaborn.regplot(data=off_diag, x=off_diag.index, label='FF', y='Calculated', fit_reg=False, ax=ax[i], color=color)
        ax[i].set_title(total_scores[i])
        max_y = max(max_y, max(off_diag['Calculated']))
        ax[i].set_ylim(top=max_y, bottom=-max_y)

    plt.show()

def plot_off_diag_scatter_(starting_scores:pd.DataFrame, starting_score:float, scored_runs:list, final_scores:list, title:str=''):
    fig, ax = plt.subplots(1, len(scored_runs)+1, figsize=(8*(len(scored_runs)+1), 10))
    fig.suptitle('Off-Diagonal Eigenmatrix terms'+title)
    palette = itertools.cycle(zesty_palette)

    off_diag_start = starting_scores.loc[starting_scores['Reference'] == 0.0000]
    off_diag_start = off_diag_start.loc[off_diag_start['Weight'] != 0.0000]
    off_diag_start = off_diag_start.sort_values(by='Calculated', ignore_index=True)
    seaborn.regplot(data=off_diag_start, x=off_diag_start.index, label='FF', y='Calculated', fit_reg=False, ax=ax[0], color='gray')
    ax[0].set_title('FUERZA - '+'{0:.3f}'.format(starting_score))
    max_y = max(off_diag_start['Calculated'])

    for i, run in enumerate(scored_runs):
        off_diag = run.loc[run['Reference'] == 0.0000]
        off_diag = off_diag.loc[off_diag['Weight'] != 0.0000]
        off_diag = off_diag.sort_values(by='Calculated', ignore_index=True)

        color = next(palette)
        seaborn.regplot(data=off_diag, x=off_diag.index, label='FF', y='Calculated', fit_reg=False, ax=ax[i+1], color=color)
        ax[i+1].set_title(final_scores[i])
        ax[i+1].set_ylim(top=max_y, bottom=-max_y)

    plt.show()

def plot_off_diag_violin(starting_scores:pd.DataFrame, starting_score:float, scored_runs:list, final_scores:list, title:str=''):

    fig, ax = plt.subplots(1, 1, figsize=(4*(len(scored_runs)+1), 8))
    off_diag_start = starting_scores.loc[starting_scores['Reference'] == 0.0000]
    off_diag_start = off_diag_start.loc[off_diag_start['Weight'] != 0.0000]
    off_diag_start = off_diag_start.sort_values(by='Calculated', ignore_index=True)

    off_diag_merged = pd.concat([run.loc[run['Reference'] == 0.0000] for run in scored_runs])
    off_diag_merged = off_diag_merged.loc[off_diag_merged['Weight'] != 0.0000]
    off_diag_merged = pd.concat([off_diag_start, off_diag_merged])
    off_diag_merged['FF'] = off_diag_merged['FF'].astype(str)

    seaborn.violinplot(data=off_diag_merged, x='FF', y='Calculated', hue='FF')#, title='Off-Diagonal Eigenmatrix terms'+title)
    ax.set_label('FF Score')
    fig.suptitle(title)

    plt.show()

def plot_fit_diag_scores(starting_scores:pd.DataFrame, starting_score:float, scored_runs:list, final_scores, title:str=''):
    # Plot Diagonal Elements with a linear fit

    fig, ax = plt.subplots(1, len(scored_runs)+1, figsize=(8*(len(scored_runs)+1),8))
    fig.suptitle('Diagonal Eigenmatrix terms after a PSO only - Rh Hyd Enamides')
    palette = itertools.cycle(zesty_palette)

    diag = starting_scores.loc[starting_scores['Reference'] != 0.0000]
    diag = diag.loc[diag['Weight'] != 0.0000]
    slope, intercept, r2, pv, se = stats.linregress(diag['Reference'], diag['Calculated'])

    seaborn.regplot(data=diag, y='Calculated', x='Reference', color=next(palette), label='FF', line_kws={'label':'$y=%3.7s*x+%3.7s   r2:%3.7s$'%(slope, intercept, r2)}, ax=ax[0])
    ax[0].legend()
    ax[0].set_title('FUERZA - '+'{0:.3f}'.format(starting_score))

    for i, run in enumerate(scored_runs):
        diag = run.loc[run['Reference'] != 0.0000]
        slope, intercept, r2, pv, se = stats.linregress(diag['Reference'], diag['Calculated'])

        seaborn.regplot(data=diag, y='Calculated', x='Reference', color=next(palette), label=final_scores[i], line_kws={'label':'$y=%3.7s*x+%3.7s   r2:%3.7s$'%(slope, intercept, r2)}, ax=ax[i+1])

        ax[i+1].legend()
        ax[i+1].set_title('Score: '+'{0:.3f}'.format(final_scores[i]) if final_scores[i] is not str else final_scores[i])

    for a in ax:
        seaborn.lineplot(data=diag, x='Reference', y='Reference', color='gray', ax=a)

def linear_fit_diag_scores(starting_scores:pd.DataFrame, starting_score:float, scored_runs:list, final_scores, title:str=''):
    # Plot Diagonal Elements with a PROPER linear fit

    fig, ax = plt.subplots(1, len(scored_runs)+1, figsize=(5*(len(scored_runs)+1),4))
    fig.suptitle(title)
    palette = itertools.cycle(zesty_palette)

    diag = starting_scores.loc[starting_scores['Reference'] != 0.0000]
    diag = diag.loc[diag['Weight'] != 0.0000]
    slope, intercept, r2, pv, se = stats.linregress(diag['Reference'], diag['Calculated'])

    seaborn.scatterplot(data=diag, y='Calculated', x='Reference', color='gray', label=starting_score, ax=ax[0])
    ax[0].legend()
    r2_ = r2_score(diag['Reference'], diag['Calculated'])
    ax[0].set_title('Score: '+'{0:.3f}'.format(starting_score)+' y=x r2:'+'{0:.3f}'.format(r2_))


    for i, run in enumerate(scored_runs):
        diag = run.loc[run['Reference'] != 0.0000]
        diag = diag.loc[diag['Weight'] != 0.0000]
        slope, intercept, r2, pv, se = stats.linregress(diag['Reference'], diag['Calculated'])

        #seaborn.scatterplot(data=diag_start, y='Calculated', x='Reference', color=c1, edgecolor=c1e, label='Estimate r2: '+str(np.round(r2_, decimals=3))+' Score: '+str(np.round(static_score, decimals=3)), ax=ax[0], marker='s', alpha = 0.8)
        seaborn.scatterplot(data=diag, y='Calculated', x='Reference', color=next(palette), label=final_scores[i], ax=ax[i+1])
        r2_ = r2_score(diag['Reference'], diag['Calculated'])
        ax[i+1].legend()
        if type(final_scores[i]) is str:
            ax[i+1].set_title(final_scores[i] + r' $y=x r^{2}$: '+'{0:.3f}'.format(r2_))
        else:
            ax[i+1].set_title('Score: '+'{0:.3f}'.format(final_scores[i])+r' $y=x r^{2}$: '+'{0:.3f}'.format(r2_))

    for a in ax:
        seaborn.lineplot(data=diag, x='Reference', y='Reference', color='gray', ax=a)

def linear_fit_diag_scores_grid(starting_scores:list, starting_score:list, final_scores:list, final_score:list, title:str=''):
    """_summary_

    Args:
        starting_scores (list[pd.DataFrame]): _description_
        starting_score (list[float]): _description_
        final_scores (list[pd.DataFrame]): _description_
        final_score (list[float]): _description_
        title (str, optional): _description_. Defaults to ''.
    """    
    # Plot Diagonal Elements with a PROPER linear fit

    fig, ax = plt.subplots(len(starting_scores), 2, figsize=(12, 10*int(len(final_scores)/2)+1))
    fig.suptitle('Diagonal Eigenmatrix terms')
    palette = itertools.cycle(zesty_palette)

    for i, start in enumerate(starting_scores):
        diag = start.loc[start['Reference'] != 0.0000]
        diag = diag.loc[diag['Weight'] != 0.0000]
        slope, intercept, r2, pv, se = stats.linregress(diag['Reference'], diag['Calculated'])
        labl = '{0:.3f}'.format(starting_score[i]) #+ diag['FF'][0]
        seaborn.regplot(data=diag, y='Calculated', x='Reference', color=next(palette), label=labl, line_kws={'label':'$y=%3.7s*x+%3.7s   r2:%3.7s$'%(slope, intercept, r2)}, ax=ax[i][0])
        ax[i][0].legend()
        seaborn.move_legend(ax[i][0], "upper left", bbox_to_anchor=(1, 1))
        r2_ = r2_score(diag['Reference'], diag['Calculated'])
        ax[i][0].set_title(str(start['FF'][0])+' - '+'{0:.3f}'.format(starting_score[i])+' y=x r2:'+'{0:.3f}'.format(r2_))
        seaborn.lineplot(data=diag, x='Reference', y='Reference', color='gray', ax=ax[i][0])


    for i, run in enumerate(final_scores):
        diag = run.loc[run['Reference'] != 0.0000]
        diag = diag.loc[diag['Weight'] != 0.0000]
        slope, intercept, r2, pv, se = stats.linregress(diag['Reference'], diag['Calculated'])

        seaborn.regplot(data=diag, y='Calculated', x='Reference', color=next(palette), label=final_score[i], line_kws={'label':'$y=%3.7s*x+%3.7s   r2:%3.7s$'%(slope, intercept, r2)}, ax=ax[int(i%2)][int(i/2)+1])
        r2_ = r2_score(diag['Reference'], diag['Calculated'])
        ax[int(i%2)][int(i/2)+1].legend()
        seaborn.move_legend(ax[int(i%2)][int(i/2)+1], "upper left", bbox_to_anchor=(1, 1))
        ax[int(i%2)][int(i/2)+1].set_title('Score: '+'{0:.3f}'.format(final_score[i])+' y=x r2:'+'{0:.3f}'.format(r2_))
        seaborn.lineplot(data=diag, x='Reference', y='Reference', color='gray', ax=ax[int(i%2)][int(i/2)+1])


def get_ff_params(base_direc:str, directories:list, filename:str, final_scores:list, bond_rows:list, angle_rows:list, title:str='', bond_cols=bond_cols, angle_cols=angle_cols):# -> tuple[list, list]:
    # Plot FCs

    bonds = []
    angles = []
    bonds_rows = [str(bond_row+1) for bond_row in bond_rows]
    angles_rows = [str(angle_row+1) for angle_row in angle_rows]

    for directory, score in zip(directories, final_scores):
        bonds.append((pd.read_csv(os.path.join(base_direc, directory, filename), skiprows=lambda x: x not in bond_rows, sep='\s+', names=bond_cols).assign(FF=title+'{0:.3f}'.format(score))))#.assign(ff_row = bonds_rows))
        angles.append(pd.read_csv(os.path.join(base_direc, directory, filename), skiprows=lambda x: x not in angle_rows, sep='\s+', names=angle_cols).assign(FF=title+'{0:.3f}'.format(score)).assign(FF=score))#.assign(ff_row=angles_rows))

    params = [pd.concat([bond, angle]) for bond, angle in zip(bonds, angles)]
    
    return bonds, angles, params

def get_frcmod_params(base_direc:str, directories:list, filename:str, final_scores:list, bond_rows:list, angle_rows:list, title:str='', bond_cols=bond_cols, angle_cols=angle_cols):# -> tuple[list, list]:
    # Plot FCs

    bonds = []
    angles = []
    bonds_rows = [str(bond_row+1) for bond_row in bond_rows]
    angles_rows = [str(angle_row+1) for angle_row in angle_rows]

    amber_cols = ["Parameter", "Force Constant", "Equilibrium Value"]

    for directory, score in zip(directories, final_scores):
        bonds.append((pd.read_csv(os.path.join(base_direc, directory, filename), skiprows=lambda x: x not in bond_rows, sep='\s+', names=amber_cols).assign(FF=title+'{0:.3f}'.format(score))))#.assign(ff_row = bonds_rows))
        angles.append(pd.read_csv(os.path.join(base_direc, directory, filename), skiprows=lambda x: x not in angle_rows, sep='\s\s+', names=amber_cols).assign(FF=title+'{0:.3f}'.format(score)).assign(FF=score))#.assign(ff_row=angles_rows))

    params = [pd.concat([bond, angle]) for bond, angle in zip(bonds, angles)]
    
    return bonds, angles, params

def get_substr_def(filename:str, row:int) -> list:
    
    with open(filename) as fld:
        ff = fld.readlines()
        substr = ff[row-1]
    substr = substr.split()
    print(substr)
    delimiters = ['.', '-', '=', '*', '(', ')']
    result = [substr[1]]
    for delimiter in delimiters: 
        temp_result = []
        for item in result:
            temp_result.extend(item.split(delimiter))
        result = temp_result
    #substr = re.split('-|=|.|*|(|)', substr[1])
    substructure = list(filter(lambda a: a != '2' and a != '', result))
    
    return substructure

def make_labels(df:pd.DataFrame, substr_def:list) -> pd.DataFrame:
    labels = []
    for ind, entry in df.iterrows():
        try:
            label = substr_def[int(entry['atom1'])-1]
        except:
            label = str(entry['atom1'])
        try:
            label = label + ' ' + substr_def[int(entry['atom2'])-1]
        except:
            print(label)
            print(type(label))
            label = str(label) + ' ' + str(entry['atom2'])
        if 'a' in entry['param_type']: label = '170\u00b0\n'+ label
        if '2' in entry['param_type']:
            if entry['atom3'] is int or str.isdigit(entry['atom3']):
                label = label + ' ' + substr_def[int(entry['atom3'])-1]
            else:
                label = label + ' ' + str(entry['atom3'])
        labels.append(label)
    df = df.assign(label=labels)
    return df
    # label = substr_def[int(df['atom1'])-1] + ' ' + substr_def[int(df['atom2'])-1]
    # if df['param_type'].contains('a'): label = 'alt '+label
    # if df['param_type'].contains('2'): label = label + substr_def[int(entry['atom3'])-1]
    # df['label'] = label

def get_ff_param_labels(directory:str, filename:str, df:pd.DataFrame, bond_rows:list, angle_rows:list, title:str=''):# -> tuple[list, list]:
    # Plot FCs



    bonds = []
    angles = []

    bonds.append(pd.read_csv(os.path.join('base_direc', directory, filename), skiprows=lambda x: x not in bond_rows, sep='\s+', names=bond_cols).assign(FF=title+'{0:.3f}'.format(score)))
    angles.append(pd.read_csv(os.path.join('base_direc', directory, filename), skiprows=lambda x: x not in angle_rows, sep='\s+', names=angle_cols).assign(FF=title+'{0:.3f}'.format(score)))

    return bonds, angles

def filter_params_by_opt(bonds, angles, final_scores):
    param_opt = pd.DataFrame()
    param_unopt = pd.DataFrame()

    for i in range(len(bonds)):
        bonds[i]["FF"] = final_scores[i]
        angles[i]["FF"]=final_scores[i]
        if any(opt_flag in final_scores[i] for opt_flag in ['Opt', 'OPT', 'GRAD', 'HO']):
            param_opt = pd.concat([param_opt, bonds[i]])
            param_opt = pd.concat([param_opt, angles[i]])
        else:
            param_unopt = pd.concat([param_unopt, bonds[i]])
            param_unopt = pd.concat([param_unopt, angles[i]])

    return param_unopt, param_opt


def filter_params_by_type_opt(bonds, angles, final_scores):
    bond_opt = pd.DataFrame()
    bond_unopt = pd.DataFrame()
    angle_opt = pd.DataFrame()
    angle_unopt = pd.DataFrame()

    for i in range(len(bonds)):
        bonds[i] = bonds[i].assign(FF=final_scores[i])
        angles[i] = angles[i].assign(FF=final_scores[i])
        if any(opt_flag in final_scores[i] for opt_flag in ['Opt', 'OPT', 'GRAD', 'HO']):
            bond_opt = pd.concat([bond_opt, bonds[i]])
            angle_opt = pd.concat([angle_opt, angles[i]])
        else:
            bond_unopt = pd.concat([bond_unopt, bonds[i]])
            angle_unopt = pd.concat([angle_unopt, angles[i]])

    return bond_unopt, angle_unopt, bond_opt, angle_opt

def score_r2s(scores, score_sums):
    r2_scores = []
    r2_score_labels = []

    for eigenmatrix, score in zip(scores, score_sums):
        diag = eigenmatrix.loc[eigenmatrix['Reference'] != 0.0000]
        diag = diag.loc[diag['Weight'] != 0.0000]
        #slope, intercept, r2, pv, se = stats.linregress(diag_start['Reference'], diag_start['Calculated'])
        r2_ = r2_score(diag['Reference'], diag['Calculated'])
        r2_scores.append(r2_)
        r2_score_labels.append(score + r' $r^{2}$: '+str(np.round(r2_, decimals=3)))

    return r2_scores, r2_score_labels

def plot_ff_params(bonds:list, angles:list, final_scores:list, title:str='', bond_labels=None, angles_labels=None):
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    fig.suptitle('Force Constants'+title)
    ax[0].set_title('Bonds')
    ax[1].set_title('Angles')

    palette = itertools.cycle(zesty_palette)

    if bond_labels is None:
        bond_labels = bonds[0][['atom1', 'atom2', 'param_type']].values
        bond_labels = [str(bl) for bl in bond_labels]
    if angles_labels is None:
        angles_labels = angles[0][['atom1', 'atom2', 'atom3', 'param_type']].values
        angles_labels = [str(al) for al in angles_labels]
    for i in range(len(bonds)):
        color = next(palette)
        seaborn.regplot(data=bonds[i], label=final_scores[i], x = bond_labels, y="Force Constant", fit_reg=False, ax=ax[0], color=color)
        seaborn.regplot(data=angles[i], label=final_scores[i], x = angles_labels, y="Force Constant", fit_reg=False, ax=ax[1], color=color)

    plt.xticks(rotation=45)
    ax[1].legend()
    ax[0].set_ylabel(r'Force Constant ($mdyn/\AA$)')
    ax[1].set_ylabel(r'Force Constant ($mdyn/\AA$)')
    plt.show()

def plot_ff_params_v_static(bonds:list, angles:list, final_scores:list, title:str='', bond_labels=None, angles_labels=None, estimate_score=None):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle('Force Constants'+title)
    ax[0].set_title('Bonds')
    ax[1].set_title('Angles')

    colors_pick = seaborn.color_palette('muted')
    # palette = itertools.cycle(seaborn.color_palette(palette=zesty))
    # palette_opt = itertools.cycle(seaborn.color_palette(palette=zesty))
    palette = itertools.cycle(seaborn.color_palette('muted'))
    palette_opt = itertools.cycle(seaborn.color_palette('muted'))
    edges = itertools.cycle(zesty2_palette)
    
    color=next(palette)
    edge=next(edges)
    estimate_label = estimate_score if estimate_score is not None else 'Estimate'
    ax[0].axhline(5, color=color, label=estimate_label)
    ax[1].axhline(0.5, color=color, label=estimate_label)


    if bond_labels is None:
        bond_labels = bonds[0][['atom1', 'atom2', 'param_type']].values
        bond_labels = [str(bl) for bl in bond_labels]
    if angles_labels is None:
        angles_labels = angles[0][['atom1', 'atom2', 'atom3', 'param_type']].values
        angles_labels = [str(al) for al in angles_labels]

    bond_unopt, angle_unopt, bond_opt, angle_opt = filter_params_by_type_opt(bonds, angles, final_scores)
    seaborn.stripplot(data=bond_unopt, x="Parameter", y="Force Constant", ax=ax[0], hue="FF", dodge=True, palette=palette)
    seaborn.stripplot(data=angle_unopt, x="Parameter", y="Force Constant", ax=ax[1], hue="FF", dodge=True, palette=itertools.cycle(colors_pick[1:]))
    seaborn.stripplot(data=bond_opt, x="Parameter", y="Force Constant", ax=ax[0], hue="FF", dodge=True, palette=itertools.cycle(colors_pick), marker='^')
    seaborn.stripplot(data=angle_opt, x="Parameter", y="Force Constant", ax=ax[1], hue="FF", dodge=True, palette=itertools.cycle(colors_pick), marker='^')


    plt.xticks(rotation=90)
    ax[0].legend_.remove()
    ax[1].legend(bbox_to_anchor=(1.0, 0.85), fancybox=True, framealpha=0.5)
    ax[0].set_ylabel(r'Force Constant ($mdyn/\AA$)')
    ax[1].set_ylabel(r'Force Constant ($mdyn/\AA$)')
    plt.show()
    return fig, ax

def plot_ff_params_v_static_(bonds:list, angles:list, final_scores:list, title:str='', bond_labels=None, angles_labels=None, estimate_score=None):
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle('Force Constants'+title)

    colors_pick = seaborn.color_palette('muted')
    # palette = itertools.cycle(seaborn.color_palette(palette=zesty))
    # palette_opt = itertools.cycle(seaborn.color_palette(palette=zesty))
    palette = itertools.cycle(seaborn.color_palette('muted'))
    palette_opt = itertools.cycle(seaborn.color_palette('muted'))
    edges = itertools.cycle(zesty2_palette)
    
    color=next(palette)
    edge=next(edges)
    estimate_label = estimate_score if estimate_score is not None else 'Estimate'
    ax.axhline(5, color=color, label=estimate_label)
    ax.axhline(0.5, color=color, label=estimate_label)


    if bond_labels is None:
        bond_labels = bonds[0][['atom1', 'atom2', 'param_type']].values
        bond_labels = [str(bl) for bl in bond_labels]
    if angles_labels is None:
        angles_labels = angles[0][['atom1', 'atom2', 'atom3', 'param_type']].values
        angles_labels = [str(al) for al in angles_labels]

    param_unopt, param_opt = filter_params_by_opt(bonds, angles, final_scores)

    seaborn.stripplot(data=param_unopt, x="Parameter", y="Force Constant", ax=ax, hue="FF", dodge=True, palette=palette)
    seaborn.stripplot(data=param_opt, x="Parameter", y="Force Constant", ax=ax, hue="FF", dodge=True, palette=itertools.cycle(colors_pick), marker='^')


    plt.xticks(rotation=90)
    ax.legend(bbox_to_anchor=(1.0, 0.85), fancybox=True, framealpha=0.5)
    ax.set_ylabel(r'Force Constant ($mdyn/\AA$)')
    plt.show()
    return fig, ax

def compare_opt_params(bonds:list, angles:list, final_scores:list, title:str='', bond_labels=None, angles_labels=None, estimate_score=None):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Force Constants'+title)
    ax[0,0].set_title('Bonds')
    ax[0,1].set_title('Angles')

    colors_pick = seaborn.color_palette('muted') #seaborn.color_palette(palette=zesty)
    # palette_opt = itertools.cycle(seaborn.color_palette(palette=zesty))
    palette = itertools.cycle(colors_pick)
    edges = itertools.cycle(zesty2_palette)
    
    color=next(palette)
    estimate_label = estimate_score if estimate_score is not None else 'Estimate'
    ax[0,0].axhline(5, color=color, label=estimate_label)
    ax[0,1].axhline(0.5, color=color, label=estimate_label)


    if bond_labels is None:
        bond_labels = bonds[0][['atom1', 'atom2', 'param_type']].values
        bond_labels = [str(bl) for bl in bond_labels]
    if angles_labels is None:
        angles_labels = angles[0][['atom1', 'atom2', 'atom3', 'param_type']].values
        angles_labels = [str(al) for al in angles_labels]

    bond_unopt, angle_unopt, bond_opt, angle_opt = filter_params_by_type_opt(bonds, angles, final_scores)

    seaborn.stripplot(data=bond_unopt, x="Parameter", y="Force Constant", ax=ax[0,0], hue="FF", dodge=True, palette=palette)
    seaborn.stripplot(data=angle_unopt, x="Parameter", y="Force Constant", ax=ax[0,1], hue="FF", dodge=True, palette=itertools.cycle(colors_pick[1:]))
    seaborn.stripplot(data=bond_opt, x="Parameter", y="Force Constant", ax=ax[1,0], hue="FF", dodge=True, palette=itertools.cycle(colors_pick))
    seaborn.stripplot(data=angle_opt, x="Parameter", y="Force Constant", ax=ax[1,1], hue="FF", dodge=True, palette=itertools.cycle(colors_pick))

    plt.xticks(rotation=90)
    ax[0,0].legend_.remove()
    ax[1,0].legend_.remove()
    ax[0,1].legend(bbox_to_anchor=(1.0, 0.85), fancybox=True, framealpha=0.5)
    ax[1,1].legend(bbox_to_anchor=(1.0, 0.85), fancybox=True, framealpha=0.5)
    ax[0,0].set_ylabel(r'Force Constant ($mdyn/\AA$)')
    ax[0,1].set_ylabel(r'Force Constant ($mdyn/\AA$)')
    ax[1,0].set_ylabel(r'Force Constant ($mdyn/\AA$)')
    ax[1,1].set_ylabel(r'Force Constant ($mdyn/\AA$)')
    plt.show()
    return fig, ax

def compare_opt_params_(bonds:list, angles:list, final_scores:list, title:str='', bond_labels=None, angles_labels=None, estimate_score=None):
    fig, ax = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Force Constants'+title)

    colors_pick = seaborn.color_palette('muted') #seaborn.color_palette(palette=zesty)
    # palette_opt = itertools.cycle(seaborn.color_palette(palette=zesty))
    palette = itertools.cycle(colors_pick)
    edges = itertools.cycle(zesty2_palette)
    
    color=next(palette)
    estimate_label = estimate_score if estimate_score is not None else 'Estimate'
    ax[0].axhline(5, color=color, label=estimate_label)
    ax[0].axhline(0.5, color=color, label=estimate_label)


    if bond_labels is None:
        bond_labels = bonds[0][['atom1', 'atom2', 'param_type']].values
        bond_labels = [str(bl) for bl in bond_labels]
    if angles_labels is None:
        angles_labels = angles[0][['atom1', 'atom2', 'atom3', 'param_type']].values
        angles_labels = [str(al) for al in angles_labels]

    param_unopt, param_opt = filter_params_by_opt(bonds, angles, final_scores)

    seaborn.stripplot(data=param_unopt, x="Parameter", y="Force Constant", ax=ax[0], hue="FF", dodge=True, palette=palette)
    seaborn.stripplot(data=param_opt, x="Parameter", y="Force Constant", ax=ax[1], hue="FF", dodge=True, palette=itertools.cycle(colors_pick))

    for axis in ax.flat:  # Iterate through all subplots
        axis.tick_params(axis='x', rotation=90)
        axis.legend(bbox_to_anchor=(1.0, 0.85), fancybox=True, framealpha=0.5)
        axis.set_ylabel(r'Force Constant ($mdyn/\AA$)')
    plt.show()
    return fig, ax

def plot_ff_params_v_static_table(bonds:list, angles:list, finalized_bonds:pd.DataFrame, finalized_angles:pd.DataFrame, substr_def:list, title:str='', bond_labels=None, angles_labels=None, estimate_score=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Force Constants'+title)
    ax[0].set_title('Bonds')
    ax[1].set_title('Angles')

    palette = itertools.cycle(zesty_palette)
    edges = itertools.cycle(zesty2_palette)
    
    color=next(palette)
    edge=next(edges)
    estimate_label = 'Estimate ' + '{0:.3f}'.format(estimate_score) if estimate_score is not None else 'Estimate'
    ax[0].axhline(5, color=color, label=estimate_label)
    ax[1].axhline(0.5, color=color, label=estimate_label)

    bonds = [make_labels(bond_set, substr_def) for bond_set in bonds]
    angles = [make_labels(angle_set, substr_def) for angle_set in angles]

    errors = pd.DataFrame()
    for bond_set in bonds:
        bond_set['Err'] = ((bond_set['Force Constant'] - finalized_bonds['Force Constant'])/finalized_bonds['Force Constant'])* 100.
        errors[bond_set['FF'][0]] = bond_set['Err']

    bonds[0].sort_values('Err', inplace=True, key=abs)
    key = pd.Series({k:v for v,k in enumerate(bonds[0].index)})
    for i in range(1, len(bonds)):
        bond_set = bonds[i]
        bonds[i] = bond_set.reindex(bonds[0].index)
    finalized_bonds = finalized_bonds.reindex(bonds[0].index)
    errors.sort_values(bonds[0]['FF'][0], inplace=True, key=abs)
    errors['label'] = bonds[0]['label'].values

    # if bond_labels is None:
    #     bond_labels
    #     #bond_labels = bonds[0][['atom1', 'atom2', 'param_type']].values
    #     #bond_labels = [str(bl) for bl in bond_labels]
    # if angles_labels is None:
    #     angles_labels = angles[0][['atom1', 'atom2', 'atom3', 'param_type']].values
    #     angles_labels = [str(al) for al in angles_labels]
    for i in range(len(bonds)):
        color = next(palette)
        edge = next(edges)
        print(bonds[i]['label'].values)
        seaborn.scatterplot(data=bonds[i], label=bonds[i]['FF'][0], x ='label', y="Force Constant", edgecolor=edge, ax=ax[0], color=color)
        seaborn.scatterplot(data=angles[i], label=angles[i]['FF'][0], x ='label', y="Force Constant", edgecolor=edge, ax=ax[1], color=color)

    row_labels = errors['label']
    print(errors.values)
    
    #ax[0].set_xticks(ax[0].get_xticks(), ax[0].get_xticklabels(), rotation=90)
    #ax[1].set_xticks(ax[1].get_xticks(), ax[1].get_xticklabels(), rotation=90)

    table = plt.table(cellText=errors.loc[:, errors.columns != 'label'].values.round(3), rowLabels=row_labels.values, colLabels=errors.columns, bbox=(-0.7, -1.4, 1, 1))
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    print(bonds[0])
    print(finalized_bonds)

    plt.xticks(rotation=90)
    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylabel(r'Force Constant ($mdyn/\AA$)')
    ax[1].set_ylabel(r'Force Constant ($mdyn/\AA$)')
    plt.show()
    return fig, ax

def plot_bond_params_v_static_table(bonds:list, finalized_bonds:pd.DataFrame, substr_def:list, title:str='', bond_labels=None, estimate_score=None):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('Force Constants'+title)
    ax[0].set_title('Bonds')

    palette = itertools.cycle(zesty_palette)
    edges = itertools.cycle(zesty2_palette)
    
    color=next(palette)
    edge=next(edges)
    estimate_label = 'Estimate ' + '{0:.3f}'.format(estimate_score) if estimate_score is not None else 'Estimate'
    ax[0].axhline(5, color=color, label=estimate_label)
    ax[1].set_axis_off()
    #ax[2].set_axis_off()

    bonds = [make_labels(bond_set, substr_def) for bond_set in bonds]

    errors = pd.DataFrame()
    for bond_set in bonds:
        bond_set['Err'] = ((bond_set['Force Constant'] - finalized_bonds['Force Constant'])/finalized_bonds['Force Constant'])
        errors[bond_set['FF'][0]] = bond_set['Err']

    errors['Estimate'] = (5. - finalized_bonds['Force Constant'])/finalized_bonds['Force Constant']

    bonds[0].sort_values('Err', inplace=True, key=abs)
    for i in range(1, len(bonds)):
        bond_set = bonds[i]
        bonds[i] = bond_set.reindex(bonds[0].index)
    finalized_bonds = finalized_bonds.reindex(bonds[0].index)
    errors.sort_values(bonds[0]['FF'][0], inplace=True, key=abs)
    errors['label'] = bonds[0]['label'].values

    # if bond_labels is None:
    #     bond_labels
    #     #bond_labels = bonds[0][['atom1', 'atom2', 'param_type']].values
    #     #bond_labels = [str(bl) for bl in bond_labels]
    # if angles_labels is None:
    #     angles_labels = angles[0][['atom1', 'atom2', 'atom3', 'param_type']].values
    #     angles_labels = [str(al) for al in angles_labels]
    plt.xticks(rotation=45)
    for i in range(len(bonds)):
        color = next(palette)
        edge = next(edges)
        seaborn.scatterplot(data=bonds[i], label=bonds[i]['FF'][0], x ='label', y="Force Constant", edgecolor=edge, ax=ax[0], color=color)


    errors = errors.replace([np.inf, -np.inf], np.nan).dropna()
    
    #ax[0].set_xticks(ax[0].get_xticks(), ax[0].get_xticklabels(), rotation=90)
    #ax[1].set_xticks(ax[1].get_xticks(), ax[1].get_xticklabels(), rotation=90)


    table_data = errors.drop(['label', finalized_bonds['FF'][0]], axis=1)
    table_data[finalized_bonds['FF'][0]] = bonds[-1]['Force Constant']
    table_data = table_data[[finalized_bonds['FF'][0]] + ['Estimate'] + [col for col in table_data.columns if col != finalized_bonds['FF'][0] and col != 'Estimate']]

    for i in range(len(bonds)-1):
        table_data[bonds[i]['FF'][0]] = table_data[bonds[i]['FF'][0]].map('{:.2%}'.format)
    table_data['Estimate'] = table_data['Estimate'].map('{:.2%}'.format)

    for bond_set in bonds:
        table_data.at['Score', bond_set['FF'][0]] = bond_set['Score'][0]
    table_data.at['Score', 'Estimate'] = 6.375
    value_labels = errors['label'].values
    value_labels = np.concatenate([value_labels, ['Score']])

    print(table_data)
    print(value_labels)
    table = plt.table(cellText=table_data.values.T, colLabels=value_labels, rowLabels=table_data.columns, bbox=(0, 0, 1, 1), cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    ax[0].legend()
    ax[0].set_ylabel(r'Force Constant ($mdyn/\AA$)')
    return fig, ax

def plot_ff_params_vert(bonds:list, angles:list, final_scores:list, title:str='', bond_labels=None, angles_labels=None):
    fig, ax = plt.subplots(2, 1, figsize=(16, 24))
    fig.suptitle('Force Constants'+title)
    ax[0].set_title('Bonds')
    ax[1].set_title('Angles')

    palette = itertools.cycle(zesty_palette)

    if bond_labels is None:
        bond_labels = bonds[0][['atom1', 'atom2', 'param_type']].values
        bond_labels = [str(bl) for bl in bond_labels]
    if angles_labels is None:
        angles_labels = angles[0][['atom1', 'atom2', 'atom3', 'param_type']].values
        angles_labels = [str(al) for al in angles_labels]
    for i in range(len(bonds)):
        color = next(palette)
        seaborn.regplot(data=bonds[i], label=final_scores[i], x = bond_labels, y="Force Constant", fit_reg=False, ax=ax[0], color=color)
        seaborn.regplot(data=angles[i], label=final_scores[i], x = angles_labels, y="Force Constant", fit_reg=False, ax=ax[1], color=color)

    plt.xticks(rotation=45)
    ax[1].legend()
    ax[0].set_ylabel(r'Force Constant ($mdyn/\AA$)')
    ax[0].set_ylim(top=6)
    ax[1].set_ylabel(r'Force Constant ($mdyn/\AA$)')
    plt.show()

def plot_last_x(history:dict, rows:list, title:str=''):
    fig, ax = plt.subplots(1, 1, figsize=(24, 8))
    fig.suptitle('Force Constants'+title)

    X_history = history["X"]
    last_X = X_history[-1]
    palette = itertools.cycle(zesty_palette)

    for i, particle in enumerate(last_X):
        color = next(palette)
        seaborn.regplot(x=rows, y=particle, fit_reg=False, label=i, ax=ax, color=color)

    ax.legend()
    ax.set_ylabel(r'Force Constant ($mdyn/\AA$)')
    plt.show()


