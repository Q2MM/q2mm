#!/usr/bin/env python3
import os
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

#src_dir = os.path.abspath("/home/mmfarrugia/repos/q2mm/q2mm")
sys.path.append("/home/mfarrugi/repos/q2mm/rh-hybrid/schrodinger.ve/lib/python3.8/site-packages/q2mm-0.0.0-py3.8.egg")

#from hybrid_optimizer import PSO_GA
import q2mm.hybrid_optimizer as hybrid_optimizer
from q2mm.hybrid_optimizer import PSO_DE
from tools.plotters import plot_cost_history, plot_contour, plot_surface, plot_summary, Mesher, Designer

bond_cols = ["param_type", "atom1", "atom2", "Equilibrium Value", "Force Constant", "Dipole Moment", "FF"]
angle_cols = ["param_type", "atom1", "atom2", "atom3", "Equilibrium Value", "Force Constant", "FF"]

def plot_history(base_direc:str, directories:list, title:str, cycle_iter_length:int, starting_score:float) -> list :
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
        loss.plot(kind='line', ax=ax[1], color=color, label='Final Score: '+str(loss.iloc[-1]))
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
        loss.plot(kind='line', ax=ax[1], color=color, label='Final Score: '+str(loss.iloc[-1]))
        #final_scores.append(loss.iloc[-1])

    ax[1].legend()

    ax[0].xaxis.set_ticks(np.arange(1, ax[0].get_xlim()[1], cycle_iter_length))
    ax[1].xaxis.set_ticks(np.arange(1, ax[0].get_xlim()[1], cycle_iter_length))

    plt.show()
    return final_scores

def plot_scores(starting_scores:pd.DataFrame, starting_score:float, scored_runs:list, final_scores:list, title:str):
    fig, ax = plt.subplots(1, len(scored_runs)+1, figsize=(8*(len(scored_runs)+1), 8))
    fig.suptitle(title)
    palette = itertools.cycle(seaborn.color_palette())
    
    seaborn.regplot(data=starting_scores, x='Reference', y='Calculated', label='FF', fit_reg=True, ax=ax[0], color='gray')
    ax[0].set_ylim(top=5000)
    ax[0].set_xlim(xmax = 5000)
    ax[0].set_title('INITIAL - '+str(starting_score))
    
    for i, run in enumerate(scored_runs):
        color = next(palette)
        seaborn.regplot(data=run, x='Reference', y='Calculated', label='FF', fit_reg=True, ax=ax[i+1], color=color)
    
        ax[i+1].set_ylim(top=5000)
        ax[i+1].set_xlim(xmax = 5000)
        ax[i+1].set_title('Score: '+str(final_scores[i]))
    
    plt.show()

def plot_off_diag_scatter(starting_scores:pd.DataFrame, starting_score:float, scored_runs:list, final_scores:list, title:str=''):
    fig, ax = plt.subplots(1, len(scored_runs)+1, figsize=(8*(len(scored_runs)+1), 10))
    fig.suptitle('Off-Diagonal Eigenmatrix terms'+title)
    palette = itertools.cycle(seaborn.color_palette())

    off_diag_start = starting_scores.loc[starting_scores['Reference'] == 0.0000]
    off_diag = off_diag.loc[off_diag['Weight'] != 0.0000]
    off_diag_start = off_diag_start.sort_values(by='Calculated', ignore_index=True)
    seaborn.regplot(data=off_diag_start, x=off_diag_start.index, label='FF', y='Calculated', fit_reg=False, ax=ax[0], color='gray')
    ax[0].set_title('FUERZA - '+str(starting_score))
    max_y = max(off_diag_start['Calculated'])

    for i, run in enumerate(scored_runs):
        off_diag = run.loc[run['Reference'] == 0.0000]
        off_diag = off_diag.loc[off_diag['Weight'] != 0.0000]
        off_diag = off_diag.sort_values(by='Calculated', ignore_index=True)

        color = next(palette)
        seaborn.regplot(data=off_diag, x=off_diag.index, label='FF', y='Calculated', fit_reg=False, ax=ax[i+1], color=color)
        ax[i+1].set_title('Score: '+str(final_scores[i]))
        ax[i+1].set_ylim(top=max_y, bottom=-max_y)

    plt.show()

def plot_off_diag_violin(starting_scores:pd.DataFrame, starting_score:float, scored_runs:list, final_scores:list, title:str=''):

    fig, ax = plt.subplots(1, 1, figsize=(8*(len(scored_runs)+1), 8))
    off_diag_start = starting_scores.loc[starting_scores['Reference'] == 0.0000]
    off_diag_start = off_diag_start.loc[off_diag_start['Weight'] != 0.0000]
    off_diag_start = off_diag_start.sort_values(by='Calculated', ignore_index=True)

    off_diag_merged = pd.concat([run.loc[run['Reference'] == 0.0000] for run in scored_runs])
    off_diag_merged = off_diag_merged.loc[off_diag_merged['Weight'] != 0.0000]
    off_diag_merged = pd.concat([off_diag_start, off_diag_merged])
    off_diag_merged['FF'] = off_diag_merged['FF'].astype(str)

    seaborn.violinplot(data=off_diag_merged, x='FF', y='Calculated')#, title='Off-Diagonal Eigenmatrix terms'+title)
    ax.set_label('FF Score')

    plt.show()

def plot_fit_diag_scores(starting_scores:pd.DataFrame, starting_score:float, scored_runs:list, final_scores, title:str=''):
    # Plot Diagonal Elements with a linear fit

    fig, ax = plt.subplots(1, len(scored_runs)+1, figsize=(8*(len(scored_runs)+1),8))
    fig.suptitle('Diagonal Eigenmatrix terms after a PSO only - Rh Hyd Enamides')
    palette = itertools.cycle(seaborn.color_palette())

    diag = starting_scores.loc[starting_scores['Reference'] != 0.0000]
    diag = diag.loc[diag['Weight'] != 0.0000]
    slope, intercept, r2, pv, se = stats.linregress(diag['Reference'], diag['Calculated'])

    seaborn.regplot(data=diag, y='Calculated', x='Reference', color=next(palette), label='FF', line_kws={'label':'$y=%3.7s*x+%3.7s   r2:%3.7s$'%(slope, intercept, r2)}, ax=ax[0])
    ax[0].legend()
    ax[0].set_title('FUERZA - '+str(starting_score))

    for i, run in enumerate(scored_runs):
        diag = run.loc[run['Reference'] != 0.0000]
        slope, intercept, r2, pv, se = stats.linregress(diag['Reference'], diag['Calculated'])

        seaborn.regplot(data=diag, y='Calculated', x='Reference', color=next(palette), label=final_scores[i], line_kws={'label':'$y=%3.7s*x+%3.7s   r2:%3.7s$'%(slope, intercept, r2)}, ax=ax[i+1])

        ax[i+1].legend()
        ax[i+1].set_title('Score: '+str(final_scores[i]))

    for a in ax:
        seaborn.lineplot(data=diag, x='Reference', y='Reference', color='gray', ax=a)

def linear_fit_diag_scores(starting_scores:pd.DataFrame, starting_score:float, scored_runs:list, final_scores, title:str=''):
    # Plot Diagonal Elements with a PROPER linear fit

    fig, ax = plt.subplots(1, len(scored_runs)+1, figsize=(8*(len(scored_runs)+1),8))
    fig.suptitle('Diagonal Eigenmatrix terms after a PSO only - Rh Hyd Enamides')
    palette = itertools.cycle(seaborn.color_palette())

    diag = starting_scores.loc[starting_scores['Reference'] != 0.0000]
    diag = diag.loc[diag['Weight'] != 0.0000]
    slope, intercept, r2, pv, se = stats.linregress(diag['Reference'], diag['Calculated'])

    seaborn.regplot(data=diag, y='Calculated', x='Reference', color=next(palette), label='FF', line_kws={'label':'$y=%3.7s*x+%3.7s   r2:%3.7s$'%(slope, intercept, r2)}, ax=ax[0])
    ax[0].legend()
    r2_ = r2_score(diag['Reference'], diag['Calculated'])
    ax[0].set_title('FUERZA - '+str(starting_score)+' y=x r2:'+str(r2_))


    for i, run in enumerate(scored_runs):
        diag = run.loc[run['Reference'] != 0.0000]
        diag = diag.loc[diag['Weight'] != 0.0000]
        slope, intercept, r2, pv, se = stats.linregress(diag['Reference'], diag['Calculated'])

        seaborn.regplot(data=diag, y='Calculated', x='Reference', color=next(palette), label=final_scores[i], line_kws={'label':'$y=%3.7s*x+%3.7s   r2:%3.7s$'%(slope, intercept, r2)}, ax=ax[i+1])
        r2_ = r2_score(diag['Reference'], diag['Calculated'])
        ax[i+1].legend()
        ax[i+1].set_title('Score: '+str(final_scores[i])+' y=x r2:'+str(r2_))

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
    palette = itertools.cycle(seaborn.color_palette())

    for i, start in enumerate(starting_scores):
        diag = start.loc[start['Reference'] != 0.0000]
        diag = diag.loc[diag['Weight'] != 0.0000]
        slope, intercept, r2, pv, se = stats.linregress(diag['Reference'], diag['Calculated'])
        labl = str(starting_score[i]) #+ diag['FF'][0]
        seaborn.regplot(data=diag, y='Calculated', x='Reference', color=next(palette), label=labl, line_kws={'label':'$y=%3.7s*x+%3.7s   r2:%3.7s$'%(slope, intercept, r2)}, ax=ax[i][0])
        ax[i][0].legend()
        seaborn.move_legend(ax[i][0], "upper left", bbox_to_anchor=(1, 1))
        r2_ = r2_score(diag['Reference'], diag['Calculated'])
        ax[i][0].set_title(str(start['FF'][0])+' - '+str(starting_score[i])+' y=x r2:'+str(r2_))
        seaborn.lineplot(data=diag, x='Reference', y='Reference', color='gray', ax=ax[i][0])


    for i, run in enumerate(final_scores):
        diag = run.loc[run['Reference'] != 0.0000]
        diag = diag.loc[diag['Weight'] != 0.0000]
        slope, intercept, r2, pv, se = stats.linregress(diag['Reference'], diag['Calculated'])

        seaborn.regplot(data=diag, y='Calculated', x='Reference', color=next(palette), label=final_score[i], line_kws={'label':'$y=%3.7s*x+%3.7s   r2:%3.7s$'%(slope, intercept, r2)}, ax=ax[int(i%2)][int(i/2)+1])
        r2_ = r2_score(diag['Reference'], diag['Calculated'])
        ax[int(i%2)][int(i/2)+1].legend()
        seaborn.move_legend(ax[int(i%2)][int(i/2)+1], "upper left", bbox_to_anchor=(1, 1))
        ax[int(i%2)][int(i/2)+1].set_title('Score: '+str(final_score[i])+' y=x r2:'+str(r2_))
        seaborn.lineplot(data=diag, x='Reference', y='Reference', color='gray', ax=ax[int(i%2)][int(i/2)+1])


def get_ff_params(base_direc:str, directories:list, filename:str, final_scores:list, bond_rows:list, angle_rows:list, title:str=''):# -> tuple[list, list]:
    # Plot FCs

    bonds = []
    angles = []

    for directory, score in zip(directories, final_scores):
        bonds.append(pd.read_csv(os.path.join(base_direc, directory, filename), skiprows=lambda x: x not in bond_rows, delim_whitespace=True, names=bond_cols).assign(FF=title+str(score)))
        angles.append(pd.read_csv(os.path.join(base_direc, directory, filename), skiprows=lambda x: x not in angle_rows, delim_whitespace=True, names=angle_cols).assign(FF=title+str(score)))

    return bonds, angles

def plot_ff_params(bonds:list, angles:list, final_scores:list, title:str='', bond_labels=None, angles_labels=None):
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))
    fig.suptitle('Force Constants'+title)
    ax[0].set_title('Bonds')
    ax[1].set_title('Angles')

    palette = itertools.cycle(seaborn.color_palette())

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
    ax[0].set_ylabel('Force Constant (mdyne/Ang)')
    ax[1].set_ylabel('Force Constant (mdyne/Ang)')
    ax[0].set_xlabel('atom1, atom2, param_type')
    ax[1].set_xlabel('atom1, atom2, atom3, param_type')
    plt.show()

def plot_ff_params_vert(bonds:list, angles:list, final_scores:list, title:str='', bond_labels=None, angles_labels=None):
    fig, ax = plt.subplots(2, 1, figsize=(16, 24))
    fig.suptitle('Force Constants'+title)
    ax[0].set_title('Bonds')
    ax[1].set_title('Angles')

    palette = itertools.cycle(seaborn.color_palette())

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
    ax[0].set_ylabel('Force Constant (mdyne/Ang)')
    ax[0].set_ylim(top=6)
    ax[1].set_ylabel('Force Constant (mdyne/Ang)')
    ax[0].set_xlabel('atom1, atom2, param_type')
    ax[1].set_xlabel('atom1, atom2, atom3, param_type')
    plt.show()

def plot_last_x(history:dict, rows:list, title:str=''):
    fig, ax = plt.subplots(1, 1, figsize=(24, 8))
    fig.suptitle('Force Constants'+title)

    X_history = history["X"]
    last_X = X_history[-1]
    palette = itertools.cycle(seaborn.color_palette())

    for i, particle in enumerate(last_X):
        color = next(palette)
        seaborn.regplot(x=rows, y=particle, fit_reg=False, label=i, ax=ax, color=color)

    ax.legend()
    ax.set_ylabel('Force Constant (mdyne/Ang)')
    plt.show()


