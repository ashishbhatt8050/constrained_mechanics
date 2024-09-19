#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:30:30 2020

@author: ashishbhatt

Copied from https://github.com/jbmouret/matplotlib_for_papers
"""

from pylab import figure, rcParams, cm
import brewer2mpl
from cycler import cycler
import os
import pickle as pickle

 # brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors

params = {
    'axes.labelsize': 20,
    'axes.prop_cycle': (cycler(color=colors[:3]) +cycler(linestyle=['-','--','-.'])),
    'font.size': 20,
    'legend.fontsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'text.usetex': True,
    # 'text.latex.preamble': r'\boldmath',
    'figure.figsize': [14, 6],
    'figure.autolayout': True
}
rcParams.update(params)

from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        _ = f(*args, **kw)
        te = time()
        # print('func:%r args:[%r, %r] took: %2.4f sec' % \
        #   (f.__name__, args, kw, te-ts))
        return te-ts
    return wrap


def plot_data(ax, x_data, y_data, xlims=None, ylabel=None, margins=None):
    # now all plot function should be applied to ax
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', length=0)

    # offset the spines
    for spine in ax.spines.values():
            spine.set_position(('outward', 5))
    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    # put the grid behind
    ax.set_axisbelow(True)


    ax.plot(x_data, y_data, linewidth=2)

    if xlims is not None: ax.set_xlim(xlims)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if margins is not None: ax.margins(y=margins)

def logplot(y_data, xlabel=None, xlims=None):
    
    fig = figure()
    ax = fig.add_subplot(111)
    
    # now all plot function should be applied to ax
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', direction='out')

    # offset the spines
    for spine in ax.spines.values():
            spine.set_position(('outward', 5))
    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    # put the grid behind
    ax.set_axisbelow(True)
    
    ax.semilogy(y_data, linewidth=2)
    
    # ax.legend((r'$\mathbb{S}$ (PCIMP)',r'$\mathbb{S}$ (PCSV)'), loc='upper right')
    if xlabel is not None: ax.set_xlabel(xlabel)
    ax.margins(y=0.1)
    if xlims is not None: ax.set_xlim(xlims)
        
    return fig, ax

def save_figure(fig, filename):
    
    # Get the current working directory
    cwd = os.getcwd()
    
    # # Construct the data folder path (adjust as needed)
    data_folder = os.path.join(cwd, "data")
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        
    fig_path = os.path.join(data_folder, filename[:-4])
    fig.savefig(fig_path)
    
    pickle.dump(fig, open(fig_path+'.fig.pickle', 'wb'))
    
    
    # edit the figure later
    # import pickle
    
    # with open('data/2024-04-08_10-10_osc_ConformalStormerVerlet_reduced.fig.pickle', 'rb') as file: figx = pickle.load(file)
    
    # figx.show() # Show the figure, edit it, etc.!
    # ax = figx.axes
    # ax[0].set_xlabel('time')
    # ax[0].set_ylim([-2e-16, 2e-16])
    # ax[2].set_xlim([1,6])


def tex_table(solver_name, array2print):
    print(solver_name, "\n", " \\\\\n".join([" & ".join(map('{0:.3f}'.format, line)) for line in array2print]))
    # print(solver_name, "\n \\num{", " \\\\\n \\num{".join(["} & \\num{".join(map('{0:.6f}'.format, line)) for line in array2print]))
    

    
def plot_3dsurface(fig, ax, xx, yy, zz):
    surf = ax.plot_surface(xx, yy, zz,\
                           cmap = cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    _ = ax.contour(xx, yy, zz, zdir='z', offset=-1, cmap=cm.coolwarm)
    ax.view_init(30, -135)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_zlim3d(-1, abs(zz).max())

#%% Testing
def test_PlotScript():
    raise NotImplementedError

if __name__ == '__main__':
    test_PlotScript()
