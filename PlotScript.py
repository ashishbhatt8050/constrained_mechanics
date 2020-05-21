#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:30:30 2020

@author: ashishbhatt

Copied from https://github.com/jbmouret/matplotlib_for_papers
"""

import glob
from pylab import *
import brewer2mpl
from cycler import cycler

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
    'figure.figsize': [14, 6],
    'figure.autolayout': True
}
rcParams.update(params)

def plot_data(ax, x_data, y_data, use_y_labels, use_legend):
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

#    ax.fill_between(x, perc_25_low_mut, perc_75_low_mut, alpha=0.25, linewidth=0, color=colors[0])
#    ax.fill_between(x, perc_25_high_mut, perc_75_high_mut, alpha=0.25, linewidth=0, color=colors[1])


    ax.plot(x_data, y_data, linewidth=2)
#    ax.plot(x, med_high_mut, linewidth=2, linestyle='--', color=colors[1])

    # change xlim to set_xlim
#    ax.set_xlim(np.amin(x_data), np.amax(x_data))
#    ax.set_ylim(np.amin(y_data), np.amax(y_data))

    #change xticks to set_xticks
    #ax.set_xticks(np.arange(np.amin(x_data), np.amax(x_data), 100))

    if not use_y_labels:
        ax.set_yticklabels([])

    if use_legend:
        frame = ax.legend().get_frame()
        frame.set_facecolor('1.0')
        frame.set_edgecolor('1.0')

def test_PlotScript():
    data_low_mut = _load('/home/ashishbhatt/Documents/matplotlib_for_papers/src/data/low_mut')
    data_high_mut = _load('/home/ashishbhatt/Documents/matplotlib_for_papers/src/data/high_mut')

    n_generations = data_low_mut.shape[1]
    x = np.arange(0, n_generations)

    med_low_mut, perc_25_low_mut, perc_75_low_mut = _perc(data_low_mut)
    med_high_mut, perc_25_high_mut, perc_75_high_mut = _perc(data_high_mut)

    fig = figure()
    fig.subplots_adjust(left=0.09, right=0.99, top=0.99, wspace=0.1)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    plot_data(ax1, x, np.array([med_low_mut, med_high_mut]).T, True, True)
    plot_data(ax2, x, np.array([med_low_mut, med_high_mut]).T, False, False)

    # labeling
    fig.text(0.01, 0.98, "A", weight="bold", horizontalalignment='left', verticalalignment='center')
    fig.text(0.54, 0.98, "B", weight="bold", horizontalalignment='left', verticalalignment='center')

def _load(dir):
    f_list = glob.glob(dir + '/*/*/bestfit.dat')
    num_lines = sum(1 for line in open(f_list[0]))
    i = 0;
    data = np.zeros((len(f_list), num_lines))
    for f in f_list:
        data[i, :] = np.loadtxt(f)[:,1]
        i += 1
    return data

def _perc(data):
    median = np.zeros(data.shape[1])
    perc_25 = np.zeros(data.shape[1])
    perc_75 = np.zeros(data.shape[1])
    for i in range(0, len(median)):
        median[i] = np.median(data[:, i])
        perc_25[i] = np.percentile(data[:, i], 25)
        perc_75[i] = np.percentile(data[:, i], 75)
    return median, perc_25, perc_75

if __name__ == '__main__':
    _test()