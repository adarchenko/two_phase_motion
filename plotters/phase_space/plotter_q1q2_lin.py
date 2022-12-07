#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import yaml
import pickle
import glob

def get_config(fname):
    conf = {}
    if os.path.exists(fname):
        with open(fname) as f:
            conf = yaml.load(f, Loader = yaml.FullLoader)
    else:
        print('No yaml config file!')
    return conf
names = {'params': '../../final_01/res_rr_eq_200_mu_eq_04/params.yaml', \
         'result': '../../final_01/res_rr_eq_200_mu_eq_04/q1q2r_*.pkl', \
         'frames': '../../final_01/res_rr_eq_200_mu_eq_04/frames/q2_dump_{0:04d}'}

params = get_config(names['params'])

a = float(params["const"]["a"])
mu = float(params["const"]["mu"])
tau = float(params["const"]["tau"])

n_r = int(params["n_r"])
n_u = int(params["n_u"])
n_v = int(params["n_v"])
L_u0 = float(params["L_u0"])
L_u1 = float(params["L_u1"])
L_v0 = float(params["L_v0"])
L_v1 = float(params["L_v1"])
L_r0 = float(params["L_r0"])
L_r1 = float(params["L_r1"])
logr = params["logr"]
logu = params["logu"]
logv = params["logv"]
logall = params["logall"]

L_u = L_u1 - L_u0
L_v = L_v1 - L_v0
L_r = L_r1 - L_r0
du = L_u / (n_u - 1)
dv = L_v / (n_v - 1)
dr = L_r / (n_r - 1)
if logr == 1 and logu == 1 and logv == 1:
    u_grid = list(map(lambda k: 10 ** (L_u0 + du * k), range(n_u)))
    v_grid = list(map(lambda k: 10 ** (L_v0 + dv * k), range(n_v)))
    r_grid = list(map(lambda k: 10 ** (L_r0 + dr * k), range(n_r)))
elif logr == 1 and logu == 1 and logv == 0:
    u_grid = list(map(lambda k: 10 ** (L_u0 + du * k), range(n_u)))
    v_grid = list(map(lambda k: L_v0 + dv * k, range(n_v)))
    r_grid = list(map(lambda k: 10 ** (L_r0 + dr * k), range(n_r)))
elif logr == 1 and logu == 0 and logv == 0:
    u_grid = list(map(lambda k: L_u0 + du * k, range(n_u)))
    v_grid = list(map(lambda k: L_v0 + dv * k, range(n_v)))
    r_grid = list(map(lambda k: 10 ** (L_r0 + dr * k), range(n_r)))
elif logr == 0 and logu == 0 and logv == 0:
    u_grid =       list(map(lambda k: L_u0 + du*k, range(n_u)))
    v_grid =       list(map(lambda k: L_v0 + dv*k, range(n_v)))
    r_grid =       list(map(lambda k: L_r0 + dr*k, range(n_r)))
n_u = len(u_grid)
n_v = len(v_grid)
n_r = len(r_grid)

points = {}
# color_names = ['black', 'blue', 'green', 'red', 'magenta', 'yellow', 'gray']
color_names = ['black', 'blue', 'green', 'red', 'blue', 'yellow', 'gray']
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 12
# font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 22}
font = {'family': 'Monospace'}
matplotlib.rc('font', **font)
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
for dump, xs in enumerate(r_grid):
    fig, ax = plt.subplots(figsize=(9, 9), dpi=300, facecolor='w', edgecolor='k')
    # ax.set_ylim([-2,1])
    ax.set_xlabel('$q_1$')
#    ax.set_ylabel('$\log_{10}(r)$')
    ax.set_ylabel('$q_2$')
    ax.grid(False)
    ax.set_title('### R = ' + "{0:5.3f}".format(xs))
    # Конец оформления и начало собственно построения графиков
    for fname in glob.glob(names['result']):
        with open(fname, 'rb') as f:
            points.update(pickle.load(f))
        colors = sorted(points.keys())
        for c, color in enumerate(colors):
            arr = np.array(points[color]['init'])
            if arr.size > 0:
                ind = np.where(arr[:, 2] == xs)
                if ind[0].size > 0:
                    arr = arr[ind]
            	    # print(arr.shape)
                    ax.scatter(arr[:, 0], arr[:, 1], c=color_names[c], s=1, \
                               label=color_names[c], alpha=1.0, edgecolors='none')
    ax.scatter([1], [xs*mu/(2 + xs*mu)],c='red',s=20, marker=r'^', label="cross")
    x1,x2,y1,y2 = plt.axis()
    ax.plot([1, 1], [-1000, 1000], c = 'black', linewidth = 1)
    ax.plot([-1000, 1000], [1, 1], c = 'black', linewidth = 1)
    plt.axis((x1,x2,y1,y2))
#    filename = './frames_q1r/q2_dump_{0:04d}'.format(dump) + '.png'
    filename = names['frames'].format(dump) + '.png'
    plt.savefig(filename, dpi=300)
    plt.gca()
