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
names = {'params': '../../final_003/q2_eq_0_mu_eq_04/params.yaml', \
         'result': '../../final_003/q2_eq_0_mu_eq_04/q1q2r_*.pkl', \
         'frames': '../../final_003/q2_eq_0_mu_eq_04/frames/q2_dump_{0:04d}'}
# /Users/vovka/Documents/math/2021/calc_box/2hydro/final_03/res_q2_eq_0
# /Users/vovka/Documents/math/2021/calc_box/2hydro/final_003/q2_eq_0_mu_eq_04
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
elif logr == 10 and logu == 0 and logv == 0:
    u_grid =       list(map(lambda k: L_u0 + du*k, range(n_u)))
    v_grid =       list(map(lambda k: L_v0 + dv*k, range(n_v)))
    r_cent =       list(map(lambda k: 2*a*tau * v_grid[k] / mu / (a - v_grid[k]), range(n_v)))
    r_grid =       list(map(lambda v: list(map(lambda k: r_cent[v] - L_r/2 + dr*k, range(n_r))), range(n_v)))

n_u = len(u_grid)
n_v = len(v_grid)
n_r = len(r_grid)

points = {}
color_names = ['black', 'blue', 'green', 'red', 'blue', 'yellow', 'gray']
SMALL_SIZE = 30
MEDIUM_SIZE = 30
BIGGER_SIZE = 30
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
limset = [[-1,-0.5], [-1,-0.5], [-1,-0.5], [-1,-0.5], [-0.75,-0.25], [-0.6,-0.2], [-0.4,0.0], \
 [-0.2,0.2], [0.0,0.5]]
for dump, xs in enumerate(v_grid):
    fig, ax = plt.subplots(figsize = (15,15), facecolor='w', edgecolor='k') # figsize=(30, 30), dpi=300,
    # ax.set_ylim([-2,1])
    ax.set_xlabel('$q_1$')
#    ax.set_ylabel('$\log_{10}(r)$')Vfgkt!20
    ax.set_ylabel('$r$')
    ax.grid(False)
    ax.set_title('### $q_2$ = ' + "{0:5.3f}".format(xs))
    # Конец оформления и начало собственно построения графиков
    for fname in glob.glob(names['result']):
        with open(fname, 'rb') as f:
            points.update(pickle.load(f))
        colors = sorted(points.keys())
        for c, color in enumerate(colors):
            arr = np.array(points[color]['init'])
            if arr.size > 0:
                ind = np.where(arr[:, 1] == xs)
                if ind[0].size > 0:
                    arr = arr[ind]
            	    # print(arr.shape)
                    # ax.set_ylim(limset[dump])
                    ax.scatter((arr[:, 0]), arr[:, 2], c=color_names[c], s=4, \
                               label=color_names[c], alpha=1.0, edgecolors='none')
    ax.scatter([1], [2*a*tau*xs/mu/(a-xs)],c='red',s=60, marker=r'^', label="cross")
    # ax.plot(np.ones(n_r), r_grid, c='white')
    t = np.linspace(0, 1, 500)
    y = 20*t
    x1 = 1 + 0.1*(1 - np.exp(-y))
    x2 = 1 - 0.1*(1 - np.exp(-y))
    ax.fill_betweenx(y, x1, x2, facecolor='black')
    filename = names['frames'].format(dump) + '.png'
    plt.savefig(filename, dpi=300)
#    plt.gca()
#    plt.show()
