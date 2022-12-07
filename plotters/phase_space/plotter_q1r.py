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

names = {'params': '../final_0008/params.yaml', \
         'result': '../final_0008/q1q2r_*.pkl', \
         'frames': '../final_0008/frames/q2_dump_{0:04d}'}
# /Users/vovka/Documents/math/2021/calc_box/2hydro/final_003/q2_eq_0_mu_eq_16_extra

# params = get_config('../calc_box/2hydro/12_11_2021/v_01/params.yaml')
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
elif logr == 0 and logu == 1 and logv == 1:
    u_grid = list(map(lambda k: 10 ** (L_u0 + du * k), range(n_u)))
    v_grid = list(map(lambda k: 10 ** (L_v0 + dv * k), range(n_v)))
    r_grid = list(map(lambda k: L_r0 + dr * k, range(n_r)))
elif logr == 1 and logu == 1 and logv == 0:
    u_grid = list(map(lambda k: 10 ** (L_u0 + du * k), range(n_u)))
    v_grid = list(map(lambda k: L_v0 + dv * k, range(n_v)))
    r_grid = list(map(lambda k: 10 ** (L_r0 + dr * k), range(n_r)))
elif logr == 1 and logu == 0 and logv == 0:
    u_grid = list(map(lambda k: L_u0 + du * k, range(n_u)))
    v_grid = list(map(lambda k: L_v0 + dv * k, range(n_v)))
    r_grid = list(map(lambda k: 10 ** (L_r0 + dr * k), range(n_r)))

n_u = len(u_grid)
n_v = len(v_grid)
n_r = len(r_grid)

points = {}
color_names = ['gray', 'deepskyblue', 'green', 'red', 'deepskyblue', 'green', 'red']
color_names = ['gray', 'deepskyblue', 'green', 'red', 'deepskyblue', 'whitesmoke', 'darkviolet']
alphas_vals = [0.3,      0.3,    0.3,     0.3,   0.3,    0.5,     1.0]
SMALL_SIZE = 18
MEDIUM_SIZE = 18
BIGGER_SIZE = 18
# font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 22}
font = {'family': 'Monospace'}
matplotlib.rc('font', **font)
plt.rc('font', size=SMALL_SIZE+2)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE+2)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE+2)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
for dump, xs in enumerate(v_grid):
    fig, ax = plt.subplots(figsize=(9, 9), dpi=300, facecolor='w', edgecolor='k')
    ax.set_xlabel('$\log_{10}(q_1)$', fontsize = 22)
    ax.set_ylabel('$\log_{10}(r)$', fontsize = 22)
    ax.grid(False)
    # ax.set_title('### $q_2$ = ' + str(xs))
    print('q2 = ', str(xs))
    # Конец оформления и начало собственно построения графиков
#    for fname in glob.glob('./res/q1q2r_*.pkl'):
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
                    clr = color_names[c]
                    alp = alphas_vals[c]
                    ax.scatter(np.log10(arr[:, 0]), np.log10(arr[:, 2]), c=clr, s=5, \
                               label=color, alpha=alp, edgecolors='none', marker='o')
    if params["cross"] == 'yes':
        ax.scatter([0], np.log10([2*a*tau*xs/mu/(a-xs)]),c='black',s=45, marker='o', label="cross")
    x1,x2,y1,y2 = plt.axis()
    # ax.plot([0, 0], [-1000, 1000], c = 'black', linewidth = 1)
    # ax.plot([-1000, 1000], [0, 0], c = 'black', linewidth = 1)
    plt.axis((x1,x2,y1,y2))
    filename = names['frames'].format(dump) + '.png'
    plt.savefig(filename, dpi=300)
    plt.gca()
