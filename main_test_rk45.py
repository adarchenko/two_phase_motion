#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import glob, os
import yaml
from scipy.integrate import solve_ivp
import pickle

def get_config(fname):
    conf = {}
    if os.path.exists(fname):
        with open(fname) as f: 
            conf = yaml.load(f)
    else:
        print('No yaml config file!')
    return conf

def get_pos(NX,NY,NZ,ind):
    i = int(( int(ind/NZ) )/NY)
    j = - i*NY + int(ind/NZ)
    k = ind - NZ*int(ind/NZ)
    position = [int(i), int(j), int(k)]
    return position

def get_share(size, rank, NX, NY, NZ):
    N = NX*NY*NZ
    if N <= size:
        if rank+1 <= N:
            position = get_pos(NX, NY, NZ,rank+1)
            print('Start to calculate at point {:d},{:d},{:d} for rank = {:d}\n'.\
                  format(position[0], position[1], position[2], rank))
            share = [rank+1, rank+1]
            return share
        else:
            print("Pass work for rank = {:d}\n".format(rank))
    else:
        ppp = math.ceil(N/size) # ppp - points_per_proc
        if rank+1 < size:
            share = [1+rank*ppp, (rank+1)*ppp]
        else:
            share = [1+rank*ppp, N]
    share = [int(share[0]-1), int(share[1]-1)]
#    print("rank is r = {:d}; share from [{:d},{:d},{:d}] to [{:d},{:d},{:d}]; total ppp is = {:d}\n".\
#           format(rank, [get_pos(NX, NY, NZ,share[0])[0], get_pos(NX, NY, NZ,share[0])[1], get_pos(NX, NY, NZ,share[0])[2]], \
#                        [get_pos(NX, NY, NZ,share[1])[0], get_pos(NX, NY, NZ,share[1])[1], get_pos(NX, NY, NZ,share[1])[2]], ppp))
    return share

def hydro(t, f, a, mu, tau):
    q1, q2, r = f
    return [q1*(2*a**2*tau*q2 - mu*(q1-q2)*q1*r),
            (q1-q2)*r*(q1**2 - a**2),
            tau*q2*r*(q1**2 - a**2)]

def is_white(point, epsilon): 
    q1, q2, r = point
    res = {'line': {}, 'int': {}, 'dist': {}}
    
    analytical = np.array([1, q2, 2*a*tau*q2/mu/(a - q2)])
    numerical = np.array([q1, q2, r])
    norm = np.sqrt(np.sum(list(map(lambda x: x**2, analytical))))
    distance = np.sqrt((a - q1)**2 + (r - 2*a*tau*q2/mu/(a - q2))**2)/norm
    # print('### analytical', analytical)
    # print('### numerical', numerical)
    # print('### distance', distance)
    
    ### правда ли, что точка вблизи линии ?
    if distance < epsilon:
        res['line'] = True
        res['dist'] = distance
    else:
        res['line'] = False
    ### правда ли, что точка попала в интервал по q2 ?   
    if q2 >= white_bounds[0] and q2 <= white_bounds[1]:
        res['int'] = True
    else:
        res['int'] = False
    # print('### ans', res)
    return res

def hydroj(t, f, a, mu, tau):
    q1, q2, r = f
    return [[2*a**2*tau*q2 - 3*mu*q1**2*r + 2*mu*q1*q2*r, 2*a**2*tau*q1 + mu*q1**2*r, -mu*q1**3 + mu*q1**2*q2], \
            [3*r*q1**2 - a**2*r - 2*r*q1*q2, -r*q1**2 + a**2*r, q1**3 - a**2*q1 - q1**2*q2 + a**2*q2], \
            [2*tau*q1*q2*r, tau*q1**2*r - tau*a**2*r, tau*q1**2*q2 - tau*a**2*q2]]

def printer(ans):
    print('### Метод РК45 \n### Движение вперед')
    print('Метод Рунге при движении вперед остановился в состоянии', ans['RK_45']['plus'].status, \
          '###', ans['RK_45']['plus'].message)
    if ans['RK_45']['plus'].status != 0:
        print('События, приведшие к остановке метода Рунге', ans['RK_45']['plus'].t_events )
    print('Количество итераций решателя Рунге до остановки', ans['RK_45']['plus'].nfev )
    print('### Движение назад')
    print('Метод Рунге при движении назад остановился в состоянии', ans['RK_45']['minus'].status, \
          '###', ans['RK_45']['minus'].message)
    if ans['RK_45']['minus'].status != 0:
        print('События, приведшие к остановке метода Рунге', ans['RK_45']['minus'].t_events )
    print('Количество итераций решателя Рунге до остановки', ans['RK_45']['minus'].nfev )

    print('### Метод LSODA \n###Движение вперед')
    print('Метод LSODA при движении вперед остановился в состоянии', ans['LSODA']['plus'].status, \
          '###', ans['LSODA']['plus'].message)
    if ans['LSODA']['plus'].status != 0:
        print('События, приведшие к остановке метода LSODA', ans['LSODA']['plus'].t_events )
    print('Количество итераций решателя LSODA до остановки', ans['LSODA']['plus'].nfev )
    print('Количество вычислений якобиана', ans['LSODA']['plus'].njev )
    print('###Движение назад')
    print('Метод LSODA при движении назад остановился в состоянии', ans['LSODA']['minus'].status, \
          '###', ans['LSODA']['minus'].message)
    if ans['LSODA']['minus'].status != 0:
        print('События, приведшие к остановке метода LSODA', ans['LSODA']['minus'].t_events )
    print('Количество итераций решателя LSODA до остановки', ans['LSODA']['minus'].nfev )

def plotter(ans, tp_end, tm_end, n_time_points):
    
    t_p = np.linspace(0, tp_end, n_time_points)
    t_m = np.linspace(0, tm_end, n_time_points)
    
    
    tp1 = t_p
    yp1 = ans['RK_45']['plus'].sol(t_p)
    tm1 = t_m
    ym1 = ans['RK_45']['minus'].sol(t_m)
    
    tp2 = t_p
    yp2 = ans['LSODA']['plus'].sol(t_p)
    tm2 = t_m
    ym2 = ans['LSODA']['minus'].sol(t_m)
    
    fig, ax = plt.subplots(4,1, figsize=(7,21), dpi= 100, facecolor='w', edgecolor='k')
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 14
    # font = {'family' : 'normal',
    #        'weight' : 'bold',
    #        'size'   : 22}
    font = {'family': 'Monospace'}
    matplotlib.rc('font', **font)
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # Конец оформления и начало собственно построения графиков
    ###                                   ***                                  ###
    ax[0].plot(tp1, yp1[0], color = 'red', linewidth = 2)
    ax[0].plot(tp2, yp2[0], color = 'red', linestyle='dashed', linewidth = 2)
    ax[0].plot(tm1, ym1[0], color = 'blue', linewidth = 2)
    ax[0].plot(tm2, ym2[0], color = 'blue', linestyle='dashed', linewidth = 2)
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('$q_1$')
    ax[0].grid(True)
    ax[0].legend(['$RK45\ to +\infty$', '$LSODA\ to +\infty$','$RK45\ to -\infty$', '$LSODA\ to -\infty$'], shadow=True)
    ###                                   ***                                  ###
    ax[1].plot(tp1, yp1[1], color = 'red', linewidth = 2)
    ax[1].plot(tp2, yp2[1], color = 'red', linestyle='dashed', linewidth = 2)
    ax[1].plot(tm1, ym1[1], color = 'blue', linewidth = 2)
    ax[1].plot(tm2, ym2[1], color = 'blue', linestyle='dashed', linewidth = 2)
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('$q_2$')
    ax[1].grid(True)
    ax[1].legend(['$RK45\ to +\infty$', '$LSODA\ to +\infty$','$RK45\ to -\infty$', '$LSODA\ to -\infty$'], shadow=True)
    ###                                   ***                                  ###
    ax[2].plot(tp1, yp1[2], color = 'red', linewidth = 2)
    ax[2].plot(tp2, yp2[2], color = 'red', linestyle='dashed', linewidth = 2)
    ax[2].plot(tm1, ym1[2], color = 'blue', linewidth = 2)
    ax[2].plot(tm2, ym2[2], color = 'blue', linestyle='dashed', linewidth = 2)
    ax[2].set_xlabel('time')
    ax[2].set_ylabel('$R$')
    ax[2].grid(True)
    ax[2].legend(['$RK45\ to +\infty$', '$LSODA\ to +\infty$','$RK45\ to -\infty$', '$LSODA\ to -\infty$'], shadow=True)
    ###                                   ***                                  ###
    ax[3].plot(yp1[0], yp1[1], color = 'red', linewidth = 2)
    ax[3].plot(yp2[0], yp2[1], color = 'red', linestyle='dashed', linewidth = 2)
    ax[3].plot(ym1[0], ym1[1], color = 'blue', linewidth = 2)
    ax[3].plot(ym2[0], ym2[1], color = 'blue', linestyle='dashed', linewidth = 2)
    ax[3].set_xlabel('$q_1$')
    ax[3].set_ylabel('$q_2$')
    ax[3].grid(True)
    ax[3].legend(['$RK45\ to +\infty$', '$LSODA\ to +\infty$','$RK45\ to -\infty$', '$LSODA\ to -\infty$'], shadow=True)
    plt.show()

### События останова: и переход через ноль по q2
### 1. уход на бесконечность по радиусу
def r_inf(t, f, a, mu, tau):
    # f[2] - это радиус
    return f[2] - r_infty
### 2. переход через ноль по q2
def q2_0(t, f, a, mu, tau):
    # f[1] - это КУ-2
    return f[1]
### трансзвуковые переходы для двух компонент
def q1_pa(t, f, a, mu, tau):
    # f[0] - это КУ-1
    return f[0] - a
def q1_ma(t, f, a, mu, tau):
    # f[0] - это КУ-1
    return f[0] - a
def q2_pa(t, f, a, mu, tau):
    # f[0] - это КУ-1
    return f[1] - a
def q2_ma(t, f, a, mu, tau):
    # f[0] - это КУ-1
    return f[1] - a

r_inf.terminal = True
r_inf.direction = +1
q2_0.terminal = True
q2_0.direction = -1
q1_pa.terminal = False
q1_pa.direction = +1
q1_ma.terminal = False
q1_ma.direction = -1
q2_pa.terminal = False
q2_pa.direction = +1
q2_ma.terminal = False
q2_ma.direction = -1

def postproc(ans, direct, result):
    result[direct] = {}
    events = ['q1_ma', 'q1_pa', 'q2_ma', 'q2_pa', 'q2_0', 'r_inf']
    mask = list(map( lambda x: x.size > 0, ans['RK_45'][direct].t_events))
    ev_tlist = np.array(ans['RK_45'][direct].t_events)[mask].tolist()
    ev_ylist = np.array(ans['RK_45'][direct].y_events)[mask].tolist()
    ev_nlist = np.array(events)[mask].tolist()
    for k, key in enumerate(ev_nlist):
        result[direct][key] = {}
        result[direct][key]['time'] = ev_tlist[k]
        result[direct][key]['vars'] = ev_ylist[k]
    if ans['RK_45'][direct].status == 1:
        result[direct]['last'] = {}
        result[direct]['last']['time'] = ev_tlist[-1][0]
        result[direct]['last']['vars'] = ev_ylist[-1][0]
    elif ans['RK_45'][direct].status == 0:
        result[direct]['last'] = {}
        result[direct]['last']['time'] = ans['RK_45'][direct].t[-1]
        var = np.array(list(map( lambda x: ans['RK_45'][direct].y[x][-1], [0,1,2])))
        result[direct]['last']['vars'] = var
        # np.array(map(lambda x: ans['RK_45'][direct].y[x][-1],[0,1,2]))
    else:
        print('### We have a problem!!! The priblem is: ', ans['RK_45'][direct].message)
    return result

def solver(system, jac, times, minstep, init_point, sys_pars, events):
    ### system - система уравнений
    ### jac - якобиан системы уравнений
    ### times - время на плюс-, минус-бесконечности, число точек на графике, признак расчета промежуточных точек
    tp_end = times['t_plus']
    tm_end = times['t_minus']
    atol = 1E-9
    rtol = 1E-6
    # atol = 1E-8
    # rtol = 1E-5
    ans = {}
    ans['RK_45'] = {}
    #ans['LSODA'] = {}
    
    ### движение вперед
    sol = solve_ivp(system, [0, tp_end], init_point, method='RK45', atol = atol, rtol = rtol, args=sys_pars,                   dense_output=True, events=events)
    ans['RK_45']['plus'] = sol
    #sol = solve_ivp(system, [0, tp_end], init_point, method='LSODA', atol = atol, rtol = rtol, jac = jac, args=sys_pars, \
    #              min_step = minstep, dense_output=True, events=events)
    #ans['LSODA']['plus'] = sol
    ### движение назад
    sol = solve_ivp(system, [0, tm_end], init_point, method='RK45',atol = atol, rtol = rtol, args=sys_pars,                   dense_output=True, events=events)
    ans['RK_45']['minus'] = sol
    #sol = solve_ivp(system, [0, tm_end], init_point, method='LSODA', atol = atol, rtol = rtol, jac = jac, args=sys_pars, \
    #              min_step = minstep, dense_output=True, events=events)
    #ans['LSODA']['minus'] = sol
    result = {}
    result = postproc(ans, 'plus', result)
    result = postproc(ans, 'minus', result)
    return result, ans

def classificator(init_point, points):
    ### отслеживаются все-на-свете события, они заданы в postproc
    ### обработка области №1
    ### задается только обработка событий для останова: пересечение нуля, уход на бесконечность
    if   init_point[0] <= a and init_point[1] >= a: 
        color, init, pend, mend = test1(init_point)
    ### обработка области №2
    ### задается только обработка событий: 
    ###   - пересечение скорости звука по  q1 "сверх -> до",
    ###   - пересечение нуля, 
    ###   - уход на бесконечность
    elif init_point[0] >= a and init_point[1] >= a:
        color, init, pend, mend = test2(init_point)
    ### обработка области №3
    ### задается только обработка событий: 
    ###   - пересечение скорости звука по  q1 "сверх -> до",
    ###   - пересечение нуля, 
    ###   - уход на бесконечность
    elif init_point[0] >= a and init_point[1] <= a:
        color, init, pend, mend = test3(init_point)
    ### обработка области №4
    ### задается только обработка событий: 
    ###   - пересечение скорости звука по  q1 "сверх -> до",
    ###   - пересечение нуля, 
    ###   - уход на бесконечность
    elif init_point[0] <= a and init_point[1] <= a:
        color, init, pend, mend = test4(init_point)
    else:
        print('Fuck!!! \nI am on the sound speed line and can\'t detect the point\'s type')
        color = 'smth'
        init = init_point
        pend = []
        mend = []
    points[color]['init'].append(init_point)
    points[color]['pend'].append(pend)
    points[color]['mend'].append(mend)
    return points  

def test1(init_point):
    ### пока для тестирование даже в априорно красной области уравнения решаются
    ### для того, чтобы можно было контролировать траектории
    ### потом после тестирования процедуру можно будет упростить
    ### запуск решения из начальной точки до заданных моментов времени в плюс и минус
    result, sln = solver(hydro, hydroj, times, minstep, init_point, sys_pars, events)
    color = 'red'
    init = [0] + init_point
    t = result['minus']['last']['time']
    q1, q2, r = result['minus']['last']['vars']
    mend = [t, q1, q2, r]
    t = result['plus']['last']['time']
    q1, q2, r = result['plus']['last']['vars']
    pend = [t, q1, q2, r]  
    return color, init, pend, mend

def test2(init_point):
    ### решаем задачку и по решению собираем все нужные события
    result, sln = solver(hydro, hydroj, times, minstep, init_point, sys_pars, events)
    if 'q2_ma' and 'q2_0' in result['minus'].keys():
        color = 'green'
    elif 'q1_ma' in result['minus'].keys():
        color = 'red'
    else:
        color = 'smth'
    
    init = [0] + init_point
    t = result['minus']['last']['time']
    q1, q2, r = result['minus']['last']['vars']
    mend = [t, q1, q2, r]
    t = result['plus']['last']['time']
    q1, q2, r = result['plus']['last']['vars']
    pend = [t, q1, q2, r] 
    return color, init, pend, mend

def test3(init_point):
    ### решаем задачку и по решению собираем все нужные события
    result, sln = solver(hydro, hydroj, times, minstep, init_point, sys_pars, events)
    mpoint = result['minus']['last']['vars']
    ppoint = result['plus']['last']['vars']
    res = is_white(ppoint, epsilon)
    if res['line']*res['int']:
        if 'q2_0' in result['minus'].keys():
            color = 'white_0'
        else:
            color = 'white_1'
    else:
        if ppoint[0] <= a:
            color = 'black'
        else:
            if mpoint[0] <= a:
                color = 'red'
            else:
                if 'q2_0' in result['minus'].keys():
                    color = 'green'
                else:
                    color = 'smth'
        
    init = [0] + init_point
    t = result['minus']['last']['time']
    q1, q2, r = result['minus']['last']['vars']
    mend = [t, q1, q2, r]
    t = result['plus']['last']['time']
    q1, q2, r = result['plus']['last']['vars']
    pend = [t, q1, q2, r] 
    return color, init, pend, mend

def test4(init_point):
    ### решаем задачку и по решению собираем все нужные события
    result, sln = solver(hydro, hydroj, times, minstep, init_point, sys_pars, events)
    mpoint = result['minus']['last']['vars']
    ppoint = result['plus']['last']['vars']    
    ### далее цепочка вопросов
    ### вопрос номер один: белая?
    res = is_white(ppoint, epsilon)
    if res['line']*res['int']:
        if 'q2_0' in result['minus'].keys():
            color = 'white_0'
        else:
            color = 'white_1'
    else:
        if ppoint[0] > a or ppoint[1] > a :
            color = 'red'
        else:
            if mpoint[0] > a:
                color = 'black'
            else:
                if 'q2_0' in result['plus'].keys() and mpoint[0] < delta*init_point[0]:
                    color = 'blue'
                else:
                    color = 'smth'
        
    init = [0] + init_point
    t = result['minus']['last']['time']
    q1, q2, r = result['minus']['last']['vars']
    mend = [t, q1, q2, r]
    t = result['plus']['last']['time']
    q1, q2, r = result['plus']['last']['vars']
    pend = [t, q1, q2, r] 
    return color, init, pend, mend

def painter(points):
    #init_points = list(map(lambda n: uvr_grid[n][:3], range(share[0],share[1])))
    ntotal = len(u_grid)*len(v_grid)*len(r_grid)
    ppnode =  share[1] - share[0]
    for npoint, point in enumerate(range(share[0], share[1])):
        pos = get_pos(n_u,n_v,n_r, point)
        i = pos[0]
        j = pos[1]
        k = pos[2]
        if logr != 10:
            init_point = [u_grid[i], v_grid[j], r_grid[k]]
        else:
            init_point = [u_grid[i], v_grid[j], r_grid[j][k]]
        # print(init_point)
        points = classificator(init_point, points)
        frac, whole = math.modf(npoint/100.0)
        if frac == 0:
            print('The current point is {:04d} of {:04d}'.format(npoint,ppnode))
    return points

def writer(points, folder):
    for color in points.keys():
        X = np.array(points[color]['init'])
        if X.size > 0: 
            X = X[X[:,1].argsort()]
        fname = folder + '/q1q2r_' + color + '_{:03d}'.format(rank) + '.txt'
        np.savetxt(fname, X, fmt='%+1.8e', delimiter=' ', newline='\n',  \
                   header='', footer='', comments='# ', encoding='UTF-8')
    fname = folder + '/q1q2r_{:03d}'.format(rank) + '.pkl'
    fdict = open(fname, "wb")
    pickle.dump(points, fdict)
    fdict.close()

def clear(points):
    points = {'red':   {'init': [], 'pend': [], 'mend': []}, \
              'green': {'init': [], 'pend': [], 'mend': []}, \
              'blue':  {'init': [], 'pend': [], 'mend': []}, \
              'black': {'init': [], 'pend': [], 'mend': []}, \
              'white_0': {'init': [], 'pend': [], 'mend': []}, \
              'white_1': {'init': [], 'pend': [], 'mend': []}, \
              'smth':  {'init': [], 'pend': [], 'mend': []}}
    return points

### РНД-шка ###
params = get_config('./params.yaml')
### Константы задачи
a = float(params["const"]["a"])
mu = float(params["const"]["mu"])
tau = float(params["const"]["tau"])
sys_pars = (a, mu, tau)
### Параметры для решателя 
### 1. времена задачи
tm = float(params["glob_left"])
tp = float(params["glob_right"])
times = {'t_plus': tp, 't_minus': tm, 'plot': False, 'n_time_points': 2}
### 2. тип решателя минимальный шаг по времени
int_meth = params['solver']['meth']
minstep =  float(params['solver']['minstep'])
### 3. обработка особых случаев
###  линии особых точек (эпсилон-окрестность и границы по  q2)
epsilon = float(params['eps_w'])
white_bounds = [a/2*(1 - np.sqrt(1 - 8/mu)), a/2*(1 + np.sqrt(1 - 8/mu))]
###  насколько конечная координата по q1 должна быть меньше начальной в признаке синих
delta = float(params['delta_blue'])
###  когда останавливать счет по признаку выхода на бесконечность
r_infty = float(params['infty'])
### 4. начальный словарь точек для заполнения
points = {'red':   {'init': [], 'pend': [], 'mend': []}, \
          'green': {'init': [], 'pend': [], 'mend': []}, \
          'blue':  {'init': [], 'pend': [], 'mend': []}, \
          'black': {'init': [], 'pend': [], 'mend': []}, \
          'white_0': {'init': [], 'pend': [], 'mend': []}, \
          'white_1': {'init': [], 'pend': [], 'mend': []}, \
          'smth':  {'init': [], 'pend': [], 'mend': []}}
events = (q1_ma, q1_pa, q2_ma, q2_pa, q2_0, r_inf)
### 5. параметры для задания пространственной сетки
n_r  = int(params["n_r"])
n_u  = int(params["n_u"])
n_v  = int(params["n_v"])
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

### Определение сетки в фазовом пространстве
L_u = L_u1 - L_u0
L_v = L_v1 - L_v0
L_r = L_r1 - L_r0
du = L_u/(n_u-1)
dv = L_v/(n_v-1)
dr = L_r/(n_r-1)
if logr == 1 and logu == 1 and logv == 1:
    u_grid =       list(map(lambda k: 10**(L_u0 + du*k), range(n_u)))
    v_grid =       list(map(lambda k: 10**(L_v0 + dv*k), range(n_v)))
    r_grid =       list(map(lambda k: 10**(L_r0 + dr*k), range(n_r)))
elif logr == 1 and logu == 1 and logv == 0:
    u_grid =       list(map(lambda k: 10**(L_u0 + du*k), range(n_u)))
    v_grid =       list(map(lambda k: L_v0 + dv*k, range(n_v)))
    r_grid =       list(map(lambda k: 10**(L_r0 + dr*k), range(n_r)))
elif logr == 1 and logu == 0 and logv == 0:
    u_grid =       list(map(lambda k: L_u0 + du*k, range(n_u)))
    v_grid =       list(map(lambda k: L_v0 + dv*k, range(n_v)))
    r_grid =       list(map(lambda k: 10**(L_r0 + dr*k), range(n_r)))
if logr == 0 and logu == 1 and logv == 1:
    u_grid =       list(map(lambda k: 10**(L_u0 + du*k), range(n_u)))
    v_grid =       list(map(lambda k: 10**(L_v0 + dv*k), range(n_v)))
    r_grid =       list(map(lambda k: L_r0 + dr*k, range(n_r)))
elif logr == 0 and logu == 0 and logv == 0:
    u_grid =       list(map(lambda k: L_u0 + du*k, range(n_u)))
    v_grid =       list(map(lambda k: L_v0 + dv*k, range(n_v)))
    r_grid =       list(map(lambda k: L_r0 + dr*k, range(n_r)))
elif logr == 10 and logu == 0 and logv == 0:
    u_grid =       list(map(lambda k: L_u0 + du*k, range(n_u)))
    v_grid =       list(map(lambda k: L_v0 + dv*k, range(n_v)))
    r_cent =       list(map(lambda k: 2*a*tau * v_grid[k] / mu / (a - v_grid[k]), range(n_v)))
    r_grid =       list(map(lambda v: list(map(lambda k: r_cent[v] - L_r/2 + dr*k + 0.000003, range(n_r))), range(n_v)))

rank = int(os.getenv("OMPI_COMM_WORLD_RANK"))
size = int(os.getenv("OMPI_COMM_WORLD_SIZE"))
# rank = 0
# size = 1
share = get_share(size, rank, n_u, n_v, n_r)
print(share)
points = painter(points)
folder = params["saveto"]
writer(points, folder)









