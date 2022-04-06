# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 09:31:15 2021

@author: jmatt
"""

import numpy as np
from dtaidistance import dtw
dtw.try_import_c(verbose=True)
#from dtaidistance.dtw_barycenter import dba
from tslearn import metrics

x = np.array([0.80619399, 0.70388858, 0.10022689])
weights = np.array([0.19443334, 0.08837178, 0.08207931])

euclidean_distance = sum((x-weights)**2)**0.5

window_size = 5
lb_keogh = metrics.lb_keogh(x, weights, radius=window_size)

dtw_distance = dtw.distance_fast(x,weights, window = window_size+1, use_pruning=False)

print('Input Sample:\n  {}'.format(x))
print('Weights:\n  {}'.format(weights))
print('distances:')
print('  Euclidean: {}'.format(euclidean_distance))
print('   lb_keogh: {}'.format(lb_keogh))
print('        DTW: {}'.format(dtw_distance))



x_vals = np.random.rand(25,100)

weights = np.array(list(np.random.rand(100)))

window_fraction = 0.1
window_size = int(x_vals.shape[1]*window_fraction)

for i in range(x_vals.shape[0]):
    x = np.array(list(x_vals[i,:]))

    euclidean_distance = round(sum((x-weights)**2)**0.5,12)

    lb_keogh = round(metrics.lb_keogh(x, weights, radius=window_size),12)

    dtw_distance = round(dtw.distance_fast(x,weights, window = window_size, use_pruning=True),12)
    
    print('SAMPLE: {}'.format(i))
    if (dtw_distance>=lb_keogh) & (dtw_distance<=euclidean_distance):
        print('DTW distance between upper & lower bounds\n')
    else:
        print('ERROR:\nDTW distance not between uppper & lower bounds')
        # print('Input Sample:\n  {}'.format(x))
        # print('Weights:\n  {}'.format(weights))
    print('distances:')
    print('  Euclidean: {}'.format(euclidean_distance))
    print('   lb_keogh: {}'.format(lb_keogh))
    print('        DTW: {}\n'.format(dtw_distance))
