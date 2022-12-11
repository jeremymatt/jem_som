# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:31:38 2021

@author: jmatt
"""


import numpy as np
import pandas as pd
import SOM
# import SOM_bkup as SOM
import os
import timeit


# X = pd.DataFrame(np.random.rand(500,3))
X = pd.read_csv('RGB_data.csv')

### MODEL SETTINGS
#Set the grid size
grid_size = [10,10]
# grid_size = [4,4]
#Set the starting learning rate and neighborhood size
alpha = 0.9
neighborhood_size = int(grid_size[0]/4)
#Set the number of training epochs
num_epochs = 10
label_col = None
toroidal = False
distance = 'cosine'
distance = 'euclidean'
distance = {'type':'dtw','window_size':0}

#Initialize the model
model = SOM.SOM(grid_size,X,label_col,alpha,neighborhood_size,toroidal,distance)

#Set the visualization parameters
#Create the output dir if it doesn't exist
output_dir = os.path.join(os.getcwd(),'rgb_test')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
sample_vis='colors'
legend_text = None
include_D = False
labels=None 
# plane_vis='u_matrix'
plane_vis='weights'
model.visualization_settings(output_dir,sample_vis,legend_text,include_D,labels,plane_vis)

    
#Train the model and save the weights  
plot_intermediate_epochs = ('both',1)
plot_intermediate_epochs = False
# ex_time = timeit.timeit(lambda: model.train(num_epochs,plot_intermediate_epochs),number=10)
# print('Execution time: {}'.format(round(ex_time,5)))

model.train(num_epochs,plot_intermediate_epochs)

model.save_weights(output_dir,'weights.pkl')
 

#Calculate the U-matrix differences
model.calc_u_matrix()
#Plot the u-matrix
model.plot_u_matrix()


#Plot the feature planes
model.plot_feature_planes()
    