# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:31:38 2021

@author: jmatt
"""


import numpy as np
import pandas as pd
import SOM
import os


X = pd.DataFrame(np.random.rand(500,3))

### MODEL SETTINGS
#Set the grid size
grid_size = [10,10]
# grid_size = [4,4]
#Set the starting learning rate and neighborhood size
alpha = 0.9
neighborhood_size = int(grid_size[0]/2)
#Set the number of training epochs
num_epochs = 500
label_col = None
toroidal = True
distance = 'cosine'
# distance = 'euclidean'

#Initialize the model
model = SOM.SOM(grid_size,X,label_col,alpha,neighborhood_size,toroidal,distance)

#Create the output dir if it doesn't exist
output_dir = os.path.join(os.getcwd(),'rgb_test')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
#Train the model and save the weights  
model.train(num_epochs)
model.save_weights(output_dir,'weights.pkl')

#Set the visualization parameters
sample_vis='colors'
legend_text = None
include_D = False
labels=None  

#Calculate the U-matrix differences
model.calc_u_matrix()
#Plot the u-matrix
model.plot_u_matrix(include_D,output_dir=output_dir,labels=labels,sample_vis=sample_vis,legend_text=legend_text)
    