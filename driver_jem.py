# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 10:33:48 2020

@author: jmatt

Jeremy Matt
CE359 - Homework 4
"""
#imports
import pandas as pd
import SOM
import matplotlib.pyplot as plt
import numpy as np
import os
import copy as cp





# def get_nonsingular_cols(df):
    
#     selected_features = []
#     for key in df.keys():
#         vals = set(df[key].dropna())
#         if len(df[key]) == len(df[key].dropna()):
#             nan_rem = False
#         else:
#             nan_rem = True
#         if not len(vals) <= 1:
#             selected_features.append(key)
            
#         print('For key: {} - nan removed:{} num_features:{}'.format(key,nan_rem,len(vals)))
    
#     return selected_features
    

t = np.ones([5,5])

cd = os.getcwd()

# data_file = 'lymphedat_bp0.csv'
data_file= 'lymphedat_bp0-20221030.csv'
data_path = os.path.join(cd,data_file)

workspace_dir = os.path.join(cd,'SOM_output')

# Make directory if it doesn't exist
if not os.path.isdir(workspace_dir):
    os.mkdir(workspace_dir)

# Read data into a Pandas Dataframe
# index_col should be the index (0 starting) of the ID of the table, this 
# is also what will be used when printing the sample ID onto the SOM grid
data_df = pd.read_csv(data_path)
data_df.set_index('record_id',inplace=True)
data_df['record_id'] = data_df.index

label_ID = 'lymphed'
data_labels = data_df[label_ID]

features_to_keep = ['record_id',
 'fs',
 'ns',
 'supraf',
 'cws',
 'bs',
 'bpe',
 'fon',
 'nd',
 'nt',
 'eue',
 'ca',
 'dt_diff',
 'dashdi1',
 'dashdi2',
 'dashdi3',
 'dashdi4',
 'dashdi5',
 'dashdi6']

legend_text = [
    'group B',
    'group A',
    'group C']



#No dt_diff
# features_to_keep = ['record_id',
#  'fs',
#  'ns',
#  'supraf',
#  'cws',
#  'bs',
#  'bpe',
#  'fon',
#  'nd',
#  'nt',
#  'eue',
#  'ca',
#  'dashdi1',
#  'dashdi2',
#  'dashdi3',
#  'dashdi4',
#  'dashdi5',
#  'dashdi6']
# to_drop = []




label_df = pd.read_csv(os.path.join(cd,'SOM_output','3clusters','cluster_results.csv'))
label_df = label_df[['record_id','cluster']]
label_df.set_index('record_id',inplace=True)
l2 = label_df['cluster']

doug_cluster_labels = False
if doug_cluster_labels:
    labels = label_df['cluster']
else:
    labels = data_labels

#Define the label column name
sample_ID_col = 'record_id' 

legend_text = {
    0:'Group A',
    1:'Group B',
    2:'Group C'}

legend_text = {}
label_set = set(labels)
for label in label_set:
    legend_text[label] = '{}={}'.format(labels.name,label)

# if labels.name == 'lymphed':
# legend_text = {
#     0:'lymphed=0',
#     1:'lymphed=1',
#     2:'Group C'}

data_df = data_df[features_to_keep]
selected_features = SOM.get_nonsingular_cols(data_df)

singular_features = [feat for feat in features_to_keep if not feat in selected_features]
 

# Get only the data from the features of interest
selected_data_feats_df = data_df.loc[:, selected_features]


# Drop rows with NaN
selected_data_feats_df.dropna(axis=0, inplace=True)


#%%

dup_cols,selected_data_feats_df = SOM.combine_duplicate_colums(selected_data_feats_df,keep='merge')

selected_features = SOM.get_nonsingular_cols(selected_data_feats_df)

X = selected_data_feats_df[selected_features]

data_labels = [feat for feat in selected_features if not feat == sample_ID_col]
selected_data_feats_df = SOM.min_max_norm(X,data_labels)



#Set the grid size
grid_size = [50,50]
# grid_size = [10,10]
# grid_size = [4,4]
#Set the starting learning rate and neighborhood size
alpha = 0.9
# neighborhood_size = 4
neighborhood_size = int(grid_size[0]/2)
# neighborhood_size = 2*int(grid_size[0]/3)
# neighborhood_size = 1
#Set the number of training epochs
num_epochs = 500
    
toroidal = True
distance='euclidean'

load_trained = True 
 #%%
num_iterations = 3
if load_trained:
    num_iterations = 1
for i in range(num_iterations):
    
    i+=12
    
    
    if load_trained:
        weights_dir = 'run3_keep'
        weights_dir = 'run9_toroidal'
        weights_dir = 'run9'
        i = weights_dir.split("_")[0].split('run')[1]
        directory = os.path.join(cd,'SOM_output',weights_dir)
    
    output_dir = os.path.join(cd,'SOM_output',f'run{i}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    #Init a SOM object
    SOM_model = SOM.SOM(grid_size,X,sample_ID_col,alpha,neighborhood_size,toroidal,distance)
    
    #boolean flag to include D in the u-matrix calculations
    include_D = False
    #Plot the u-matrix
    sample_vis='labels'
    # sample_vis='symbols'
    plane_vis='u_matrix'
    # plane_vis='weights'
    
    plot_legend = True
    SOM_model.visualization_settings(output_dir,sample_vis,legend_text,include_D,labels,plane_vis,plot_legend)
    
    #Train the SOM
    
    if load_trained:
        # weights_dir = 'run3_keep'
        # i = weights_dir.split("_")[0].split('run')[1]
        # directory = os.path.join(cd,'SOM_output',weights_dir)
        
        # weights_dir = r'SOM_output\run3_keep'
        # weights_dir = weights_dir.split('\\')
        # directory = cd
        # for folder in weights_dir:
        #     directory = os.path.join(directory,folder)
            
        fn = 'weights.pkl'
        
        SOM_model.load_weights(directory,fn)
        
        
        
            
    else:
        SOM_model.train(num_epochs)
    
    SOM_model.save_weights(output_dir,'weights.pkl')
    
    SOM_model.plot_weight_hist(output_dir)
    
    #Plot the samples to the grid
    # SOM_model.plot_samples()
    
    #Calculate the U-matrix differences
    SOM_model.calc_u_matrix()
    
    
    #boolean flag to include D in the u-matrix calculations
    include_D = False
    #Plot the u-matrix
    sample_vis='labels'
    sample_vis='symbols'
    plane_vis='u_matrix'
    plane_vis='weights'
    SOM_model.visualization_settings(output_dir,sample_vis,legend_text,include_D,labels,plane_vis,plot_legend)
    
    # SOM_model.plot_u_matrix(include_D,output_dir=output_dir,labels=labels,sample_vis=sample_vis,legend_text=legend_text)
    SOM_model.plot_u_matrix()
    #Plot the feature planes
    SOM_model.plot_feature_planes()
    
    
    #Plot the feature planes
    SOM_model.plot_feature_planes()
    
    n_clusters = 6
    SOM_model.plot_clusters(n_clusters)
    
    
    # t = SOM_model.grid_updates
#%%
runtest = False

if runtest:
    SOM_model.gen_sample_grid()
    SOM_model.show_samples = False
    SOM_model.plot_u_matrix()
    SOM_model.plot_clusters()
