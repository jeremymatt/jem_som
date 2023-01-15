# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 19:16:09 2021

@author: jmatt
"""

import numpy as np
import pandas as pd
import os

import sys

from somappy import somutils


def get_nonsingular_cols(df):
    
    selected_features = []
    for key in df.keys():
        vals = set(df[key].dropna())
        if len(df[key]) == len(df[key].dropna()):
            nan_rem = False
        else:
            nan_rem = True
        if not len(vals) <= 1:
            selected_features.append(key)
            
        print('For key: {} - nan removed:{} num_features:{}'.format(key,nan_rem,len(vals)))
    
    return selected_features
    

t = np.ones([5,5])

cd = os.getcwd()

data_file = 'lymphedat_bp0.csv'
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

data_df = data_df[features_to_keep]

to_drop = ['record_id','breast_lymphedema','identifers_complete']
# to_drop = []

features_to_keep = [key for key in data_df.keys() if not key in to_drop]


selected_features = get_nonsingular_cols(data_df)

selected_features = [feat for feat in selected_features if not feat in to_drop]
 

# Get only the data from the features of interest
selected_data_feats_df = data_df.loc[:, selected_features]

# Drop rows with NaN
selected_data_feats_df.dropna(axis=0, inplace=True)


selected_features = get_nonsingular_cols(selected_data_feats_df)

selected_data_feats_df = selected_data_feats_df[selected_features]

# The number of features for the SOM                     
num_feats = len(selected_features) 


# NORMALIZE DATA by min, max normalization approach
selected_feats_df_norm = somutils.normalize(selected_data_feats_df)

# Display statistics on our normalized data
print(selected_feats_df_norm.describe())

# Initial learning rate for SOM. Will decay to 0.01 linearly
init_learning_rate = 0.05

# The number of rows for the grid and number of columns. This dictates 
# how many nodes the SOM will consist of. Currently not calculated 
# using PCA or other analyses methods.
nrows = 50
ncols = 50
# Create the SOM grid (which initializes the SOM network)
grid_type = 'hex'
grid_type = 'square'
som_grid = somutils.create_grid(nrows, ncols, grid_type=grid_type)

# Initial neighbourhood radius is defaulted to 2/3 of the longest distance
# Should be set up similar to R package Kohonen
# https://cran.r-project.org/web/packages/kohonen/kohonen.pdf
# Radius will decay to 0.0 linearly
init_radius = somutils.default_radius(som_grid)

# Get the data as a matrix dropping the dataframe wrapper
data = selected_feats_df_norm.values

# Number of iterations to run SOM
niter = 500

# Run SOM
som_weights, object_distances = somutils.run_som(
    data, som_grid, grid_type, niter, init_radius, init_learning_rate)

# It's possible that some data samples were not selected for training, thus do
# do not have a latest bmu
object_distances = somutils.fill_bmu_distances(
    data, som_weights, object_distances)

# Number of clusters to cluster SOM
nclusters = 3

# Cluster SOM nodes
clustering = somutils.cluster_som(som_weights, nclusters)

# Let's save the clusters corresponding to the samples now
results_path = os.path.join(workspace_dir, 'cluster_results.csv')

# To help track clusters to original ID, say ID column of SGAT_PID_P2
data_id = data_df.loc[:, ['record_id']]
merged_df = pd.merge(
    selected_data_feats_df, data_id, left_index=True, right_index=True)

somutils.save_cluster_results(
    merged_df, results_path, clustering.labels_, (nrows, ncols), 
    object_distances)

# Display the SOM, coloring the nodes into different clusters from 
# 'clustering' above
# Optional: pass in original dataframe to plot 
# the IDs onto their respective nodes
save_fig_path = os.path.join(workspace_dir, "som_figure.png")
somutils.basic_som_figure(data, som_weights, som_grid, clustering.labels_,
                            'hex', save_fig_path, dframe=data_df)
                            

print("Finished")