# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 00:54:01 2023

@author: jmatt
"""

import numpy as np
import pandas as pd

def find_one_cluster_mapping(df,cluster,col1,col2):
    temp = df.loc[df[col1] == cluster].copy()
    all_mappings = list(zip(temp[col1],temp[col2]))
    map_set = set(all_mappings)
    max_count = 0
    best_mapping = None
    for mapping in map_set:
        matches = np.array(all_mappings) == mapping
        num_matches = sum(matches.all(axis=1))
        if sum(matches.all(axis=1))>max_count:
            best_mapping = mapping
            max_count = num_matches
            
    return best_mapping,max_count
            

def one2one_mapping(mapping_list):
    start,end = zip(*mapping_list)
    return len(set(start)) == len(set(end))


def mappings_match(mapping1,mapping2):
    return np.all([cur_map in mapping2 for cur_map in mapping1]) and np.all([cur_map in mapping1 for cur_map in mapping2])
    
    
        

def find_all_cluster_mappings(df1,df2):
    ###
    # Returns mapping of df1 clusters to df2 clusters
    ###
    
    n_clusters = len(set(df1.cluster))
    df = df1.copy()
    df['cluster2'] = df2['cluster']
    
    col1 = 'cluster'
    col2 = 'cluster2'
    
    forward_mappings = []
    backward_mappings = []
    forward_max_count_list = []
    backward_max_count_list = []
    
    for cluster in range(n_clusters):
        cluster = n_clusters - cluster -1
        forward_best_mapping,forward_max_count = find_one_cluster_mapping(df,cluster,col1,col2)
        backward_best_mapping,backward_max_count = find_one_cluster_mapping(df,cluster,col2,col1)
        
        forward_mappings.append(forward_best_mapping)
        backward_mappings.append(backward_best_mapping)
        
        forward_max_count_list.append((cluster,forward_max_count))
        backward_max_count_list.append((cluster,backward_max_count))
        
    forward_valid = one2one_mapping(forward_mappings)
    backward_valid = one2one_mapping(backward_mappings)
    
    stable = False
    
    
    if forward_valid and backward_valid:
        if not mappings_match(forward_mappings,backward_mappings):
            print('\n  WARNING: mappings are valid but different - returning forward mapping')
            print('    forward mapping: {}'.format(forward_mappings))
            print('    backward mapping: {}\n'.format(backward_mappings))
        
        final_mapping =  forward_mappings
        max_counts = forward_max_count_list
        stable = True
    elif forward_valid and not backward_valid:
        print("\n  Warning: forward mapping valid but backward mapping invalid")
        print('    Returning forward mapping: {}'.format(forward_mappings))
        print('    backward mapping: {}\n'.format(backward_mappings))
        final_mapping =  forward_mappings
        max_counts = forward_max_count_list
    elif not forward_valid and backward_valid:
        print("\n  Warning: backward mapping valid but forward mapping invalid")
        print('    Returning backward mapping: {}'.format(forward_mappings))
        print('    forward mapping: {}\n'.format(backward_mappings))
        s,e = zip(*backward_mappings)
        final_mapping =  list(zip(e,s))
        max_counts = backward_max_count_list
    else:
        print('  ERROR: no valid mapping found')
        print('    forward mapping: {}'.format(forward_mappings))
        print('    backward mapping: {}\n'.format(backward_mappings))
        final_mapping = None
            
        
    if isinstance(final_mapping,type(None)):
        cluster_map_dict = None
        max_counts = None
    else:
        cluster_map_dict = dict(final_mapping)
        for ind,row in df.iterrows():
            df.loc[ind,col1] = cluster_map_dict[df.loc[ind,col1]]
    
    return cluster_map_dict,df,max_counts,stable
            
    
    
        
    
    
    
