# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 18:24:13 2020

@author: jmatt

Jeremy Matt
CE359 - Homework 4
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy as cp
import os
import pickle
import tqdm
import itertools


def min_max_norm(df,data_labels,categorical_labels = None):
    """
    Normalizes non-categorical features between 0 and 1

    Parameters
    ----------
    df : TYPE pandas dataframe
        DESCRIPTION.
        Contains the data to be normalized
    data_labels : TYPE list
        DESCRIPTION.
        The dataframe headers for each of the features
    categorical_labels : TYPE list, optional
        DESCRIPTION. The default is None.
        A list of dataframe headers containing categorical values

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    
    if categorical_labels == None:
        #If there are no categorical features, normalize all of the features
        labels_to_norm = data_labels
    else:
        #If there are categorical features, remove the categorical labels from
        #the list of features to normalize
        labels_to_norm = [key for key in data_labels if not key in categorical_labels]
    for key in labels_to_norm:
        #Find the min/max values of the current feature
        min_val = min(df[key])
        max_val = max(df[key])
        #Normalize the feature
        df[key] = (df[key]-min_val)/(max_val-min_val)
        
    return df
        
        
def get_nonsingular_cols(df):
    """
    First finds the features that have NaN values and reports that to the user
    and then finds all of the non-singular columns in a dataframe

    Parameters
    ----------
    df : TYPE pandas dataframe
        DESCRIPTION.
        The dataframe containing the features to check for singularity

    Returns
    -------
    selected_features : TYPE list
        DESCRIPTION.
        List of the dataframe headers that are non-singular

    """
    
    selected_features = []
    #For each column in the input dataframe
    for key in df.keys():
        #Extract the set of non-NaN values in the current dataframe column
        vals = set(df[key].dropna())
        if len(df[key]) == len(df[key].dropna()):
            #If the length of the column is the same before and after dropping
            #NaN values, then there are no NaN values
            nan_rem = False
        else:
            #If the lengths are different after dropping NaN, then there are
            #NaN values
            nan_rem = True
        if len(vals) > 1:
            #If there are more than 1 value after dropping NaN values, then
            #the feature is non-singular
            selected_features.append(key)
        #Report the results to the user
        print('For key: {} - nan removed:{} num_features:{}'.format(key,nan_rem,len(vals)))
        
        
    #Build a list of singular features, which is the list of dataframe column
    #headers that are not in the list of non-singular features
    singular_features = [feat for feat in df.keys() if not feat in selected_features]
    #Report the results to the user
    if len(singular_features)>0:
        print('\nWARNING: The following features have only one value:')
        for feat in singular_features:
            print('    {}'.format(feat))
        print('\n')
    else:
        print('\nAll features have at least two values\n')
    
    return selected_features
    


class SOM:
    def __init__(self,grid_size,X,label_col,alpha,neighborhood_size,toroidal = False,distance='euclidean'):
        """
        Initialize a SOM object

        Parameters
        ----------
        grid_size : TYPE list
            DESCRIPTION.
            list of the x,y map dimensions
        X : TYPE pandas dataframe
            DESCRIPTION.
            A pandas dataframe containing the training samples
        label_col : TYPE string
            DESCRIPTION.
            The name of the column in the pandas dataframe containing the
            labels
        alpha : TYPE float
            DESCRIPTION.
            The starting learning rate
        neighborhood_size : TYPE int
            DESCRIPTION.
            The starting neighborhood size (the number of left/right/up/down
                                            from the current cell)
        toroidal : TYPE boolean
            DESCRIPTION.
            Flag to use toroidal boundaries (true) or non-toroidal boundaries
            (false)
        distance : TYPE string
            DESCRIPTION.
            The type of distance metric to use, either "euclidean" or "cosine"

        Returns
        -------
        None.

        """
        #for debugging
        self.write_winning = False
        
        #Build lists of colors and marker styles for plotting
        self.build_color_marker_lists()
        
        #Store the grid size and the label column in self
        self.grid_size = grid_size
        self.label_col = label_col
        
        #Store the boundary type and distance metric in self
        self.toroidal = toroidal
        self.distance = distance
        
        #Set to true for debugging
        self.print_weight_range = False
        
        #Store the learning rate in self and init a variable learning rate 
        #to be updated as training progresses
        self.alpha = alpha
        self.starting_alpha = alpha
        
        #Store the neighborhood size in self and init a variable neighborhood 
        #to be updated as training progresses
        self.neighborhood_size = neighborhood_size
        self.starting_neighborhood_size = neighborhood_size
        #Store the training patterns in self
        self.X = X
        
        #Generate a list of the names of the columns containing the data
        #features of the training patterns
        self.get_data_cols(label_col)
        #Store the number of features (includeing the extra column added for
        #cosine similarity)
        self.num_features = len(self.data_cols_mod)
        
        if distance == 'cosine':
            #Add the dimension required by cosine similarity
            self.add_dimension()
            #Set the method of finding winning to cosine
            self.find_winning = self.find_winning_cosine
        elif distance == 'euclidean':
            #Copy the input training variables without modification into the
            #working dataframe
            self.X_mod = cp.deepcopy(self.X)
            #Set the method of finding winning to euclidean
            self.find_winning = self.find_winning_euclidean
            
        
        #Initialize a matrix of random numbers in [-0.1, 0.1]
        # self.grid = 0.1*(np.random.rand(self.num_features,grid_size[0],grid_size[1])-0.5)
        self.grid = 0.2*(np.random.rand(self.num_features,grid_size[0],grid_size[1]))
        
        if self.print_weight_range:
            print('\nRANDOM INITIALIZATION')
            self.print_grid_range()
        #Determine the mean of each of the normalized data features (including
        #the D column)
        x_mean = self.X_mod[self.data_cols_mod].mean()
        
        center_weights = False
        #If center_weights is true:
        #For each feature, shift the grid values around the mean of that 
        #feature  The self.grid variable now contains the starting random
        #weights
        if center_weights:
            for ind in range(self.num_features):
                self.grid[ind,:,:] = x_mean[x_mean.keys()[ind]]+x_mean[x_mean.keys()[ind]]*self.grid[ind,:,:]
                    
            if self.print_weight_range:
                print('\nPLACE AROUND MEAN')
                self.print_grid_range()
        
        #Init a variable to store the previous grid values
        self.prev_grid = np.array(self.grid)
        #Init an empty list to hold the weight change trace
        self.weight_change = []
        #init empty lists to hold the changes to the neighborhood size and 
        #the learning rate
        self.LR_trace = []
        self.NH_trace = []
        
    def train(self,num_epochs,plot_intermediate_epochs = False):
        """
        Train the SOM

        Parameters
        ----------
        num_epochs : TYPE int
            DESCRIPTION.
            The number of epochs to train the SOM

        Returns
        -------
        None.

        """
        
        if plot_intermediate_epochs == False:
            plot_intermediate_type = None
            intermediate_plot_step = 1
        else:
            plot_intermediate_type,intermediate_plot_step = plot_intermediate_epochs
        
        with open('winning.txt','w') as self.f:
            
            for epoch in tqdm.tqdm(range(num_epochs),total = len(range(num_epochs))):
                    
                self.epoch = epoch
                
                if (plot_intermediate_epochs != None) & (epoch%intermediate_plot_step == 0):
                    if plot_intermediate_type == 'both':
                        self.calc_u_matrix()
                        #Plot the u-matrix
                        self.plot_u_matrix(False)
                        #Plot the feature planes
                        self.plot_feature_planes(False)
                    
                    if plot_intermediate_type == 'u_matrix':
                        self.calc_u_matrix()
                        #Plot the u-matrix
                        self.plot_u_matrix(False)
                    
                    if plot_intermediate_type == 'feature_planes':
                        self.calc_u_matrix()
                        #Plot the feature planes
                        self.plot_feature_planes(False)
                    
                    
                
                #Generate a ordered list of all index values into X
                order = self.X_mod.index.tolist()
                #Loat a random number generator object
                rng = np.random.default_rng()
                #Shuffle the indices into a random order
                rng.shuffle(order)
                
                #Update the neighborhood size and the learning rate based on the 
                #current epoch
                self.update_neighborhood_size(epoch,num_epochs)
                self.update_alpha(epoch,num_epochs)
                self.ctr = 0
                #Present each training sample in random order
                max_inds = [0,0]
                for ind in order:
                    
                    #Extract the data associated with the current sample into a
                    #numpy array
                    cur_x = np.array(self.X_mod.loc[ind,self.data_cols_mod])
                    #Find the winning index
                    self.find_winning(cur_x)
                    #Used for debugging
                    if self.winning[0]>max_inds[0]:
                        max_inds[0] = self.winning[0]
                    if self.winning[1]>max_inds[1]:
                        max_inds[1] = self.winning[1]
                    #Find the neighborhood indices around the winner
                    self.get_neighborhood()
                    #Update the weights of the nodes within the neighborhood
                    self.update_weights(cur_x)
                    self.ctr+=1
                    
                if self.print_weight_range:
                    print('\nAfter weight update for epoch {}'.format(epoch))
                    self.print_grid_range()
                    
                #Print winning indices for debugging purposes
                self.f.write('  MAX INDS: {}\n\n'.format(max_inds))
                
                self.calc_weight_change()
                
    def calc_weight_change(self):
        """
        Calculates the change in the weights from the previous epoch and stores
        the weight change, the learning rate, and the neighborhood size
        in lists

        Returns
        -------
        None.

        """
        #Calculate the difference between the current weights and the 
        #previous-epoch weights
        temp = self.grid-self.prev_grid
        #Square each element in the matrix and take the sum
        temp = np.power(temp,2)
        temp = np.sum(temp)
        
        #Append the square root to the weight change list
        self.weight_change.append(np.sqrt(temp))
        #Update the previous grid to the current grid
        self.prev_grid = np.array(self.grid)
        
        #Store the learning rate and neighborhood sizes
        self.LR_trace.append(self.alpha*10)
        self.NH_trace.append(self.neighborhood_size)
                
        
    def update_alpha(self,epoch,num_epochs):
        """
        Updates the learning rate based on the current epoch

        Parameters
        ----------
        epoch : TYPE int
            DESCRIPTION.
            The current epoch
        num_epochs : TYPE int
            DESCRIPTION.
            The total number of epochs the network will train for

        Returns
        -------
        None.

        """
        #Update the learning rate
        self.alpha = self.starting_alpha*(1-epoch/num_epochs)
        
        
    def update_neighborhood_size(self,epoch,num_epochs):
        """
        Updates the neighborhood size

        Parameters
        ----------
        epoch : TYPE int
            DESCRIPTION.
            The current epoch
        num_epochs : TYPE int
            DESCRIPTION.
            The total number of epochs the network will train for

        Returns
        -------
        None.

        """
        #UPdate the neighborhood size based on the formula given in class
        #and round to the nearest integer.  Then convert to an integer to 
        #allow the value to be used to calculate matrix indices
        self.neighborhood_size = int(round(self.starting_neighborhood_size*(1-epoch/num_epochs),0))
        #Make sure the neighborhood size is never less than one
        if self.neighborhood_size<0:
            self.neighborhood_size = 0
        
        
    def update_weights(self,cur_x):
        """
        Updates the weights based on the current neighborhood indices

        Parameters
        ----------
        cur_x : TYPE numpy array
            DESCRIPTION.
            The feature values of the current training pattern

        Returns
        -------
        None.

        """
            
            
        testing_neighborhood = True
        testing_neighborhood = False
        #Load sample grid to test neighborhood code - set a breakpoint, set
        #testing_neighborhood = True, and run the if/else statement in debug
        #mode
        if testing_neighborhood:
            self.neighborhood_size = 3
            for self.winning in [(5,5),(1,1),(1,9),(9,9),(9,1),(5,9),(9,5),(5,1),(1,5)]:
                for self.toroidal in [True,False]:
                    self.make_test_grid()
                    self.grid = self.test_grid
                    
                    self.get_neighborhood()
                    
                    to_update = self.get_weights_to_update()
                    
                    to_update *= 0
                    
                    self.put_updated_weights(to_update)
                    
                    self.grid[0,self.winning[0],self.winning[1]] =-1
                    print('toroidal:{}, centered at: {}'.format(self.toroidal,self.winning))
                    print(self.grid)
                    print('\n')
        else:
            #Extract a grid of the weights to update
            to_update = self.get_weights_to_update()
            #For each variable in the current input pattern, update the weights
            #for that variable at all of the in-neighborhood nodes
            for ind,val in enumerate(cur_x):
                to_update[ind,:,:] += self.alpha*(cur_x[ind]-to_update[ind,:,:])
            #Store the updated weights back in weights matrix
            self.put_updated_weights(to_update)
            
            
            
    def get_weights_to_update(self):
        """
        Extracts the weights of the in-neighborhood nodes and stores in a matrix

        Returns
        -------
        to_update : TYPE numpy array
            DESCRIPTION.
            The weights of the in-neighborhood nodes

        """
        #Store in shorter variables to limit line length
        #Extract the range of indices in the non-wrapped zone for the 0-axis
        a0_range = self.a0_range
        #Extract the range of indices in the non-wrapped zone for the 1-axis
        a1_range = self.a1_range
        #Extract the range of indices in the wrapped zone for the 0-axis
        a0_wrap = self.a0_wrap
        #Extract the range of indices in the wrapped zone for the 1-axis
        a1_wrap = self.a1_wrap
    
        #determine the shape of the in-neighborhood nodes.  np.diff calculates
        #the differences between element 0 & 1, element 1 & 2, etc. in a list
        #or 1d array.  the [0] extracts the first element in the returned
        #list of differences.
        a0_size = np.diff(a0_range)[0]+np.diff(a0_wrap)[0]
        a1_size = np.diff(a1_range)[0]+np.diff(a1_wrap)[0]
        
        #Initialize a matrix of zeros to hold the weights
        to_update = np.zeros([self.grid.shape[0],a0_size,a1_size])
        
        #Extract the weights in the non-wrapped zone
        to_update[:,0:np.diff(a0_range)[0],0:np.diff(a1_range)[0]] = self.grid[:,a0_range[0]:a0_range[1],a1_range[0]:a1_range[1]]
        #Extract the weights in the wrapped zone along axis 0
        to_update[:,np.diff(a0_range)[0]:a0_size,0:np.diff(a1_range)[0]] = self.grid[:,a0_wrap[0]:a0_wrap[1],a1_range[0]:a1_range[1]]
        #Extract the weights in the wrapped zone along axis 1
        to_update[:,0:np.diff(a0_range)[0],np.diff(a1_range)[0]:a1_size] = self.grid[:,a0_range[0]:a0_range[1],a1_wrap[0]:a1_wrap[1]]
        #Extract the weights in the wrapped zone along both axis 0 & axis 1
        to_update[:,np.diff(a0_range)[0]:a0_size,np.diff(a1_range)[0]:a1_size] = self.grid[:,a0_wrap[0]:a0_wrap[1],a1_wrap[0]:a1_wrap[1]]
        
        return to_update
        
    def put_updated_weights(self,to_update):
        """
        Stores the updated weights back in in the weights matrix

        Parameters
        ----------
        to_update : TYPE numpy array
            DESCRIPTION.
            The matrix of updated weights

        Returns
        -------
        None.

        """
        #Store in shorter variables to limit line length
        #Extract the range of indices in the non-wrapped zone for the 0-axis
        a0_range = self.a0_range
        #Extract the range of indices in the non-wrapped zone for the 1-axis
        a1_range = self.a1_range
        #Extract the range of indices in the wrapped zone for the 0-axis
        a0_wrap = self.a0_wrap
        #Extract the range of indices in the wrapped zone for the 1-axis
        a1_wrap = self.a1_wrap
    
        #determine the shape of the in-neighborhood nodes.  np.diff calculates
        #the differences between element 0 & 1, element 1 & 2, etc. in a list
        #or 1d array.  the [0] extracts the first element in the returned
        #list of differences.
        a0_size = np.diff(a0_range)[0]+np.diff(a0_wrap)[0]
        a1_size = np.diff(a1_range)[0]+np.diff(a1_wrap)[0]
        
           
        #Store the updated weights in the non-wrapped zone
        self.grid[:,a0_range[0]:a0_range[1],a1_range[0]:a1_range[1]] = to_update[:,0:np.diff(a0_range)[0],0:np.diff(a1_range)[0]]
        #Store the weights in the wrapped zone along axis 0
        self.grid[:,a0_wrap[0]:a0_wrap[1],a1_range[0]:a1_range[1]] = to_update[:,np.diff(a0_range)[0]:a0_size,0:np.diff(a1_range)[0]]
        #Store the weights in the wrapped zone along axis 1
        self.grid[:,a0_range[0]:a0_range[1],a1_wrap[0]:a1_wrap[1]] = to_update[:,0:np.diff(a0_range)[0],np.diff(a1_range)[0]:a1_size]
        #Store the weights in the wrapped zone along both axis 0 and axis 1
        self.grid[:,a0_wrap[0]:a0_wrap[1],a1_wrap[0]:a1_wrap[1]] = to_update[:,np.diff(a0_range)[0]:a0_size,np.diff(a1_range)[0]:a1_size]
        
        
        
        
    def find_winning_cosine(self,cur_x):
        """
        Finds the winning node for the current input using cosine similarity

        Parameters
        ----------
        cur_x : TYPE numpy array
            DESCRIPTION.
            The feature values of the current training pattern

        Returns
        -------
        None.

        """
        #Generate a temporary variable to contain the sum of the products of 
        #the inputs and the weights
        temp = np.zeros(self.grid_size)
        
        #For each feature in the current training pattern
        for ind,val in enumerate(cur_x):
            #Multiply the current feature value by the grid of weights 
            #associated with that feature and add the result to the temporary 
            #grid
            temp += self.grid[ind,:,:]*cur_x[ind]
        
        #Find the x,y coordinates of the winning node.  argmax finds the 
        #1D index based on row-major order.  unravel_index converts the 1D
        #index to x,y coordinates based on the shape of the grid
        self.winning = np.unravel_index(np.argmax(temp),self.grid_size)
        if self.write_winning:
            self.f.write('epoch: {} item: {} ==> winning node: {}\n'.format(self.epoch,self.ctr,self.winning))
            
    def find_winning_euclidean(self,cur_x):
        """
        Finds the winning node for the current input using euclidean distance

        Parameters
        ----------
        cur_x : TYPE numpy array
            DESCRIPTION.
            The feature values of the current training pattern

        Returns
        -------
        None.

        """
        
        #Store the weights matrix in a temporary variable
        temp = cp.deepcopy(self.grid)
        
        #For each feature in the current training pattern, calculate the 
        #distance between the feature value and every associated weight
        for ind,val in enumerate(cur_x):
            temp[ind,:,:] -= val
        #Square the distances
        temp = np.power(temp,2)
        #Sum the squared distances associated with each node
        temp = np.sum(temp,axis=0)    
        #Take the square root of the summed distances
        temp = np.power(temp,0.5)
        
        #Find the x,y coordinates of the winning node.  argmin finds the 
        #1D index based on row-major order.  unravel_index converts the 1D
        #index to x,y coordinates based on the shape of the grid
        self.winning = np.unravel_index(np.argmin(temp),self.grid_size)
        if self.write_winning:
            self.f.write('epoch: {} item: {} ==> winning node: {}\n'.format(self.epoch,self.ctr,self.winning))
        
    def get_data_cols(self,label_col):
        """
        Finds the labels of the data feature columns

        Parameters
        ----------
        label_col : TYPE string
            DESCRIPTION.
            The name of the column containing the training pattern labels

        Returns
        -------
        None.

        """
        #self.X.keys() is a list of the column headers in the pandas dataframe
        #containing the training patterns.  This line of code stores in a list 
        #those column headers that are not equal to the label header
        self.data_cols = [val for val in self.X.keys() if not val == label_col]
        #Copy the data columns to a separate list and append the string 'D'
        #The 'D' column will contain the extra dimension for the cosine 
        #similarity calculation
        self.data_cols_mod = list(self.data_cols)
        if self.distance == 'cosine':
            self.data_cols_mod.append('D')
        
        
    def add_dimension(self):
        """
        Adds the dimension required by the cosine similarity calculation

        Returns
        -------
        None.

        """
        
        #Calculate the length of each input training pattern.  This line
        #raises each element of the training patterns to the power of 2, takes 
        #the sum of each row, and then calculates the square root of each sum
        length = np.sqrt(np.sum(np.power(self.X[self.data_cols],2),axis=1))
        #Find the max length and add a little bit
        self.max_length = np.max(length)+0.01
        #Calculate the D vector 
        D = np.sqrt(self.max_length**2-np.power(length,2))
        #make a copy of the X dataframe and add the D column to the new 
        #dataframe
        self.X_mod = cp.deepcopy(self.X)
        self.X_mod['D'] = D
        #Normalize by the maximum length
        self.X_mod[self.data_cols_mod] = self.X_mod[self.data_cols_mod]/self.max_length
        
        
        
    def get_neighborhood(self):
        """
        Finds the indices of the neighborhood based on the current 
        neighborhood size and the indices of the current winning node.  The
        neighborhood boundaries may or may not be toroidal

        Returns
        -------
        None.

        """
        
        #Set the range of wrapping along axis 0 to (0,0) - this corresponds
        #to no wrapping
        self.a0_wrap = (0,0)
        #Find the lower index of Axis 0
        a0_start = self.winning[0]-self.neighborhood_size
        if a0_start<0:
            #If the lower index of Axis 0 is less than 0, set the start of the
            #non-wrapped zone to 0
            a0_start = 0
            if self.toroidal:
                #If using toroidal, Calculate the distance to wrap.  Since
                #the left edge of the grid is zero-indexed, the wrap is the 
                #difference between the neighborhood size and the winning
                #node
                a0_wrap = self.neighborhood_size-self.winning[0]
                #Wrapping on the right side of the grid.  The start is the size
                #of the grid along axis 0 minus the distance to wrap and the
                #end is the size of the grid
                self.a0_wrap = (self.grid.shape[1]-a0_wrap,self.grid.shape[1])
        #Find the upper index of Axis 0
        a0_end = self.winning[0]+self.neighborhood_size+1
        if a0_end>self.grid.shape[1]:
            #If the axis 0 upper bound is beyond the grid, set the upper index
            #of the non-wrapped zone to be the limit of the grid
            a0_end = self.grid.shape[1]
            if self.toroidal:
                #If using toroidal, calculate the the distance to wrap
                #as the location of the winning node plus the neighborhood
                #size plus 1 (to account for python's indexing) minus the
                #location of the grid boundary
                a0_wrap = self.winning[0]+self.neighborhood_size+1-self.grid.shape[1]
                #Wrapping on the left side of the grid.  The first value is
                #0 and the second value is the distance to wrap
                self.a0_wrap = (0,a0_wrap)
        
        #Store the start and end indices along axis 0 of the non-wrapped zone
        self.a0_range = (a0_start,a0_end)
        
           
        #Set the range of wrapping along axis 1 to (0,0) - this corresponds
        #to no wrapping
        self.a1_wrap = (0,0)
        #Find the lower index of Axis 1
        a1_start = self.winning[1]-self.neighborhood_size
        if a1_start<0:
            #If the lower index of Axis 1 is less than 0, set the start of the
            #non-wrapped zone to 0
            a1_start = 0
            if self.toroidal:
                #If using toroidal, Calculate the distance to wrap.  Since
                #the left edge of the grid is zero-indexed, the wrap is the 
                #difference between the neighborhood size and the winning
                #node
                a1_wrap = self.neighborhood_size-self.winning[1]
                #Wrapping on the bottom of the grid.  The start is the size
                #of the grid along axis 1 minus the distance to wrap and the
                #end is the size of the grid
                self.a1_wrap = (self.grid.shape[2]-a1_wrap,self.grid.shape[2])
        #Find the upper index of Axis 1
        a1_end = self.winning[1]+self.neighborhood_size+1
        if a1_end>self.grid.shape[2]:
            #If the axis 1 upper bound is beyond the grid, set the upper index
            #of the non-wrapped zone to be the limit of the grid
            a1_end = self.grid.shape[2]
            if self.toroidal:
                #If using toroidal, calculate the the distance to wrap
                #as the location of the winning node plus the neighborhood
                #size plus 1 (to account for python's indexing) minus the
                #location of the grid boundary
                a1_wrap = self.winning[1]+self.neighborhood_size+1-self.grid.shape[2]
                #Wrapping on the top of the grid.  The first value is
                #0 and the second value is the distance to wrap
                self.a1_wrap = (0,a1_wrap)
            
        #Store the start and end indices along axis 1 of the non-wrapped zone
        self.a1_range = (a1_start,a1_end)
        
        
                
    def calc_u_matrix(self):
        """
        Calculate the u-matrix of the trained weights

        Returns
        -------
        None.

        """
        #SEt the neighborhood size to one
        self.neighborhood_size = 1
        #Initialize a matrix of zeros to hold the u-matrix values.  
        self.u_matrix = np.zeros(self.grid.shape)
        #loop through each row of the node matrix
        for ii in range(self.grid_size[0]):
            #loop through each node in the current row
            for iii in range(self.grid_size[1]):
                #Set the center of the neighborhood calculation to the current
                #node and get the neighborhood indices
                self.winning = [ii,iii]
                self.get_neighborhood()
                
                #Extract the weights in the neighborhood around the current 
                #point
                neighborhood = self.get_weights_to_update()
                
                #For each input feature, calculate the difference between
                #the weight for that feature at the current node and the 
                #weights for the feature at the nodes in the neighborhood.
                #This results in a 3D matrix that contains all the information
                #for both the U-matrix and the feature planes.  Each "layer"
                #in the matrix contains the sums of neighborhood distances
                #for a particular input feature
                #NOTE: this includes the different between the node weight and
                #itself; however, this difference is zero, so including the 
                #current node in the calculation doesn't affect the results
                #NOTE2: This includes the feature plane for the "D" feature
                for i in range(self.num_features):
                    #Extract the neighborhood weights for the current feature
                    cw_neighborhood = np.array(neighborhood[i,:,:])
                    #Extract the weight for the current node and feature
                    cur_weight = self.grid[i,ii,iii]
                    #Subtract the current weight from the weights in the 
                    #neighborhood
                    cw_neighborhood -= cur_weight
                    #Take the absolute value to convert from a difference to 
                    #a distance
                    cw_neighborhood = np.abs(cw_neighborhood)
                    #Store the sum in the u matrix
                    self.u_matrix[i,ii,iii] = np.sum(cw_neighborhood)
                    
                    
    def open_new_fig(self): 
        """
        Opens a new figure for plotting

        Returns
        -------
        None.

        """
        
        #Determine the aspect ratio of the grid
        wh_ratio = self.grid_size[0]/self.grid_size[1]
        
        #Determine the x and y sizes of the figure
        y_size = 15
        x_size = wh_ratio*y_size
        
        #Open a new figure and store the figure and axis handles
        self.fig,self.ax = plt.subplots(figsize=(x_size,y_size))
        #Adjust the min/max values to be beyond the grid so there's white 
        #space when plotting
        x_adjust = max([self.grid_size[0]*.15,0.25])
        y_adjust = max([self.grid_size[1]*.15,0.25])
        xlim = [0-x_adjust,self.grid_size[0]-1+x_adjust]
        ylim = [0-y_adjust,self.grid_size[1]-1+y_adjust]
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        #Get rid of the x and y axis labels
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])  
        #Set the marker size
        self.marker_size =  min([x_size,y_size])*40   
        #Set the text size
        self.text_size = min([x_size,y_size])   

    def add_heatmap(self,grid,colormap,cbar_label=None):
        """
        Adds a heat map behind the plotted points and adds a colorbar legend 
        to the figure

        Parameters
        ----------
        grid : TYPE numpy array
            DESCRIPTION.
            The heatmap values
        colormap : TYPE python colormap
            DESCRIPTION.
            The color map to use when plotting the grid
        cbar_label : TYPE string, optional
            DESCRIPTION. The default is None.
            Add a label to the colorbar legend

        Returns
        -------
        None.

        """
        #plot the heat map
        im = plt.imshow(grid,cmap=colormap,origin='lower')
        #Add the colorbar legend
        if cbar_label:
            cbar = self.ax.figure.colorbar(im,ax=self.ax)
            cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", size=self.text_size*1.25)
            cbar.ax.tick_params(labelsize=self.text_size*.9)
        else:
            print(cbar_label)
        
                
    def plot_u_matrix(self,final_plot = True):
        """
        Generates a figure showing the u matrix

        Parameters
        ----------
        include_D : TYPE boolean, optional
            DESCRIPTION. The default is False.
            Include the weights associated with the D feature when calculating
            and plotting the u-matrix


        Returns
        -------
        None.

        """
        
        #Open a new figure
        self.open_new_fig()
        
        # #Store the legend text in self
        # self.legend_text = legend_text
        # #Store the category labels in self
        # self.labels = labels
        
        #plot the sample labels
        if self.sample_vis == 'labels':
            #Plot with text strings (IE cat, tiger, dog, horse, etc) at the 
            #winning nodes
            self.plot_samples(False)
        elif self.sample_vis == 'colors':
            #Used for plotting the RGB test case
            self.plot_samples_colormap()
        else:
            #Plot symbols at the winning nodes
            self.plot_samples_symbols(False)
        
        if self.include_D or self.distance == 'euclidean':
            #Take the sum of the u matrix along the 0th axis; this sums the 
            #differences for each feature for each node.
            #Transpose the u-matrix to align the heatmap correctly
            grid = np.sum(self.u_matrix,axis=0).T
        else:
            #Take the sum of the u matrix along the 0th axis; this sums the 
            #differences for each feature for each node, excluding D. 
            #Transpose the u-matrix to align the heatmap correctly
            grid = np.sum(self.u_matrix[:-1,:,:],axis=0).T
        
        #Set the colormap, colorbar label, and add the heatmap to the figure
        colormap = 'gray_r'
        cbar_label = 'Weight Change'
        # cbar_label = None
        self.add_heatmap(grid, colormap,cbar_label)
        #save the figure to file
        plt.tight_layout()
        if final_plot:
            fn = 'u_matrix_final_{}_epochs.png'.format(self.epoch+1)
        else:
            fn = 'u_matrix_ep-{}.png'.format(self.epoch)
            
        if self.output_dir == None:
            plt.savefig(fn)
        else:
            plt.savefig(os.path.join(self.output_dir,fn))
            
        plt.close(self.fig)
        
    def plot_feature_planes(self,final_plot = True):
        """
        Plots each of the feature planes, including the plane associated with
        the D feature

        Returns
        -------
        None.

        """
        
        
        # self.legend_text = legend_text
        # self.labels = labels
        #For each feature in the training data set
        for i in range(len(self.data_cols_mod)):
            
            self.open_new_fig()
            
            #Plot the sample labels
            if self.sample_vis == 'labels':
                self.plot_samples(False)
            elif self.sample_vis == 'colors':
                self.plot_samples_colormap(False)
            else:
                self.plot_samples_symbols(False)
            
            if self.plane_vis == 'u_matrix':
                #extract the u-matrix values associated with the current feature, 
                #transposing to make the data line up when plotting the heatmap
                grid = self.u_matrix[i,:,:].T
                #Add the feature plane to the figure
                colormap = 'gray_r'
                cbar_label = 'Weight Change'
                self.add_heatmap(grid, colormap,cbar_label)
            else:
                #Extract the weight values associated with the current feature
                #transposing to make the data line up when plotting the heatmap
                grid = self.grid[i,:,:].T
                #Set the colormap and colorbar label and add the heatmap to 
                #the plot
                colormap = 'viridis'
                cbar_label = 'Weight Value'
                self.add_heatmap(grid, colormap,cbar_label)
                
            #Add a plot title
            plt.title('Feature: {}'.format(self.data_cols_mod[i]),size = self.text_size*1.5)
            #Save the figure to file
            plt.tight_layout()
            
            if final_plot:
                fn = 'feature_plane-{}_final_{}-epochs.png'.format(self.data_cols_mod[i],self.epoch+1)
            else:
                fn = 'feature_plane-{}_ep-{}.png'.format(self.data_cols_mod[i],self.epoch)
            
            if self.output_dir == None:
                plt.savefig(fn)
            else:
                plt.savefig(os.path.join(self.output_dir,fn))
                    
            plt.close(self.fig)
        
                
    def plot_samples(self,print_coords = True):
        """
        Generates a figure with the sample labels at the winning coordinates

        Parameters
        ----------
        print_coords : TYPE boolean, optional
            DESCRIPTION. The default is True.
            Whether or not to print a list of the winning coordinates to the
            terminal

        Returns
        -------
        None.

        """
        
        #Initialize a list to store the indices where a feature was plotted
        used_inds = []
        
        #Loop through each training pattern
        for ind in self.X_mod.index:
            #Extract the data associated with the current training pattern
            cur_x = np.array(self.X_mod.loc[ind,self.data_cols_mod])
                    
            #Extract the label text
            label_txt = self.X_mod.loc[ind,self.label_col]
            #Find the winning node (the coordinates where the feature will be
            #plotted)
            self.find_winning(cur_x)
            #Determine if the text needs to be offset to avoid plotting over
            #a previous label
            method = 'points'
            xytext = self.get_text_offset(used_inds,method)
            #Add the winning coordinates to the list of used coordinates
            used_inds.append(self.winning)
            #Add the label to the winning point on the map
            if np.all(self.labels == None):
                self.ax.annotate(label_txt,self.winning,xytext = xytext, textcoords='offset points')
            else:
                bkg_color = self.colors[self.labels[ind]]
                self.ax.annotate(label_txt,[self.winning[0],self.winning[1]],xytext = xytext, backgroundcolor=bkg_color,textcoords='offset points')
                
            if print_coords:
                print("label: {} at {}".format(label_txt,self.winning)) 
        
                
    def plot_samples_colormap(self,print_coords = True):
        """
        Generates a figure for the RGB test case

        Parameters
        ----------
        print_coords : TYPE boolean, optional
            DESCRIPTION. The default is True.
            Whether or not to print a list of the winning coordinates to the
            terminal

        Returns
        -------
        None.

        """
        
        #Initialize a list to store the indices where a feature was plotted
        used_inds = []
        
        #Loop through each training pattern
        for ind in self.X_mod.index:
            #Extract the data associated with the current training pattern
            cur_x = np.array(self.X_mod.loc[ind,self.data_cols_mod])
                    
            #Find the winning node (the coordinates where the feature will be
            #plotted)
            self.find_winning(cur_x)
            #Determine if the text needs to be offset to avoid plotting over
            #a previous label
            method = 'grid_fraction'
            xytext = self.get_text_offset(used_inds,method)
            #Add the winning coordinates to the list of used coordinates
            used_inds.append(self.winning)
            #Add the label to the winning point on the map
            self.ax.scatter(self.winning[0]+xytext[0],self.winning[1]+xytext[1],marker='o',s = self.marker_size,color=tuple(cur_x))
                       
                
    def plot_samples_symbols(self,print_coords = True):
        """
        Generates a figure with the sample labels at the winning coordinates

        Parameters
        ----------
        print_coords : TYPE boolean, optional
            DESCRIPTION. The default is True.
            Whether or not to print a list of the winning coordinates to the
            terminal

        Returns
        -------
        None.

        """
        
        self.get_plot_sets(print_coords)
        
        
        if self.have_labels:
            font = {'size': self.text_size*2}
            for ind,label in enumerate(self.label_list):
                color = self.colors[ind]
                marker = self.markers[ind]
                self.ax.scatter(self.data_dict[label]['x'],
                                self.data_dict[label]['y'],
                                c = color,
                                marker = marker,
                                s = self.marker_size,
                                edgecolors = "black",
                                linewidth = self.marker_size/150,
                                label = self.data_dict[label]['legend_text'])
            # num_labels = len(self.label_list)
            target_cols = 2
            if len(self.label_list)<=target_cols:
                ncol = len(self.label_list)
                # nrow = 1
            else:
                ncol = target_cols
                # nrow = int(np.ceil(num_labels/target_cols))
            self.ax.legend(shadow=True,fancybox=True,loc='center', ncol=ncol,bbox_to_anchor=(0.5,-.05),prop=font)    
        else:
            self.ax.scatter(self.data_dict[0]['x'],self.data_dict[0]['y'],c=self.colors[0])
                
    def get_plot_sets(self,print_coords):
        """
        Generates a set of x,y node coordinates for each overlay category type

        Parameters
        ----------
        print_coords : TYPE boolean
            DESCRIPTION.
            Flag to print the coordinates used for debugging

        Returns
        -------
        None.

        """
        
        #Initialize a list to store the indices where a feature was plotted
        used_inds = []
        
        if np.all(self.labels == None):
            #The data is not labeled, just plot one type of symbol
            self.label_list = [0]
            self.have_labels = False
        else:
            #The data is labeled, plot a different symbol for each category
            self.label_list = list(set(self.labels))
            self.have_labels = True
        
        #Init an empty dictionary
        self.data_dict = {}
        #init empty lists for each catetory and store in the dict
        for ind,label in enumerate(self.label_list):
            self.data_dict[label] = {}
            self.data_dict[label]['x'] = []
            self.data_dict[label]['y'] = []
            if not self.legend_text:
                self.data_dict[label]['legend_text'] = f'label_{ind}'
            else:
                self.data_dict[label]['legend_text'] = self.legend_text[label]
        
        #Loop through each training pattern
        for ind in self.X_mod.index:
            #Extract the data associated with the current training pattern
            cur_x = np.array(self.X_mod.loc[ind,self.data_cols_mod])
                    
            #Extract the label text
            label_txt = self.X_mod.loc[ind,self.label_col]
            #Find the winning node (the coordinates where the feature will be
            #plotted)
            self.find_winning(cur_x)
            #Determine if the text needs to be offset to avoid plotting over
            #a previous label
            method = 'grid_fraction'
            xytext = self.get_text_offset(used_inds,method)
            #Add the winning coordinates to the list of used coordinates
            used_inds.append(self.winning)
            #Add the label to the winning point on the map
            label = self.labels[ind]
            
            self.data_dict[label]['x'].append(self.winning[0]+xytext[0])
            self.data_dict[label]['y'].append(self.winning[1]+xytext[1])
            if print_coords:
                print("label: {} at {}".format(label_txt,self.winning))
            
        
    def get_text_offset(self,used_inds,method='points'):
        """
        Determines how far the text label should be offset based on the 
        number of items plotted to the current winning node.

        Parameters
        ----------
        used_inds : TYPE list
            DESCRIPTION.
            A list of the nodes where previous samples were plotted.  Each 
            element is a list containing x,y coordinates
        method : TYPE string, optional
            DESCRIPTION.
            'points' for how far text values shoudl be offset
            anything else for how far symbols should be offset and in what 
            direction

        Returns
        -------
        xytext : TYPE list
            DESCRIPTION.

        """
        #If the length of used_inds is greater than 0, at least one point was
        #previously plotted; check if the current point is an overlap
        if len(used_inds)>0:
            #Convert the used inds list to a numpy array.  This becomes a 2D
            #array with two columns (one for x and one for y) and a number of
            #rows equal to the number of previously-plotted labels
            temp = np.array(used_inds)
            #Generates an array of boolean values where each element in
            #the first column is either true or false based on whether or not
            #it matches the first index in self.winning.  The same is true for
            #the second column
            temp = temp == self.winning
            #np.all along axis 1 finds all the rows where all elements are true
            #(rows where both the x and y coordinates match).  The sum finds
            #the number of rows where both coordinates match - this is the 
            #number of times a label has been plotted to the current point
            used_prev = sum(np.all(temp,axis=1))
        else:
            #If there are no elements in used_ind, the current point was not 
            #used previously
            used_prev = 0
        #The number of points to offset each subsequent label
        if method == 'points':
            #Determine how far to offset text labels
            offset = 10
            
            xytext = (0,used_prev*offset)
            return xytext
        else:
            #Determine how far and in what direction to offset symbols
            fraction = 0.025
            yval = self.grid_size[1]
            xval = self.grid_size[0]
            yoffset = fraction*yval
            xoffset = fraction*xval
            #Store the total offset (the offset per previous label times the number
            #of previous labels) in a tuple
            dir_tpls = [
                (1,1),
                (1,0),
                (1,-1),
                (0,1),
                (0,-1),
                (-1,1),
                (-1,0),
                (-1,-1)]
            
            #If the node has not yet "won", set the x/y offset to zero
            if used_prev == 0:
                first = 0
            else:
                first = 1
               
            #determine the direction to offset
            ind = used_prev%8
            #determine the distance to offset
            mult = int((used_prev-1)/8)+1
            x_off = first*mult*dir_tpls[ind][0]*xoffset
            y_off = first*mult*dir_tpls[ind][1]*yoffset
            xytext = (x_off,y_off)
            
            return xytext
    
    def plot_weight_change(self):
        """
        Plot the change in weights, the change in neighborhood size, and the 
        change in the learning rate per epoch

        Returns
        -------
        None.

        """
        #Init a blank figure
        plt.figure()
        #plot the weight change, neighborhood size, and learning rage
        plt.plot(range(1,1+len(self.weight_change)),self.weight_change,label='sqrt of sum of squares of weight changes')
        plt.plot(range(1,1+len(self.weight_change)),self.NH_trace,label='Neighborhood size')
        plt.plot(range(1,1+len(self.weight_change)),self.LR_trace,label='10 times the learning rate')
        #Add a legend, grid, axis labels, and save the figure
        plt.legend()
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('value')
        plt.savefig('weight_change.png')

    def save_weights(self,directory,fn):
        """
        Save the weights in a pickle

        Parameters
        ----------
        directory : TYPE path string
            DESCRIPTION.
            The directory to store the weight sin
        fn : TYPE string
            DESCRIPTION.
            The name of the weights file

        Returns
        -------
        None.

        """
        #Open the pickle file
        with open(os.path.join(directory,fn),'wb') as file:
            #Store the grid, grid size, the working sample dataframe, the
            #list of data columns, the label column header, and the number of 
            #features in the pickle
            pickle.dump(self.grid,file)
            pickle.dump(self.grid_size,file)
            pickle.dump(self.X_mod,file)
            pickle.dump(self.data_cols_mod,file)
            pickle.dump(self.label_col,file)
            pickle.dump(self.num_features,file)
            pickle.dump(self.epoch,file)


    def load_weights(self,directory,fn):
        """
        Loads pre-trained weights and the associated data structures
        needed to print plots

        Parameters
        ----------
        directory : TYPE
            DESCRIPTION.
        fn : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        weights_file = os.path.join(directory,fn)
        with open(weights_file, 'rb') as file:
            self.grid = pickle.load(file)
            self.grid_size = pickle.load(file)
            self.X_mod = pickle.load(file)
            self.data_cols_mod = pickle.load(file)
            self.label_col = pickle.load(file)
            self.num_features = pickle.load(file)
            self.epoch = pickle.load(file)
            
    def plot_weight_hist(self,directory):
        """
        Plots the weight history.

        Parameters
        ----------
        directory : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        num_vals = self.grid.shape[0]
        
        hist_dir = os.path.join(directory,'weight_histograms')
        if not os.path.exists(hist_dir):
            os.makedirs(hist_dir)
        
        for i in range(num_vals):
            data = self.grid[i,:,:].flatten()
            
            plt.hist(data)
            plt.title('feature: {}'.format(self.data_cols_mod[i]))
            fn = 'weight_hist_{}.png'.format(self.data_cols_mod[i])
            fn = os.path.join(hist_dir,fn)
            plt.savefig(fn)
            plt.tight_layout()
            plt.close()
            
        
    def make_test_grid(self):
        """
        Builds a 10x10 grid of numbers between 1 and 100.  Used to test the 
        neighborhood code

        Returns
        -------
        None.

        """
        
        self.test_grid = np.array(range(1,101))
        self.test_grid = self.test_grid.reshape((1,10,10))
        
    def print_grid_range(self):
        """
        Prints the range of weight values in the node grid for each feature

        Returns
        -------
        None.

        """
        
        str_len = 16
        
        for i in range(self.grid.shape[0]-1):
            str_start = 'range of {}'.format(self.data_cols_mod[i]).rjust(str_len,' ')
            print('{}: {} to {}'.format(str_start,
                                                 np.round(np.max(self.grid[i,:,:]),4),
                                                 np.round(np.min(self.grid[i,:,:]),4)))
            
    def build_color_marker_lists(self):
        #Lists of colors and markers for plotting
        self.colors = ['c',
                       'gold',
                       'salmon',
                       'tab:orange',
                       'lime',
                       'teal',
                       'darkgreen',
                       'cornflowerblue',
                       'blue',
                       'navy',
                       'mediumpurple',
                       'darkviolet',
                       'magenta',
                       'darkmagenta',
                       'deeppink',
                       'hotpink',
                       'firebrick',
                       'peru',
                       'sienna',
                       'gray']
        self.markers = ['s',
                        'o',
                        'P',
                        'X',
                        'D',
                        4,
                        5,
                        6,
                        '*',
                        'H',
                        '1',
                        '2',
                        '3',
                        '4',
                        'd',
                        '+',
                        'x',
                        'p',
                        '.',
                        '$V$']
        
    def visualization_settings(self,output_dir,sample_vis,legend_text,include_D,labels,plane_vis):
        self.output_dir = output_dir
        self.sample_vis = sample_vis
        self.legend_text = legend_text
        self.include_D = include_D
        self.labels = labels
        self.plane_vis = plane_vis
        