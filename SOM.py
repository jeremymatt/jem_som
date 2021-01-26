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


def min_max_norm(df,data_labels):
    for key in data_labels:
        min_val = min(df[key])
        max_val = max(df[key])
        
        df[key] = (df[key]-min_val)/(max_val-min_val)
        
        
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
    


class SOM:
    def __init__(self,grid_size,X,label_col,alpha,neighborhood_size):
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

        Returns
        -------
        None.

        """
        #Store the grid size and the label column in self
        self.grid_size = grid_size
        self.label_col = label_col
        
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
        
        #Add the dimension required by cosine similarity
        self.add_dimension()
        
        #Initialize a matrix of random numbers in [-0.1, 0.1]
        self.grid = 0.1*(np.random.rand(self.num_features,grid_size[0],grid_size[1])-0.5)
        self.grid_updates = np.zeros([self.num_features,grid_size[0],grid_size[1]])
        if self.print_weight_range:
            print('\nRANDOM INITIALIZATION')
            self.print_grid_range()
        # self.grid = np.random.rand(self.num_features,grid_size[0],grid_size[1])
        #Determine the mean of each of the normalized data features (including
        #the D column)
        x_mean = self.X_mod[self.data_cols_mod].mean()
        #For each feature, shift the grid values around the mean of that 
        #feature  The self.grid variable now contains the starting random
        #weights
        for ind in range(self.num_features):
            self.grid[ind,:,:] = x_mean[ind]+x_mean[ind]*self.grid[ind,:,:]
            
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
        
    def print_grid_range(self):
        
        str_len = 16
        
        for i in range(self.grid.shape[0]-1):
            str_start = 'range of {}'.format(self.data_cols_mod[i]).rjust(str_len,' ')
            print('{}: {} to {}'.format(str_start,
                                                 np.round(np.max(self.grid[i,:,:]),4),
                                                 np.round(np.min(self.grid[i,:,:]),4)))
        
    def train(self,num_epochs):
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
        
        with open('winning.txt','w') as self.f:
            
            for epoch in range(num_epochs):
                self.epoch = epoch
                
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
                    self.write_winning = True
                    self.find_winning(cur_x)
                    # self.winning = [8,8]
                    if self.winning[0]>max_inds[0]:
                        max_inds[0] = self.winning[0]
                    if self.winning[1]>max_inds[1]:
                        max_inds[1] = self.winning[1]
                    self.write_winning = False
                    #Find the neighborhood indices around the winner
                    self.get_neighborhood()
                    #Update the weights of the nodes within the neighborhood
                    self.update_weights(cur_x)
                    self.ctr+=1
                    
                if self.print_weight_range:
                    print('\nAfter weight update for epoch {}'.format(epoch))
                    self.print_grid_range()
                    
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
        if self.neighborhood_size<1:
            self.neighborhood_size = 1
        
        
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
        #Extract the portion of the weights that need to be updated
        to_update = self.grid[:,self.axis0_start:self.axis0_end,self.axis1_start:self.axis1_end]
        
        #For each feature, subtract the weights associated with the feature
        #from the feature value, multiply the difference by the current 
        #learning rate and add the result to the weighs in the temporary
        #variable holding the weigths to be updated
        for ind,val in enumerate(cur_x):
            to_update[ind,:,:] += self.alpha*(cur_x[ind]-to_update[ind,:,:])
            
            
         
        #Insert the updated weights into the whole weight matrix
        self.grid[:,self.axis0_start:self.axis0_end,self.axis1_start:self.axis1_end] = to_update
        
        self.grid_updates[:,self.axis0_start:self.axis0_end,self.axis1_start:self.axis1_end] += to_update*0+1
        
        
        
    def find_winning(self,cur_x):
        """
        Finds the winning node for the current input

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
        neighborhood boundaries are not toroidal

        Returns
        -------
        None.

        """
        #Find the lower index of Axis 0 and ensure it is not less than 0
        self.axis0_start = self.winning[0]-self.neighborhood_size
        if self.axis0_start<0:
            self.axis0_start = 0
        #Find the upper index of Axis 0 and ensure that it is not greater than
        #the maximum Axis 0 dimension
        self.axis0_end = self.winning[0]+self.neighborhood_size+1
        if self.axis0_end>self.grid.shape[1]:
            self.axis0_end = self.grid.shape[1]
            
           
        #Find the lower index of Axis 1 and ensure it is not less than 0 
        self.axis1_start = self.winning[1]-self.neighborhood_size
        if self.axis1_start<0:
            self.axis1_start = 0
        #Find the upper index of Axis 1 and ensure that it is not greater than
        #the maximum Axis 1 dimension
        self.axis1_end = self.winning[1]+self.neighborhood_size+1
        if self.axis1_end>self.grid.shape[2]:
            self.axis1_end = self.grid.shape[2]
            
            
                
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
                neighborhood = self.grid[:,self.axis0_start:self.axis0_end,self.axis1_start:self.axis1_end]
                
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
        
        wh_ratio = self.grid_size[0]/self.grid_size[1]
        
        y_size = 15
        x_size = wh_ratio*y_size
        
        self.fig,self.ax = plt.subplots(figsize=(x_size,y_size))
        x_adjust = max([self.grid_size[0]*.05,0.25])
        y_adjust = max([self.grid_size[1]*.05,0.25])
        xlim = [0-x_adjust,self.grid_size[0]-1+x_adjust]
        ylim = [0-y_adjust,self.grid_size[1]-1+y_adjust]
        # print('fig limits:\n  X: {}\n  Y: {}'.format(xlim,ylim))
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])  
        self.marker_size =  min([x_size,y_size])*20   

        self.text_size = min([x_size,y_size])   

    def add_heatmap(self,grid,colormap,cbar_label=None):
        im = plt.imshow(grid,cmap=colormap,origin='lower')
        if cbar_label:
            cbar = self.ax.figure.colorbar(im,ax=self.ax)
            cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", size=self.text_size*1.25)
            cbar.ax.tick_params(labelsize=self.text_size*.9)
        else:
            print(cbar_label)
        
                
    def plot_u_matrix(self,include_D=False,output_dir=None,labels=None,sample_vis='labels'):
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
        
        # wh_ratio = self.grid_size[0]/self.grid_size[1]
        
        # y_size = 15
        # x_size = wh_ratio*y_size
        
        # self.fig,self.ax = plt.subplots(figsize=(x_size,y_size))
        # x_adjust = self.grid_size[0]*.1
        # y_adjust = self.grid_size[1]*.1
        # self.ax.set_xlim([0-x_adjust,self.grid_size[0]+x_adjust])
        # self.ax.set_ylim([0-y_adjust,self.grid_size[0]+y_adjust])
        # self.ax.get_xaxis().set_ticks([])
        # self.ax.get_yaxis().set_ticks([])
        
        self.open_new_fig()
        
        
        self.labels = labels
        #plot the sample labels
        if sample_vis == 'labels':
            self.plot_samples(False)
        else:
            self.plot_samples_symbols(False)
        
        if include_D:
            #Take the sum of the u matrix along the 0th axis; this sums the 
            #differences for each feature for each node.
            grid = np.sum(self.u_matrix,axis=0).T
        else:
            #Take the sum of the u matrix along the 0th axis; this sums the 
            #differences for each feature for each node, excluding D.
            grid = np.sum(self.u_matrix[:-1,:,:],axis=0).T
        
        #Add the matrix to the figure (transpose to align the values correctly)
        colormap = 'gray_r'
        cbar_label = 'Weight Change'
        cbar_label = None
        self.add_heatmap(grid, colormap,cbar_label)
        #save the figure to file
        plt.tight_layout()
        if output_dir == None:
            plt.savefig('u_matrix.png')
        else:
            plt.savefig(os.path.join(output_dir,'u_matrix.png'))
            
        plt.close(self.fig)
        
    def plot_feature_planes(self,output_dir=None,labels=None,sample_vis='labels',plane_vis='u_matrix'):
        """
        Plots each of the feature planes, including the plane associated with
        the D feature

        Returns
        -------
        None.

        """
        self.labels = labels
        #For each feature in the training data set
        for i in range(len(self.data_cols_mod)):
            
            self.open_new_fig()
            
            #Plot the sample labels
            if sample_vis == 'labels':
                self.plot_samples(False)
            else:
                self.plot_samples_symbols(False)
            
            if plane_vis == 'u_matrix':
                #extract the feature plan from the feature plane matrix, 
                #transposing to make the data line up
                grid = self.u_matrix[i,:,:].T
                #Add the feature plane to the figure
                colormap = 'gray_r'
                cbar_label = 'Weight Change'
                self.add_heatmap(grid, colormap,cbar_label)
            else:
                grid = self.grid[i,:,:].T
                colormap = 'viridis'
                cbar_label = 'Weight Value'
                self.add_heatmap(grid, colormap,cbar_label)
                # im = plt.imshow(grid,cmap='viridis',origin='lower')
                # cbar = self.ax.figure.colorbar(im,ax=self.ax)
                # cbar.ax.set_ylabel('Weight values', rotation=-90, va="bottom", size=self.text_size*1.25)
                # cbar.ax.tick_params(labelsize=self.text_size*.9)
                
            #Add a plot title
            plt.title('Feature: {}'.format(self.data_cols_mod[i]),size = self.text_size*1.5)
            #Save the figure to file
            plt.tight_layout()
            if output_dir == None:
                plt.savefig('feature_plane-{}.png'.format(self.data_cols_mod[i]))
            else:
                plt.savefig(os.path.join(output_dir,'feature_plane-{}.png'.format(self.data_cols_mod[i])))
                    
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
        
        colors = ['c','gold','tomato','tab:orange','lime']
        #Initialize a figure and set the x and y limits to the grid extents
        # plt.figure(figsize = [5,5])
        # plt.ylim(-5,self.grid_size[0]+5)
        # plt.xlim(-5,self.grid_size[1]+5)
        
        #Initialize a list to store the indices where a feature was plotted
        used_inds = []
        
        #Loop through each training pattern
        for ind in self.X_mod.index:
            #Extract the data associated with the current training pattern
            try:
                cur_x = np.array(self.X_mod.loc[ind,self.data_cols_mod])
            except:
                breakhere=1
                    
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
                bkg_color = colors[self.labels[ind]]
                self.ax.annotate(label_txt,[self.winning[0],self.winning[1]],xytext = xytext, backgroundcolor=bkg_color,textcoords='offset points')
                
            if print_coords:
                print("label: {} at {}".format(label_txt,self.winning))        
                
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
        
        colors = ['c','gold','tomato','tab:orange','lime']
        markers = ['s','o','P','X','D']
        #Initialize a figure and set the x and y limits to the grid extents
        # plt.figure(figsize = [5,5])
        # plt.ylim(0-1,self.grid_size[0])
        # plt.xlim(0-1,self.grid_size[1])
        
        #Initialize a list to store the indices where a feature was plotted
        used_inds = []
        
        #Loop through each training pattern
        for ind in self.X_mod.index:
            #Extract the data associated with the current training pattern
            try:
                cur_x = np.array(self.X_mod.loc[ind,self.data_cols_mod])
            except:
                breakhere=1
                    
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
            if np.all(self.labels == None):
                self.ax.scatter(self.winning[0]+xytext[0],self.winning[1]+xytext[1],c=colors[0])
            else:
                color = colors[self.labels[ind]]
                marker = markers[self.labels[ind]]
                self.ax.scatter(self.winning[0]+xytext[0],
                                self.winning[1]+xytext[1],
                                c=color,
                                marker=marker,
                                s = self.marker_size,
                                edgecolors= "black",
                                linewidth=self.marker_size/150)
                
            # ax = plt.gca()
            # ylim = ax.get_ylim
            # xlim = ax.get_xlim
            # ax.set_ylim([ylim[0]-1,ylim[1]])
            # ax.set_xlim([xlim[0]-1,xlim[1]])
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
            offset = 10
            
            xytext = (0,used_prev*offset)
            return xytext
        else:
            fraction = 0.025
            yval = self.grid_size[1]
            xval = self.grid_size[0]
            yoffset = fraction*yval
            xoffset = fraction*yval
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
            
            if used_prev == 0:
                first = 0
            else:
                first = 1
                
            ind = used_prev%8
            mult = int((used_prev-1)/8)+1
            x_off = first*mult*dir_tpls[ind][0]*xoffset
            y_off = first*mult*dir_tpls[ind][1]*yoffset
            xytext = (x_off,y_off)
            
            # print('xytext:{}'.format(xytext))
            
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
        with open(os.path.join(directory,fn),'wb') as file:
            pickle.dump(self.grid,file)
            pickle.dump(self.grid_size,file)
            pickle.dump(self.X_mod,file)
            pickle.dump(self.data_cols_mod,file)
            pickle.dump(self.label_col,file)
            pickle.dump(self.num_features,file)


    def load_weights(self,directory,fn):
        weights_file = os.path.join(directory,fn)
        self.write_winning = False
        with open(weights_file, 'rb') as file:
            self.grid = pickle.load(file)
            self.grid_size = pickle.load(file)
            self.X_mod = pickle.load(file)
            self.data_cols_mod = pickle.load(file)
            self.label_col = pickle.load(file)
            self.num_features = pickle.load(file)
            
    def plot_weight_hist(self,directory):
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
            
            

        