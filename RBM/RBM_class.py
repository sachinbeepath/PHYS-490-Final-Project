import json, argparse, torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt
import numpy as np
import random


class RBM(nn.Module):
    
    def __init__(self, num_visible_nodes, num_hidden_nodes, num_visible_states, num_hidden_states, 
                 num_samples):
        super(RBM, self).__init__()
        
        self.num_samples=num_samples 
        
        self.num_visible_nodes = num_visible_nodes #number of visible units
        self.num_visible_states = num_visible_states # number of states of the visible units
        self.num_hidden_nodes = num_hidden_nodes   #number of hidden units
        self.num_hidden_states = num_hidden_states # number of states of the hidden units

        #visible bias:
        self.visible_bias = nn.Parameter(torch.zeros(self.num_visible_nodes*self.num_visible_states, 1))

        #hidden bias:
        self.hidden_bias = nn.Parameter(torch.zeros(self.num_hidden_nodes*self.num_hidden_states, 1))

        #weights:
        self.weights = nn.Parameter(torch.normal(0, 0.05, size=(self.num_visible_nodes*self.num_visible_states, 
                                                   self.num_hidden_nodes*self.num_hidden_states)))
        

        # randomly initialize visible units for sampling
        v = np.random.uniform(0,self.num_visible_states,size=self.num_visible_nodes*self.num_samples).astype(np.int32)
        v_one_hot = np.zeros((self.num_visible_nodes*self.num_samples, self.num_visible_states))
        v_one_hot[np.arange(v.size),v] = 1

        self.visible_samples = torch.from_numpy(np.reshape(v_one_hot,(self.num_samples,self.num_visible_states*self.num_visible_nodes)))


        
        # randomly initialize hidden units for sampling
        h = np.random.uniform(0,self.num_hidden_states,size=self.num_hidden_nodes*self.num_samples).astype(np.int32)
        h_one_hot = np.zeros((self.num_hidden_nodes*self.num_samples, self.num_hidden_states))
        h_one_hot[np.arange(h.size),h] = 1
        
        self.hidden_samples = torch.from_numpy(np.reshape(h_one_hot,(self.num_samples,self.num_hidden_states*self.num_hidden_nodes)))
        

    ### Generate hidden samples given visible samples

    
    def generate_h_samples(self,v,nsamples=None):
        if nsamples == None:
            nsamples = self.num_samples
            
        a = torch.matmul(v.float(), self.weights) + torch.t(self.hidden_bias)
        p = torch.sigmoid(a.view(nsamples*self.num_hidden_nodes,self.num_hidden_states))
        sig = torch.distributions.one_hot_categorical.OneHotCategorical(probs=p)
        samples = sig.sample()
        return samples.view(nsamples,self.num_hidden_states*self.num_hidden_nodes)
    
    def generate_v_samples(self,h,nsamples=None):
        if nsamples == None:
            nsamples = self.num_samples
            
        a = torch.matmul(h.float(), torch.t(self.weights)) + torch.t(self.visible_bias)
        p = torch.sigmoid(a.view(nsamples*self.num_visible_nodes,self.num_visible_states))
        sig = torch.distributions.one_hot_categorical.OneHotCategorical(probs=p)
        samples = sig.sample()
        return samples.view(nsamples,self.num_visible_states*self.num_visible_nodes)
                         
    def CD_sample(self, num_iterations,samples,nsamples=None):
        if samples is None:
            v = self.visible_samples
        else:
            v = samples

        for i in range(num_iterations):
            h = self.generate_h_samples(v,nsamples)
            v = self.generate_v_samples(h,nsamples)

        self.hidden_samples = h
        self.visible_samples = v

        return v, h
    def energy(self,v,h):
        wvh = torch.sum(torch.matmul(v.float(),self.weights) * h,axis=1)
        vb = torch.t(torch.matmul(v.float(),self.visible_bias))[0]
        hb = torch.t(torch.matmul(h,self.hidden_bias))[0]
        return -wvh - vb - hb      
                                    
    def update_weights(self, data_visible, num_gibbs=2,lr=1e-3,samples=None):
        data_hidden = self.generate_h_samples(data_visible)
        data_mean = torch.mean(self.energy(data_visible, data_hidden))
        model_visible, model_hidden = self.CD_sample(num_gibbs,samples)
        model_mean = torch.mean(self.energy(model_visible,model_hidden))
        #self.weights += lr*(data_mean-model_mean)
        return data_mean-model_mean
