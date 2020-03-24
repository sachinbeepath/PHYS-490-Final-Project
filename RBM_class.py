import json, argparse, torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt
import numpy as np
import random


class RBM(object):
    
    def __init__(self, num_visible_nodes, num_hidden_nodes, num_visible_states, num_hidden_states, 
                 num_samples, weights, visible_bias, hidden_bias):
        
        self.num_samples=num_samples 
        
        self.num_visible_nodes = num_visible_nodes #number of visible units
        self.num_visible_states = num_visible_states # number of states of the visible units
        self.num_hidden_nodes = num_hidden_nodes   #number of hidden units
        self.num_hidden_states = num_hidden_states # number of states of the hidden units

        #visible bias:
        self.visible_bias = torch.zeros(self.num_visible_nodes*self.num_visible_states, 1)

        #hidden bias:
        self.hidden_bias = torch.zeros(self.num_hidden_nodes*self.num_hidden_states, 1)

        #weights:
        self.weights = torch.normal(0, 0.05, size=(self.num_visible_nodes*self.num_visible_states, 
                                                   self.num_hidden_nodes*self.num_hidden_states))
        

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
    
    ### old sampling
    '''
    def generate_h_samples(v):
        pv = torch.matmul(v.float(), self.weights) + torch.t(self.hidden_bias)
        v_r = pv.view(self.num_samples*self.num_hidden_nodes,self.num_hidden_states)
        sig = torch.distributions.multinomial.Multinomial(1,logits=v_r)
        return torch.reshape(sig.sample(),(self.num_samples,self.num_hidden_states*self.num_hidden_nodes))  

    ### Generate visible samples given hidden samples
    def sample_v_given(self, h):
        ph = torch.matmul(h.float(), torch.t(self.weights)) + torch.t(self.visible_bias)
        h_r = ph.view(self.num_samples*self.num_visible_nodes,self.num_visible_states)
        sig = torch.distributions.multinomial.Multinomial(1,logits=h_r)
        return torch.reshape(sig.sample(),(self.num_samples,self.num_visible_states*self.num_visible_nodes))  

    '''
    
    def generate_h_samples(self,v):
        a = torch.matmul(v.float(), self.weights) + torch.t(self.hidden_bias)
        p = torch.sigmoid(a.view(self.num_samples*self.num_hidden_nodes,self.num_hidden_states))
        sig = torch.distributions.one_hot_categorical.OneHotCategorical(probs=p)
        samples = sig.sample()
        return samples.view(self.num_samples,self.num_hidden_states*self.num_hidden_nodes)
    
    def generate_v_samples(self,h):
        a = torch.matmul(h.float(), torch.t(self.weights)) + torch.t(self.visible_bias)
        p = torch.sigmoid(a.view(self.num_samples*self.num_visible_nodes,self.num_visible_states))
        sig = torch.distributions.one_hot_categorical.OneHotCategorical(probs=p)
        samples = sig.sample()
        return samples.view(self.num_samples,self.num_visible_states*self.num_visible_nodes)
                         
    def CD_sample(self, num_iterations):
        v = self.visible_samples

        for i in range(num_iterations):
            h = self.generate_h_samples(v)
            v = self.generate_v_samples(h)

        self.hidden_samples = h
        self.visible_samples = v

        return v, h
    def energy(self,v,h):
        wvh = torch.sum(torch.matmul(v,self.weights))
        vb = torch.matmul(v,self.visible_bias)
        hb = torch.matmul(h,self.hidden_bias)
        return -wvbh - torch.t(vb) - torch.t(hb) 
                                    
    def update_weights(self, data_visible, num_gibbs=2,lr=1e-3):
        data_hidden = self.sample_h_given(data_visible)
        data_mean = torch.mean(self.energy(data_visible, data_hidden))
        model_visible, model_hidden = self.CD_sample(num_gibbs)
        model_mean = torch.mean(self.energy(model_hidden, model_visible))
        self.weights += lr*(data_mean-data_model)
