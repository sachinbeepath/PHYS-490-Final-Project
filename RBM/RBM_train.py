import json, argparse, torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt
import numpy as np
import random
from RBM_class import RBM
import os

def get_unique(data):
    if not isinstance(data, list):
        data_list = data.tolist()
    else:
        data_list = data
    dres = [] 
    for i in data_list: 
        if i not in dres: 
            dres.append(i) 
    return dres

def get_freq(data,dres):
    if not isinstance(data, list):
        data_list = data.tolist()
    else:
        data_list = data
    prob = []
    for i in dres:
        p = data_list.count(i)/len(data_list)
        if p == 0:
            prob.append(1e-8)
        else:
            prob.append(p)
    return prob

def kl(data,model_data):
    dres = get_unique(data)
    prob = get_freq(data,dres)
    mp = get_freq(model_data,dres)
    kl_div = np.sum([prob[i] * np.log(prob[i]/mp[i]) for i in range(len(prob))])
    return kl_div
    
### Function to save weights and biases to a parameter file ###
def save_parameters(rbm,epoch):
    weights, visible_bias, hidden_bias = [rbm.weights, rbm.visible_bias, rbm.hidden_bias]
    parameter_dir = 'RBM_parameters'
    if not(os.path.isdir(parameter_dir)):
        os.mkdir(parameter_dir)
    parameter_file_path = '{}/parameters_nH{}_q{}_p{}_epoch{}'.format(parameter_dir,num_hidden_nodes,q,str(p),str(epoch))
    np.savez_compressed(parameter_file_path, weights=weights.detach(), visible_bias=visible_bias.detach(), hidden_bias=hidden_bias.detach())

# Input parameters:
with open('param.json') as paramfile:
    param = json.load(paramfile)
p = 0
q = param['q']  # number of qubits
num_visible_nodes = q  # number of visible nodes
num_visible_states = param['num_visible_states']  # number of visible states (4 for the tetrahedral POVM)
num_hidden_nodes = param['num_hidden_nodes']  # number of hidden nodes
num_hidden_states = param['num_hidden_states']  # number of states (binary)        
num_train = param['num_train'] # number of training steps
lr = param['lr']  # learning rate 
batch_size = param['batch_size']# batch size
num_gibbs = param['num_gibbs']# number of Gibbs iterations (steps of contrastive divergence)
num_samples = batch_size
file_name = param['file_name']
out_dir = "output/" + str(q) + "_qubit"

# load data
data_train = np.loadtxt(param['file_name'])
it_per_epoch = data_train.shape[0]/batch_size

# Initialize the RBM 
model = RBM(num_visible_nodes, num_hidden_nodes, num_visible_states, num_hidden_states, num_samples)
batch_count = 0  
epoch = 0
num_iter = 0
optimizer = optim.Adam(model.parameters(), lr)
model.train()
kl_div = []
loss_list = []

data = data_train
for i in range(1,num_train+1):
    #check if entire dataset has been used to train
    if (batch_count * batch_size + batch_size) > data_train.shape[0]:
        batch_count = 0
        num_iter = 0
        data = np.random.permutation(data_train) # randomize data

    #load batch
    batch =  torch.from_numpy(data[batch_count*batch_size: batch_count*batch_size+ batch_size,:])
    
    #train on batch
    loss = model.update_weights(batch,num_gibbs=num_gibbs,lr=lr,samples=batch)
    loss_list.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    batch_count += 1
    num_iter += 1

    if num_iter % (it_per_epoch) == 0:
        model_data,_ = model.CD_sample(num_gibbs,torch.from_numpy(data),nsamples = data.shape[0])
        model_data = model_data.detach().numpy()
        kl_div.append(kl(data,model_data))
        epoch += 1
        if not(os.path.isdir(out_dir)):
            os.mkdir(out_dir)
        outfile = out_dir + "/epoch_" + str(epoch)+ ".txt"
        np.savetxt(outfile,model_data,fmt='%1.0f')
        print ('Epoch =',epoch)
        save_parameters(model,epoch)

plt.plot(list(range(1,len(kl_div)+1)),kl_div)
plt.grid()
plt.xlabel("Epoch")
plt.ylabel("KL Divergence")
plt.savefig(out_dir + '/KL_div.pdf')
plt.close()
