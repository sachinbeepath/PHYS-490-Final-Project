import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import gc
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import os
import argparse
import scipy
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('-q', help="qubits")
parser.add_argument('-lr', help="learning rate")

args = parser.parse_args()

with open('param.json') as file:
    params = json.load(file)

K = params['K']
#num_qubits = params['q']
latent_size = params['latent_size']
batch_size = params['batch_size']
#data = params['data'].format(num_qubits)
epochs = params['epochs']
#lr = params['lr']

# try:
#     os.mkdir("outputs")
# except:
#     "exists"
#
# try:
#     os.mkdir(out_dir)
# except:
#     "exists"

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
    print(prob)
    return prob

def kl(data,model_data):
    unbatched_data = []
    unbatched_model_data = []
    for batch in data:
        for item in batch:
            unbatched_data.append(item)
    for batch in model_data:
        for item in batch:
            unbatched_model_data.append(item)
    dres = get_unique(data)
    prob = get_freq(data,dres)
    mp = get_freq(model_data,dres)
    kl_div = np.sum([prob[i] * np.log(prob[i]/mp[i]) for i in range(len(prob))])
    return kl_div

def onehot_encode(l, K):
    onehot = []
    for i in l:
        onehot_encoded = np.zeros(K)
        index = np.where(l == i)[0]#[0]
        onehot_encoded[int(i)] = 1
        onehot.append(onehot_encoded.tolist())
    return onehot

def build_training_set(data_path, batch_size, K):
    dataset = np.loadtxt(data_path) # limit to 100 for testing
    dataset_onehot = [onehot_encode(datapoint, K) for datapoint in dataset]
    train_set = [dataset_onehot[i:i + batch_size] for i in range(0, len(dataset_onehot), batch_size)]

    return train_set

class RNN_Class(nn.Module):
    def __init__(self, K = 4, num_qubits = 2, latent_size = 100, batch_size = 20):
        super(RNN_Class, self).__init__()
        self.charset_length = K
        self.num_qubits = num_qubits
        self.batch_size = batch_size

        self.latent_size = latent_size

        self.GRU_layers = nn.GRU(self.charset_length, latent_size, 3)
        self.output_layers = nn.Sequential(
            nn.Linear(self.latent_size, self.charset_length),
            nn.Softmax(dim=2),
        )

    def forward(self, input, h):

        gru_output, h_i = self.GRU_layers(input, h)
        output = self.output_layers(gru_output)
        return output, h_i

    def initialize_hiddens(self):
        weight = next(self.parameters()).data
        h = weight.new(3, self.num_qubits, self.latent_size).zero_()
        return h

def generate(rnn, batched_data):
    unbatched_data = []
    for batch in batched_data:
        for item in batch:
            unbatched_data.append(item)
    h = rnn.initialize_hiddens().data
    data = torch.Tensor(unbatched_data)
    outputs, _ = rnn(data, h)
    _, max_indices = torch.max(outputs.data, -1)
    onehot_outputs = [onehot_encode(datapoint, K) for datapoint in max_indices.tolist()]

    onehot = [data[0] + data[1] for data in onehot_outputs]

    gc.collect()

    return unbatched_data, outputs, onehot

def train_rnn(rnn, train_data, epochs, learning_rate, out_dir):

    directory = out_dir + "/lr_{}".format(learning_rate)

    try:
        os.mkdir(directory)
    except:
        "exists"

    optimizer = torch.optim.Adam(rnn.parameters(), learning_rate)
    criterion = nn.KLDivLoss(size_average=False) #input this for now, may change

    rnn.train()
    print("Training Starts")
    for epoch in range(1, epochs+1):
        h = rnn.initialize_hiddens().data
        for i, batch in enumerate(train_data):
            rnn.zero_grad()
            batch = torch.Tensor(batch)
            generated_output, h = rnn(batch, h.detach())
            loss = criterion(generated_output.log(), batch)
            loss.backward(retain_graph=True)
            optimizer.step()
            print("Epoch {}, Batch {}".format(epoch, i))

        unbatched_data, generated_probs, generated_data = generate(rnn, train_data)

        print("Epoch {} complete".format(epoch))

        output_path = directory + "/epoch_{}.txt".format(epoch)

        np.savez_compressed(output_path, generated_data)

        gc.collect()

lr = float(args.lr)
q = int(args.q)

try:
    os.mkdir("outputs")
except:
    "exists"

out_dir = "outputs/{}_qubits".format(q)

try:
    os.mkdir(out_dir)
except:
    "exists"

data = params['data'].format(q)

train_set = build_training_set(data, batch_size, K)

rnn = RNN_Class(K, q, latent_size, batch_size)

train_rnn(rnn, train_set, epochs, lr, out_dir)
