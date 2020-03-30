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
parser.add_argment('-s', help="max training size in int")
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
max_size = int(args.s)
                  

# try:
#     os.mkdir("outputs")
# except:
#     "exists"
#
# try:
#     os.mkdir(out_dir)
# except:
#     "exists"

def onehot_encode(l, K):
    onehot = []
    for i in l:
        onehot_encoded = np.zeros(K)
        index = np.where(l == i)[0]#[0]
        onehot_encoded[int(i)] = 1
        onehot.append(onehot_encoded.tolist())
    return onehot

def build_training_set(data_path, batch_size, K):
    dataset = np.loadtxt(data_path)[0:max_size] # limit to 100 for testing
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


    unbatched_outputs = []
    for batch in outputs.tolist():
        for item in batch:
            unbatched_outputs.append(item)

    onehot = []
    for inner_list in onehot_outputs:
         onehot.append([item for sublist in inner_list for item in sublist])

    gc.collect()

    return unbatched_data, unbatched_outputs, onehot

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

        np.savetxt(output_path, generated_data, fmt='%1.0f')

        output_path_probs = directory + "/epoch_probs_{}.txt".format(epoch)

        onehot_probs = []
        for i in range(0, len(generated_probs), 4):
            onehot_probs.append([item for sublist in generated_probs[i:i + 4] for item in sublist])
#            onehot_probs.append([item for sublist in inner_list for item in sublist])
        np.savetxt(output_path_probs, onehot_probs, fmt='%1.4f')

        gc.collect()

lr = float(args.lr)
q = int(args.q)

try:
    os.mkdir("outputs_2")
except:
    "exists"

out_dir = "outputs_2/{}_qubits".format(q)

try:
    os.mkdir(out_dir)
except:
    "exists"

data = params['data'].format(q)

train_set = build_training_set(data, batch_size, K)

rnn = RNN_Class(K, q, latent_size, batch_size)

train_rnn(rnn, train_set, epochs, lr, out_dir)
