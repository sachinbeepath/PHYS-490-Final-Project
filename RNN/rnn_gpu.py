import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import os

def get_default_device():
  if torch.cuda.is_available:
    return torch.device("cuda")
  else:
    return torch.device("cpu")

device = get_default_device()

def to_device(data, device):
  if isinstance(data, (list, tuple)):
    return [to_device(x, device) for x in data]
  return data.to(device, non_blocking=True)

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


def onehot_encode(l, K):
    onehot = []
    for i in l:
        onehot_encoded = np.zeros(K)
        index = np.where(l == i)[0]#[0]
        onehot_encoded[int(i)] = 1
        onehot.append(onehot_encoded.tolist())
    return onehot

def build_training_set(data_path, batch_size, K):
    dataset = np.loadtxt(data_path)[0:100] # limit to 100 for testing
    dataset_onehot = [onehot_encode(datapoint, K) for datapoint in dataset]
    train_set = [dataset_onehot[i:i + batch_size] for i in range(0, len(dataset_onehot), batch_size)]
    train_torch = [torch.Tensor(data) for data in train_set]
    return to_device(train_torch, device)

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
    h = rnn.initialize_hiddens()
    data = torch.Tensor(unbatched_data)
    outputs, _ = rnn.forward(data, h)
    _, max_indices = torch.max(outputs.data, -1)
    onehot_outputs = [onehot_encode(datapoint, K) for datapoint in max_indices.tolist()]

    onehot = [data[0] + data[1] for data in onehot_outputs]

    return unbatched_data, outputs, onehot

def train_rnn(rnn, train_data, epochs, learning_rate, out_dir):

    kl_loss = []

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
        total_loss = []
        for i, batch in enumerate(train_data):
            rnn.zero_grad()
            batch = torch.Tensor(batch)
            generated_output, h = rnn(batch, h)
            loss = criterion(generated_output.log(), batch)
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss.append(loss.item())
            print("Epoch {}, Batch {}".format(epoch, i))

        unbatched_data, generated_probs, generated_data = generate(rnn, train_data)
        kl_div = criterion(torch.Tensor(generated_probs).log(), torch.Tensor(unbatched_data))
        kl_loss.append(kl_div)
        print("Epoch {} complete, KL loss = {}".format(epoch, kl_div))

        output_path = directory + "/epoch_{}.txt".format(epoch)

        np.savetxt(output_path, generated_data, fmt='%1.0f')

    np.savetxt(directory + "/kl.txt", kl_loss)

    return kl_loss

lrate = [0.01, 0.001, 0.0001]

qubits = [2, 3, 4, 5]

for lr in lrate:
    for q in qubits:

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

        kl_loss = train_rnn(rnn, train_set, epochs, lr, out_dir)

        plt.plot(kl_loss)

        directory = out_dir + "/lr_{}".format(lr)

        plt.savefig(directory + "/KL_div.pdf")

        plt.close()
