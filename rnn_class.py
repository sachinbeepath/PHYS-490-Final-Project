import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader

data = "/Users/yilda/Documents/PHYS490/PHYS-490-Final-Project/Data/2 Qubit/data.txt"

K = 4
num_qubits = 2
latent_size = 100
batchsize = 20
data = "/Users/yilda/Documents/PHYS490/PHYS-490-Final-Project/Data/2 Qubit/data.txt"


def onehot_encode(l, K):
    onehot = []
    for i in l:
        onehot_encoded = np.zeros(K)
        index = np.where(l == i)[0][0]
        onehot_encoded[int(i)] = 1
        onehot.append(onehot_encoded)
    return onehot

def build_dataloader(data_path, batch_size, K):
    dataset = np.loadtxt(data_path)[0:100] # limit to 100 for testing
    dataset_onehot = [onehot_encode(datapoint, K) for datapoint in dataset]
    train_set = [dataset_onehot[i:i + batch_size] for i in range(0, len(dataset_onehot), batch_size)]

    train_dataset = TensorDataset(torch.FloatTensor(train_set))
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size = batchsize)

    return train_loader

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

        #h = torch.zeros(self.charset_length, input.size(0), self.latent_size)

        gru_output, h_i = self.GRU_layers(input, h)

        output = self.output_layers(gru_output)
        return output, h_i

    def initialize_hiddens(self):
        weight = next(self.parameters()).data
        h = weight.new(3, self.batch_size, self.latent_size).zero_()
        return h

def train_rnn(train_dataloader, epochs, learning_rate):

    rnn = RNN_Class(K = K, num_qubits = num_qubits, latent_size = latent_size, batch_size = batchsize)

    optimizer = torch.optim.Adam(rnn.parameters(), learning_rate)
    criterion = nn.KLDivLoss() #input this for now, may change

    rnn.train()
    print("Training Starts")
    for epoch in range(1, epochs+1):
        h = rnn.initialize_hiddens().data
        total_loss = []
        for i, batch in enumerate(train_dataloader):

            rnn.zero_grad()
            generated_output = rnn(batch, h)
            loss = criterion(generated_output, batch)
            loss.backwards()
            optimizer.step()
            total_loss.append(loss.item())
        print("Epoch {} complete, average loss = {}".format(epoch + 1, sum(total_loss)/len(total_loss)))

train_dataloader = build_dataloader(data, batchsize, K)

rnn = RNN_Class(K = 4, num_qubits = 2, latent_size = 100, batch_size = 20)

train_rnn(train_dataloader, 3, 0.002)

#output = rnn.generation(train_dataloader)

#print(output)
#print(output.size())
