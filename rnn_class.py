import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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
        index = np.where(l == i)[0]#[0]
        onehot_encoded[int(i)] = 1
        onehot.append(onehot_encoded.tolist())
    return onehot

def build_training_set(data_path, batch_size, K):
    dataset = np.loadtxt(data_path)[0:100] # limit to 100 for testing
    dataset_onehot = [onehot_encode(datapoint, K) for datapoint in dataset]
    print(dataset[0], onehot_encode(dataset[0], K))
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

def train_rnn(train_data, epochs, learning_rate):

    rnn = RNN_Class(K = K, num_qubits = num_qubits, latent_size = latent_size, batch_size = batchsize)

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
        print("Epoch {} complete, average loss = {}".format(epoch, sum(total_loss)/len(total_loss)))

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

    return outputs, onehot_outputs

train_set = build_training_set(data, batchsize, K)

rnn = RNN_Class(K = 4, num_qubits = 2, latent_size = 100, batch_size = 20)

train_rnn(train_set, 3, 0.002)

output, output_onehot = generate(rnn, train_set)
print(output, output_onehot)
