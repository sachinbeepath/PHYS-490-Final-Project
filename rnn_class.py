import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader

data = "/Users/yilda/Documents/PHYS490/PHYS-490-Final-Project/Data/2 Qubit/data.txt"

def onehot_encode(l, K, q):
    onehot = np.zeros(K*q)
    for i in l:
        index = np.where(l == i)[0][0]
        onehot[int(i)-1 + index*K] = 1
    return onehot

class RNN_Class(nn.Module):
    def __init__(self, data_path = data, K = 3, num_qubits = 2, latent_size = 100, batch_size = 20):
        super(RNN_Class, self).__init__()
        self.charset_length = K
        self.num_qubits = num_qubits
        self.batch_size = batch_size
        dataset = np.loadtxt(data_path)[0:100] # limit to 100 for testing
        dataset_onehot = [onehot_encode(datapoint, self.charset_length, self.num_qubits) for datapoint in dataset]
        train_set = [dataset_onehot[i:i + batch_size] for i in range(0, len(dataset_onehot), batch_size)]
        self.training_data = torch.Tensor(train_set)

        self.latent_size = latent_size
        self.num_samples = self.training_data.shape[0]

        # this part doesn't need to be defined up here yet

        # self.POVM_measurement = torch.zeros([batch_size, self.num_qubits*self.charset_length], dtype=torch.float32)
        # self.molecules = torch.zeros([batch_size, self.num_qubits, self.charset_length], dtype=torch.float32)
        #
        # self.generated_molecules = self.generation(decoder, self.molecules)
        #
        # self.sample_onehot = self.generation(decoder)
        #
        #self.sample_RNN = torch.argmax(self.sample_onehot, axis=2)
        #
        # self.generation_loss = -torch.sum(self.molecules * torch.log(1e-10 + self.generated_molecules), [1, 2])
        #
        # self.cost = torch.mean(self.generation_loss)

        self.GRU_layers = nn.GRU(self.num_qubits*self.charset_length, latent_size, 3)
        self.output_layers = nn.Sequential(
            nn.Linear(self.latent_size, self.num_qubits*self.charset_length),
            nn.Softmax(dim=2),
        )

    def generation_time_distributed(self, input):
        logP = torch.zeros([self.batch_size], dtype=torch.float32)
        gru_output, state = self.GRU_layers(input)
        output = self.output_layers(gru_output)
        return output, state

rnn = RNN_Class(data, K = 4, num_qubits = 2, latent_size = 100, batch_size = 20)

output, h = rnn.generation_time_distributed(rnn.training_data)

print(output)
print(h)
