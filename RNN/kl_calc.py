import numpy as np
import torch.nn.functional as F
import torch
import argparse
import matplotlib
import matplotlib.pyplot as plt

epochs = 50

parser = argparse.ArgumentParser()
parser.add_argument('-q', help="qubits")
parser.add_argument('-lr', help="learning rate")

args = parser.parse_args()

q = int(args.q)
lr = float(args.lr)

data_path = "Data/{} Qubit/train.txt".format(q)

def onehot_encode(l, K):
    onehot = []
    for i in l:
        onehot_encoded = np.zeros(K)
        index = np.where(l == i)[0]#[0]
        onehot_encoded[int(i)] = 1
        onehot.append(onehot_encoded.tolist())
    return onehot

kls = []

epoch = list(range(1,epochs + 1))

for i in epoch:
    output_path = "outputs_50epochs/{}_qubits/lr_{}/epoch_probs_{}.txt".format(q, lr, i)

    dataset = np.loadtxt(data_path)

    for i, list in enumerate(dataset):
        for j, item in enumerate(list):
            if item == 0:
                dataset[i][j] = 1e-8

    model_data = np.loadtxt(output_path)

    for i, list in enumerate(model_data):
        for j, item in enumerate(list):
            if item == 0:
                model_data[i][j] = 1e-8

    model_tensor = torch.FloatTensor(model_data).reshape(-1, q, 4)

    data_tensor = torch.FloatTensor(dataset).reshape(-1, q, 4)

    kl = F.kl_div(model_tensor.log(), data_tensor) 

    kls.append(kl.item())

plt.plot(epoch,kls)
plt.xlabel('Number of Epochs')
plt.ylabel('KL Divergence')
plt.savefig('KLs/KL_{}Q_LR{}.pdf'.format(q, lr))

