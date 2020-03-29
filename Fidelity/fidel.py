from math import sqrt
import matplotlib.pyplot as plt

# Function to calculate classical fidelity
def fid(data = 'C:\\Users\\antch\\Downloads\\PHYS-490-Final-Project-master\\RBM\\data\\2_qubit_train.txt', 
        gen = 'C:\\Users\\antch\\Downloads\\PHYS-490-Final-Project-master\\RBM\\output\\2_qubit\\epoch_50.txt',  
        POVM = 'Tetra', Nq = 2, p = 0.0, Ns = 6e4, epochs = 50):
    
    # Read in bell state data storing probability of each measurement
    try:
        f1 = open(data, 'r')
    except:
        print ('ERROR: No data file found')
        return 0
    
    bell = {}
    for line in f1:
        if line[:-2] in bell:
            bell[line[:-2]] += 1./ Ns
        else:
            bell[line[:-2]] = 1. /Ns
    f1.close()
    
    # Read in model data storing prob of each measurement
    try:
        f2 = open(gen, 'r')
    except:
        print ('ERROR: No model file found')
        return 0
    
    model = {}
    for line in f2:
        if line[:-1] in model:
            model[line[:-1]] += 1. /Ns
        else:
            model[line[:-1]] = 1. / Ns
    f2.close()
    
    # Compute fidelity
    fidelity = 0
    for item in bell:
        # If model does not generate data for a particular measurement 
        # then prob(a) of model is 0
        try:
            fidelity +=  bell[item] * sqrt(model[item] / bell[item])
        except:
            fidelity +=  bell[item] * sqrt(0.0 / bell[item])
    return fidelity
    
epochs = 50
Nq = 2
model = 'RNN'

if model == 'RBM':
    data = 'C:\\Users\\antch\\Downloads\\PHYS-490-Final-Project-master\\RBM\\data\\{0}_qubit_train.txt'.format(Nq)
    direct = 'C:\\Users\\antch\\Downloads\\PHYS-490-Final-Project-master\\RBM\\output\\{0}_qubit\\'.format(Nq)
    
elif model == 'RNN':
    data = 'C:\\Users\\antch\\Downloads\\PHYS-490-Final-Project-master\\RNN\\Data\\{0} Qubit\\train.txt'.format(Nq)
    direct = 'C:\\Users\\antch\\Downloads\\PHYS-490-Final-Project-master\\RNN\\outputs\\{0}_qubits\\lr_0.01\\'.format(Nq)

fidels = []
epoch = list(range(1,epochs + 1))

for i in epoch:
    gen = direct + 'epoch_{0}.txt'.format(i)
    fidels.append(fid(data,gen))

plt.plot(epoch,fidels)
plt.xlabel('Number of Epochs')
plt.ylabel('Fidelity')
plt.savefig('Fidelity_{0}Q.pdf'.format(Nq))

#gen = direct + 'epoch_50.txt'
#fid(data,gen)

