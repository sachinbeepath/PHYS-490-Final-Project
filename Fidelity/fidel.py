from math import sqrt
import matplotlib.pyplot as plt

# Function to calculate classical fidelity
def fid(data = 'C:\\Users\\antch\\Downloads\\PHYS-490-Final-Project-master\\RBM\\data\\2_qubit_train.txt', 
        gen = 'C:\\Users\\antch\\Downloads\\PHYS-490-Final-Project-master\\RBM\\output\\2_qubit\\epoch_50.txt',  
        POVM = 'Tetra', Nq = 2, p = 0.0, Nsbell = 6e4, Nsmodel = 6e4):
    
    # Read in bell state data storing probability of each measurement
    try:
        f1 = open(data, 'r')
    except:
        print ('ERROR: No data file found')
        return 0
    
    bell = {}
    for line in f1:
        if line[:-2] in bell:
            bell[line[:-2]] += 1./ Nsbell
        else:
            bell[line[:-2]] = 1. /Nsbell
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
            model[line[:-1]] += 1. /Nsmodel
        else:
            model[line[:-1]] = 1. / Nsmodel
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

model = 'RNN'

if model == 'RBM':
    Nq = 2
    epochs = 30
    hn = 100
    data = 'C:\\Users\\antch\\Downloads\\PHYS-490-Final-Project-master\\RBM\\data\\{0}_qubit_train.txt'.format(Nq)
    direct = 'C:\\Users\\antch\\Downloads\\PHYS-490-Final-Project-master\\RBM\\output\\Hn{0}\\{1}_qubit\\'.format(hn,Nq)
    
    epoch = list(range(1,epochs + 1))
    fidels = []
    
    for i in epoch:
        gen = direct + 'epoch_{0}.txt'.format(i)
        fidels.append(fid(data,gen))
        
    plt.plot(epoch,fidels)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Fidelity')
    plt.grid()
    plt.savefig('Fidelity_{0}Q.pdf'.format(Nq))
    
elif model == 'RNN':
    epochs = 15
    samples = [1000,5000,10000,30000,60000]
    
    for i in range (2,7):
        fidels = []
        data = 'C:\\Users\\antch\\Downloads\\PHYS-490-Final-Project-master\\RNN\\Data\\{0} Qubit\\train.txt'.format(i)
        direct = 'C:\\Users\\antch\\Downloads\\PHYS-490-Final-Project-master\\RNN\\outputs\\{0}_qubits\\lr_0.0001\\'.format(i)
        
        gen1 = direct + 'max_size_1000\\epoch_{0}.txt'.format(epochs)
        gen2 = direct + 'max_size_5000\\epoch_{0}.txt'.format(epochs)
        gen3 = direct + 'max_size_10000\\epoch_{0}.txt'.format(epochs)
        gen4 = direct + 'max_size_30000\\epoch_{0}.txt'.format(epochs)
        gen5 = direct + 'max_size_60000\\epoch_{0}.txt'.format(epochs)
    
        fidels.append(fid(data,gen1, Nsmodel = 1000))
        fidels.append(fid(data,gen2, Nsmodel = 5000))
        fidels.append(fid(data,gen3, Nsmodel = 10000))
        fidels.append(fid(data,gen4, Nsmodel = 30000))
        fidels.append(fid(data,gen5, Nsmodel = 60000))
        
        plt.plot(samples,fidels, '-o')
        
    plt.legend(["N=2","N=3","N=4","N=5","N=6"])
    plt.xlabel('Number of Samples')
    plt.ylabel('Fidelity')
    plt.grid()
    plt.savefig('Fidelity.pdf'.format(Nq))

