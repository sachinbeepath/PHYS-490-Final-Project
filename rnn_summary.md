# RNN Summary

## Reminders about RNNs
- processes sequences while retaining memory of what came before
- at each step, consider the new input plus the output of the previous step

## Structure of their RNN
- three GRU units followed by a fully-connected layer with softmax activation
- input is sequential data of the single-qubit outcomes of the ordered qubits (e.g the data a = (a1, a2, a3...) where a1 = the outcome of qubit 1 etc)
- the process updates a recurrent hidden state h_i of size 100 whose value depends on itself at previous steps, and the new input
  - e.g. at step i, we have h_i = f(W[a_i; h_(i-1)]) where a_i is a one-hot encoding of the integer outcome of qubit i, and W is a matrix of parameters to update each time
- the ordering of the lattice sites (qubits) is the sequence. in GHZ state, the sequence is arbitrary. in 2d, it is a 1d path filling the 2d lattice.
- what is a GRU unit?
  - we have update gates z_i, and reset gates r_i. we calculate these from the value of h_(i-1):
    - z_i = sigmoid(W_z[h_(i-1), a_i])$$
    - r_i = sigmoid(W_r[h_(i-1), a_i])
  - then calculate a "candidate hidden state" h'_i using these:
    - h'_i = tanh(W_c[r_i * h _(i-1), a_i])
  - then you interpolate the h_i value from these:
    - h_i = (1-z_i)*h_(i-1) + z_i * h'_i

## What's Going on in the Code rnn.py
- RNN class is called LatentAttention
- defines an __init__ function, as well as a generation function and a training function
- __init__
  - loads data and defines the values stored in the model (eg number of samples, sample size, optimizer, cost etc)
- generation
  - takes in 'decoder' and 'molecules'
    - decoder is either 'TimeDistributed' or 'TimeDistributed_mol'
      - decoder='TimeDistributed' is the simplest time distributed decoder feeding z z z as input to the decoder at every time step
      - decoder='TimeDistributed_mol' during training, input of the decoder is concatenating [z,symbol_t]; during
        generation input should be [z,predicted_symbol_t] where predicted molecule is a sample of the probability
        vector (or a greedy sample where we convert predicted_molecule_t to one hot vector of the most likely symbol)
      *still confused about this*
  -
