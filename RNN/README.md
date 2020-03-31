# RNN

``rnn.py`` contains the definition of the RNN class and the RNN training script.

``kl_calc.py`` calculates the KL divergence of the RNN models for each epoch using the outputs saved during training.

#### Running ``rnn.py``

The path to the training dataset, latent layer size, batch size and number of epochs can be edited in ``param.json``.

The number of qubits, learning rate and training data size are defined upon running the Python file ``rnn.py``. ``rnn.py`` runs as follows:

```sh
usage: rnn.py [-h] [-q Q] [-lr LR] [-s S]

arguments:
  -h, --help  show this help message and exit
  -q Q        number of qubits
  -lr LR      learning rate
  -s S        maximum training dataset size
  ```
The dependencies in ``rnn.py`` are:
  - torch
  - numpy
  - json
  - gc
  - sklearn
  - argparse
  
#### Running ``kl_calc.py``

The path to the training dataset, generated dataset main folder and number of epochs can be edited within ``kl_calc.py`` itself.

The number of qubits and learning rate (used to locate the generated data for the specific model within the dataset main folder) are defined upon running the Python file ``kl_calc.py``. ``kl_calc.py`` runs as follows:

```sh
usage: rnn.py [-h] [-q Q] [-lr LR] [-s S]

arguments:
  -h, --help  show this help message and exit
  -q Q        number of qubits
  -lr LR      learning rate
  ```
The dependencies in ``kl_calc.py`` are:
  - torch
  - numpy
  - argparse
  - matplotlib
