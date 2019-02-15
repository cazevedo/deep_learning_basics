import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from pathlib import Path
import pickle
import torch
from torch.utils import data
from torch.autograd import Variable

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, y):
        'Initialization'
        self.X = X
        self.y = y

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

  def __getitem__(self, index):
        'Generates one sample of data'
        return Variable(self.X[index]), self.y[index]
        # return self.X.view(index, len(self.X[0])), self.y[index]

class NeuralNet(torch.nn.Sequential):
    '''Create a Neural Net model'''
    def __init__(self, num_features, num_classes, hidden_units=[1], activate_f='tanh', dropout=True, dropout_prob=0.5, out_activate_f='softmax'):
        super(NeuralNet, self).__init__()

        if not hidden_units:
            raise Exception('Define at least one hidden unit')

        self.add_module('inputlayer', torch.nn.Linear(num_features, hidden_units[0], bias=True))

        for hidden_index in range(len(hidden_units)):
            name = 'hiddenlayer%i' % (hidden_index)

            if activate_f == 'tanh':
                self.add_module(name+'_act', torch.nn.Tanh())
            elif activate_f == 'relu':
                self.add_module(name+'_act', torch.nn.ReLU())
            elif activate_f == 'sigmoid':
                self.add_module(name+'_act', torch.nn.Sigmoid())

            if dropout:
                self.add_module(name+'_drop', torch.nn.Dropout(p=dropout_prob))

            if hidden_index < len(hidden_units)-1:
                self.add_module(name+'_lin', torch.nn.Linear(hidden_units[hidden_index], hidden_units[hidden_index+1], bias=True))

        self.add_module('outputlayer', torch.nn.Linear(hidden_units[-1], num_classes, bias=True))

        if out_activate_f == 'softmax':
            self.add_module('output_act', torch.nn.Softmax())

def train_neural_net(X_train, y_train, X_dev, y_dev, labels, neural_net, loss_f='CEL', optimizer_type='SGD', eta=1e-3, reg=0.,
                       num_epochs=20, batch_size=32):
    """Train Neural Net model"""

    # Create a train loader
    train_dataset = Dataset(X_train, y_train)
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 2}
    train_loader = data.DataLoader(train_dataset, **params)

    # Loss Function
    if loss_f == 'CEL':
        loss_function = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
    else:
        raise Exception('No loss function defined')

    # Optimizer
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(neural_net.parameters(), lr=eta, weight_decay=reg)
    elif optimizer_type == 'ADAM':
        optimizer = torch.optim.Adam(neural_net.parameters(), lr=eta, weight_decay=reg)
    elif optimizer_type == 'ADAGRAD':
        optimizer = torch.optim.Adagrad(neural_net.parameters(), lr=eta, weight_decay=reg)
    else:
        raise Exception('No optimizer defined')

    losses = []
    accuracies = []
    for epoch in range(num_epochs):
        # set training mode to true to activate dropouts
        neural_net.train(mode=True)

        total_loss = 0.
        for X,y in train_loader:
            # forward pass
            logits = neural_net(X)

            # compute the loss
            loss = loss_function(logits, y)

            # zero the gradients before recomputing them again
            neural_net.zero_grad()
            # compute the gradient
            loss.backward()
            total_loss += loss.item()

            # take a step toward the gradient direction
            optimizer.step()

        losses.append(total_loss)
        accuracy = evaluate(neural_net, X_dev, y_dev)
        accuracies.append(accuracy)
        print('Epoch: %d, Total Loss: %f, Dev accuracy: %f' %
                    (epoch + 1, total_loss, accuracy) )

    return neural_net, accuracies, losses

def evaluate(model, X_test, y_test):
    """Evaluate model on data."""
    model.train(mode=False)

    dev_dataset = Dataset(X_test, y_test)
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 2}
    dev_loader = data.DataLoader(dev_dataset, **params)

    correct = 0
    for X, y in dev_loader:
        # forward pass
        logits = model(X)

        y_hat = np.argmax(logits)
        if y == y_hat:
            correct += 1

    return 100 * float(correct) / len(y_test)

def load_dataset(filepath, pairwise=True):
    print('Loading data...')
    pickled_file = Path("ocr_{}.pkl".format('raw' if not pairwise else 'pairwise'))
    if pickled_file.is_file():
        print('Found pickled data {}'.format(pickled_file))
        with Path.open(pickled_file, 'rb') as f:
            X_train, y_train, labels, X_dev, y_dev, X_test, y_test = pickle.load(f)

    else:
        print('Reading raw dataset')
        X_train, y_train, labels = read_data(filepath, partitions=set(range(8)),
                                             pairwise_features=pairwise)
        X_dev, y_dev, _ = read_data(filepath, partitions={8},
                                    pairwise_features=pairwise)
        X_test, y_test, _ = read_data(filepath, partitions={9},
                                      pairwise_features=pairwise)

        X_train = torch.tensor(X_train, dtype=torch.float, requires_grad=True)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_dev = torch.tensor(X_dev, dtype=torch.float, requires_grad=True)
        y_dev = torch.tensor(y_dev, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float, requires_grad=True)
        y_test = torch.tensor(y_test, dtype=torch.long)

        with Path.open(pickled_file, 'wb') as f:
            pickle.dump( (X_train, y_train, labels, X_dev, y_dev, X_test, y_test), f)

    return X_train, y_train, labels, X_dev, y_dev, X_test, y_test

def read_data(filepath, partitions=None, pairwise_features=False):
    """Read the OCR dataset."""
    labels = {}
    f = open(filepath)
    X = []
    y = []
    for line in f:
        line = line.rstrip('\t\n')
        fields = line.split('\t')
        letter = fields[1]
        if letter in labels:
            k = labels[letter]
        else:
            k = len(labels)
            labels[letter] = k
        partition = int(fields[5])
        if partitions is not None and partition not in partitions:
            continue
        x = np.array([float(v) for v in fields[6:]])
        if pairwise_features:
            x = x[:, None].dot(x[None, :]).flatten()
        X.append(x)
        y.append(k)
    f.close()
    l = ['' for k in labels]
    for letter in labels:
        l[labels[letter]] = letter
    return X, y, l


def main(X_train = 0, y_train = 0, labels = 0, X_dev = 0, y_dev = 0, X_test = 0, y_test = 0,
        activation_function = 'tanh', dropout_p = 0.1, n_hidden_unts = [50], optimizer = 'ADAGRAD',
        load = True, filepath = 'letter.data', pairwise = False, num_epochs = 20, learn_rate = 0.1, reg_const = 0., batch_size = 50):

    print('% s %s hidden units = %i \n drop = %.2f learn_rate = %.2E regularization = %.2E' %
               (activation_function, optimizer, n_hidden_unts[0], dropout_p, learn_rate, reg_const))

    if load:
        X_train, y_train, labels, X_dev, y_dev, X_test, y_test = load_dataset(filepath, pairwise)

    num_features = len(X_train[0])
    num_classes = len(labels)

    print('--Neural Net--')
    nn_obj = NeuralNet(num_features, num_classes, hidden_units=n_hidden_unts, activate_f=activation_function,
                       dropout=True, dropout_prob=dropout_p, out_activate_f=None)
    print(nn_obj)

    print('--Training--')
    trained_model, accuracies, losses = train_neural_net(X_train, y_train, X_dev, y_dev, labels, nn_obj,
                        loss_f='CEL', optimizer_type=optimizer, eta=learn_rate, reg=reg_const,
                        num_epochs=num_epochs, batch_size=batch_size)

    print('--Evaluating on the test set--')
    test_accuracy = evaluate(trained_model, X_test, y_test)
    print('Test accuracy: %f' % (test_accuracy))

    file = open('neuralnet_test_accuracy', 'a')
    file.write('ID: %s_%s_%i_%.2f_%.2E_%.2E\t->\tTest ACC: %f \n' %
              (activation_function, optimizer, n_hidden_unts[0], dropout_p, learn_rate, reg_const, test_accuracy))
    file.close()

    print('--Plotting--')
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(range(1, num_epochs+1), accuracies, 'bo-')
    plt.title('Validation Accuracy \n $act = $ %s $opt = $ %s $hidden$ $units = $ %i $hidden$ $layers = $ %i \n $drop = $ %.2f $\eta = $ %.2E $\lambda = $ %.2E' %
              (activation_function, optimizer, n_hidden_unts[0], len(n_hidden_unts), dropout_p, learn_rate, reg_const))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')

    name = 'neuralnet_acc_%s_%s_%i_%.2f_%.2E_%.2E' % \
           (activation_function, optimizer, n_hidden_unts[0], dropout_p, learn_rate, reg_const)
    if pairwise:
        name += '_pairwise'
    plt.savefig('neuralnet/%s.pdf' % name, bbox_inches="tight")

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(range(1, num_epochs+1), losses, 'bo-')
    plt.title('Training Loss \n $act = $ %s $opt = $ %s $hidden$ $units = $ %i $hidden$ $layers = $ %i \n $drop = $ %.2f $\eta = $ %.2E $\lambda = $ %.2E' %
              (activation_function, optimizer, n_hidden_unts[0], len(n_hidden_unts), dropout_p, learn_rate, reg_const))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    name = 'neuralnet_loss_%s_%s_%i_%.2f_%.2E_%.2E' % \
           (activation_function, optimizer, n_hidden_unts[0], dropout_p, learn_rate, reg_const)

    if pairwise:
        name += '_pairwise'
    plt.savefig('neuralnet/%s.pdf' % name, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    main()