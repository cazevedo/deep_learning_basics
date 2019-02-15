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

def train_logistic_sgd(X_train, y_train, X_dev, y_dev, labels, eta=1e-3, reg=0.,
                       num_epochs=20, batch_size=32):
    """Train logistic regression model with SGD."""

    # Create a train loader
    train_dataset = Dataset(X_train, y_train)
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 2}
    train_loader = data.DataLoader(train_dataset, **params)

    num_features = len(X_train[0])
    num_classes = len(labels)

    # Linear Model
    linear_layer = torch.nn.Linear(num_features, num_classes, bias=False)
    linear_model = torch.nn.Sequential(linear_layer)

    # Cross Entropy Loss Function
    # loss_function = torch.nn.CrossEntropyLoss(reduction='sum')
    loss_function = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')

    # Stochastic Gradient Descent Optimizer
    optimizer = torch.optim.SGD(linear_model.parameters(), lr=eta, weight_decay=reg)

    losses = []
    accuracies = []
    for epoch in range(num_epochs):
        total_loss = 0.
        for X,y in train_loader:
            # forward pass
            logits = linear_model(X)

            # compute the loss
            loss = loss_function(logits, y)

            # zero the gradients before recomputing them again
            linear_model.zero_grad()
            # compute the gradient
            loss.backward()
            total_loss += loss.item()

            # take a step toward the gradient direction
            optimizer.step()

        losses.append(total_loss)
        accuracy = evaluate(linear_model, X_dev, y_dev)
        accuracies.append(accuracy)
        print('Epoch: %d, Total Loss: %f, Dev accuracy: %f' %
                    (epoch + 1, total_loss, accuracy) )

    return linear_model, accuracies, losses

def evaluate(model, X_test, y_test):
    """Evaluate model on data."""

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

def main():
    filepath = 'letter.data'
    num_epochs = 40
    learn_rate = 1e-2
    reg_const = 1e-3
    batch_size = 32
    pairwise = True
    X_train, y_train, labels, X_dev, y_dev, X_test, y_test = load_dataset(filepath, pairwise)

    print('--Training--')
    trained_model, accuracies, losses = train_logistic_sgd(X_train, y_train, X_dev, y_dev, labels, eta=learn_rate, reg=reg_const,
                       num_epochs=num_epochs, batch_size=batch_size)

    print('--Evaluating on the test set--')
    test_accuracy = evaluate(trained_model, X_test, y_test)
    print('Test accuracy: %f' % (test_accuracy))

    print('--Plotting--')
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(range(1, num_epochs+1), accuracies, 'bo-')
    plt.title('Validation Accuracy \n $\eta = $ %.2E $\lambda = $ %.2E' % (learn_rate, reg_const))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')

    name = 'logistic_acc_%s_%s' % (learn_rate, reg_const)
    if pairwise:
        name += '_pairwise'
    plt.savefig('%s.pdf' % name)

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(range(1, num_epochs+1), losses, 'bo-')
    plt.title('Training Loss \n $\eta = $ %.2E $\lambda = $ %.2E' % (learn_rate, reg_const))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    name = 'logistic_loss_%s_%s' % (learn_rate, reg_const)
    if pairwise:
        name += '_pairwise'
    plt.savefig('%s.pdf' % name)


if __name__ == "__main__":
    main()