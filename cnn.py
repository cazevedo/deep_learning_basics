import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import random
from pathlib import Path
import pickle
import torch
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

def load_dataset(filepath, pairwise=False):
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

        x = np.zeros((16,8))
        for row in range(16):
            x[row] = fields[6:][row*8:row*8+8]

        # x = np.array([float(v) for v in fields[6:]])
        # if pairwise_features:
        #     x = x[:, None].dot(x[None, :]).flatten()

        # X.append([x])
        X.append(x)
        y.append(k)
    f.close()
    l = ['' for k in labels]
    for letter in labels:
        l[labels[letter]] = letter
    return X, y, l

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


class ConvNetOrig(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNetOrig, self).__init__()
        # -----------------------------First Convolution---------------------------------------- #

        # input=16x8x1 ; channels=20 filters=5x5 padding=(F-1)/2=(5-1)/2=2 ; output=16x8x20
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=20,
                                       kernel_size=5, stride=1,
                                       padding=2)

        # in=W1xH1xD1 out=W2xH2xD2 ; W2=(W1-F)/S + 1 H2=(H1-F)/S + 1 D2=D1
        # input=16x8x20 ; F=2 S=2 ; output=8x4x20
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


        # -----------------------------Second Convolution---------------------------------------- #

        # input=8x4x20 ; channels=30 filters=7x7 padding=(F-1)/2=(7-1)/2=3 ; output=8x4x30
        self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=30,
                                       kernel_size=7, stride=1,
                                       padding=3)

        # in=W1xH1xD1 out=W2xH2xD2 ; W2=(W1-F)/S + 1 H2=(H1-F)/S + 1 D2=D1
        # input=8x4x30 ; F=2 S=2 ; output=4x2x30
        # self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # in=W1xH1xD1 out=W2xH2xD2 ; W2=(W1-F)/S + 1 H2=(H1-F)/S + 1 D2=D1
        # input=8x4x30 ; F=3 S=3 ; output=2x1x30
        self.pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=3, padding=0)

        # -----------------------------Fully Connected Layer----------------------------------- #
        # input=flatten(4x2x30)=1x240 output=26
        # self.fc1 = torch.nn.Linear(4*2*30, num_classes)

        # input=flatten(2x1x30)=1x60 output=26
        self.fc1 = torch.nn.Linear(2*1*30, num_classes)

    def forward(self, x):
        # first convolution
        x = self.conv1(x)
        # activation of the first convolution
        # input=16x8x20 output=16x8x20
        x = F.relu(x)
        # first pooling
        x = self.pool1(x)

        # second convolution
        x = self.conv2(x)
        # activation of the second convolution
        # input=8x4x20 output=8x4x20
        x = F.relu(x)
        # second pooling
        x = self.pool2(x)

        # reshape data to input to the input layer of the neural net
        # x = x.view(-1, 4*2*30)
        x = x.view(-1, 2*1*30)

        # fully connected layer
        x = self.fc1(x)

        return x

class ConvNetMod(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNetMod, self).__init__()
        # -----------------------------First Convolution---------------------------------------- #
        # input=16x8x1 ; channels=20 filters=3x3 padding=(F-1)/2=(3-1)/2=1 ; output=16x8x20
        self.conv1 = torch.nn.Sequential( torch.nn.Conv2d(in_channels=1, out_channels=20,
                                       kernel_size=3, stride=1,
                                       padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(p=0.1)
                                        )

        # -----------------------------Second Convolution---------------------------------------- #
        # input=16x8x20 ; channels=30 filters=5x5 padding=(F-1)/2=(5-1)/2=2 ; output=16x8x30
        self.conv2 = torch.nn.Sequential( torch.nn.Conv2d(in_channels=20, out_channels=30,
                                       kernel_size=5, stride=1,
                                       padding=2),
                                          torch.nn.ReLU(),
                                          torch.nn.Dropout(p=0.1)
                                        )

        # -----------------------------Third Convolution---------------------------------------- #
                                            # input=16x8x30; channels=50 filters=7x7 padding=(F-1)/2=(7-1)/2=3 ; output=16x8x50
        self.conv3 = torch.nn.Sequential( torch.nn.Conv2d(in_channels=30, out_channels=50,
                                           kernel_size=7, stride=1,
                                           padding=3),
                                          torch.nn.ReLU(),
                                            # in=W1xH1xD1 out=W2xH2xD2 ; W2=(W1-F)/S + 1 H2=(H1-F)/S + 1 D2=D1
                                            # input=16x8x50 ; F=6 S=2 ; output=6x2x50
                                            torch.nn.MaxPool2d(kernel_size=6, stride=2, padding=0),
                                            torch.nn.Dropout(p=0.1)
                                            )

        # -----------------------------Fully Connected Layer----------------------------------- #
        self.fc = torch.nn.Sequential(torch.nn.Linear(6*2*50, 2*1*25),
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.1),
                                      torch.nn.Linear(2 * 1 * 25, num_classes)
                                      )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # reshape data to input to the input layer of the neural net
        x = x.view(-1, 6*2*50)
        x = self.fc(x)

        return x

def train_cnn(X_train, y_train, X_dev, y_dev, cnn, device, optimizer_type='SGD', eta=1e-2, reg=0.,
                       num_epochs=20, batch_size=1):
    """Train Convolutional Neural Net model"""
    if device != 'cpu':
        cnn = cnn.to(device)

    # Create a train loader
    train_dataset = Dataset(X_train, y_train)
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 2}
    train_loader = data.DataLoader(train_dataset, **params)

    # Loss Function
    loss_function = torch.nn.CrossEntropyLoss(reduction='elementwise_mean').to(device)

    # Optimizer
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(cnn.parameters(), lr=eta, weight_decay=reg)
    elif optimizer_type == 'ADAM':
        optimizer = torch.optim.Adam(cnn.parameters(), lr=eta, weight_decay=reg)
    elif optimizer_type == 'ADAGRAD':
        optimizer = torch.optim.Adagrad(cnn.parameters(), lr=eta, weight_decay=reg)
    else:
        raise Exception('No optimizer defined')

    losses = []
    accuracies = []
    for epoch in range(num_epochs):
        cnn.train()
        total_loss = 0.
        for X,y in train_loader:
            if device != 'cpu':
                X, y = X.to(device), y.to(device)

            # forward pass
            logits = cnn(X)

            # compute the loss
            loss = loss_function(logits, y)

            # zero the gradients before recomputing them again
            cnn.zero_grad()
            # compute the gradient
            loss.backward()
            total_loss += loss.item()

            # take a step toward the gradient direction
            optimizer.step()

        losses.append(total_loss)
        accuracy = evaluate(cnn, device, X_dev, y_dev)
        accuracies.append(accuracy)
        print('Epoch: %d, Total Loss: %f, Dev accuracy: %f' %
                    (epoch + 1, total_loss, accuracy) )

    return cnn, accuracies, losses

def evaluate(model, device, X_test, y_test):
    """Evaluate model on data."""
    model.eval()

    dev_dataset = Dataset(X_test, y_test)
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 2}
    dev_loader = data.DataLoader(dev_dataset, **params)

    correct = 0
    for X, y in dev_loader:
        if device != 'cpu':
            X, y = X.to(device), y.to(device)

        # forward pass
        logits = model(X)

        y_hat = np.argmax(logits)
        if y == y_hat:
            correct += 1

    return 100 * float(correct) / len(y_test)


def main(load = True, filepath = 'letter.data', generate_plot=True,
         X_train = 0, y_train = 0, labels = 0, X_dev = 0, y_dev = 0, X_test = 0, y_test = 0,
         optimizer='ADAM',
         num_epochs = 100, learn_rate = 1e-2, reg_const = 1e-4, batch_size = 2048):

    conv_original = True

    if load:
        X_train, y_train, labels, X_dev, y_dev, X_test, y_test = load_dataset(filepath)

    print('--Convolutional Neural Net--')
    num_classes = len(labels)

    # Select convolutional model
    if conv_original:
        cnn_obj = ConvNetOrig(num_classes)
    else:
        cnn_obj = ConvNetMod(num_classes)
    print(cnn_obj)

    processing_unit = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print('Processing Unit : ', processing_unit)

    print('--Training--')
    trained_model, accuracies, losses = train_cnn(X_train, y_train, X_dev, y_dev, cnn_obj, processing_unit,
                                                  optimizer_type=optimizer,
                                                  eta=learn_rate,reg=reg_const,
                                                  num_epochs=num_epochs, batch_size=batch_size)

    print('--Evaluating on the test set--')
    test_accuracy = evaluate(trained_model, processing_unit, X_test, y_test)
    print('Test accuracy: %f' % (test_accuracy))

    # save test accuracy to a file
    file = open('cnn_test_accuracy', 'a')
    file.write('ID: %s_%i_%.2E_%.2E\t->\tTest ACC: %f \n' %
               (optimizer, num_epochs, learn_rate, reg_const, test_accuracy))
    file.close()

    plot_filters = False
    if plot_filters:
        print('--Plotting filters--')
        cnn_obj = cnn_obj.to('cpu')
        layer1_weights = cnn_obj.conv1.weight.data.numpy()
        layer2_weights = cnn_obj.conv2.weight.data.numpy()

        layer = 'layer1'
        for i in range(3):
            j = random.randint(0,20)
            plt.imshow(layer1_weights[j][0], extent=[0,5,0,5], origin='lower', interpolation='nearest')
            plt.savefig(layer+'_weight' + str(j) + '.pdf')

        layer = 'layer2'
        for i in range(3):
            j = random.randint(0,20)
            plt.imshow(layer2_weights[j][0], extent=[0,7,0,7], origin='lower', interpolation='nearest')
            plt.savefig(layer+'_weight' + str(j) + '.pdf')

        plt.close()

    if generate_plot:
        print('--Plotting--')
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(range(1, num_epochs+1), accuracies, 'bo-')
        plt.title('Validation Accuracy \n $opt = $ %s $\eta = $ %.2E $\lambda = $ %.2E' %
                  (optimizer, learn_rate, reg_const))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')

        name = 'acc_%s_%i_%.2E_%.2E' % \
               (optimizer, num_epochs, learn_rate, reg_const)

        plt.savefig('cnn_%s.pdf' % name, bbox_inches="tight")

        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(range(1, num_epochs+1), losses, 'bo-')
        plt.title('Training Loss \n $opt = $ %s $\eta = $ %.2E $\lambda = $ %.2E' %
                  (optimizer, learn_rate, reg_const))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        name = 'loss_%s_%i_%.2E_%.2E' % \
               (optimizer, num_epochs, learn_rate, reg_const)

        plt.savefig('cnn_%s.pdf' % name, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()