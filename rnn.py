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

def load_dataset(filepath, sort=True):
    print('Loading data...')
    pickled_file = Path("ocr_words.pkl")
    if pickled_file.is_file():
        print('Found pickled data {}'.format(pickled_file))
        with Path.open(pickled_file, 'rb') as f:
            words_train, words_dev, words_test, labels = pickle.load(f)

    else:
        print('Reading raw dataset')
        words_train, labels = read_data(filepath, partitions=set(range(8)))
        words_dev, _ = read_data(filepath, partitions={8})
        words_test, _ = read_data(filepath, partitions={9})

        with Path.open(pickled_file, 'wb') as f:
            pickle.dump( (words_train, words_dev, words_test, labels), f)

    return words_train, words_dev, words_test, labels

def read_data(filepath, partitions=None, sort=True):
    """Read the OCR dataset."""
    labels = {}
    f = open(filepath)
    X = []
    y = []
    words = []

    for line in f:
        line = line.rstrip('\t\n')
        fields = line.split('\t')
        letter = fields[1]
        next_id = int(fields[2])
        if letter in labels:
            k = labels[letter]
        else:
            k = len(labels)
            labels[letter] = k
        partition = int(fields[5])
        if partitions is not None and partition not in partitions:
            continue
        x = np.array([int(v) for v in fields[6:]], dtype=int)
        # pairwise
        # x = x[:, None].dot(x[None, :]).flatten()

        X.append(x)
        y.append(k)

        if next_id == -1:  # current word has finished
            words.append((torch.tensor(X, dtype=torch.float, requires_grad=True), torch.tensor(y, dtype=torch.long, requires_grad=False)))
            X = []
            y = []

    f.close()

    l = ['' for _ in labels]
    for letter in labels:
        l[labels[letter]] = letter

    # sort the dataset in decreasing length
    if sort:
        words.sort(key=lambda vec: len(vec[1]), reverse=True)

    words = np.array(words)
    return words, l

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, words):
        'Initialization'
        self.words = words

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.words)

  def __getitem__(self, index):
        'Generates one sample of data'
        # X, y = self.words[index]
        # return X, y
        return self.words[index] # (X,y) tuple

def pad_fn(data):
    # separate source and target sequences
    X_seqs, y_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    X_len = [len(seq) for seq in X_seqs]
    padded_X = Variable(torch.zeros(max(X_len), len(X_seqs), len(X_seqs[0][0]) , dtype=torch.float, requires_grad=True))
    for i, seq in enumerate(X_seqs):
        end = X_len[i]
        padded_X[:end, i, :] = seq[:end, :]

    y_len = [len(seq) for seq in y_seqs]
    padded_y = torch.zeros(max(y_len), len(y_seqs)).long()
    for i, seq in enumerate(y_seqs):
        end = y_len[i]
        padded_y[:end, i] = seq[:end]

    return padded_X, X_len, padded_y, y_len


class CNN_BILSTM(torch.nn.Module):
    '''Create a BILSTM model'''
    def __init__(self, num_features, num_classes, lstm_units=64, lstm_layers=1, dropout_prob=0.1, device='cpu'):
        super(CNN_BILSTM, self).__init__()

        self.device = device
        self.lstm_layers = lstm_layers
        self.lstm_units = lstm_units

        # -----------------------------First Convolution---------------------------------------- #
        # input=16x8x1 ; channels=20 filters=5x5 padding=(F-1)/2=(5-1)/2=2 ; output=16x8x20

        # in=W1xH1xD1 out=W2xH2xD2 ; W2=(W1-F)/S + 1 H2=(H1-F)/S + 1 D2=D1
        # input=16x8x20 ; F=2 S=2 ; output=8x4x20

        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=20,
                                                         kernel_size=5, stride=1,
                                                         padding=2),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                                         )

        # -----------------------------Second Convolution---------------------------------------- #
        # input=8x4x20 ; channels=30 filters=7x7 padding=(F-1)/2=(7-1)/2=3 ; output=8x4x30

        # in=W1xH1xD1 out=W2xH2xD2 ; W2=(W1-F)/S + 1 H2=(H1-F)/S + 1 D2=D1
        # input=8x4x30 ; F=3 S=3 ; output=2x1x30

        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=20, out_channels=30,
                                                         kernel_size=7, stride=1,
                                                         padding=3),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
                                         )

        # -----------------------------Fully Connected Layer----------------------------------- #
        # input=flatten(2x1x30)=1x60 output=lstm_units
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(2*1*30, lstm_units, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_prob)
        )

        # -----------------------------Recurrent Layer----------------------------------- #
        self.bilstm = torch.nn.LSTM(input_size=lstm_units, hidden_size=lstm_units,
                                    num_layers=lstm_layers, dropout=dropout_prob, bidirectional=True)

        self.output_linear = torch.nn.Linear(2*lstm_units, num_classes, bias=True)

    def forward(self, x, orig_len):
        max_len, batch_size, num_features = x.size()

        # reshape data to the convolution layer
        output = x.view(-1, 1, 16, 8)

        # first convolution
        output = self.conv1(output)
        # second convolution
        output = self.conv2(output)

        # reshape data to input to the input layer of the fully connected layer
        output = output.view(-1, 2*1*30)

        # fully connected layer
        output = self.fc1(output)

        # reshape data to the recurrent layer
        output = output.view(-1, batch_size, self.lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        output = torch.nn.utils.rnn.pack_padded_sequence(output, orig_len)

        h = torch.nn.init.xavier_uniform_(
            torch.FloatTensor(2*self.lstm_layers, batch_size, self.lstm_units))
        c = torch.nn.init.xavier_uniform_(
            torch.FloatTensor(2*self.lstm_layers, batch_size, self.lstm_units))

        h = Variable(h).to(self.device)
        c = Variable(c).to(self.device)

        # now run through LSTM
        output, h = self.bilstm(output, (h,c))

        # undo the packing operation
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)

        # reshape the data so it goes into the linear layer
        output = output.view(max_len, batch_size, -1)
        output = self.output_linear(output)

        return output


class FF_BILSTM(torch.nn.Module):
    '''Create a BILSTM model'''
    def __init__(self, num_features, num_classes, lstm_units=64, lstm_layers=1, dropout_prob=0.1, device='cpu'):
        super(FF_BILSTM, self).__init__()

        self.device = device
        self.lstm_layers = lstm_layers
        self.lstm_units = lstm_units

        self.input_linear = torch.nn.Sequential(
            torch.nn.Linear(num_features, lstm_units, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_prob)
        )

        self.bilstm = torch.nn.LSTM(input_size=lstm_units, hidden_size=lstm_units,
                                    num_layers=lstm_layers, dropout=dropout_prob, bidirectional=True)

        self.output_linear = torch.nn.Linear(2*lstm_units, num_classes, bias=True)

    def forward(self, x, orig_len):
        max_len, batch_size, num_features = x.size()

        output = self.input_linear(x)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        output = torch.nn.utils.rnn.pack_padded_sequence(output, orig_len)

        h = torch.nn.init.xavier_uniform_(
            torch.FloatTensor(2*self.lstm_layers, batch_size, self.lstm_units))
        c = torch.nn.init.xavier_uniform_(
            torch.FloatTensor(2*self.lstm_layers, batch_size, self.lstm_units))

        h = Variable(h).to(self.device)
        c = Variable(c).to(self.device)

        # now run through LSTM
        output, h = self.bilstm(output, (h,c))

        # undo the packing operation
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)

        # reshape the data so it goes into the linear layer
        output = output.view(max_len, batch_size, -1)
        output = self.output_linear(output)

        return output

def train_rnn(rnn, words_train, words_dev, device,
              optimizer_type='SGD', eta=1e-2, reg=1e-4, num_epochs=20, batch_size=1):

    """Train Convolutional Neural Net model"""
    if device != 'cpu':
        rnn = rnn.to(device)

    # Create a train loader
    train_dataset = Dataset(words_train)
    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 2,
              'pin_memory' : True,
              'collate_fn' : pad_fn}
    train_loader = data.DataLoader(train_dataset, **params)

    # Loss Function
    loss_function = torch.nn.CrossEntropyLoss(reduction='elementwise_mean').to(device)

    # Optimizer
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(rnn.parameters(), lr=eta, weight_decay=reg)
    elif optimizer_type == 'ADAM':
        optimizer = torch.optim.Adam(rnn.parameters(), lr=eta, weight_decay=reg)
    elif optimizer_type == 'ADAGRAD':
        optimizer = torch.optim.Adagrad(rnn.parameters(), lr=eta, weight_decay=reg)
    else:
        raise Exception('No optimizer defined')

    losses = []
    accuracies = []
    for epoch in range(num_epochs):
        rnn.train()
        total_loss = 0.
        # for X, y, original_len in train_loader:
        for X, X_orig_len, y, y_orig_len in train_loader:
            if device != 'cpu':
                X, y = X.to(device), y.to(device)

            # forward pass
            logits = rnn(X, X_orig_len)

            logits = logits.view(-1, 26)
            y = y.view(-1)

            # compute the loss
            loss = loss_function(logits, y)

            # zero the gradients before recomputing them again
            rnn.zero_grad()
            # compute the gradient
            loss.backward()
            total_loss += loss.item()

            # take a step toward the gradient direction
            optimizer.step()

        losses.append(total_loss)
        accuracy = evaluate(rnn, device, words_dev)
        accuracies.append(accuracy)
        print('Epoch: %d, Total Loss: %f, Dev accuracy: %f' %
                    (epoch + 1, total_loss, accuracy) )

    return rnn, accuracies, losses

def evaluate(model, device, words_dev):
    """Evaluate model on data."""
    model.eval()

    dev_dataset = Dataset(words_dev)
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 2,
              'collate_fn': pad_fn}
    dev_loader = data.DataLoader(dev_dataset, **params)

    correct = 0
    total = 0
    for X, X_orig_len, y, y_orig_len in dev_loader:
        if device != 'cpu':
            X, y = X.to(device), y.to(device)

        # forward pass
        logits = model(X, X_orig_len)

        logits = logits.view(-1, 26)
        y = y.view(-1)

        _, y_hat = torch.max(logits, 1)

        correct += (y == y_hat).sum()
        total += np.sum(X_orig_len)

    return 100 * float(correct) / total


def main(filepath = 'letter.data', generate_plot=True,
         optimizer='ADAM',
         num_epochs = 500, learn_rate = 1e-4, reg_const = 1e-6, batch_size = 64):

    pre_ff = True

    words_train, words_dev, words_test, labels = load_dataset(filepath)

    processing_unit = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print('Processing Unit : ', processing_unit)

    print('--Recurrent Neural Net--')
    num_classes = len(labels)
    num_features = 128

    # Select between a pre ffnn or a pre cnn
    if pre_ff:
        units = 128
        layers = 2
        rnn_obj = FF_BILSTM(num_features, num_classes, lstm_units=units, lstm_layers=layers, dropout_prob=0.1, device=processing_unit)
    else:
        units = 64
        layers = 2
        rnn_obj = CNN_BILSTM(num_features, num_classes, lstm_units=units, lstm_layers=layers, dropout_prob=0.1, device=processing_unit)
    print(rnn_obj)

    print('--Training--')
    trained_model, accuracies, losses = train_rnn(rnn_obj, words_train, words_dev, processing_unit,
                                                  optimizer_type=optimizer,
                                                  eta=learn_rate,reg=reg_const,
                                                  num_epochs=num_epochs, batch_size=batch_size)

    print('--Evaluating on the test set--')
    test_accuracy = evaluate(trained_model, processing_unit, words_test)
    print('Test accuracy: %f' % (test_accuracy))

    # save test accuracy to a file
    file = open('rnn_test_accuracy', 'a')
    file.write('ID: %s_%i_%.2E_%.2E\t->\tTest ACC: %f \n' %
               (optimizer, num_epochs, learn_rate, reg_const, test_accuracy))
    file.close()

    if generate_plot:
        print('--Plotting--')
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(range(1, num_epochs+1), accuracies, 'bo-')
        plt.title('Validation Accuracy \n $opt = $ %s $\eta = $ %.2E $\lambda = $ %.2E \n BILSTM units$= %i$ BILSTM layers$= %i$' %
                  (optimizer, learn_rate, reg_const, units, layers))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')

        name = 'acc_%s_%i_%.2E_%.2E' % \
               (optimizer, num_epochs, learn_rate, reg_const)

        plt.savefig('rnn_%s.pdf' % name, bbox_inches="tight")

        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(range(1, num_epochs+1), losses, 'bo-')
        plt.title('Training Loss \n $opt = $ %s $\eta = $ %.2E $\lambda = $ %.2E \n BILSTM units$= %i$ BILSTM layers$= %i$' %
                  (optimizer, learn_rate, reg_const, units, layers))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        name = 'loss_%s_%i_%.2E_%.2E' % \
               (optimizer, num_epochs, learn_rate, reg_const)

        plt.savefig('rnn_%s.pdf' % name, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()