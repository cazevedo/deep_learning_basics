import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from random import shuffle
from pathlib import Path
import pickle
import torch
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

def read_data(filepath):
    """Read the Google Transliteration dataset."""
    f = open(filepath)
    X = []
    y = []
    in_vocab = set()
    out_vocab = set()
    for line in f:
        fields = line.split('\t')

        in_word = fields[0]
        out_word = fields[1][0:-1]

        for c in in_word:
            in_vocab.add(c)
        for c in out_word:
            out_vocab.add(c)

        in_word = list(in_word)
        in_word.append('EOW')
        out_word = list(out_word)
        out_word.append('EOW')

        X.append(in_word)
        y.append(out_word)
    f.close()

    # convert to a numpy array so it can be indexable
    in_vocab = np.array(list(in_vocab))
    out_vocab = np.array(list(out_vocab))

    return X, y, in_vocab, out_vocab

def char2index(char, vocab, type, grad):
    return torch.tensor(np.where(vocab == char)[0][0], dtype=type, requires_grad=grad)

def word2index(word, vocab, type, grad):
    word_index = [0]*len(word)
    for i, c in enumerate(word):
        word_index[i] = char2index(c, vocab, type, grad)
    return torch.tensor(word_index, dtype=type, requires_grad=grad)

def list2index(list_words, vocab, type=torch.long, grad=False):
    for i, word in enumerate(list_words):
        list_words[i] = word2index(word, vocab, type, grad)
    return list_words

def index2char(char, vocab):
    return vocab[char]

def index2word(word_index, vocab):
    word = [0]*len(word_index)
    for i, c in enumerate(word_index):
        word[i] = index2char(c, vocab)
    return word

def load_dataset(train_filepath='ar2en-train.txt', eval_filepath='ar2en-eval.txt', test_filepath='ar2en-test.txt'):
    print('Loading data...')

    pickled_file = Path("google_transliteration_dataset.pkl")

    if pickled_file.is_file():
        print('Found pickled data {}'.format(pickled_file))
        with Path.open(pickled_file, 'rb') as f:
            X_train, y_train, X_dev, y_dev, X_test, y_test, in_vocab, out_vocab = pickle.load(f)

    else:
        print('Reading raw dataset')

        X_train, y_train, in_vocab_train, out_vocab_train = read_data(train_filepath)
        X_dev, y_dev, in_vocab_dev, out_vocab_dev = read_data(eval_filepath)
        X_test, y_test, in_vocab_test, out_vocab_test = read_data(test_filepath)

        in_vocab = np.unique(np.concatenate((in_vocab_train, in_vocab_dev, in_vocab_test), axis=0))
        out_vocab = np.unique(np.concatenate((out_vocab_train, out_vocab_dev, out_vocab_test), axis=0))

        special_char = ['SOW', 'EOW']
        in_vocab = np.concatenate((in_vocab, special_char), axis=0)
        out_vocab = np.concatenate((out_vocab, special_char), axis=0)

        X_train, X_dev, X_test = list2index(X_train, in_vocab, type=torch.long, grad=False), \
                                 list2index(X_dev, in_vocab, type=torch.long, grad=False), \
                                 list2index(X_test, in_vocab, type=torch.long, grad=False)

        y_train, y_dev, y_test = list2index(y_train, out_vocab, type=torch.long, grad=False), \
                                 list2index(y_dev, out_vocab, type=torch.long, grad=False), \
                                 list2index(y_test, out_vocab, type=torch.long, grad=False)

        print('Source Vocabulary : ')
        print(in_vocab)
        print('Target Vocabulary : ')
        print(out_vocab)

        # with Path.open(pickled_file, 'wb') as f:
        #     pickle.dump( (X_train, y_train, X_dev, y_dev, X_test, y_test, in_vocab, out_vocab), f)

    return X_train, y_train, X_dev, y_dev, X_test, y_test, in_vocab, out_vocab

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
        return self.X[index], self.y[index]


class UnitEncoderRNN(torch.nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size, dropout_prob, device):
        super(UnitEncoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                    num_layers=1, dropout=dropout_prob, bidirectional=False)

    def forward(self, input, h_n, c_n):
        input = input.view(1, 1, -1)
        embedded = self.embedding(input)
        output = embedded.view(len(input), 1, -1)
        output, (h_n, c_n) = self.lstm(output, (h_n, c_n))
        return output, h_n, c_n

    def initHidden(self, rand=False):
        if rand:
            h = torch.rand(1, 1, self.hidden_size, device=self.device)
            c = torch.rand(1, 1, self.hidden_size, device=self.device)
        else:
            h = torch.zeros(1, 1, self.hidden_size, device=self.device)
            c = torch.zeros(1, 1, self.hidden_size, device=self.device)

        return h, c

class UnitDecoderRNN(torch.nn.Module):
    def __init__(self, input_size, vocab_size, hidden_size, dropout_prob, device):
        super(UnitDecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                    num_layers=1, dropout=dropout_prob, bidirectional=False)

        self.linear = torch.nn.Linear(hidden_size, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, h_n, c_n):
        input = input.view(1, 1, -1)
        output = self.embedding(input).view(len(input), 1, -1)
        output = F.relu(output)
        output, (h_n, c_n) = self.lstm(output, (h_n, c_n))
        output = self.linear(output[0])
        output = self.softmax(output)
        return output, h_n, c_n

    def initHidden(self, rand=False):
        if rand:
            h = torch.rand(1, 1, self.hidden_size, device=self.device)
            c = torch.rand(1, 1, self.hidden_size, device=self.device)
        else:
            h = torch.zeros(1, 1, self.hidden_size, device=self.device)
            c = torch.zeros(1, 1, self.hidden_size, device=self.device)
        return h, c

def encoder(encoding_obj, word, init_h, init_c):
    encoder_h = init_h
    encoder_c = init_c
    for i, c in enumerate(word):
        encoder_output, encoder_h, encoder_c = encoding_obj(Variable(c), encoder_h, encoder_c)

    return encoder_output, encoder_h, encoder_c

def decoder(decoding_obj, decoder_input, init_h, init_c, vocab_out, expected_output, criterion=None, teacher_forcing_prob=1.0):
    decoder_h = init_h
    decoder_c = init_c
    outputs = [] # list of outputs indexes (each index corresponds to a char according to the position in the vocab)
    loss = 0

    if np.random.rand() < teacher_forcing_prob:
        for index_char in expected_output:
            decoder_output, decoder_h, decoder_c = decoding_obj(Variable(decoder_input), decoder_h, decoder_c)
            topv, topi = decoder_output.topk(1)
            output_char_index = topi.squeeze().detach()  # detach from history as input

            outputs.append(output_char_index.item())

            index_char = index_char.unsqueeze(0)

            if criterion:
                # compute the loss
                loss += criterion(decoder_output, index_char)

            # next iteration input
            decoder_input = output_char_index
    else:
        stop_char_index = char2index('EOW', vocab_out, type=torch.long, grad=False)

        for index_char in expected_output:
            decoder_output, decoder_h, decoder_c = decoding_obj(Variable(decoder_input), decoder_h, decoder_c)
            topv, topi = decoder_output.topk(1)
            output_char_index = topi.squeeze().detach()  # detach from history as input

            outputs.append(output_char_index.item())

            index_char = index_char.unsqueeze(0)

            if criterion:
                # compute the loss
                loss += criterion(decoder_output, index_char)

            if output_char_index.item() == stop_char_index:
                break

            # next iteration input
            decoder_input = output_char_index

    return loss, outputs, decoder_h, decoder_c

def train_rnn(encode_obj, decode_obj, X_train, y_train, X_dev, y_dev, vocab_in, vocab_out,
              device, eta=1e-2, reg=1e-4, num_epochs=20, reverse_input=False, teacher_forcing=1.0):
    """Train Neural Net model"""

    if device != 'cpu':
        encode_obj = encode_obj.to(device)
        decode_obj = decode_obj.to(device)

    # Loss Function
    # loss_function = torch.nn.NLLLoss().to(device)
    loss_function = torch.nn.CrossEntropyLoss(reduction='elementwise_mean').to(device)

    # Optimizer
    # encoder_optimizer = torch.optim.SGD(encode_obj.parameters(), lr=eta, weight_decay=reg)
    # decoder_optimizer = torch.optim.SGD(decode_obj.parameters(), lr=eta, weight_decay=reg)
    encoder_optimizer = torch.optim.Adam(encode_obj.parameters(), lr=eta, weight_decay=reg)
    decoder_optimizer = torch.optim.Adam(decode_obj.parameters(), lr=eta, weight_decay=reg)

    losses = []
    accuracies = []
    for epoch in range(num_epochs):
        encode_obj.train()
        decode_obj.train()

        total_loss = 0.
        rand_list = list(range(len(X_train)))
        shuffle(rand_list)  # shuffle the indexes so each training example is drawn randomly
        for index in rand_list:
            if reverse_input:
                X, y = X_train[index], y_train[index]
                X, y = np.fliplr(X[:-1].unsqueeze(0))[0], np.fliplr(y[:-1].unsqueeze(0))[0]
                X, y = np.append(X, X_train[index][-1]), np.append(y, y_train[index][-1])
                X = torch.tensor(X, dtype=torch.long, requires_grad=False)
                y = torch.tensor(y, dtype=torch.long, requires_grad=False)
            else:
                X, y = X_train[index], y_train[index]

            h, c = encode_obj.initHidden(rand=False)

            # encode
            encoder_output, h, c = encoder(encode_obj, X, h, c)

            # decode
            decoder_input = char2index('SOW', vocab_out, type=torch.long, grad=False)
            loss, outputs, decoder_h, decoder_c = decoder(decode_obj, decoder_input, h, c, vocab_out,
                                                          expected_output=y, criterion=loss_function,
                                                          teacher_forcing_prob=teacher_forcing)

            # zero the gradients before recomputing them again
            encode_obj.zero_grad()
            decode_obj.zero_grad()

            # compute the gradient
            loss.backward()
            total_loss += loss.item()

            # take a step toward the gradient direction
            encoder_optimizer.step()
            decoder_optimizer.step()

        losses.append(total_loss)
        word_accuracy = evaluate(encode_obj, decode_obj, X_dev, y_dev, vocab_out, device, reverse_input=reverse_input)
        accuracies.append(word_accuracy)
        print('Epoch: %d, Total Loss: %f, Dev accuracy: %f' %
                    (epoch + 1, total_loss, word_accuracy) )

    return encode_obj, decode_obj, accuracies, losses

def evaluate(encode_obj, decode_obj, X_dev, y_dev, vocab_out, device, reverse_input):
    """Evaluate model on data."""
    encode_obj.eval()
    decode_obj.eval()

    incorrect = 0
    total = len(X_dev)
    with torch.no_grad():
        for index in range(total):
            if reverse_input:
                X, y = X_dev[index], y_dev[index]
                X, y = np.fliplr(X[:-1].unsqueeze(0))[0], np.fliplr(y[:-1].unsqueeze(0))[0]
                X, y = np.append(X, X_dev[index][-1]), np.append(y, y_dev[index][-1])
                X = torch.tensor(X, dtype=torch.long, requires_grad=False)
                y = torch.tensor(y, dtype=torch.long, requires_grad=False)
            else:
                X, y = X_dev[index], y_dev[index]

            h, c = encode_obj.initHidden()

            # encode
            encoder_output, h, c = encoder(encode_obj, X, h, c)

            # decode
            decoder_input = char2index('SOW', vocab_out, type=torch.long, grad=False)
            loss, outputs, decoder_h, decoder_c = decoder(decode_obj, decoder_input, h, c, vocab_out,
                                                          expected_output=y, criterion=None,
                                                          teacher_forcing_prob=0.0)
            old_incorrect = incorrect
            for i, y_hat in enumerate(outputs):
                if y_hat != y[i].item():
                    incorrect += 1
                    break

            if incorrect == old_incorrect:
                correct_word = index2word(outputs, vocab_out)
                # print(outputs, correct_word)


    word_accuracy = (100.0 * (total-incorrect)) / total
    return word_accuracy


def main(num_epochs = 40, learn_rate = 1e-4, reg_const = 1e-6, teacher_forcing_prob=0.5, reverse_input=True, generate_plot=True):

    X_train, y_train, X_dev, y_dev, X_test, y_test, in_vocab, out_vocab = load_dataset()

    # processing_unit = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    processing_unit = 'cpu'
    print('Processing Unit : ', processing_unit)

    print('--Encoder Decoder LSTM--')
    emb_dim = 30
    encode_obj = UnitEncoderRNN(input_size=emb_dim, vocab_size=len(in_vocab),  hidden_size=128,
                                dropout_prob=0.0, device=processing_unit)
    decode_obj = UnitDecoderRNN(input_size=emb_dim, vocab_size=len(out_vocab),
                                hidden_size=128, dropout_prob=0.0, device=processing_unit)

    trained_encoder, trained_decoder, accuracies, losses = train_rnn(encode_obj, decode_obj, X_train, y_train, X_dev, y_dev,
                                                                     in_vocab, out_vocab, device=processing_unit,
                                                                     eta=learn_rate, reg=reg_const, num_epochs=num_epochs,
                                                                     reverse_input=reverse_input, teacher_forcing=teacher_forcing_prob)

    pickled_file = Path("trained_model_embed.pkl")
    with Path.open(pickled_file, 'wb') as f:
        pickle.dump( (trained_encoder, trained_decoder, accuracies, losses), f)

    print('--Evaluating on the test set--')
    test_accuracy = evaluate(trained_encoder, trained_decoder, X_test, y_test, out_vocab, processing_unit, reverse_input=reverse_input)
    print('Test accuracy: %f' % (test_accuracy))

    # save test accuracy to a file
    file = open('rnn_test_accuracy', 'a')
    file.write('ID: embed_%i_%.2E_%.2E\t->\tTest ACC: %f \n' %
               (num_epochs, learn_rate, reg_const, test_accuracy))
    file.close()

    if generate_plot:
        print('--Plotting--')
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(range(1, num_epochs+1), accuracies, 'bo-')
        plt.title('Validation Accuracy \n $\eta = $ %.2E $\lambda = $ %.2E' %
                  (learn_rate, reg_const))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')

        name = 'translit_embed_acc_%i_%.2E_%.2E' % \
               (num_epochs, learn_rate, reg_const)

        plt.savefig('translit_embed_%s.pdf' % name, bbox_inches="tight")

        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(range(1, num_epochs+1), losses, 'bo-')
        plt.title('Training Loss \n $\eta = $ %.2E $\lambda = $ %.2E' %
                  (learn_rate, reg_const))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        name = 'translit_embed_loss_%i_%.2E_%.2E' % \
               (num_epochs, learn_rate, reg_const)

        plt.savefig('translit_embed_%s.pdf' % name, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    main()