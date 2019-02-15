import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from pathlib import Path
import pickle
import string
import random

def get_char_index(char):
    if char == 'start':
        return 26
    elif char == 'stop':
        return 27
    else:
        return string.ascii_lowercase.index(char)


def read_data(filepath, partitions=None):
    """Read the OCR dataset."""
    labels = {}
    f = open(filepath)
    X = []
    y = []
    B = []
    words = []
    num_labels = 26
    new_word = True
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
        x = x[:, None].dot(x[None, :]).flatten()

        # bigrams
        if new_word:
            new_word = False
            first_tag = get_char_index('start')
        else:
            first_tag = y[-1]

        next_tag = k

        b = np.zeros((num_labels+2, num_labels+2), dtype=int)
        b[next_tag, first_tag] = 1

        X.append(x)
        B.append(b)
        y.append(k)

        if next_id == -1:  # current word has finished
            new_word = True  # for the next loop

            stop_bigram = np.zeros((num_labels+2, num_labels+2), dtype=int)
            stop_bigram[get_char_index('stop'), k] = 1  # next is the current tag
            B.append(stop_bigram)

            words.append((np.array(X), np.array(B), np.array(y)))
            X = []
            B = []
            y = []

    f.close()

    l = ['' for _ in labels]
    for letter in labels:
        l[labels[letter]] = letter

    return words, l

def load_dataset(filepath):
    print('Loading data...')
    pickled_file = Path("ocr_structured.pkl")
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

    return words_train, labels, words_dev, words_test

def fmap(y):
    feature_map = np.zeros((28, 28), dtype=int)
    feature_map[y[0], get_char_index('start')] = 1

    for i in range(1, len(y)):
        feature_map[y[i], y[i-1]] = 1

    feature_map[get_char_index('stop'), y[-1]] = 1
    return feature_map


def train_structured_perceptron(words_train, words_dev, labels, num_epochs=20):
    """Train with perceptron."""
    accuracies = []
    mstks = []
    Wuni = np.zeros((len(labels)+2, len(words_train[0][0][0])), dtype=float)
    Wbi = np.zeros((len(labels)+2, len(labels)+2), dtype=float)

    for epoch in range(num_epochs):
        mistakes = 0
        nchars = 0
        random.shuffle(words_train)

        for i, (Xuni, Xbi, y) in enumerate(words_train):
            # predict
            word_predicted = viterbi(Xuni, Xbi, Wuni, Wbi)

            # update
            for k in range(len(word_predicted)):
                y_cor = y[k]
                y_hat = word_predicted[k]
                if y_cor != y_hat:
                    mistakes += 1
                    # perform an update for this letter
                    Wuni[y_cor] += Xuni[k]
                    Wuni[y_hat] -= Xuni[k]

            # mistakes += np.sum(y != word_predicted)

            Wbi += fmap(y)
            Wbi -= fmap(word_predicted)

            nchars += len(y)

        accuracy = evaluate(Wuni, Wbi, words_dev)
        accuracies.append(accuracy)
        mstks.append(mistakes)
        print('Epoch: %d, Dev accuracy: %f, Mistakes: %d/%d' % (epoch+1, accuracy, mistakes, nchars))

    return Wuni, Wbi, accuracies, mstks

def viterbi(Xuni, Xbi, Wuni, Wbi):
    num_hidden = 28

    V = np.zeros((len(Xuni), num_hidden), dtype=float)
    phi = np.zeros((len(Xuni), num_hidden), dtype=int)

    # Initialize forward pass
    V[0, :] = Wbi[:, get_char_index('start')].dot(Xbi[0, :, get_char_index('start')]) + Wuni.dot(Xuni[0])

    # Forward pass: incrementally fill the table
    for i in range(1, len(Xuni)):
        for j in range(num_hidden):
            p = Wbi[j, :].dot(Xbi[i, j, :]) + V[i - 1, :]
            V[i,j] = np.max(p + Wuni.dot(Xuni[i]))
            phi[i,j] = np.argmax(p)

    # Initialize backward pass
    probs_last = np.zeros(num_hidden)
    for k in range(num_hidden):
        probs_last[k] = Wbi[get_char_index('stop'), k] * Xbi[-1, get_char_index('stop'), k] + V[-1, k]

    y_hat = np.zeros(len(Xuni), dtype=int)
    y_hat[-1] = np.argmax(probs_last)

    # backward pass: follow backpointers
    for i in range(len(Xuni)-2, -1, -1):
        y_hat[i] = phi[i+1, y_hat[i+1]]

    return y_hat

def evaluate(Wu, Wb, words):
    """Evaluate model on data."""
    correct = 0
    chars = 0

    for i, (xuni, xbi, y) in enumerate(words):
        word_predicted = viterbi(xuni, xbi, Wu, Wb)
        correct += (word_predicted == y).sum()
        chars += len(y)
    return 100.0 * correct / chars

def main():
    filepath = 'letter.data'
    num_epochs = 20

    words_train, labels, words_dev, words_test = load_dataset(filepath)

    print('--Training--')
    Wuni, Wbi, accuracies, mistakes = train_structured_perceptron(words_train, words_dev, labels, num_epochs)

    print('--Evaluating on the test set--')
    test_accuracy = evaluate(Wuni, Wbi, words_test)
    print('Test accuracy: %f' % (test_accuracy))

    print('--Plotting--')
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(range(1, num_epochs+1), accuracies, 'bo-')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')

    name = 'sequential_accuracies'
    plt.savefig('%s.pdf' % name)

    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(range(1, num_epochs+1), mistakes, 'bo-')
    plt.title('Mistakes')
    plt.xlabel('Epochs')
    plt.ylabel('Number of Mistakes')

    name = 'sequential_mistakes'
    plt.savefig('%s.pdf' % name)

if __name__ == "__main__":
    main()