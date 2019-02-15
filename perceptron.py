import numpy as np
from random import shuffle
import string
import load_dataset
import matplotlib.pyplot as plt

class SingleLayerPerceptron(object):
    def __init__(self):
        self.list_of_char = list(string.ascii_lowercase)
        self.features_size = 128
        self.classes_size = 26
        self.weights = np.zeros(self.features_size*self.classes_size, dtype=int)
        self.feature_vector =  np.zeros(self.features_size*self.classes_size, dtype=int)
        self.feature_vector_hat = np.zeros(self.features_size*self.classes_size, dtype=int)

    def reset_weights(self):
        self.weights.fill(0)

    def joint_feature_map(self, data, letter_index, hat=False):
        if hat:
            self.feature_vector_hat.fill(0)
            self.feature_vector_hat[letter_index*self.features_size : letter_index*self.features_size+self.features_size] = data
        else:
            self.feature_vector.fill(0)
            self.feature_vector[letter_index*self.features_size : letter_index*self.features_size+self.features_size] = data

    def classify(self, example):
        value = np.zeros(26)
        for class_letter in range(26):
            self.joint_feature_map(example, class_letter)
            value[class_letter] = self.weights.dot(self.feature_vector)

        return np.argmax(value)

    def train(self, labeled_data, n_epochs):
        k = 0 # number of mistakes

        for step in range(n_epochs):
            rand_list = list(range(len(labeled_data)))
            shuffle(rand_list) # shuffle the indexes so each training example is drawn randomly
            for index in rand_list:
                training_example = labeled_data[index]
                y = self.list_of_char.index(training_example[1].decode('UTF-8'))
                training_example = training_example[6:128+6]
                training_example = training_example.astype(np.int)

                y_hat = self.classify(training_example)

                if y_hat != y:
                    self.joint_feature_map(training_example, y_hat, True)
                    self.joint_feature_map(training_example, y)
                    self.weights = self.weights + self.feature_vector - self.feature_vector_hat
                    k = k + 1

        return self.weights , k

    def test(self, test_set):
        test_mistakes = 0
        # test with the given set
        for test_index in range(len(test_set)):
            test_example = test_set[test_index]
            y = self.list_of_char.index(test_example[1].decode('UTF-8'))
            test_example = test_example[6:128+6]
            test_example = test_example.astype(np.int)

            y_hat = self.classify(test_example)

            if y_hat != y:
                test_mistakes = test_mistakes + 1

        return 1 - float(test_mistakes)/len(test_set)

def main():
    print('---Loading Dataset---')
    training_data, validation_data, test_data = load_dataset.load()

    validation_accuracy = []
    test_accuracy = []
    number_epochs = 20
    perceptron_obj = SingleLayerPerceptron()
    for i in range(number_epochs):
        print('---Epoch #', i,'---')
        print('---Training---')
        perceptron_obj.train(training_data, 1)

        print('---Validating---')
        acc = perceptron_obj.test(validation_data)*100
        validation_accuracy.append(acc)
        print(acc)

        print('---Testing---')
        acc = perceptron_obj.test(test_data)*100
        test_accuracy.append(acc)
        print(acc)

    # plot accuracies
    plt.plot(validation_accuracy, label='validation')
    plt.plot(test_accuracy, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.suptitle('Perceptron (binary feature representation)')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__": main()

# names = ('id', 'letter', 'nest_id', 'word_id', 'position', 'fold')