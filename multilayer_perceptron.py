import numpy as np
import string
import load_dataset
import matplotlib.pyplot as plt
import sys

class MultiLayerPerceptron(object):
    def __init__(self, n_hidden_layers, n_neurons):
        self.n_hidden_layers = n_hidden_layers
        self.features_size = 128
        self.classes_size = 26
        self.input_weights = np.random.rand(n_neurons[0] , self.features_size)
        self.input_bias = np.zeros((n_neurons[0],1))
        self.output_weights = np.random.rand(self.classes_size, n_neurons[-1])
        self.output_bias = np.zeros((self.classes_size,1))
        self.output = np.zeros((self.classes_size,1))
        self.h_vector = np.zeros((n_neurons[0],1))
        self.z_vector = np.zeros((n_neurons[0],1))
        self.o_vector = np.zeros((self.classes_size,1))

    def preactivation(self, feature_vector):
        self.z_vector = self.input_weights.dot(feature_vector) + self.input_bias

    def activation(self):
        self.h_vector = np.tanh(self.z_vector)

    def output_activation(self):
        self.o_vector = self.output_weights.dot(self.h_vector) + self.output_bias
        self.softmax()

    def softmax(self):
        exp_values = np.exp(self.o_vector)
        self.output = exp_values/exp_values.sum()

    def feedforward(self, example):
        self.preactivation(example)
        self.activation()
        self.output_activation()

    def predict(self, example):
        self.feedforward(example)
        return np.argmax(self.output)

    def backpropagation(self, example, y, eta, reg_lambda):
        self.feedforward(example)

        bgrad_1 = self.output
        bgrad_1[y] -= 1
        d_output_weights = self.h_vector.dot(bgrad_1.T)
        d_output_bias = bgrad_1

        bgrad_2 = bgrad_1.T.dot(self.output_weights) * (1 - np.power(self.h_vector, 2).T)

        d_input_weigths = np.dot(example, bgrad_2)
        d_input_bias = bgrad_2

        # Regularization
        d_output_weights += reg_lambda * self.output_weights.T
        d_input_weigths += reg_lambda * self.input_weights.T

        # Parameters update
        self.input_weights += -eta * d_input_weigths.T
        self.input_bias += -eta * d_input_bias.T
        self.output_weights += -eta * d_output_weights.T
        self.output_bias += -eta * d_output_bias

def draw_example(labeled_data, dataset_index=-1):
    list_of_char = list(string.ascii_lowercase)
    if dataset_index == -1:
        training_example = labeled_data[int(np.random.rand() * len(labeled_data))]
    else:
        training_example = labeled_data[dataset_index]

    y = list_of_char.index(training_example[1].decode('UTF-8'))
    training_example = training_example[6:128 + 6]
    training_example = training_example.astype(np.int)

    return training_example.reshape(-1,1), y

def main():
    print('---Loading Dataset---')
    training_data, validation_data, test_data = load_dataset.load()

    test_accuracy = []
    mlp = MultiLayerPerceptron(n_hidden_layers=1, n_neurons=[50])

    epochs = 20
    for k in range(epochs):
        print('---Training---')
        for i in range(len(training_data)):
            training_example, y = draw_example(training_data)
            mlp.backpropagation(training_example, y, eta=1e-2, reg_lambda=1e-3)

        print('---Testing---')
        mistakes = 0
        where_ok = [0] * 26
        for j in range(len(test_data)):
            validation_example, y = draw_example(test_data, j)
            y_hat = mlp.predict(validation_example)

            if y_hat != y:
                mistakes += 1

            else:
                where_ok[y] +=1

        test_accuracy.append(1-mistakes/len(test_data))
        print('Acc : ', 1-mistakes/len(test_data))
        print(where_ok)

    plt.plot(test_accuracy, label='test set')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.suptitle('MultiLayer Perceptron')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__": main()