import numpy as np
import string
import load_dataset
import matplotlib.pyplot as plt

class LogisticRegression(object):
    def __init__(self):
        self.list_of_char = list(string.ascii_lowercase)
        self.features_size = int((128+1)*128.0/2)
        self.classes_size = 26
        self.weights = np.zeros(self.features_size*self.classes_size)
        self.feature_vector =  np.zeros(self.features_size*self.classes_size, dtype=int)
        self.feature_vector_hat = np.zeros(self.features_size*self.classes_size, dtype=int)
        self.pair_combination = np.zeros(self.features_size)
        self.sum_all_classes = np.zeros(self.features_size*self.classes_size)
        self.lambda_ = 0.01
        self.regularization = False

    def reset_weights(self):
        self.weights = np.zeros(self.features_size*self.classes_size)

    def joint_feature_map(self, data, letter_index, hat=False):
        self.pair_combination = data.reshape(data.size,1)*data
        self.pair_combination = self.pair_combination[np.triu_indices(data.size)].flatten()

        if hat:
            self.feature_vector_hat.fill(0)
            self.feature_vector_hat[letter_index * self.features_size: letter_index * self.features_size + self.features_size] = self.pair_combination
        else:
            self.feature_vector.fill(0)
            self.feature_vector[letter_index * self.features_size: letter_index * self.features_size + self.features_size] = self.pair_combination

    def classify(self, example):
        value = np.zeros(self.classes_size)
        for class_letter in range(self.classes_size):
            self.joint_feature_map(example, class_letter)
            value[class_letter] = self.weights.dot(self.feature_vector)

        return np.argmax(value)

    def update(self, training_example, y, step):
        normalizing_const = 0
        for class_letter in range(self.classes_size):
            self.joint_feature_map(training_example, class_letter)
            normalizing_const += np.exp(self.weights.dot(self.feature_vector))

        self.sum_all_classes.fill(0)
        for class_letter in range(self.classes_size):
            self.joint_feature_map(training_example, class_letter)
            multiplier = np.exp(self.weights.dot(self.feature_vector))/float(normalizing_const)
            self.sum_all_classes += multiplier * self.feature_vector

        self.joint_feature_map(training_example, y)

        if self.regularization:
            self.weights += step*(self.feature_vector - self.sum_all_classes + self.lambda_*self.weights)
        else:
            self.weights += step * (self.feature_vector - self.sum_all_classes)

    def draw_example(self, labeled_data, dataset_index=-1):
        if dataset_index == -1:
            training_example = labeled_data[int(np.random.rand() * len(labeled_data))]
        else:
            training_example = labeled_data[dataset_index]

        y = self.list_of_char.index(training_example[1].decode('UTF-8'))
        training_example = training_example[6:128 + 6]
        training_example = training_example.astype(np.int)

        return training_example, y

    def train(self, labeled_data, learning_rate, validation_set, epsilon):
        k = 0 # number of mistakes

        n_trained_examples = 0
        accuracy = [0]
        accuracy_variation = 100
        previous_accuracy = 0
        while accuracy_variation > epsilon:
            # draw a random labeled example from the training set
            training_example, y = self.draw_example(labeled_data, -1)

            self.update(training_example, y, learning_rate)
            n_trained_examples += 1

            if (n_trained_examples%10000 == 0):
                accuracy.append( self.test(validation_set)*100 )

                accuracy_variation = accuracy[-1] - previous_accuracy
                previous_accuracy = accuracy[-1]

                print(n_trained_examples)
                print(accuracy[-1])

        print(n_trained_examples)
        return accuracy, n_trained_examples

    def test(self, test_set):
        test_mistakes = 0
        # test with the given set
        for test_index in range(len(test_set)):
            test_example, y = self.draw_example(test_set, test_index)

            y_hat = self.classify(test_example)

            if y_hat != y:
                test_mistakes = test_mistakes + 1

        return 1 - float(test_mistakes)/len(test_set)

def main():
    print('---Loading Dataset---')
    training_data, validation_data, test_data = load_dataset.load()

    test_accuracy = []
    lr_obj = LogisticRegression()

    print('---Training---')
    vaidation_accuracy, n_trained_examples = lr_obj.train(training_data, 0.001, validation_data, 0.1)

    print('---Testing---')
    acc = lr_obj.test(test_data)*100
    test_accuracy.append(acc)
    print(acc)

    plt.plot(np.arange(10000, n_trained_examples+1, 10000), vaidation_accuracy[1:])
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy (%)')
    plt.suptitle('Logistic Regression (pairwise feature representation)')
    plt.show()

if __name__ == "__main__": main()

# names = ('id', 'letter', 'nest_id', 'word_id', 'position', 'fold')
