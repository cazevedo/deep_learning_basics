import numpy as np
import csv
from time import perf_counter
import pickle
import os

def load():
    if os.path.exists('dataset.pkl'):
        with open('dataset.pkl', 'rb') as f:
            training_data = pickle.load(f)
            validation_data = pickle.load(f)
            test_data = pickle.load(f)

        return training_data, validation_data, test_data

    start = perf_counter()
    path = 'letter.data'
    dataset = np.genfromtxt(path, delimiter='\t', dtype=None)
    training_data = [list()] * 41679
    validation_data = [list()] * 5331
    test_data = [list()] * 5142

    print('Gen took {}s'.format(perf_counter() - start))
    start2 = perf_counter()

    counters = [0,0,0]

    for row in dataset:
        fold = row[5]
        if fold < 8:
            training_data[counters[0]] = list(row)
            counters[0]+=1
        elif fold == 8:
            validation_data[counters[1]] = list(row)
            counters[1] += 1
        elif fold == 9:
            test_data[counters[2]] = list(row)
            counters[2] += 1

    training_data = np.asarray(training_data)
    validation_data = np.asarray(validation_data)
    test_data = np.asarray(test_data)

    with open('dataset.pkl', 'wb') as f:
        pickle.dump(training_data, f)
        pickle.dump(validation_data, f)
        pickle.dump(test_data, f)

    print('Fill Took {}s'.format(perf_counter() - start2))

    return training_data, validation_data, test_data
