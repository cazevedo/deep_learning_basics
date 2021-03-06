# deep_learning_basics
Some basic examples of deep learning neural networks that ilustrate the basic architectures and algorithms and respective implementations in python/pytorch.

## Setup
install pytorch: https://pytorch.org/get-started/locally/
```bash
pip install -r requirements.txt --upgrade
```

## Dataset
The OCR dataset used in these examples can be found here: http://ai.stanford.edu/~btaskar/ocr
The Arabic-English transliteration dataset can be found here: https://github.com/googlei18n/transliteration

## Description
##### perceptron.py and perceptron_joint_features.py

Perceptron algorithm implemented using just plain algebra. This linear classifier is used for optical character recognition in the OCR dataset. The first implementation used the binary pixel values as feature representation, while the second takes advantage of the correlation between neighbour pixels and uses pairwise combinations of all pixels.

##### logistic_regression.py

Logistic regression algorithm, with stochastic gradient descent as tranining algorithm, implemented using just plain algebra. This linear classifier is used for optical character recognition in the OCR dataset.

##### multilayer_perceptron.py

Multilayer perceptron (feed-forward neural network) implementation using just plain algebra. This classifier is used for optical character recognition in the OCR dataset.

##### logistic_regression_pytorch.py

l2-regularized logistic regression implemented using pytorch. This classifier is used for optical character recognition in the OCR dataset.

##### feedforward_pytorch.py

Feed-forward neural network using dropout regularization, implemented using pytorch. This classifier is used for optical character recognition in the OCR dataset.

##### dynamic_programming.py

Implementation of viterbi and forward-backward algorithms in order to predict the most likely weather, by knowing John's past activities (emission and transition probabilities can be found in the load_dataset() method).

##### structured_perceptron.py

Implementation of a structured perceptron, that takes advantage of the sequential structure of the characters (as they form words) in the OCR dataset. It makes use of the viterbi algorithm to perform the word prediction and keeps the perceptron weight update to improve both the unigram and the bigram weigths.

##### cnn.py

Convolutional Neural Network implemented in Pytorch for OCR optical character recognition, with the following structure:
* Convolutional layer with 20 channels and filters of size 5x5, stride of 1, and padding chosen to preserve the original image size.
* Relu activation applied to the end of this layer.
* Max pooling of size 2x2, with stride of 2.
* Convolutional layer with 30 channels and filters of size 7x7, stride of 1, and padding chosen to preserve the original image size.
* Relu activation applied to the end of this layer.
* Max pooling of size 3x3, with stride of 3.
* Affine transformation followed by an output softmax layer.

##### rnn.py

BILSTM tagger, that consists in a feedforward layer, followed by a bidirectional LSTM and an affine transformation with an output softmax layer. This architecture explores the sequential order of the characters.

##### transliteration_onehot.py

Sequence-to-sequence model to transliterate Arabic to English words. Implemented using an encoder-decoder architecture with two unidirectional LSTMs (one encoder LSTM and one decoder LSTM), where the input/output is represented by one-hot embeddings.

##### transliteration_embed.py

Sequence-to-sequence model to transliterate Arabic to English words. Implemented using an encoder-decoder architecture with two unidirectional LSTMs (one encoder LSTM and one decoder LSTM), where the input/output embeddings representation was learned through a fully connected layer.

##### transliteration_att.py

Sequence-to-sequence model to transliterate Arabic to English words. Implemented using an encoder-decoder architecture with a bidirectional LSTM (encoder) and an unidirectional LSTM (decoder) with an attention mechanism.
