# deep_learning_basics
Some basic examples of deep learning neural networks architectures and algorithms

## Setup
```bash
pip install -r requirements.txt --upgrade
```

## Description
####### cnn.py

Convolutional Neural Network implemented in Pytorch for OCR optical character recognition, with the following structure:
A first convolutional layer with 20 channels and filters of size 5x5, stride of 1, and padding chosen to preserve the original image size.
A relu activation applied to the end of this layer.
Max pooling of size 2x2, with stride of 2.
A second convolutional layer with 30 channels and filters of size 7x7, stride of 1, and
padding chosen to preserve the original image size.
A relu activation applied to the end of this layer.
Max pooling of size 3x3, with stride of 3.
An affine transformation followed by an output softmax layer.
