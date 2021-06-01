# Detection of Defects on Metallic Surfaces with Convolutional Neural Networks with Pytorch

## Purpose:
The purpose of this project was to train a Neural Network to recognize images, in this case, matalic surfaces with defects. 

![image]("./Fig.jpg")

## Dataset:
The dataset used can be found at [Kaggle](https://www.kaggle.com/fantacher/neu-metal-surface-defects-data). It was created by Northeastern University, and consists of 1,800 grayscale images of 6 different types of typical surface defects of the hot-rolled steel strip: rolled-in scale (RS), patches (Pa), crazing (Cr), pitted surface (PS), inclusion (In) and scratches (Sc). Each category had 300 samples to train the model on. 

## Method:
The methodology used was the creation of a convolutional neural network consisting of 3 convolutional layers and 3 dense layers, and applying rectified linear units(ReLU) on each convolutional layers with max pooling over 3x3 pixels.

For optimizer, a Adam Optimizer was used. It is a combination of the ‘gradient descent with momentum’ algorithm and the ‘RMSP’ algorithm. For Momentum part, it takes the ‘exponentially weighted average’ of the gradients into account which helps the algorithm to reach the minima faster. For Adaptive Learning Rate part, the learning rate is adjusted in the training phase by reducing the learning rate to a pre-defined schedule using ‘exponential moving average’. For loss function, the mean squared error was used because the targets are one-hot vectors.

## Results:
95% accuracy was achieved of detection by using 70% of total number of images were used to train the convolutional neural networks and using the rest 30% of images for testing. 

## Conclusion:

While this does have a high percentage of accuracy in highly critical industries, space industry for example, this accuracy should be increased. This can be done through a larger dataset.
