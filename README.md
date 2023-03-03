# SingleLayerPreceptron

Any neural network consists of interconnected nodes arranged in layers. The nodes in the input layer distribute data, and the nodes in other layers perform summation and then apply an activation function.

The connections between these nodes are weighted, meaning that each connection multiplies the transferred datum by a scalar value.

### Dataset
The data set consists of 50 samples from each of three species of Penguins (Adelie, Gentoo and Chinstrap).  

Five features were measured from each sample: bill_length, bill_depth, flipper_length, gender and body_mass, (in millimeter).

### GUI
User Input:
 Select two features
 Select two classes (C1 & C2 or C1 & C3 or C2 & C3)
 Enter learning rate (eta)
 Enter number of epochs (m)
 Add bias or not (Checkbox)

Initialization:
Number of features = 2.
Number of classes = 2.
Weights + Bias = small random numbers

Classification:
 Sample (single sample to be classified).

### Description
Visualize Penguins dataset
Penguins' dataset contains 150 samples (50 samples/class). Each sample consists of 5 features.
The first part of task(1) is analyzing the data and making a simple report to know the linear/non-linear separable features. HINT: Drawing all possible combinations of features like (X1, X2), (X1, X3), (X1, X4), (X2, X3), (X2, X4), etc as shown in the following figure and determine which features are discriminative between which classes

Implement the Perceptron learning algorithm
Single layer neural networks which can be able to classify a stream of input data to one of a set of predefined classes.

Use the penguins data in both your training and testing processes. (Each class has 50 samples: train NN with 30 non-repeated samples randomly selected, and test it with the remaining 20 samples)

After training
Draw a line that can discriminate between the two learned classes. You should also scatter the points of both classes to visualize the behavior of the line.








Test the classifier with the remaining 20 samples of each selected classes and find confusion matrix and compute overall accuracy.

