# image_regression_karpathy
A Tensorflow implementation of "A neural network paints an image" as seen at http://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html

#### How to run the regressor
~$ python painter.py [filename.jpg]

#### Technical Details
The code implements a 2 hidden layer ANN to regress on the pixel position of a single image to its RGB mapping.

The input is thus has 2 features (x,y values of the pixel location) and the output has 3 features (R,G and B values of the pixel)
The hidden layers have presently been set to have 20 neurons each. This can be changed at line 35 of the code.
The optimizer used is Adam. This can be changed in line 60 of the code.

The number of hidden layers is kept fixed though. As a further extension to this project, a list of weight,bias and activation tensors can be created to accomodate more hidden layers.

#### Interesting Observations

In the examples folder, I have included 2 examples that I trained on. One was a picture of Dora The Explorer and other was a logo of Reddit.
If we observe the painting after 900 epochs, we see that the painting of Dora is starting to look like the original though Reddit is not even close.
Notice that the picture of Dora covers a lot of the image area though the Reddit logo is just a set of curves.
This is probably because the neural network is not looking for features in the image (unlike a CNN) but rather just regressing on the color maps. As a result, the painting of Dora which is originally filled with colors is starting to look like the original but the picture of Reddit Logo which occupies just a small space doesn't ; we just see blackish regions aroung the place where the curve should be.
