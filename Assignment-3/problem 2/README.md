# Problem-2 for Assignment-3
In this we were supposed to identify the number from the a given 28x28 image. I used tensorflow.keras to create this neural network. 
I have used a convolutional neural net to predict the digits. There are 2 convolution layers with 32 and 64 filters each of 3x3 size. Each of them is followed by a max pooling layer. 
After that is a fully connected layer with 512 units which finally concets to the output layer with 10 units corresponding to the predicted digits.
I have used categorical cross-entropy loss function with softmax for the gradient descent and ReLU activation in each convolution layer.
I have used a batch size of 128 and ran the program for 10 epochs. I was able to achieve a validation accuracy of 98.4% and here is the screenshot of my Kaggle score:
![Kaggle Score](https://github.com/Karrthik-Arya/Moodify-Learning/blob/master/Assignment-3/problem%202/Kaggle_score.png)
