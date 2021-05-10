# Problem -1 for Assignment-3 
In this we we were supposed to make a simple fully connected neural netwrok to identify the letter from the given 16 features. 
I made a network with 3 layers: 1 input layer with 16 units, 1 hidden layer with 20 units and 1 output layer with 26 units.The loss function I used is categorical cross entropy with softmax.
I have used He initialization to initialize the weights and used ReLU activation in the hidden layer. I got a validation accuracy of 84% and validation loss of 0.53.
Using the chain rule I first calculated the partial derivatives with respect to the weights and biases for neurons going from the 2nd to 3rd layer and also with respect to the actiavtions of the previous layer. 
Using the last partial derivative I then calculated the partial derivatives with respect to weights and biases for neurons going from 1st to 2nd layer.
I took the average of these derivatives over a batch size of 130 and using an appropriate learning rate changed the weights and biasses. I ran this program for 20 epochs to get this accuracy. I have saved the predictions for the test set as test_predictions.csv in this folder. 
