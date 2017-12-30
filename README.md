# Language-Identifier
Language Identifier among English, French and Italian with 1-layer neural network

Part 1:

I take all the 5 adjacent letters as my input to train the NN.  When predicting a sentense, I first break it down to the 5-letters and then predict the outcome for all the 5-letters.  The prediction for the sentense is the simple majority.
Accuracy of test set after 3 epochs: 0.67

Part 2:

I tried to increase d to see whether it improves, and I tried a high and a low learning rate as well:

d = 200, eta = 0.1, epoch = 4.  dev accuracy: 0.991
d = 100, eta = 0.01, epoch = 4.  dev accuracy: 0.988
d = 100, eta = 0.5, epoch = 4.  dev accuracy: 0.983
d = 200, eta = 0.01, epoch = 4.  dev accuracy: 0.978
d = 200, eta = 0.5, epoch = 4.  dev accuracy: 0.988

The 1st and 4th model produce the highest dev accuracy.  Using the 4th model to predict the test set, the final accuracy rate for the test set is 0.68
