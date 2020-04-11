from Tokenization import Corpus
import pandas as pd
import numpy as np
import os
from Vectorization import WordsToVec
import torch
import pickle


class SentimentAnalysis:
    """performs crude sentiment analysis using Logistic regression on the vectorized sum of text data"""
    def __init__(self, corpus, vectorizer):
        self.corpus = corpus
        self.vectorizer = vectorizer
        self.model = LogisticRegression(self.vectorizer.embed_size, 2)

    def train_model(self, inputs, targets, epochs=2, batch_size=100):
        """Train the logistic regression model. Inputs should be array of texts,
        targets should be numpy array of 0's and 1's
        Uses stochastic gradient descend and Cross Entropy. lr=0.01"""

        print('Pre-processing')

        X = torch.stack([self.pre_processing(text) for text in inputs])
        Y = torch.from_numpy(targets).type(torch.LongTensor)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        print('start Training')

        for epoch in range(epochs):

            epoch_loss = 0

            #randomize data every epoch
            ran = np.arange(len(Y))
            np.random.shuffle(ran)

            X_ran = X[ran]
            Y_ran = Y[ran]

            batch_count = len(Y) // batch_size

            for k in range(batch_count):
                #training step
                optimizer.zero_grad()

                X_in = X_ran[k*batch_size : (k+1)*batch_size]
                Y_tar = Y_ran[k*batch_size : (k+1)*batch_size]

                outputs = self.model(X_in)
                loss = criterion(outputs, Y_tar)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print('epoch {} done, loss: {}'.format(epoch+1, epoch_loss/batch_count))

    def predict(self, array_of_texts):
        """predicts outcomes, Input: list of texts, output: tensor of {0, 1}'s"""
        X = torch.stack([self.pre_processing(text) for text in array_of_texts])

        prob_predicted = self.model.pred(X)

        predictions = (prob_predicted[:, 1] > prob_predicted[:, 0]).type(torch.IntTensor)

        return predictions

    def test(self, array_of_texts, targets):
        """Tests the model on list of texts against numpy of ground truth values {0,1}
        returns accuracy"""

        predictions = self.predict(array_of_texts)
        Y = torch.from_numpy(targets).type(torch.LongTensor)

        correct = (predictions == Y).sum().item()

        return correct/len(Y)

    def save_model_state(self, path):
        """path must exist"""
        torch.save(self.model.state_dict(), path)

    def load_model_state(self, path):
        self.model.load_state_dict(torch.load(path))

    def pre_processing(self, text):
        """Converts words in texts to vectors and returns sum"""

        return torch.sum(self.vectorizer.vectorize(text), dim=0)

class LogisticRegression(torch.nn.Module):
    """Logistic Regression Model with no hidden layers"""
    def __init__(self, embed_size, class_count):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(embed_size, class_count)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        return x

    def pred(self, x):
        with torch.no_grad():
            x =self.linear(x)
            x=self.softmax(x)
        return x