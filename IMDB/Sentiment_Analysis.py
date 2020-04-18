
import pandas as pd
import numpy as np
import torch
import torch.nn as nn



def save_model_state(path, model):
    """path must exist"""
    torch.save(model.state_dict(), path)

def load_model_state(path, model):
    model.load_state_dict(torch.load(path))


class SentimentAnalysis:
    """Implements sentiment analysis using two models:
    self.model is linear regression on context vector sums
    self.GRU uses two layer GRU with dropout"""
    def __init__(self, corpus, vectorizer, train_embedding=False):
        self.corpus = corpus
        self.vectorizer = vectorizer

        self.train_embedding = train_embedding
        self.embed_size = self.vectorizer.embed_size+2

        #Initialize embedding and add extra dim for STOP and UNKNOWN
        self.embedding = (nn.functional.pad(self.vectorizer.model.embed.weight,
                                                  (0, 2, 0, 2),
                                                  'constant',
                                                  0)
                                             ).detach()
        self.embedding[self.corpus.word_to_ind[self.corpus.STOP]][-2] = 1
        self.embedding[self.corpus.word_to_ind[self.corpus.UNKNOWN]][-1] = 1

        self.model = LogisticRegression(self.vectorizer.embed_size, 2)
        self.GRU = GRU(embed_size=self.embed_size, class_count=2,
                       embed=self.train_embedding, vocab_size=self.vectorizer.vocab_size)

        if train_embedding:
            self.GRU.learn_embed.weight.data = self.embedding

    def reg_train(self, inputs, targets, epochs=2, batch_size=100):
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
            X_ran, Y_ran = SentimentAnalysis.shuffle(X, Y)

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

    def reg_predict(self, array_of_texts):
        """predicts outcomes, Input: list of texts, output: tensor of {0, 1}'s"""
        X = torch.stack([self.pre_processing(text) for text in array_of_texts])

        prob_predicted = self.model.pred(X)

        predictions = (prob_predicted[:, 1] > prob_predicted[:, 0]).type(torch.IntTensor)

        return predictions

    def reg_test(self, array_of_texts, targets):
        """Tests the model on list of texts against numpy of ground truth values {0,1}
        returns accuracy"""

        predictions = self.reg_predict(array_of_texts)
        Y = torch.from_numpy(targets).type(torch.LongTensor)

        correct = (predictions == Y).sum().item()

        return correct/len(Y)

    def gru_pre_processing(self, inputs, targets):
        """precprocesses data for gru training
        input = List of strings (inputs), Numpy of labels (targets)
        output = pd DataFrame with columns 'X' ->containing word indices of input strings
                                            'Y' -> containing labels
                                             'Lengths' -> containing length of sequences in 'X' """

        print('Pre-processing')
        # turn texts into list of lists of indices:
        length = len(inputs)
        indexed_data = []
        for i, text in enumerate(inputs):
            indexed_data.append(self.corpus.parse_to_index(text))

            # print user update:
            if (i % 1000) == 999:
                print('Done with entry {} of {}'.format(i + 1, length))

        print('Creating DataFrame')
        data = pd.DataFrame()
        data['X'] = indexed_data
        data['Y'] = targets
        data['Lengths'] = data.X.map(lambda x: len(x))  # record sequence length for each training example

        return data

    def gru_batch_prep(self, batch, batch_size):
        """prepares batch for forward step and loss evaluation in gru training
        Input: DataFrame with columns 'X', 'Y', 'Lengths'
        Output: (packed padded sequence of x values, target tensor Y (size (batch_size))"""

        # sort batch by sequence lengths in batch['X']
        batch = batch.sort_values(by='Lengths', ascending=False)
        batch = batch.reset_index(drop=True)

        # create targets:
        Y = torch.tensor(batch.Y.values).type(torch.LongTensor)

        # create tensor of padded sequences of input values:
        max_length = batch.Lengths[0]
        X = torch.zeros((max_length, batch_size, self.embed_size))

        for k in range(batch_size):
            if self.train_embedding:
                seq_as_tensor = torch.Tensor(batch.X[k]).type(torch.LongTensor)
                X[range(batch.Lengths[k]), k] = self.GRU.learn_embed(seq_as_tensor)
            else:
                X[range(batch.Lengths[k]), k] = self.embedding[batch.X[k]]

        lengths = torch.tensor(batch.Lengths.values)

        # pack sequences
        X_packed = nn.utils.rnn.pack_padded_sequence(X, lengths=lengths, enforce_sorted=True, batch_first=False)

        return X_packed, Y

    # noinspection PyTypeChecker
    def gru_train(self, inputs, targets, epochs=2, batch_size=100):
        """Trains GRU unit using inputs and targets:
        Inputs:  List of strings (texts)
        Targets: Numpy of labels"""

        data = self.gru_pre_processing(inputs, targets)

        # randomizing data for train - valid split
        ran = np.arange((len(data)))
        np.random.shuffle(ran)
        data_ran = data.iloc[ran]
        data_ran = data_ran.reset_index(drop=True)

        #split data
        train_data = data_ran.iloc[:(int(-0.2*len(data)))]
        valid_data = data_ran.iloc[(int(-0.2 * len(data))):]
        valid_data = valid_data.reset_index(drop=True)

        #batch-prep validation data once
        X_valid, Y_valid = self.gru_batch_prep(valid_data, len(valid_data))

        #Initializing parameters:
        batch_count = len(train_data) // batch_size
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.GRU.parameters(), lr=0.001)
        min_valid_loss = 1.0
        impatience = 0
        patience = 5

        print('Start Training')

        for epoch in range(epochs):

            print('Starting epoch {}'.format(epoch+1))

            running_loss = 0

            #randomizing data
            ran = np.arange((len(train_data)))
            np.random.shuffle(ran)
            data_ran = train_data.iloc[ran]
            data_ran = data_ran.reset_index(drop=True)

            #training loop
            for batch_num in range(batch_count):

                print('Working on batch {} of {}'.format(batch_num, batch_count))

                #initialize gradients
                optimizer.zero_grad()

                #Allocate batch
                batch = data_ran.iloc[batch_num*batch_size : (batch_num+1)*batch_size]

                #Sort and pack batch
                X, Y = self.gru_batch_prep(batch, batch_size)

                #forward step
                Y_pred = self.GRU(X)

                #print(Y_pred)

                loss = criterion(Y_pred, Y)

                #print(loss.item())

                #backprop:
                loss.backward()

                #update
                optimizer.step()

                #user update
                running_loss += loss.item()
                if (batch_num % 50) == 49:
                    print('Epoch: {}, batch: {}, loss: {}'.format(epoch+1, batch_num+1, running_loss/50))
                    running_loss = 0


            #running validation data:
            print('Validating model')
            with torch.no_grad():
                Y_valid_pred = self.GRU(X_valid)

            #print validation loss:
            loss_valid = criterion(Y_valid_pred, Y_valid).item()
            print('Epoch: {}, validation loss: {}'.format(epoch+1, loss_valid))

            #print validation accuracy:
            pred_valid = (Y_valid_pred[:, 1] > Y_valid_pred[:, 0]).type(torch.IntTensor)
            correct_valid = (pred_valid == Y_valid.type(torch.IntTensor)).sum().item()
            print('Epoch: {}, validation accuracy: {}'.format(epoch+1, float(correct_valid)/len(Y_valid)))

            #check for early stopping:
            impatience += 1                                                              #update impatience
            if loss_valid < min_valid_loss:
                min_valid_loss = loss_valid                                              #update loss
                save_model_state('./IMDB/Parameters/analyzerGRU_autosave.pt', model=self.GRU) #save model config
                impatience = 0                                                           #reset impatience
            elif impatience > patience:
                print('Training ended due to early stopping, best model was saved')
                break

    def gru_predict(self, inputs, targets=None):
        """Predicts labels for list of strings input"""

        if targets is None:
            Y = np.zeros(len(inputs)) #dummy
        else:
            Y = targets

        data = self.gru_pre_processing(inputs, Y)

        X, Y =self.gru_batch_prep(data, len(data)) #prepare data, shuffle Y

        print('Predicting')
        self.GRU.eval()
        with torch.no_grad():
            probs = self.GRU(X)

        self.GRU.train() #immediately put model back in train mode

        predictions = (probs[:, 1] > probs[:, 0]).type(torch.IntTensor)

        if targets is None:
            return predictions
        else:
            return predictions, Y

    def gru_test(self, inputs, targets):
        """Returns accuracy fraction of model on inputs (list of strings) and targets (numpy of labels)"""

        predictions, Y = self.gru_predict(inputs, targets=targets)

        correct = (predictions == Y).sum().item()

        return correct / len(targets)

    def pre_processing(self, text):
        """Converts words in texts to vectors and returns sum"""

        return torch.sum(self.vectorizer.vectorize(text), dim=0)

    def text_to_seq_of_sentence_vects(self, text):
        """Splits texts along stop words and returns a tensor of word-vector-sums over sentences.
        Input = String
        Output = Tensor of embedding vectors, shape is (#sentences, embed_dim)"""

        #split text at stop words:
        for word in self.corpus.stop_words:
            text = text.replace(word, '.')

        sentences = text.split('. ')

        result = [torch.sum(self.vectorizer.vectorize(sentence), dim=0) for sentence in sentences]

        return torch.stack(result, dim=0)

    def vectorize(self, text):
        """Takes text and returns tensor of word vectors of shape (word_count, embed_size)"""
        parsed = self.corpus.parse_to_index(text, use_UNKNOWN=True, use_STOP=True)

        result = self.embedding[parsed]

        return result

    @staticmethod
    def shuffle(X, Y):
        """Shuffle input and target data X, Y"""
        ran = np.arange(len(Y))
        np.random.shuffle(ran)

        X_ran = X[ran]
        Y_ran = Y[ran]

        return X_ran, Y_ran

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
            x = self.linear(x)
            x = self.softmax(x)
        return x

class GRU(torch.nn.Module):
    """Bi-directional GRU with a final linear layer:
    Input: packed sequence of inputs of size embed_size (packed from (max_seq_len, batch_size, embed_size))
    Step: bi-directional GRU outputting mean of all sequential outputs to final linear layer
    Output: tensor of shapee (batch_size, class_count)"""
    def __init__(self, embed_size, class_count, embed=False, vocab_size=None):
        super(GRU, self).__init__()

        #hidden size is same as output/input sizee
        self.hidden_size = embed_size

        if embed: #create embedding layer, this is called in batch-prep
            self.learn_embed = nn.Embedding(vocab_size, embed_size)

        self.GRU_layer = nn.GRU(input_size=embed_size, hidden_size=self.hidden_size, bias=True,
                                batch_first=False, bidirectional=True, num_layers=2, dropout=0.5)
        self.dropout = nn.Dropout(0.3)
        self.linear_layer = nn.Linear(self.hidden_size, class_count)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        #x is packed sequence
        x, _ = self.GRU_layer(x)
        x, lengths = nn.utils.rnn.pad_packed_sequence(x) #x has shape (max_len, batch_size, 2*hidden_size
                                                         #lengths has shape (batch_size)
        x = x[:, :, :self.hidden_size] + x[:, :, self.hidden_size:] #sum forward and backward vectors
        lengths = 2*lengths                              #double lengths as each sequence has forward and backward vect
        x = x.sum(0)                                     #sum over all output vectors, x has shape (batch_size, hidden_size)
        x = x / lengths.view(-1, 1)                      #divide out length of sequences

        x = self.dropout(x)
        x = self.linear_layer(x)                         #(b_s, h_s) -> (b_s, class_count)
        x = self.softmax(x)

        return x









