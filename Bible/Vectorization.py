import numpy as np
import torch
import torch.nn as nn
import torch.optim


class WordsToVec:
    """Vectorization class, contains and learns skipGram model with embed_size"""
    def __init__(self, corpus, embed_size):
        self.model = None
        self.parameters = None
        self.corpus = corpus
        self.vocab_size = len(self.corpus.vocab)
        self.embed_size = embed_size
        self.model = SkipGramModel(self.vocab_size, self.embed_size)

    def pre_process(self, array_of_texts, window_size):
        """Takes (numpy) array of strings, parses and tokenizes each string, returns data tensor for skipGram training:
        result[0] = numeric labels of input
        result[1] = numeric labels of target context words
        Note that for any input label x there is one line in results for every possible context word y_k"""

        print('read in, parse data and convert to numbers')
        #result is list of lists
        X = [self.corpus.parse_to_index(text) for text in array_of_texts]

        result = [[], []]

        print('creating result')
        # for every array in X, rotate it by 1,..., window_size/2 in both directions
        # and collect all results in a single pair of lists
        for array in X:
            for i in range(1, window_size // 2 + 1):
                result[0].extend(array[:-i])
                result[0].extend(array[i:])
                result[1].extend(array[i:])
                result[1].extend(array[:-i])

        print('converting to tensor')

        return torch.tensor(result)

    def train_word_vects(self, data, epochs=5, batch_size=1000):
        """Trains skipGram model on data using AdaptiveLogSoftmax, AdaGrad, and gradient clipping. lr=1
        data should be a output from self.pre_process (pair of lists of input and context words)"""

        optimizer = torch.optim.Adagrad(self.model.parameters(), lr=1)

        batch_count = len(data[0]) // batch_size

        print('Start Training')

        for epoch in range(epochs):

            epoch_loss = 0

            #randomize data in every epoch
            data = WordsToVec.data_randomizer(data)

            running_loss = 0

            for k in range(batch_count):

                optimizer.zero_grad()

                #find batch
                inputs = data[0][k * batch_size: (k + 1) * batch_size]
                targets = data[1][k * batch_size: (k + 1) * batch_size]

                #forward step
                outputs = self.model(inputs, targets)

                loss = outputs.loss

                #backprop
                loss.backward()

                #update parameters
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.01)

                optimizer.step()

                #printing status update
                running_loss += loss.item()

                if k % 100 == 99:
                    print('Epoch: {}, batch: {}, loss: {}'.format(epoch + 1, k + 1, running_loss / 100))
                    epoch_loss += running_loss
                    running_loss = 0
            epoch_loss += running_loss
            print('Epoch: {}, epoch loss: {}'.format(epoch + 1, epoch_loss / batch_count))

        print('Finished training')

    def save_model_state(self, path):
        """path must exist"""
        torch.save(self.model.state_dict(), path)

    def load_model_state(self, path):
        self.model.load_state_dict(torch.load(path))

    def vectorize(self, text, use_UNKOWN=False, use_STOP=False):
        """Takes text and returns tensor of word vectors of shape (word_count, embed_size)"""
        parsed = self.corpus.parse_to_index(text, use_UNKNOWN=use_UNKOWN, use_STOP=use_STOP)

        if self.model == None:
            print('Not trained yet')
            return
        with torch.no_grad():
            result = self.model.embed(torch.Tensor(parsed).type(torch.LongTensor))

        return result

    @staticmethod
    def data_randomizer(data):
        """input: (2, N) tensor data
        output: (2, N) tensor with rows randomized"""

        ran = np.arange(len(data[0]))
        np.random.shuffle(ran)

        result = data[:, ran]

        return result


class SkipGramModel(nn.Module):
    """SkipGram module with 1 hidden layer"""
    def __init__(self, vocab_size, embed_size):
        super(SkipGramModel, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.adaptive = nn.AdaptiveLogSoftmaxWithLoss(embed_size, vocab_size,
                                                      cutoffs=[round(vocab_size / 20), 4 * round(vocab_size / 20)])

    def forward(self, x, target):
        x = self.embed(x)
        x = self.adaptive(x, target)

        return x
