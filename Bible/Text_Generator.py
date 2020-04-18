import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class TextGenerator:
    """Initiates LSTM based model for text generation"""
    def __init__(self, corpus, vectorizer, LSTM_count=4):
        self.corpus = corpus
        self.vocab_size = vectorizer.vocab_size
        self.embed_size = vectorizer.embed_size+len(self.corpus.punctuation)

        #add more dimensions to the vectorization embedding
        self.embedding = nn.functional.pad(vectorizer.model.embed.weight,
                                           (0, len(self.corpus.punctuation)),
                                           'constant',
                                           0).detach()


        #clear and set punctuation embeddings
        for k, word in enumerate(self.corpus.punctuation):
            self.embedding[self.corpus.word_to_ind[word], :] = 0
            self.embedding[self.corpus.word_to_ind[word], vectorizer.embed_size+k] = 1

        #clear and set UNKNOWN embedding:
        self.embedding[self.corpus.word_to_ind[self.corpus.UNKNOWN], :] = 0
        self.embedding[self.corpus.word_to_ind[self.corpus.UNKNOWN], -1] = 1

        self.LSTM_model = LSTMModel(input_size=self.vocab_size, embed_size=self.embed_size,
                                    hidden_size=128, class_count=self.vocab_size, LSTM_count=LSTM_count)

        self.LSTM_model.embedding.weight.data = self.embedding

    def pre_processing(self, text, batch_size, section_length):
        """Takes in string of text and prepares it for training:
        -parses text
        -generates input and target sequence
        -splits sequences into batches
        -returns batch list of (input, target tensor of shape (b-s, section_length)), int batch_count"""

        parse_text = self.corpus.parse_to_index(text, use_UNKNOWN=True, use_punctuation=True)

        batch_count = len(parse_text) // (batch_size*section_length)

        #crop text:
        input_seq = parse_text[:(batch_count*batch_size*section_length)]

        #shift to create target
        target_seq = input_seq[1:]
        target_seq.append(input_seq[0])

        #break sequences into batch many sequences viewed as rows of a tensor
        X = torch.Tensor(input_seq).view(batch_size, -1).type(torch.LongTensor)
        Y = torch.Tensor(target_seq).view(batch_size, -1).type(torch.LongTensor)

        #create batch list
        batches = []
        for k in range(batch_count):
            X_batch = X[:, k*section_length : (k+1)*section_length]
            Y_batch = Y[:, k*section_length : (k+1)*section_length]
            batches.append((X_batch, Y_batch))

        return batches, batch_count

    def train_model(self, text, section_length=32, batch_size=16, epochs=100, lr=0.001):
        """trains model, input should be text file"""

        print('Pre-processing')

        batches, batch_count = self.pre_processing(text, batch_size=batch_size, section_length=section_length)

        #initialize parameters:
        optimizer = torch.optim.Adam(self.LSTM_model.parameters(), lr=lr)

        print('Begin Training')

        for epoch in range(epochs):

            # model in training  mode:
            self.LSTM_model.train()

            #initialize model state:
            state = self.LSTM_model.initialize_state(batch_size)

            running_loss = 0

            #training loop
            for k, batch in enumerate(batches):

                #reset grad:
                optimizer.zero_grad()

                X, Y = batch

                #forward step:
                loss, stat = self.LSTM_model(X, prev_state=state, target=Y)

                running_loss += loss.item()

                #backward prop:
                loss.backward()

                #gradient clipping:
                _ = torch.nn.utils.clip_grad_norm_(self.LSTM_model.parameters(), 5)

                #step
                optimizer.step()

                if (k % 100) == 99:
                    print('Epoch: {}, batch: {} of {}, loss: {}'.format(epoch+1, k+1, batch_count, running_loss/100))
                    running_loss = 0

            #User update and sample printing:
            creativity = int(np.random.choice(range(2, 6)))
            print('Epoch {} done, here is a sample (creativity {}):'.format(epoch+1, creativity))

            sample_indices = self.generate('jesus', 200, creativity=creativity)

            print(self.unparse(sample_indices))

            #save model state every 5 epochs:
            if (epoch % 10) == 9:
                path = './Parameters/text_generator_autosave_epoch'+str(epoch+1)+'.pt'
                torch.save(self.LSTM_model.state_dict(), path)


    def generate(self, start_words, max_length, creativity=5):
        """Generates text sample starting with start words, text sample will be returned as list of indices"""

        start_indices = self.corpus.parse_to_index(start_words, use_punctuation=True)

        #initialize hidden states
        state = self.LSTM_model.initialize_state(1)

        result_indices = start_indices

        self.LSTM_model.eval()

        for k in range(max_length):

            # create first input into network
            X = torch.Tensor(start_indices).view(1, -1).type(torch.LongTensor)

            #run through network:
            with torch.no_grad():
                probs, state = self.LSTM_model(X, state)

            #find indices for next word
            probs = probs[-1] #only care about the last entry if X was a list
            _, candidates = torch.topk(probs, creativity)
            candidates = candidates.tolist()

            #drop UNKNOWN from candidates:
            if self.corpus.word_to_ind[self.corpus.UNKNOWN] in candidates:
                candidates.remove(self.corpus.word_to_ind[self.corpus.UNKNOWN])

            choice = np.random.choice(candidates)

            result_indices.append(choice)
            start_indices = [choice]

        return result_indices

    def unparse(self, indices):
        """turns list of indices into text:"""

        words = [self.corpus.ind_to_word[index] for index in indices]

        text = ' '.join(words)

        for word in self.corpus.punctuation:
            text = text.replace(' '+word, word)

        return text

    def load_model(self, path):
        self.LSTM_model.load_state_dict(torch.load(path))

class LSTMModel (nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, class_count, LSTM_count):
        super(LSTMModel, self).__init__()

        self.LSTM_layers = LSTM_count
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, embed_size)
        self.LSTM = nn.LSTM(embed_size, self.hidden_size, num_layers=self.LSTM_layers,
                            bias=True, batch_first=True, dropout=0.25)

        self.Adaptive_Softmax = nn.AdaptiveLogSoftmaxWithLoss(self.hidden_size, class_count,
                                                              [round(class_count / 20), 4*round(class_count / 20)])

    def forward(self, x, prev_state, target=None):
        #x, target have shape (b-s, section_length), with values  < input_size


        x = self.embedding(x) #-> (b-s, s-l, embed_size)

        x, state = self.LSTM(x, prev_state)   #->(b-s, s-l, hidden_size), (state variables)


        x = x.reshape(-1, self.hidden_size) #->(b-s*s-l, h-s)


        if self.training: #only calculate loss
            target = target.reshape(-1)  # ->(b-s*s-l)
            loss = self.Adaptive_Softmax(x, target).loss
            return loss, state

        else:  #return probability distribution over classes
            probs = self.Adaptive_Softmax.log_prob(x)
            return probs, state

    def initialize_state(self, batch_size):
        # initialize internal states of LSTM:
        h = torch.zeros(self.LSTM_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.LSTM_layers, batch_size, self.hidden_size)
        state = (h, c)

        return state
