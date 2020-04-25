import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from Beaming import Beam
from nltk.translate.bleu_score import sentence_bleu


class Translator:
    def __init__(self, source_corpus, target_corpus, source_vect, target_vect,
                 dec_GRU_count=2, enc_GRU_count=2, hidden_size=128, use_feedforward=True):
        self.source_corpus = source_corpus
        self.target_corpus = target_corpus
        self.source_vect = source_vect
        self.target_vect = target_vect
        self.source_embed_size = self.source_vect.embed_size + len(self.source_corpus.punctuation) + 1
        self.target_embed_size = self.target_vect.embed_size + len(self.target_corpus.punctuation) + 1

        # add more dimensions to the vectorization embeddings
        self.source_embedding = nn.functional.pad(source_vect.model.embed.weight,
                                                  (0, len(self.source_corpus.punctuation) + 1),
                                                  'constant',
                                                  0).detach()
        self.target_embedding = nn.functional.pad(target_vect.model.embed.weight,
                                                  (0, len(self.target_corpus.punctuation) + 1),
                                                  'constant',
                                                  0).detach()

        # clear and set punctuation embeddings
        for k, word in enumerate(self.source_corpus.punctuation):
            self.source_embedding[self.source_corpus.word_to_ind[word], :] = 0
            self.source_embedding[self.source_corpus.word_to_ind[word], source_vect.embed_size + k] = 1

        for k, word in enumerate(self.target_corpus.punctuation):
            self.target_embedding[self.target_corpus.word_to_ind[word], :] = 0
            self.target_embedding[self.target_corpus.word_to_ind[word], target_vect.embed_size + k] = 1

        # clear and set UNKNOWN embedding:
        self.source_embedding[self.source_corpus.word_to_ind[self.source_corpus.UNKNOWN], :] = 0
        self.source_embedding[self.source_corpus.word_to_ind[self.source_corpus.UNKNOWN], -1] = 1

        self.target_embedding[self.target_corpus.word_to_ind[self.target_corpus.UNKNOWN], :] = 0
        self.target_embedding[self.target_corpus.word_to_ind[self.target_corpus.UNKNOWN], -1] = 1

        self.model = TranslatorModel(self.source_vect.vocab_size, self.target_vect.vocab_size, self.source_embed_size,
                                     hidden_size=hidden_size, GRU_count_dec=dec_GRU_count, GRU_count_enc=enc_GRU_count,
                                     ignore_class=self.target_corpus.word_to_ind[self.target_corpus.UNKNOWN],
                                     use_feedforward=use_feedforward)

        self.model.enc_embed.weight.data = self.source_embedding
        self.model.dec_embed.weight.data = self.target_embedding

    def pre_processing(self, data):
        """Input should be data frame with column 'German' and 'English'
        Data will be split into test data and validation data (last 20k examples are validation data)"""

        data['German_parsed'] = [self.source_corpus.parse_to_index(text, use_UNKNOWN=True, use_punctuation=True)
                                 for text in data['German']]
        data['German_length'] = [len(seq) for seq in data['German_parsed']]

        # split off validation set:
        data_train = data.iloc[:-(len(data) // 40)].copy()
        data_valid = data.iloc[-(len(data) // 40):].reset_index(drop=True)

        data_train['English_parsed'] = [self.target_corpus.parse_to_index(text, use_UNKNOWN=True, use_punctuation=True)
                                        for text in data_train['English']]
        data_train['English_length'] = [(len(seq) - 1) for seq in
                                        data_train['English_parsed']]  # subtract 1 to ignore STOP token

        return data_train, data_valid

    def batch_prep(self, batch, target_as_text=False):
        """turn data frame batch into tensors needed for training. Batch must have fields
        'German_parsed', 'English_parsed', 'German_length', 'English_length', 'English_shifted'"""

        batch_size = len(batch)

        batch_sorted = batch.sort_values(by='German_length', ascending=False)
        batch_sorted = batch_sorted.reset_index(drop=True)

        ger_length = batch_sorted.German_length[0]

        # create Tensor containing all the German indices padded with #STOP index
        ger_tensor = torch.ones(0)
        ger_tensor = ger_tensor.new_full((ger_length, batch_size),
                                         fill_value=self.source_corpus.word_to_ind[self.source_corpus.STOP])
        ger_tensor = ger_tensor.type(torch.LongTensor)
        for k in range(batch_size):
            seq_as_tensor = torch.Tensor(batch_sorted.German_parsed[k]).type(torch.LongTensor)
            ger_tensor[range(batch_sorted.German_length[k]), k] = seq_as_tensor

        ger_lengths = torch.Tensor(batch_sorted['German_length'].values)

        # sort by English lengths and keep indices to redo sorting
        batch_eng = batch_sorted.sort_values(by='English_length', ascending=False)
        eng_sort = batch_eng.index.values

        batch_eng = batch_eng.reset_index(drop=True)

        eng_length = batch_eng.English_length[0]

        # create Tensor containing all the English indices padded with #STOP index
        eng_tensor = torch.ones(0)
        eng_tensor = eng_tensor.new_full((eng_length, batch_size),
                                         fill_value=self.target_corpus.word_to_ind[self.target_corpus.UNKNOWN])

        eng_tensor = eng_tensor.type(torch.LongTensor)
        for k in range(batch_size):
            seq_as_tensor = torch.LongTensor(batch_eng.English_parsed[k][:-1])
            eng_tensor[range(batch_eng.English_length[k]), k] = seq_as_tensor

        eng_lengths = torch.Tensor(batch_eng['English_length'].values)

        if target_as_text:
            target = batch_sorted['English']
        else:
            target = torch.LongTensor([index for seq in batch_eng.English_parsed for index in seq[1:]])

        return ger_tensor, ger_lengths, eng_sort, eng_tensor, eng_lengths, target

    def train(self, data, epochs=5, batch_size=32, lr=0.001, lr_reduce=0):
        """Train model"""

        print('pre-processing')
        data_test, data_valid = self.pre_processing(data)

        # initialize
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        batch_count = len(data_test) // batch_size

        print('Start training')

        for epoch in range(epochs):
            # randomize data
            ran = np.arange(len(data_test))
            np.random.shuffle(ran)

            data_shuffled = data_test.iloc[ran]
            data_shuffled = data_shuffled.reset_index(drop=True)

            running_loss = 0

            # update learning rate:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # training loop
            for k in range(batch_count):

                # reset
                optimizer.zero_grad()

                self.model.train()

                batch = data_shuffled[k * batch_size: (k + 1) * batch_size]

                ger_tensor, ger_lengths, eng_sort, eng_tensor, eng_lengths, target = self.batch_prep(batch)

                # forward step:
                loss = self.model(eng_sort, eng_tensor, eng_lengths, targets=target,
                                  context=None, inputs=ger_tensor, input_lengths=ger_lengths)

                running_loss += loss.item()

                # backprop
                loss.backward()

                # update
                optimizer.step()

                if (k % 100) == 99:
                    print('Done with epoch {}, batch {} of {}, loss: {}'.format(epoch + 1, k + 1, batch_count,
                                                                                running_loss / 100))
                    running_loss = 0

            # save model state every epoch:
            path = './Parameters/translator_autosave_epoch' + str(epoch + 1) + '.pt'
            torch.save(self.model.state_dict(), path)

            print("Done with epoch {}, here's a sample:".format(epoch+1))
            ran = np.random.randint(low=0, high=len(data_valid))
            text = data_valid.German[ran]
            print(text)
            print(self.unparse(self.beam_translate(text, beam_width=25)))


            print('Validating:')
            score = self.test(data_valid, print_samples=True)

            print('Validation score is: {}'.format(score))

            # update learning rate
            lr = lr*(1-lr_reduce)

    def get_context(self, German_text):
        input_list = self.source_corpus.parse_to_index(German_text, use_UNKNOWN=True, use_punctuation=True)

        inputs = torch.LongTensor(input_list).unsqueeze(1)  # shape (seq-l, bs=1)
        # noinspection PyArgumentList
        lengths = torch.Tensor([inputs.shape[0]])

        self.model.eval()

        # get context
        with torch.no_grad():
            context = self.model.encode(inputs, lengths)

        return context

    def greedy_translate(self, German_text):
        """Text must start with #START and end with #STOP"""

        context = self.get_context(German_text)

        self.model.eval()

        #initializing
        result = []
        current_word = self.target_corpus.word_to_ind[self.target_corpus.START]
        current_word = torch.LongTensor([current_word]).unsqueeze(0)
        lengths = torch.Tensor([1])
        sort_index = [0]
        counter = 0
        state = None

        while current_word != self.target_corpus.word_to_ind[self.target_corpus.STOP] and counter < 100:
            # forward
            with torch.no_grad():
                probs, state = self.model(sort_ind=sort_index, output=current_word, output_lengths=lengths,
                                          state=state, context=context)

            probs = probs.squeeze(0) # (1=bs, class-count) -> (class_count)

            _, candidates = torch.topk(probs, 2)

            if candidates[0].item() != self.target_corpus.word_to_ind[self.target_corpus.STOP]:
                choice = candidates[0].item()
            else:
                choice = candidates[1].item()

            result.append(choice)

            current_word = torch.LongTensor([choice]).unsqueeze(0)
            counter += 1

        return result

    def beam_translate(self, German_text, max_length=150, beam_width=25):
        """Text must start with #START and end with #STOP"""

        context = self.get_context(German_text)

        my_beam = Beam(self.model, start_index=self.target_corpus.word_to_ind[self.target_corpus.START],
                       stop_index=self.target_corpus.word_to_ind[self.target_corpus.STOP],
                       ignore_index=self.target_corpus.word_to_ind[self.target_corpus.UNKNOWN])

        return my_beam.generate(context, max_length=max_length, beam_width=beam_width)

    def unparse(self, indices):
        """turns list of indices into text:"""

        words = [self.target_corpus.ind_to_word[index] for index in indices]

        text = ' '.join(words)

        for word in self.target_corpus.punctuation:
            text = text.replace(' '+word, word)

        return text

    def test(self, data_test, print_samples=False):
        """data_test must be data frame with "German" and "English" field"""
        German_data = data_test['German']
        English_data = data_test['English']
        length = len(data_test)
        score = 0

        for k in range(length):
            ref = English_data[k].lower().split()[1:-1]  # cut off #START and #STOP
            hyp = self.unparse(self.beam_translate(German_data[k], max_length=150, beam_width=25))

            if print_samples and (k % 100) == 99:
                print('Input: ' + German_data[k])
                print('Translation: ' + hyp)

            hyp = hyp.split()

            score += sentence_bleu([ref], hyp)

            if (k % 100) == 99:
                print('Done with validation {} of {}'.format(k+1, length))

        return score/length

    def test_batch(self, batch):

        my_beam = Beam(self.model, start_index=self.target_corpus.word_to_ind[self.target_corpus.START],
                       stop_index=self.target_corpus.word_to_ind[self.target_corpus.STOP],
                       ignore_index=self.target_corpus.word_to_ind[self.target_corpus.UNKNOWN])

        input, input_lengths, _, _, _, target = batch

        context = self.model.encode(input, input_lengths)

        preds = my_beam.batch_generate(context, 100, 25)

        for k, text in enumerate(target):
            print(text)
            print(self.unparse(preds[k]))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def translate(self, text):
        """translate text"""

        text = self.source_corpus.START + ' ' + text + ' ' + self.source_corpus.STOP

        return self.unparse(self.beam_translate(text))


class TranslatorModel(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size, embed_size, hidden_size,
                 GRU_count_enc=2, GRU_count_dec=2, ignore_class=None, use_feedforward=True):
        super(TranslatorModel, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.GRU_count_enc = GRU_count_enc
        self.GRU_count_dec = GRU_count_dec
        self.ignore_class = ignore_class
        self.use_feedforwad = use_feedforward

        self.enc_embed = nn.Embedding(in_vocab_size, self.embed_size)
        self.enc_GRU = nn.GRU(self.embed_size, self.hidden_size, num_layers=self.GRU_count_enc, bidirectional=True,
                              dropout=0.2)

        self.dec_embed = nn.Embedding(out_vocab_size, self.embed_size)
        self.dec_ReLU = nn.ReLU()
        self.dec_GRU = nn.GRU(self.embed_size + self.hidden_size, self.hidden_size, num_layers=self.GRU_count_dec,
                              dropout=0.2)
        self.Adaptive_Softmax = nn.AdaptiveLogSoftmaxWithLoss(self.hidden_size, out_vocab_size,
                                                              [round(out_vocab_size / 20),
                                                               4 * round(out_vocab_size / 20)])
        if self.use_feedforwad:
            self.feedforward_dense = nn.Linear(2*self.hidden_size, self.hidden_size, bias=False)
        self.att_Softmax = nn.Softmax(dim=1)
        self.bridge = nn.Linear(2*self.hidden_size, self.GRU_count_dec*self.hidden_size)

    def forward(self, sort_ind, output, output_lengths, targets=None, state=None,
                context=None, inputs=None, input_lengths=None, prev_context=None):
        """Calculates forward step. If context = None, inputs must be given and context will be calculated
        context must have shape (in-seq-l, bs, 2*hidden-size)
        Inputs:     sort_ind: list to resort bs many elements
                    output: teacher forcing seq of shape (out-seq-len, b-s)
                    output_lengths: tensor of lengths of seq in output; shape (bs), sorted descending
                    targets: None or tensor of target classes of shape (bs*seq-len), seq stacked head to toe
                    state: None or previous hidden state; shape (num-layers, bs, h-s)
                    context: None or encoded context; shape (in-seq-l, bs, 2*hidden-size)
                    inputs: None or padded input seq; shape (in-seq-l, bs)
                    input_lengths: None or length of seq in input; shape (bs)"""

        # encode context
        if context == None:
            context = self.encode(inputs, input_lengths)   # (in-seq-l, b-s, 2*hidden_size)

        # resort to fit output order
        context = context[:, sort_ind]

        # embed teacher input
        teacher = self.dec_embed(output)  # (seq-l, bs)-> (seq-l, bs, embed-size)
        teacher = self.dec_ReLU(teacher)

        # decode
        output, state, prev_context = self.decode(teacher, context, state=state, prev_context=prev_context)
                                                                            # ->(seq-len, bs, hidden_size)
        lengths = output_lengths.type(torch.LongTensor)                    # (bs)

        #reshape
        output = torch.transpose(output, 0, 1)  # ->(bs, seq-l, h-s)
        output = output.reshape(-1, self.hidden_size)  # ->(bs*seq-l, h-s), seq stacked head to toe

        # drop all instances longer than target seq
        mask = (torch.arange(output.shape[0]) % lengths[0]) < lengths.repeat_interleave(lengths[0])
        output = output[mask]

        if self.training:
            # return loss only
            # drop any incidence where target = unknown:
            output = output[targets != self.ignore_class]
            targets = targets[targets != self.ignore_class]

            loss = self.Adaptive_Softmax(output, targets).loss

            return loss
        else:
            # return probability distribution and state
            probs = self.Adaptive_Softmax.log_prob(output)
            return probs, state, prev_context

    def encode(self, inputs, input_lengths):
        embedded = self.enc_embed(inputs)  # (seq-l, bs) -> (seq-l, bs, embed-size)
        inputs_packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=True,
                                                          batch_first=False)

        context, _ = self.enc_GRU(inputs_packed)
        context, _ = nn.utils.rnn.pad_packed_sequence(context)  # ->(seq-len, bs, 2*hidden_size)

        return context

    def decode(self, teacher, context, state=None, prev_context=None):
        """Takes teacher sequences and context to return prediction sequences
        Input:  teacher (out-seq-len, b-s, embed-size)
                context (in-seq-len, b-s, 2*h-s)
                state (num-layers, b-s, h-s)
                prev_context (b-s, h-s)
        Output: predictions (out-seq-len, b-s, h-s)"""

        if state == None:
            # initialize state from the context
            state = self.initialize_state(context[-1, :, :self.hidden_size],    # final context of forward dir
                                          context[0, :, self.hidden_size:])     # final context of backward dir
                                                                                # state has shape (num-lay, b-s, h-s)

        # initialize result:
        result = torch.zeros(teacher.shape[0], teacher.shape[1], self.hidden_size)  # shape (seq-l, b-s, h-s)

        # initialize prev context:
        if prev_context == None and self.use_feedforwad:
            prev_context = torch.zeros(teacher.shape[1], self.hidden_size)          # shape (b-s, h-s)


        # loop over teacher sequence
        for k in range(teacher.shape[0]):
            # apply attention (only use the bottom layer of hidden states):
            att_context = self.apply_attention(context, state[-1], prev_context=prev_context)      # ->(bs, hidden-size)

            # concat with context
            contextualized = torch.cat([att_context, teacher[k]], dim=1).unsqueeze(0)  # ->(1, bs, h-s+embed_size)

            # keep context
            if self.use_feedforwad:
                prev_context = att_context

            # apply GRU
            output, state = self.dec_GRU(contextualized, state)     # ->(1, bs, h-s), (num-lay, b-s, h-s)

            # save output
            result[k] = output.squeeze(0)

        return result, state, prev_context

    def initialize_state(self, context_for, context_back):
        """initializes a hidden state from the context.
        Input: Contexts of shape (bs, hidden-size) - the final left and right context
        Output: Hidden state (num_layers, bs, hidden_size)"""

        # stack contexts:
        context = torch.cat([context_for, context_back], 1)     # ->(bs, 2*hidden_size)

        # apply bridge layer:
        state = self.bridge(context)        # ->(bs, num_layers*hidden_size)

        # reshape
        state = state.view(state.shape[0], self.GRU_count_dec, self.hidden_size)    # ->(bs, num_layers, h-s)
        state = torch.transpose(state, 0, 1)            # ->(num_layers, b-s, h-s)

        return state

    def apply_attention(self, context, prev_hidden, prev_context):
        """Calculates attention as dot product of prev_hidden and encoded context vectors. Returns attentive context
            Inputs: context (in-seq-len, bs, 2*hidden-size)
                    prev_hidden (bs, hidden-size)
                    prev_context (bs, h-s)
            Output: att_context (bs, hidden-size)"""

        # reshape context:
        context = torch.transpose(context, 0, 1)    # ->(bs, s-l, 2*h-s)
        context = context.reshape(context.shape[0], -1, self.hidden_size)  # ->(bs, 2*s-l, h-s)

        #apply linear layer
        if self.use_feedforwad:
            prev_hidden = self.feedforward_dense(torch.cat([prev_context, prev_hidden], dim=1))  # (bs, 2h-s) -> (bs, h-s)

        # reshape prev_hidden
        prev_hidden = prev_hidden.unsqueeze(2)     # (bs, h-s, 1)

        # calculate attention as the dot product of prev-hidden and all the contexts:
        attention = torch.bmm(context, prev_hidden)     # ->(bs, 2*s-l, 1)

        # normalize and apply softmax to get distribution:
        attention = self.att_Softmax(attention / np.sqrt(self.hidden_size))   # ->(bs, 2*s-l, 1), sum along dim 1 is 1

        # reshape context again
        context = torch.transpose(context, 1, 2)    # ->(bs, h-s, 2*s-l)

        # take weighted average
        att_context = torch.bmm(context, attention)     # ->(bs, h-s, 1)
        att_context = att_context.squeeze(2)            # ->(bs, h-s)

        return att_context

class TranslatorModelOld(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size, embed_size, hidden_size, source_dict, target_dict,
                 GRU_count_enc=2, GRU_count_dec=2, ignore_class=None):
        super(TranslatorModel, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.GRU_count_enc = GRU_count_enc
        self.GRU_count_dec = GRU_count_dec
        self.ignore_class = ignore_class
        self.source_dict = source_dict
        self.target_dict = target_dict

        self.enc_embed = nn.Embedding(in_vocab_size, self.embed_size)
        self.enc_GRU = nn.GRU(self.embed_size, self.hidden_size, num_layers=self.GRU_count_enc, bidirectional=True,
                              dropout=0.2)

        self.dec_embed = nn.Embedding(out_vocab_size, self.embed_size)
        self.dec_ReLU = nn.ReLU()
        self.dec_GRU = nn.GRU(self.embed_size + self.hidden_size, self.hidden_size, num_layers=self.GRU_count_dec,
                              dropout=0.2)
        self.Adaptive_Softmax = nn.AdaptiveLogSoftmaxWithLoss(self.hidden_size, out_vocab_size,
                                                              [round(out_vocab_size / 20),
                                                               4 * round(out_vocab_size / 20)])

    def forward(self, sort_ind, output, output_lengths, targets=None, state=None,
                context=None, inputs=None, input_lengths=None):
        """Calculates forward step. If context = None, inputs must be given and context will be calculated
        context must have shape (bs, hidden-size)"""

        # encode context
        if context == None:
            context = self.encode(inputs, input_lengths)

        context = context[sort_ind]  # resort to fit output order
        context = torch.unsqueeze(context, 0).expand(int(output_lengths[0].item()), -1,
                                                     -1)  # tile to (seq-l, bs, hidden_size)

        #grab batch-size:
        batch_size = output.shape[1]

        # embed teacher input
        teacher = self.dec_embed(output)  # (seq-l, bs)-> (seq-l, bs, embed-size)
        teacher = self.dec_ReLU(teacher)

        #concat with context
        contextualized = torch.cat([context, teacher], dim=2)  # ->(seq-l, bs, hidden_size+embed_size)

        if state == None:
            state = self.initialize_state(batch_size)

        #decoding GRU
        contextualized = nn.utils.rnn.pack_padded_sequence(contextualized, output_lengths, enforce_sorted=True)
        output, state = self.dec_GRU(contextualized, state)
        output, lengths = nn.utils.rnn.pad_packed_sequence(output)  # ->(seq-len, bs, hidden_size), (bs)

        #reshape
        output = torch.transpose(output, 0, 1)  # ->(bs, seq-l, h-s)
        output = output.reshape(-1, self.hidden_size)  # ->(bs*seq-l, h-s), seq stacked head to toe

        # drop all instances longer than target seq
        mask = (torch.arange(output.shape[0]) % lengths[0]) < lengths.repeat_interleave(lengths[0])
        output = output[mask]

        if self.training:
            # return loss only
            # drop any incidence where target = unknown:
            output = output[targets != self.ignore_class]
            targets = targets[targets != self.ignore_class]

            loss = self.Adaptive_Softmax(output, targets).loss

            return loss
        else:
            # return probability distribution and state
            probs = self.Adaptive_Softmax.log_prob(output)
            return probs, state


    def encode(self, inputs, input_lengths):
        embedded = self.enc_embed(inputs)  # (seq-l, bs) -> (seq-l, bs, embed-size)
        inputs_packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=True,
                                                          batch_first=False)

        context, _ = self.enc_GRU(inputs_packed)
        context, lengths = nn.utils.rnn.pad_packed_sequence(context)  # ->(seq-len, bs, 2*hidden_size), (bs)
        lengths = 2 * lengths  # double to account for back and forth direction

        context = context[:, :, :self.hidden_size] + context[:, :,
                                                     self.hidden_size:]  # sum forward and backward vectors
        context = context.sum(0)  # ->(bs, hidden_size)
        context = context / lengths.view(-1, 1)  # average out

        return context

    def initialize_state(self, batch_size):
        state = torch.zeros(self.GRU_count_dec, batch_size, self.hidden_size)

        return state

