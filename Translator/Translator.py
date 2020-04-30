import torch
import torch.distributions as dist
import torch.nn as nn
import pandas as pd
import numpy as np
from Beaming import Beam
from nltk.translate.bleu_score import sentence_bleu
import copy


class Translator:
    """Implements a German-to-English translator based on the encoder-decoder model. Both use layered GRU's
    The decoder uses attention based on dot products. If use_feedforward=True, the decoder will take previous attention
    into consideration via a dense linear layer (prev-att, hidden state) -> (hidden-size)
    The model can be trained both with teacher forcing and reinforcement learning"""
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

        self.critic = copy.deepcopy(self.model)

    def pre_processing(self, data):
        """Input should be data frame with column 'German' and 'English'
        Data will be split into test data and validation data (1/40th is validation data)"""

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
        """Train model using teacher forcing"""

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

    def pre_processing_for_reinforcement(self, data):
        """Prepares data for reinforcement training. Data should have fields 'German' and 'English'"""

        data_train = data.iloc[:-(len(data) // 40)].copy()
        data_valid = data.iloc[-(len(data) // 40):].copy().reset_index(drop=True)

        data_train['German_parsed'] = [self.source_corpus.parse_to_index(text, use_UNKNOWN=True, use_punctuation=True)
                                       for text in data_train['German']]
        data_train['German_length'] = [len(seq) for seq in data_train['German_parsed']]

        data_train['English_bleu'] = [self.target_corpus.parse(text, use_punctuation=True, for_BLEU=True)[1:-1]
                                      for text in data_train['English']]

        return data_train, data_valid

    def reinf_batch_prep(self, batch):
        """Prepares batch for model. Batch must have fields 'German_parsed', 'German_length', 'English_bleu'
        returns tensor ger_tensor (seq-len, bs), ger_len (bs), and list targets"""

        batch_size = len(batch)

        batch_sorted = batch.sort_values(by='German_length', ascending=False)
        batch_sorted = batch_sorted.reset_index(drop=True)

        # longest length
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

        targets = batch_sorted.English_bleu.tolist()

        return ger_tensor, ger_lengths, targets

    def get_samples(self, ger_tensor, ger_lengths):
        """Encodes context of German input and generates translations by sampling from learned distribution
        Input should be:    ger_tensor (seq-l, b-s)
                            ger_lengths (bs)    (descending!)
        Output is:  samples (list of lists of words)
                    probs (tensor of log likelihoods of samples, shape (bs))"""

        # eval mode to get correct behavior
        self.model.eval()

        # get context
        context_col = self.model.encode(ger_tensor, ger_lengths, final_states=True)
        context, final_forward, final_backward = context_col    # (seq-l, bs, 2*hs), 2 x (bs, hs)

        # grab batch_size:
        batch_size = ger_lengths.shape[0]

        # abbreviations
        start = self.target_corpus.word_to_ind[self.target_corpus.START]
        stop = self.target_corpus.word_to_ind[self.target_corpus.STOP]
        max_length = 150

        # initialize
        inputs = torch.LongTensor([start]*batch_size)     # (1, bs)
        state = None
        remain_ind = torch.arange(batch_size)               # (bs)
        candidate_seq = torch.zeros(0)
        candidate_seq = candidate_seq.new_full((batch_size, max_length), fill_value=stop).type(torch.LongTensor)
        prev_context = None
        log_probs = torch.zeros(batch_size)                 # (bs)

        for k in range(max_length):
            # reshape the inputs
            inputs = inputs.reshape(1, -1)  # ->(1, b-s)

            # make lengths
            lengths = torch.Tensor([1] * batch_size)  # (b-s)
            sort_index = torch.arange(batch_size)  # (b-s)

            # forward step
            probs, state, prev_context = self.model(sort_ind=sort_index, output=inputs,
                                                    output_lengths=lengths, state=state,
                                                    context=context, prev_context=prev_context,
                                                    final_backward=final_backward, final_forward=final_forward)
                                                    # ->(bs, count), (layers, bs, h-s), (bs, hs)

            # drop final states:
            final_forward = None
            final_backward = None

            # initialize prob distr:
            my_dist = dist.categorical.Categorical(logits=probs)

            # sample
            inputs = my_dist.sample().type(torch.LongTensor)   # -> (bs)

            # update probabilies
            log_probs[remain_ind] += probs[:, inputs].diag()

            # find seq that did not end
            keep = (inputs != stop)

            # update and drop rows
            remain_ind = remain_ind[keep]
            context = context[:, keep]
            state = state[:, keep]
            prev_context = prev_context[keep]
            inputs = inputs[keep]
            batch_size = remain_ind.shape[0]

            # save samples
            candidate_seq[remain_ind, k] = inputs

            if batch_size == 0:
                break

        result = candidate_seq.tolist()

        for k in range(len(result)):
            if stop in result[k]:
                result[k] = result[k][:result[k].index(stop)]

        return result, log_probs, context_col

    def reinforce_train(self, data, epochs=50, batch_size=128, lr=0.0001):
        """Further trains the model using reinforcement learning and policy gradients"""

        print('pre-processing:')
        data_train, data_valid = self.pre_processing_for_reinforcement(data)

        # initialize
        batch_count = len(data_train) // batch_size
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        best_score = 0
        best_score_num = 0
        self.critic = copy.deepcopy(self.model)

        print('Start Training')

        for epoch in range(epochs):
            # initialize critic
            critic = Beam(self.critic, start_index=self.target_corpus.word_to_ind[self.target_corpus.START],
                          stop_index=self.target_corpus.word_to_ind[self.target_corpus.STOP],
                          ignore_index=self.target_corpus.word_to_ind[self.target_corpus.UNKNOWN])

            # randomize data
            ran = np.arange(len(data_train))
            np.random.shuffle(ran)

            data_shuffled = data_train.iloc[ran]
            data_shuffled = data_shuffled.reset_index(drop=True)

            running_score = 0
            beat_critic = 0

            # training loop
            for l in range(batch_count):
                # reset
                optimizer.zero_grad()

                batch = data_shuffled[l * batch_size: (l + 1) * batch_size].reset_index(drop=True)

                ger_tensor, ger_lengths, targets = self.reinf_batch_prep(batch)

                # get samples and critic translation
                samples, log_probs, context_col = self.get_samples(ger_tensor, ger_lengths)
                critic_translations = critic.batch_generate(context_col, 150, beam_width=10)

                loss = 0

                for k, sample in enumerate(samples):
                    # get scores
                    sample_text = [self.target_corpus.ind_to_word[word] for word in sample]
                    critic_text = [self.target_corpus.ind_to_word[word] for word in critic_translations[k]]
                    sample_score = sentence_bleu([targets[k]], sample_text)
                    critic_score = sentence_bleu([targets[k]], critic_text)

                    # update user info
                    running_score += sample_score
                    if sample_score > critic_score:
                        beat_critic += 1

                    # get loss, assign much higher score if critic was beaten
                    loss -= ((sample_score - critic_score) + 10 * max(0, sample_score - critic_score)) * log_probs[k]

                # average loss
                loss = loss / batch_size

                # backward step
                loss.backward()

                # update
                optimizer.step()

                if (k % 2) == 1:
                    print('Done with epoch {}, batch {}, score: {}, beat critic: {}'.format(epoch + 1, l + 1,
                                                                                          running_score / (batch_size * 2),
                                                                                          beat_critic))
                    running_score = 0
                    beat_critic = 0

            print("Evaluating")
            score = self.test(data_train, print_samples=False)

            print('Validation score is {}'.format(score))

            if score > best_score:
                best_score = score

                # save model:
                path = './Parameters/translator_reinf_autosave_epoch' + str(epoch + 1) + '.pt'
                torch.save(self.model.state_dict(), path)

                best_score_num += 1

                if best_score_num == 3:
                    # update critic
                    self.critic = copy.deepcopy(self.model)

    def get_context(self, German_text):
        """Takes a single line of German text and returns its context. Output is a triple
            context (seq-l, bs, 2*h-s)
            last_forward (bs, h-s)
            last_backward (bs, h-s)

            last_forward and last_backward are necessary to to calculate the first hiddens state of decoder"""
        input_list = self.source_corpus.parse_to_index(German_text, use_UNKNOWN=True, use_punctuation=True)

        inputs = torch.LongTensor(input_list).unsqueeze(1)  # shape (seq-l, bs=1)
        # noinspection PyArgumentList
        lengths = torch.Tensor([inputs.shape[0]])

        self.model.eval()

        # get context
        with torch.no_grad():
            context = self.model.encode(inputs, lengths, final_states=True)

        return context

    def beam_translate(self, German_text, max_length=150, beam_width=25):
        """Uses beam search to translate a single line of text. Text must start with #START and end with #STOP
        output is list of indices and must be unparsed"""

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
        """Tests model on given data. Returns average BLEU score.
        Data_test must be data frame with "German" and "English" field"""
        German_data = data_test['German']
        English_data = data_test['English']
        length = len(data_test)
        score = 0

        for k in range(length):
            # cut off #START and #STOP
            ref = English_data[k].lower().split()[1:-1]
            hyp = self.unparse(self.beam_translate(German_data[k], max_length=150, beam_width=25))

            if print_samples and (k % 100) == 99:
                print('Input: ' + German_data[k])
                print('Translation: ' + hyp)
                print('Target' + ref)

            hyp = hyp.split()

            score += sentence_bleu([ref], hyp)

            if (k % 100) == 99:
                print('Done with validation {} of {}'.format(k+1, length))

        return score/length

    def test_batch(self, batch):
        """Translates an entire batch using beam search. This is mainly for debugging batch_generate.
        input must be output of self.batch_prep. Output is None, but will print all line translation and their
        targets"""

        my_beam = Beam(self.model, start_index=self.target_corpus.word_to_ind[self.target_corpus.START],
                       stop_index=self.target_corpus.word_to_ind[self.target_corpus.STOP],
                       ignore_index=self.target_corpus.word_to_ind[self.target_corpus.UNKNOWN])

        input, input_lengths, _, _, _, target = batch

        self.model.eval()

        with torch.no_grad():
            context = self.model.encode(input, input_lengths, final_states=True)

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
                context=None, final_forward=None, final_backward=None,
                inputs=None, input_lengths=None, prev_context=None):
        """Calculates forward step. If context = None, inputs must be given and context will be calculated
        context must have shape (in-seq-l, bs, 2*hidden-size)
        Inputs:     sort_ind: list to resort bs many elements
                    output: teacher forcing seq of shape (out-seq-len, b-s)
                    output_lengths: tensor of lengths of seq in output; shape (bs), sorted descending
                    targets: None or tensor of target classes of shape (bs*seq-len), seq stacked head to toe
                    state: None or previous hidden state; shape (num-layers, bs, h-s)
                    context: None or encoded context; shape (in-seq-l, bs, 2*hidden-size)
                    final_forward, final_backward: final hidden states from encode (bs, h-s)
                    inputs: None or padded input seq; shape (in-seq-l, bs)
                    input_lengths: None or length of seq in input; shape (bs)"""

        # encode context
        if context == None:
            context_col = self.encode(inputs, input_lengths, final_states=True)   # (in-seq-l, b-s, 2*hidden_size)

            # break up context
            context, final_forward, final_backward = context_col

        # resort to fit output order
        context = context[:, sort_ind]
        if final_backward != None:
            final_forward = final_forward[sort_ind]
            final_backward = final_backward[sort_ind]

        # embed teacher input
        teacher = self.dec_embed(output)  # (seq-l, bs)-> (seq-l, bs, embed-size)
        teacher = self.dec_ReLU(teacher)

        # decode
        output, state, prev_context = self.decode(teacher, context, final_forward=final_forward,
                                                  final_backward=final_backward, state=state, prev_context=prev_context)
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

    def encode(self, inputs, input_lengths, final_states=False):
        embedded = self.enc_embed(inputs)  # (seq-l, bs) -> (seq-l, bs, embed-size)
        inputs_packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=True,
                                                          batch_first=False)

        context, state = self.enc_GRU(inputs_packed)        # ->packed seq, (layers*2, b-s, h-s)
        context, _ = nn.utils.rnn.pad_packed_sequence(context)  # ->(seq-len, bs, 2*hidden_size)

        # get the right most hidden state of the forward and the left most hidden state of the backward direction
        forward_context = state.view(self.GRU_count_enc, 2, -1, self.hidden_size)[-1, 0]    # ->(b-s, h-s)
        backward_context = context[0, :, self.hidden_size:]     # ->(b-s, h-s)
        if final_states:
            return context, forward_context, backward_context
        else:
            return context

    def decode(self, teacher, context, final_forward=None, final_backward=None, state=None, prev_context=None):
        """Takes teacher sequences and context to return prediction sequences
        Input:  teacher (out-seq-len, b-s, embed-size)
                context (in-seq-len, b-s, 2*h-s)
                final_forward, final_backward (bs, hs)
                state (num-layers, b-s, h-s)
                prev_context (b-s, h-s)
        Output: predictions (out-seq-len, b-s, h-s)"""

        if state == None:
            # initialize state from the context
            state = self.initialize_state(final_forward, final_backward)         # final context of backward dir
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



