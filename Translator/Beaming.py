import torch

class Beam:
    def __init__(self, model, start_index, stop_index, ignore_index=None):
        """Implements beam search for single- and multi-line batches."""
        self.model = model
        self.ignore_index = ignore_index
        self.start_index = start_index
        self.stop_index = stop_index

    def generate(self, context, max_length, beam_width=10):
        """generates beam-width many sequences and chooses best one
        context must be triple (in-seq-l, bs=1, 2*hidden_size), (bs=1, h-s), (bs=1, h-s)"""

        context, final_forward, final_backward = context

        self.model.eval()

        # reshape start:
        start = torch.LongTensor([[self.start_index]]) # shape (1, 1)

        #lengths is always 1:
        lengths = torch.Tensor([1])
        sort_index = [0]

        #initialize first hidden state:
        state = None

        #first forward step
        with torch.no_grad():
            probs, state, prev_context = self.model(sort_ind=sort_index, output=start, output_lengths=lengths,
                                                    state=state, context=context, final_forward=final_forward,
                                                    final_backward=final_backward)
                                                    # ->(1*1, class-count), (layers, 1, hidden-size), (1, h-s)


        #get last output state and squeeze:
        probs = probs.squeeze(0)  #->(class-count)

        #eliminate bad index:
        probs[self.ignore_index] = float('-inf')


        # find first batch of candidates and their probabilities:
        prob_scores, inputs = torch.topk(probs, beam_width)  # both are of shape (beam-width)

        # stack hidden states to size
        state = state.repeat(1, beam_width, 1)      # ->(layers, b-w, h-s)
        if prev_context != None:
            prev_context = prev_context.repeat(beam_width, 1)   # ->(b-w, h-s)

        # initialize candidate sequence tensor
        candidate_seq = torch.zeros(beam_width, max_length)
        candidate_seq[:, 0] = inputs

        choice_prob = float('-inf')
        choice = None

        for k in range(1, max_length):

            # reshape the inputs
            inputs = inputs.unsqueeze(0) #->(1, b-w)

            #make lengths
            lengths = torch.Tensor([1]*beam_width)     # (b-w)
            sort_index = torch.arange(beam_width)     # (b-w)

            #stack context:
            this_context = context.repeat(1, beam_width, 1)  # ->(in-seq-len, b-w, 2*h-s)

            #forward step
            with torch.no_grad():
                probs, intermediate_state, int_context = self.model(sort_ind=sort_index, output=inputs,
                                                                    output_lengths=lengths,
                                                                    state=state, context=this_context,
                                                                    prev_context=prev_context)
                                                        # ->(b-w*1, class-count), (layers,b-w, hidden-size)

            #grab class_count
            class_count = probs.shape[1]

            # eliminate bad index
            probs[:, self.ignore_index] = float('-inf')

            # add previous probabilities to all entries in rows:
            probs = probs + prob_scores.unsqueeze(1)

            # reshape into one dimension:
            probs = probs.reshape(-1)           # ->((b-w)*(class-count)) b-w many blocks of class-count

            # pick top probabilities
            prob_scores, candidates = probs.topk(beam_width)    # both shape (b-w)

            # break if top_prob lower than choice:
            if prob_scores[0] < choice_prob:
                break

            # convert candidates index (0<...<b-w*class-count) to class index:
            inputs = candidates % class_count

            # find seq that contributed:
            prev_seq = candidates // class_count

            # grab inputs that ended:
            prob_scores_end = prob_scores[inputs == self.stop_index]
            prev_seq_end = prev_seq[inputs == self.stop_index]

            #update choice if necessary
            if len(prob_scores_end) > 0 and prob_scores_end[0] > choice_prob:
                choice_prob = prob_scores_end[0]
                choice = candidate_seq[prev_seq_end[0]].tolist()[:k]

            # drop any seq that reached stop index:
            prev_seq = prev_seq[inputs != self.stop_index]
            beam_width = prev_seq.shape[0]
            prob_scores = prob_scores[inputs != self.stop_index]
            inputs = inputs[inputs != self.stop_index]

            if beam_width == 0:
                break

            # update saved sequences:
            candidate_seq = candidate_seq[prev_seq]
            candidate_seq[:, k] = inputs

            # update hidden state:
            state = intermediate_state[:, prev_seq]
            if prev_context != None:
                prev_context = int_context[prev_seq]

        if choice == None:
            choice = candidate_seq[0].tolist()

        return choice

    def batch_generate(self, context, max_length, beam_width=10):
        """generates beam-width many sequences and chooses best one for each example in batch
        context must be triple(in-seq-l, bs, 2*hidden_size), (bs, hs), (bs, hs)"""

        context, final_forward, final_backward = context

        self.model.eval()

        # grab batch size
        batch_size = context.shape[1]

        # reshape start:
        start = torch.LongTensor([[self.start_index]*batch_size])   # shape (1, b-s)

        # lengths is always 1:
        lengths = torch.Tensor([1]*batch_size)                      # shape (b-s)
        sort_index = range(batch_size)

        #initialize first hidden state:
        state = None

        #first forward step
        with torch.no_grad():
            probs, state, prev_context = self.model(sort_ind=sort_index, output=start, output_lengths=lengths,
                                                    state=state, context=context, final_forward=final_forward,
                                                    final_backward=final_backward)
                                                    # ->(1*b-s, class-count), (layers, b-s, hidden-size), (b-s, h-s)

        # eliminate bad index:
        probs[:, self.ignore_index] = float('-inf')         # shape (b-s, class-count)

        # find first batch of candidates and their probabilities:
        prob_scores, inputs = torch.topk(probs, beam_width, dim=1)  # both are of shape (b-s, beam-width)

        # stack hidden states to size
        state = torch.repeat_interleave(state, beam_width, dim=1)      # ->(layers, b-w*b-s, h-s) grouped by batches


        if prev_context != None:
            prev_context = torch.repeat_interleave(prev_context, beam_width, dim=0) # ->(bw*bs, hs)

        # initialize candidate sequence tensor
        candidate_seq = torch.zeros(0)
        candidate_seq = candidate_seq.new_full((batch_size, beam_width, max_length), fill_value=self.stop_index)
        candidate_seq[:, :, 0] = inputs

        choices = torch.zeros(batch_size, max_length)
        remaining_batches = torch.arange(batch_size)

        for k in range(1, max_length):

            # reshape the inputs
            inputs = inputs.reshape(1, -1)  # ->(1, b-s*b-w)

            #make lengths
            lengths = torch.Tensor([1]*batch_size*beam_width)     # (b-s*b-w)
            sort_index = torch.arange(batch_size*beam_width)     # (b-s*b-w)

            #stack context:
            this_context = context.repeat_interleave(beam_width, dim=1)  # ->(in-seq-len, b-s*b-w, 2*h-s)

            #forward step
            with torch.no_grad():
                probs, intermediate_state, prev_context = self.model(sort_ind=sort_index, output=inputs,
                                                                     output_lengths=lengths, state=state,
                                                                     context=this_context, prev_context=prev_context)
                                                            # ->(b-s*b-w, class-count), (layers,b-s*b-w, hs), (bs, hs)

            #grab class_count
            class_count = probs.shape[1]

            # eliminate bad index
            probs[:, self.ignore_index] = float('-inf')

            # sequences that stopped must continue with STOP:
            fixed_probs = torch.zeros(0).new_full((class_count, ), fill_value=float('-inf'))
            fixed_probs[self.stop_index] = 0
            probs[inputs.squeeze(0) == self.stop_index] = fixed_probs

            # add previous probabilities to all entries in rows:
            probs = probs + prob_scores.reshape(-1, 1)

            # reshape:
            probs = probs.reshape(batch_size, -1)         # ->((b-s),(b-w)*(class-count)) b-w many blocks of class-count

            # pick top probabilities
            prob_scores, candidates = probs.topk(beam_width)    # both shape (b-s, b-w)

            # convert candidates index (0<...<b-w*class-count) to class index:
            inputs = candidates % class_count  # ->(b-s, b-w)

            # find seq that contributed:
            prev_seq = candidates // class_count  # ->(b-s, b-w)

            # find batches with choice that ended
            finished_batches = torch.arange(inputs.shape[0])[inputs[:, 0] == self.stop_index]
            future_batches = torch.arange(inputs.shape[0])[inputs[:, 0] != self.stop_index]

            # save choices:
            if finished_batches.shape[0] > 0:
                choices[remaining_batches[finished_batches]] = candidate_seq[finished_batches,
                                                                             prev_seq[finished_batches, 0]]

            if future_batches.shape[0] == 0:
                break

            # update everything for next loop:
            inputs = inputs[future_batches]
            remaining_batches = remaining_batches[future_batches]
            prev_seq = prev_seq[future_batches]
            prob_scores = prob_scores[future_batches]

            # update cand sequences
            candidate_seq = candidate_seq[future_batches]       # -> drop any batches with no future relevance
            candidate_seq = candidate_seq.reshape(-1, candidate_seq.shape[2])   #(b-s*b-w, max-length)
            prev_seq = prev_seq + beam_width*torch.arange(prev_seq.shape[0]).view(-1,1)   # -> add batch-# x b-w to every batch
            prev_seq = prev_seq.reshape(-1)                   # -> (b-s*b-w)
            candidate_seq = candidate_seq[prev_seq]            # -> pick prev seq arrangements
            candidate_seq = candidate_seq.reshape(-1, beam_width, candidate_seq.shape[1])   # -> (bs, bw, max length)

            candidate_seq[:, :, k] = inputs                     # -> add the last value

            # update hidden state:
            # drop unused batches
            intermediate_state = intermediate_state.reshape(intermediate_state.shape[0], batch_size, beam_width, -1)
            intermediate_state = intermediate_state[:, future_batches].reshape(intermediate_state.shape[0], -1,
                                                                               intermediate_state.shape[3])
            # choose relevant batches in order
            state = intermediate_state[:, prev_seq]

            # same for prev_context
            if prev_context != None:
                prev_context = prev_context.reshape(batch_size, beam_width, -1)
                prev_context = prev_context[future_batches].reshape(-1, prev_context.shape[2])
                prev_context = prev_context[prev_seq]

            # update batch size
            batch_size = future_batches.shape[0]

        result = choices.tolist()

        for k in range(len(result)):
            if self.stop_index in result[k]:
                result[k] = result[k][:result[k].index(self.stop_index)]

        return result













