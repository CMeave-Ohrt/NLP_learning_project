import torch

class Beam:
    def __init__(self, model, ignore_index=None):
        """Model must sequential model with functions:
        -initialize_state"""
        self.model = model
        self.ignore_index = ignore_index


    def generate(self, start, length, beam_width=10, creativity=0):
        """generates beam-width many sequences and randomly chooses one
        start should be tensor of indices of form (start_length)"""

        self.model.eval()

        result = start.tolist()

        # reshape start:
        start = start.reshape(1, -1) #-> (1, start_length)

        #initialize first hidden state:
        state = self.model.initialize_state(1)

        #first forward step
        with torch.no_grad():
            probs, state = self.model(start, state)  #->(1*s-l, class-count), (layers, 1, hidden-size)


        #get last output state and squeeze:
        probs = probs[-1]  #->(class-count)

        #eliminate bad index:
        probs[self.ignore_index] = float('-inf')


        #find first batch of candidates and their probabilities:
        prob_scores, inputs = torch.topk(probs, beam_width)  #both are of shape (beam-width)

        #stack hidden states to size
        h, c = state
        state = h.repeat(1, beam_width, 1), c.repeat(1, beam_width, 1)  #->(layers, b-w, h-s)

        #initialize candidate sequence tensor
        candidate_seq = torch.zeros(beam_width, length)
        candidate_seq[:, 0] = inputs

        for k in range(1, length):

            # reshape the inputs
            inputs = inputs.unsqueeze(1) #->(b-w, 1)

            #forward step
            with torch.no_grad():
                probs, intermediate_state = self.model(inputs, state)#->(b-w*1, class-count), (layers,b-w, hidden-size)

            #grab class_count
            class_count = probs.shape[1]

            #eliminate bad index
            probs[:, self.ignore_index] = float('-inf')

            #add previous probabilities to all entries in rows:
            probs = probs + prob_scores.unsqueeze(1)

            #add noise:
            probs = probs + torch.randn_like(probs)*creativity

            #reshape into one dimension:
            probs = probs.reshape(-1)           #->((b-w)*(class-count)) b-w many blocks of class-count

            #pick top probabilities
            prob_scores, candidates = probs.topk(beam_width) #both shape (b-w)

            #convert candidates index (0<...<b-w*class-count) to class index:
            inputs = candidates % class_count

            #find seq that contributed:
            prev_seq = candidates // class_count

            #update saved sequences:
            candidate_seq = candidate_seq[prev_seq]
            candidate_seq[:, k] = inputs

            #update hidden state:
            h, c = intermediate_state
            state = h[:, prev_seq], c[:, prev_seq]

        # pick random line:
        ran = torch.randint(high=beam_width, size=(1,))
        #ran = torch.Tensor([9]).type(torch.LongTensor)
        choice = candidate_seq[ran.item()].tolist()

        print(ran.item())

        result += choice

        return result

    def generate_with_var(self, start, length, beam_width=10, creativity=0, window=2):
        """generates beam-width many sequences,
        at time t, randomly locks in one of the possibilities created at time t-window
        additionally adds noise to probability distributions before settling on the best choice
        (noise is guarded by creativity)"""

        self.model.eval()

        result = start.tolist()

        # reshape start:
        start = start.reshape(1, -1) #-> (1, start_length)

        #initialize first hidden state:
        state = self.model.initialize_state(1)

        #first forward step
        with torch.no_grad():
            probs, state = self.model(start, state)  #->(1*s-l, class-count), (layers, 1, hidden-size)


        #get last output state and squeeze:
        probs = probs[-1]  #->(class-count)

        #eliminate bad index:
        probs[self.ignore_index] = float('-inf')


        #find first batch of candidates and their probabilities:
        prob_scores, inputs = torch.topk(probs, beam_width)  #both are of shape (beam-width)

        #stack hidden states to size
        h, c = state
        state = h.repeat(1, beam_width, 1), c.repeat(1, beam_width, 1)  #->(layers, b-w, h-s)

        #initialize candidate sequence tensor
        candidate_seq = torch.zeros(beam_width, length).type(torch.LongTensor)
        candidate_seq[:, 0] = inputs

        for k in range(1, length):

            if (k % 100)==99:
                print('Generating word {}/{}'.format(k+1, length))

            # reshape the inputs
            inputs = inputs.unsqueeze(1) #->(b-s, 1)

            #forward step
            with torch.no_grad():
                probs, intermediate_state = self.model(inputs, state)#->(b-s*1, class-count), (layers,b-s, hidden-size)

            #grab class_count
            class_count = probs.shape[1]

            #eliminate bad index
            probs[:, self.ignore_index] = float('-inf')

            #add previous probabilities to all entries in rows:
            probs = probs + prob_scores.unsqueeze(1)   #Attention

            #add noise:
            probs = probs + torch.randn_like(probs)*creativity

            #reshape into one dimension:
            probs = probs.reshape(-1)           #->((b-s)*(class-count)) b-w many blocks of class-count

            #pick top probabilities
            prob_scores, candidates = probs.topk(beam_width) #both shape (b-w)

            #convert candidates index (0<...<b-w*class-count) to class index:
            inputs = candidates % class_count

            #find seq that contributed:
            prev_seq = candidates // class_count  #entries are smaller than b-s

            #update saved sequences:
            candidate_seq = candidate_seq[prev_seq]
            candidate_seq[:, k] = inputs

            #pick k-window element
            if k >= window:
                #choose element
                ran = torch.randint(high=beam_width, size=(1,)).item()
                choice = candidate_seq[ran, k-window]
                result.append(choice.item())

                #update all data
                mask = candidate_seq[:, k-window] == choice
                candidate_seq = candidate_seq[mask] #shape (b-s, length)
                inputs = candidate_seq[:, k] #shape(b-s)
                prob_scores = prob_scores[mask]
                prev_seq = prev_seq[mask]

            #update hidden state:
            h, c = intermediate_state
            state = h[:, prev_seq], c[:, prev_seq]

        return result













