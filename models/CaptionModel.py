# This file contains ShowAttendTell and AllImg model

# ShowAttendTell is from Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
# https://arxiv.org/abs/1502.03044

# AllImg is a model where
# img feature is concatenated with word embedding at every time step as the input of lstm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils


class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    def beam_search(self, state, logprobs, *args, **kwargs):
        # args are the miscelleous inputs to the core in addition to embedded word and state
        # kwargs only accept opt
        
        # state:
        # logprobs: (slen, output_vocab_size)

        def beam_step(logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            # One sample?
            # INPUTS:
            # logprobsf: (slen, output_vocab_size), probabilities augmented after diversity
            # beam_size: K
            # t        : time instant
            # beam_seq : (slen, K), beam sequence indices, valid up to (t-1, K) 
            # beam_seq_logprobs: (slen, K), beam sequence logits, valid up to (t-1, K) 
            # beam_logprobs_sum: (K, ), accumulated logits

            # OUPUTS:
            # beam_seq : (slen, K), beam sequence indices, valid up to (t, K) 
            # beam_seq_logprobs : (slen, K), beam sequence logits, valid up to (t, K) 
            # beam_logprobs_sum : (K, ), accumulated logits

            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size if t != 0 else 1

            # for each top K candidate words
            for c in range(cols): 
                # for each beam path
                for q in range(rows): 
                    # compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q, c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob, 'r': local_logprob})
            candidates = sorted(candidates,  key=lambda x: -x['p'])
            
            new_state = [foo.clone() for foo in state]
            
            if t >= 1:
            # we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            
            # top K candidates
            for vix in range(beam_size):
                v = candidates[vix]
                # fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                # rearrange recurrent states
                for state_ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
            state = new_state
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, state, candidates

        # start beam search
        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)

        # beam_seq_indices
        beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        # beam_seq_logits
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
        # running sum of logprobs for each beam
        beam_logprobs_sum = torch.zeros(beam_size) 
        # beam paths that are terminated
        done_beams = []

        for t in range(self.seq_length):
            """maintaining the top K sequences with the highest likelihoods"""
            logprobsf = logprobs.cpu().data.float() # lets go to CPU for more efficiency in indexing operations
            # suppress UNK tokens in the decoding
            logprobsf[:, logprobsf.size(1)-1] = logprobsf[:, logprobsf.size(1)-1] - 1000  
        
            beam_seq,\
            beam_seq_logprobs,\
            beam_logprobs_sum,\
            state, \
            candidates_divm = beam_step(logprobsf,
                                        beam_size,
                                        t,
                                        beam_seq,
                                        beam_seq_logprobs,
                                        beam_logprobs_sum,
                                        state)

            for vix in range(beam_size):
                # if time's up... or if end token is reached then copy beams
                if beam_seq[t, vix] == 0 or t == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(), 
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': beam_logprobs_sum[vix]
                    }
                    done_beams.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_logprobs_sum[vix] = -1000

            # encode as vectors
            it = beam_seq[t]
            logprobs, state = self.get_logprobs_state(Variable(it.cuda()), *(args + (state,)))

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams
