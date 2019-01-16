import theano
import theano.tensor as T
import numpy as np
import pickle as pickle
import os
import datetime
from collections import OrderedDict
import datetime
from theano import config
from utils.features import numpy_floatX
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from optimiser.grad_optimiser import create_optimization_updates
# from recursive_net_utils.tree_rnn import TreeRNN

SEED = 123
np.random.seed(SEED)


class EB_LSTM(object):
    def __init__(self, nin, n_hidden=100, nout=2, learning_rate_decay=1.0, activation='tanh', optimiser='sgd',
                 output_type='softmax', L1_reg=0.00, L2_reg=0.00001, state=None, vocab_size=None,
                 dim_emb=50, context_win_size=5, embedding=None, use_dropout=False,
                 w_hh_initialise_strategy='identity', w_hh_type='independent', reload_model=None,
                 reload_path=None,
                 w2v_embedding=None, position_feat= False, entity_presence_feat=False,
                 pos_feat_embedding=False, pos_indicator_embedding=False, ent_pres_feat_embedding=False,
                 dim_ent_pres_emb = 50, dim_pos_emb = 50, pos_vocab_size = None,
                 pos_emb_type=None, ent_pres_emb_type=None, context_window_usage=False,
                 postag = False, postag_vocab_size = None, entity_class = False,
                 entity_class_vocab_size = None, dim_postag_emb = 5, dim_entity_class_emb = 5,
                 update_pos_emb = None, update_ner_emb = None, add_subtree_emb = False,
                 dim_subtree_emb = None, max_degree=None, treernn_weights = 'independent',
                 margin_pos=2.5, margin_neg=0.5, scale=2,
                 batch_size=1, ignore_class=18, ranking=False):

        rng = np.random.RandomState(1234)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.activ = activation
        self.output_type = output_type
        self.use_dropout = use_dropout
        self.optimiser = optimiser
        self.w_hh_type = w_hh_type
        self.position_feat = position_feat
        self.entity_presence_feat = entity_presence_feat
        self.pos_feat_embedding = pos_feat_embedding
        self.pos_indicator_embedding = pos_indicator_embedding
        self.ent_pres_feat_embedding = ent_pres_feat_embedding
        self.dim_ent_pres_emb = dim_ent_pres_emb
        self.dim_pos_emb = dim_pos_emb
        self.pos_vocab_size = pos_vocab_size
        self.pos_emb_type = pos_emb_type
        self.ent_pres_emb_type = ent_pres_emb_type
        self.postag = postag
        self.dim_postag_emb = dim_postag_emb
        self.postag_vocab_size = postag_vocab_size
        self.entity_class = entity_class
        self.dim_entity_class_emb = dim_entity_class_emb
        self.entity_class_vocab_size = entity_class_vocab_size
        self.update_pos_emb = update_pos_emb
        self.update_ner_emb = update_ner_emb
        self.add_subtree_emb = add_subtree_emb
        self.dim_subtree_emb = dim_subtree_emb
        self.num_emb = len(w2v_embedding)
        self.emb_dim = dim_subtree_emb
        self.hidden_dim = dim_subtree_emb
        self.max_degree = max_degree
        self.treernn_weights = treernn_weights



        if embedding == 'word2vec_update':
            # vocab_size : size of vocabulary
            # dim_emb : dimension of the word embeddings
            # context_win_size : word window context size
            self.vocab_size = vocab_size
            self.dim_emb = dim_emb
            self.context_win_size = context_win_size
            self.emb = theano.shared(name='embeddings',
                                     value=np.array(w2v_embedding).astype(theano.config.floatX))

            if entity_presence_feat == True:
                ent_pres_feat = T.fmatrix()# as many rows as words in sentence, columns are two

            if position_feat == True:
                pos_feat = T.fmatrix() # as many rows as words in sentence, columns are two

            if pos_feat_embedding == True:
                if self.pos_emb_type == 'COUPLED':
                    self.pos_emb_e1_e2 = theano.shared(name='pos_emb_e1_e2',
                                                       value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                                     (2, self.pos_vocab_size+1, self.dim_pos_emb)).astype(theano.config.floatX))
                    # add one for padding at the end

                elif self.pos_emb_type == 'DECOUPLED':
                    self.pos_emb_e1 = theano.shared(name='pos_emb_e1',
                                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                                  (self.pos_vocab_size+1, self.dim_pos_emb)).astype(theano.config.floatX))
                    # add one for padding at the end
                    self.pos_emb_e2 = theano.shared(name='pos_emb_e2',
                                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                                  (self.pos_vocab_size+1, self.dim_pos_emb)).astype(theano.config.floatX))
                    # add one for padding at the end
            if ent_pres_feat_embedding == True:
                if self.ent_pres_emb_type == 'COUPLED':
                    self.ent_pres_emb_e1_e2 = theano.shared(name='ent_pres_emb_e1_e2',
                                                            value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                                          (2, self.vocab_size+1, self.dim_ent_pres_emb)).astype(theano.config.floatX))
                    # add one for padding at the end

                elif self.ent_pres_emb_type == 'DECOUPLED':
                    self.ent_pres_emb_e1 = theano.shared(name='ent_pres_emb_e1',
                                                         value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                                       (self.vocab_size+1, self.dim_ent_pres_emb)).astype(theano.config.floatX))
                    # add one for padding at the end
                    self.ent_pres_emb_e2 = theano.shared(name='ent_pres_emb_e2',
                                                         value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                                       (self.vocab_size+1, self.dim_ent_pres_emb)).astype(theano.config.floatX))
                    # add one for padding at the end


            if postag == True:
                self.postag_emb = theano.shared(name='postag_emb',
                                                value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                              ( self.postag_vocab_size, self.dim_postag_emb)).astype(theano.config.floatX))
                # add one for padding at the end

            if entity_class == True:
                self.entity_class_emb = theano.shared(name='entity_class_emb',
                                                      value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                                    ( self.entity_class_vocab_size, self.dim_entity_class_emb)).astype(theano.config.floatX))
                # add one for padding at the end


        if embedding == 'theano_word_embeddings':
            # vocab_size : size of vocabulary
            # dim_emb : dimension of the word embeddings
            # context_win_size : word window context size
            self.vocab_size = vocab_size
            self.dim_emb = dim_emb
            self.context_win_size = context_win_size
            self.emb = theano.shared(name='embeddings',
                                     value=0.2 * np.random.uniform(-1.0, 1.0,
                                                                   (self.vocab_size+1, self.dim_emb)).astype(theano.config.floatX))
            # add one for padding at the end
        #define shared variables
        # for embedding type word2vec_update, nin = self.dim_emb*self.context_win_size
        # else nin = vocab_size (for one-hot encoding) and nin = dim_emb+pos+entity_pres

        self.W_xi_f = np.asarray(rng.normal(size=(nin, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
        self.W_xf_f = np.asarray(rng.normal(size=(nin, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
        self.W_xo_f = np.asarray(rng.normal(size=(nin, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
        self.W_xc_f = np.asarray(rng.normal(size=(nin, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)

        self.W_xi_b = np.asarray(rng.normal(size=(nin, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
        self.W_xf_b = np.asarray(rng.normal(size=(nin, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
        self.W_xo_b = np.asarray(rng.normal(size=(nin, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
        self.W_xc_b = np.asarray(rng.normal(size=(nin, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)

        ##Cell Unit weights
        self.W_ci_f = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
        self.W_co_f = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
        self.W_cf_f = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)

        self.W_ci_b = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
        self.W_co_b = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
        self.W_cf_b = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)


        if self.use_dropout == True:
            # Used for dropout
            use_noise = theano.shared(numpy_floatX(0.))
            mask = T.matrix('mask', dtype=config.floatX)
            trng = RandomStreams(SEED)

        max_norm = None
        if reload_model != True:
            if state['clipstyle'] == 'rescale':
                self.clip_norm_cutoff = state['cutoff']
                max_norm = self.clip_norm_cutoff


        if self.w_hh_type == 'shared' or self.w_hh_type == 'transpose':
            if w_hh_initialise_strategy == 'identity':
                # to handle 'vanishing gradient' problem
                self.W_hi = np.asarray(np.identity(n_hidden, dtype = theano.config.floatX))
                self.W_hi_bi = np.asarray(np.identity(n_hidden, dtype = theano.config.floatX))

                self.W_hf = np.asarray(np.identity(n_hidden, dtype=theano.config.floatX))
                self.W_hf_bi = np.asarray(np.identity(n_hidden, dtype=theano.config.floatX))

                self.W_hc = np.asarray(np.identity(n_hidden, dtype=theano.config.floatX))
                self.W_hc_bi = np.asarray(np.identity(n_hidden, dtype=theano.config.floatX))

                self.W_ho = np.asarray(np.identity(n_hidden, dtype=theano.config.floatX))
                self.W_ho_bi = np.asarray(np.identity(n_hidden, dtype=theano.config.floatX))

            elif w_hh_initialise_strategy == 'ortho':
                self.W_hi = np.asarray(self.ortho_weight(n_hidden), dtype = theano.config.floatX)
                self.W_hi_bi = np.asarray(self.ortho_weight(n_hidden), dtype = theano.config.floatX)

                self.W_hf = np.asarray(self.ortho_weight(n_hidden), dtype=theano.config.floatX)
                self.W_hf_bi = np.asarray(self.ortho_weight(n_hidden), dtype=theano.config.floatX)

                self.W_hc = np.asarray(self.ortho_weight(n_hidden), dtype=theano.config.floatX)
                self.W_hc_bi = np.asarray(self.ortho_weight(n_hidden), dtype=theano.config.floatX)

                self.W_ho = np.asarray(self.ortho_weight(n_hidden), dtype=theano.config.floatX)
                self.W_ho_bi = np.asarray(self.ortho_weight(n_hidden), dtype=theano.config.floatX)

            elif w_hh_initialise_strategy == 'uniform':
                self.W_hi = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0), dtype = theano.config.floatX)
                self.W_hi_bi = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0), dtype = theano.config.floatX)

                self.W_hf = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0), dtype=theano.config.floatX)
                self.W_hf_bi = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0), dtype=theano.config.floatX)

                self.W_hc = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0), dtype=theano.config.floatX)
                self.W_hc_bi = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0), dtype=theano.config.floatX)

                self.W_ho = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0), dtype=theano.config.floatX)
                self.W_ho_bi = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0), dtype=theano.config.floatX)
            else:
                self.W_hi = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc = .0), dtype = theano.config.floatX)
                self.W_hi_bi =  np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc = .0), dtype = theano.config.floatX)

                self.W_hf = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
                self.W_hf_bi = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)

                self.W_hc = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
                self.W_hc_bi = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)

                self.W_ho = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
                self.W_ho_bi = np.asarray(rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)




        else:
            if w_hh_initialise_strategy == 'identity':
                # to handle 'vanishing gradient' problem
                self.W_hi_f = np.asarray(np.identity(n_hidden, dtype = theano.config.floatX))
                self.W_hi_b = np.asarray(np.identity(n_hidden, dtype = theano.config.floatX))
                self.W_hi_bi = np.asarray(np.identity(n_hidden, dtype = theano.config.floatX))

                self.W_hf_f = np.asarray(np.identity(n_hidden, dtype=theano.config.floatX))
                self.W_hf_b = np.asarray(np.identity(n_hidden, dtype=theano.config.floatX))
                self.W_hf_bi = np.asarray(np.identity(n_hidden, dtype=theano.config.floatX))

                self.W_hc_f = np.asarray(np.identity(n_hidden, dtype=theano.config.floatX))
                self.W_hc_b = np.asarray(np.identity(n_hidden, dtype=theano.config.floatX))
                self.W_hc_bi = np.asarray(np.identity(n_hidden, dtype=theano.config.floatX))

                self.W_ho_f = np.asarray(np.identity(n_hidden, dtype=theano.config.floatX))
                self.W_ho_b = np.asarray(np.identity(n_hidden, dtype=theano.config.floatX))
                self.W_ho_bi = np.asarray(np.identity(n_hidden, dtype=theano.config.floatX))


            elif w_hh_initialise_strategy == 'ortho':
                self.W_hi_f = np.asarray(self.ortho_weight(n_hidden), dtype = theano.config.floatX)
                self.W_hi_b = np.asarray(self.ortho_weight(n_hidden), dtype = theano.config.floatX)
                self.W_hi_bi = np.asarray(self.ortho_weight(n_hidden), dtype = theano.config.floatX)

                self.W_hf_f = np.asarray(self.ortho_weight(n_hidden), dtype=theano.config.floatX)
                self.W_hf_b = np.asarray(self.ortho_weight(n_hidden), dtype=theano.config.floatX)
                self.W_hf_bi = np.asarray(self.ortho_weight(n_hidden), dtype=theano.config.floatX)

                self.W_hc_f = np.asarray(self.ortho_weight(n_hidden), dtype=theano.config.floatX)
                self.W_hc_b = np.asarray(self.ortho_weight(n_hidden), dtype=theano.config.floatX)
                self.W_hc_bi = np.asarray(self.ortho_weight(n_hidden), dtype=theano.config.floatX)

                self.W_ho_f = np.asarray(self.ortho_weight(n_hidden), dtype=theano.config.floatX)
                self.W_ho_b = np.asarray(self.ortho_weight(n_hidden), dtype=theano.config.floatX)
                self.W_ho_bi = np.asarray(self.ortho_weight(n_hidden), dtype=theano.config.floatX)


            elif w_hh_initialise_strategy == 'uniform':
                self.W_hi_f = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0), dtype = theano.config.floatX)
                self.W_hi_b = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0), dtype = theano.config.floatX)
                self.W_hi_bi = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0), dtype = theano.config.floatX)

                self.W_hf_f = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0),
                                         dtype=theano.config.floatX)
                self.W_hf_b = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0),
                                         dtype=theano.config.floatX)
                self.W_hf_bi = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0),
                                          dtype=theano.config.floatX)

                self.W_hc_f = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0),
                                         dtype=theano.config.floatX)
                self.W_hc_b = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0),
                                         dtype=theano.config.floatX)
                self.W_hc_bi = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0),
                                          dtype=theano.config.floatX)

                self.W_ho_f = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0),
                                         dtype=theano.config.floatX)
                self.W_ho_b = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0),
                                         dtype=theano.config.floatX)
                self.W_ho_bi = np.asarray(rng.uniform(size=(n_hidden, n_hidden), low=-1.0, high=1.0),
                                          dtype=theano.config.floatX)


            else:
                self.W_hi_f = np.asarray(
                    rng.normal(size=(n_hidden, n_hidden), scale=.01, loc = .0), dtype = theano.config.floatX)
                self.W_hi_b = np.asarray(
                    rng.normal(size=(n_hidden, n_hidden), scale=.01, loc = .0), dtype = theano.config.floatX)
                self.W_hi_bi = np.asarray(
                    rng.normal(size=(n_hidden, n_hidden), scale=.01, loc = .0), dtype = theano.config.floatX)

                self.W_hf_f = np.asarray(
                    rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
                self.W_hf_b = np.asarray(
                    rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
                self.W_hf_bi = np.asarray(
                    rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)

                self.W_hc_f = np.asarray(
                    rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
                self.W_hc_b = np.asarray(
                    rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
                self.W_hc_bi = np.asarray(
                    rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)

                self.W_ho_f = np.asarray(
                    rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
                self.W_ho_b = np.asarray(
                    rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)
                self.W_ho_bi = np.asarray(
                    rng.normal(size=(n_hidden, n_hidden), scale=.01, loc=.0), dtype=theano.config.floatX)



        self.W_hy_f = np.asarray(
            rng.normal(size=(n_hidden, nout), scale=.01, loc=0.0), dtype=theano.config.floatX)
        self.W_hy_b = np.asarray(
            rng.normal(size=(n_hidden, nout), scale=.01, loc=0.0), dtype=theano.config.floatX)
        self.W_hy_bi = np.asarray(
            rng.normal(size=(n_hidden, nout), scale=.01, loc=0.0), dtype=theano.config.floatX)

        self.b_hi_f = np.zeros((n_hidden,), dtype=theano.config.floatX)
        self.b_hi_b = np.zeros((n_hidden,), dtype=theano.config.floatX)

        self.b_hf_f = np.zeros((n_hidden,), dtype=theano.config.floatX)
        self.b_hf_b = np.zeros((n_hidden,), dtype=theano.config.floatX)

        self.b_hc_f = np.zeros((n_hidden,), dtype=theano.config.floatX)
        self.b_hc_b = np.zeros((n_hidden,), dtype=theano.config.floatX)

        self.b_ho_f = np.zeros((n_hidden,), dtype=theano.config.floatX)
        self.b_ho_b = np.zeros((n_hidden,), dtype=theano.config.floatX)



        self.b_hy_f = np.zeros((nout,), dtype=theano.config.floatX)
        self.b_hy_b = np.zeros((nout,), dtype=theano.config.floatX)
        self.b_hy_bi = np.zeros((nout,), dtype=theano.config.floatX)


        if self.activ == 'tanh':
            activation = T.tanh
        elif self.activ == 'sigmoid':
            activation = T.nnet.sigmoid
        elif self.activ == 'relu':
            activation = lambda x: x * (x > 0)
        elif self.activ == 'cappedrelu':
            activation = lambda x: T.minimum(x * (x > 0), 6)
        else:
            raise NotImplementedError

        self.activ = activation

        if embedding == 'theano_word_embeddings' or embedding == 'word2vec_update':
            idx_f = T.imatrix() # as many columns as context window size/lines as words in the sentence
            idx_b = T.imatrix() # as many columns as context window size/lines as words in the sentence

            if context_window_usage == True:
                x_f = self.emb[idx_f].reshape((idx_f.shape[0], self.dim_emb*self.context_win_size))
                x_b = self.emb[idx_b].reshape((idx_b.shape[0], self.dim_emb*self.context_win_size))
            else:
                x_f = self.emb[idx_f].reshape((idx_f.shape[1], self.dim_emb*self.context_win_size))
                x_b = self.emb[idx_b].reshape((idx_b.shape[1], self.dim_emb*self.context_win_size))

            if self.position_feat == True and self.entity_presence_feat == True:
                x_f = theano.tensor.concatenate([x_f, ent_pres_feat, pos_feat], axis=1)
                x_b = theano.tensor.concatenate([x_b, ent_pres_feat[::-1], pos_feat[::-1]], axis=1)


            elif self.entity_presence_feat == True:
                # concatenate entity presence features
                # concatenate entity presence features to embedding of x (list of word indices)
                x_f = theano.tensor.concatenate([x_f, ent_pres_feat], axis=1)
                x_b = theano.tensor.concatenate([x_b, ent_pres_feat[::-1]], axis=1)

            elif self.position_feat == True:
                # concatenate position features
                x_f = theano.tensor.concatenate([x_f, pos_feat], axis=1)
                x_b = theano.tensor.concatenate([x_b, pos_feat[::-1]], axis=1)

            if pos_feat_embedding == True:
                pos_feat_idx = T.imatrix()
                if self.pos_emb_type == 'COUPLED':
                    pos_feat_emb_f_e1 = self.pos_emb_e1_e2[0][pos_feat_idx[:,0]].reshape((pos_feat_idx[:,0].shape[0],
                                                                                          self.dim_pos_emb*self.context_win_size))
                    pos_feat_emb_f_e2 = self.pos_emb_e1_e2[1][pos_feat_idx[:,1]].reshape((pos_feat_idx[:,1].shape[0],
                                                                                          self.dim_pos_emb*self.context_win_size))
                    pos_feat_emb_b_e1 = self.pos_emb_e1_e2[0][pos_feat_idx[:,0][::-1]].reshape((pos_feat_idx[:,0].shape[0],
                                                                                                self.dim_pos_emb*self.context_win_size))

                    pos_feat_emb_b_e2 = self.pos_emb_e1_e2[1][pos_feat_idx[:,1][::-1]].reshape((pos_feat_idx[:,1].shape[0],
                                                                                                self.dim_pos_emb*self.context_win_size))
                elif self.pos_emb_type == 'DECOUPLED':
                    pos_feat_emb_f_e1 = self.pos_emb_e1[pos_feat_idx[:, 0]].reshape((pos_feat_idx[:, 0].shape[0],
                                                                                     self.dim_pos_emb*self.context_win_size))
                    pos_feat_emb_f_e2 = self.pos_emb_e2[pos_feat_idx[:, 1]].reshape((pos_feat_idx[:,1].shape[0],
                                                                                     self.dim_pos_emb*self.context_win_size))
                    pos_feat_emb_b_e1 = self.pos_emb_e1[pos_feat_idx[:, 0][::-1]].reshape((pos_feat_idx[:, 0].shape[0],
                                                                                           self.dim_pos_emb*self.context_win_size))
                    pos_feat_emb_b_e2 = self.pos_emb_e2[pos_feat_idx[:, 1][::-1]].reshape((pos_feat_idx[:,1].shape[0],
                                                                                           self.dim_pos_emb*self.context_win_size))
            if ent_pres_feat_embedding == True:
                ent_pres_feat_idx = T.imatrix()
                if self.pos_emb_type == 'COUPLED':
                    ent_pres_feat_emb_f_e1 = self.ent_pres_emb_e1_e2[0][ent_pres_feat_idx[:,0]].reshape((ent_pres_feat_idx[:,0].shape[0],
                                                                                                         self.dim_ent_pres_emb*self.context_win_size))

                    ent_pres_feat_emb_f_e2 = self.ent_pres_emb_e1_e2[1][ent_pres_feat_idx[:,1]].reshape((ent_pres_feat_idx[:,1].shape[0],
                                                                                                         self.dim_ent_pres_emb*self.context_win_size))

                    ent_pres_feat_emb_b_e1 = self.ent_pres_emb_e1_e2[0][ent_pres_feat_idx[:,0][::-1]].reshape((ent_pres_feat_idx[:,0].shape[0],
                                                                                                               self.dim_ent_pres_emb*self.context_win_size))

                    ent_pres_feat_emb_b_e2 = self.ent_pres_emb_e1_e2[1][ent_pres_feat_idx[:,1][::-1]].reshape((ent_pres_feat_idx[:,1].shape[0],
                                                                                                               self.dim_ent_pres_emb*self.context_win_size))

                elif self.pos_emb_type == 'DECOUPLED':
                    ent_pres_feat_emb_f_e1 = self.pos_emb_e1[ent_pres_feat_idx[:, 0]].reshape((ent_pres_feat_idx[:, 0].shape[0],
                                                                                               self.dim_ent_pres_emb*self.context_win_size))
                    ent_pres_feat_emb_f_e2 = self.pos_emb_e2[ent_pres_feat_idx[:, 1]].reshape((ent_pres_feat_idx[:,1].shape[0],
                                                                                               self.dim_ent_pres_emb*self.context_win_size))
                    ent_pres_feat_emb_b_e1 = self.pos_emb_e1[ent_pres_feat_idx[:, 0][::-1]].reshape((ent_pres_feat_idx[:, 0].shape[0],
                                                                                                     self.dim_ent_pres_emb*self.context_win_size))
                    ent_pres_feat_emb_b_e2 = self.pos_emb_e2[ent_pres_feat_idx[:, 1][::-1]].reshape((ent_pres_feat_idx[:,1].shape[0],
                                                                                                     self.dim_ent_pres_emb*self.context_win_size))

            if self.pos_feat_embedding == True and self.ent_pres_feat_embedding == True:
                x_f = theano.tensor.concatenate([x_f, ent_pres_feat_emb_f_e1, ent_pres_feat_emb_f_e2,
                                                 pos_feat_emb_f_e1, pos_feat_emb_f_e2], axis=1)
                x_b = theano.tensor.concatenate([x_b, ent_pres_feat_emb_b_e1, ent_pres_feat_emb_b_e2,
                                                 pos_feat_emb_b_e1, pos_feat_emb_b_e2], axis=1)
            elif self.ent_pres_feat_embedding == True:
                x_f = theano.tensor.concatenate([x_f, ent_pres_feat_emb_f_e1, ent_pres_feat_emb_f_e2], axis=1)
                x_b = theano.tensor.concatenate([x_b, ent_pres_feat_emb_b_e1, ent_pres_feat_emb_b_e2], axis=1)

            elif self.pos_feat_embedding == True:
                x_f = theano.tensor.concatenate([x_f, pos_feat_emb_f_e1, pos_feat_emb_f_e2], axis=1)
                x_b = theano.tensor.concatenate([x_b, pos_feat_emb_b_e1, pos_feat_emb_b_e2], axis=1)

            if self.postag == True:
                # postag_idx = T.imatrix()
                postag_idx = T.ivector()

                # self.postag_emb_val = self.postag_emb[postag_idx].reshape((postag_idx.shape[0],
                                                        # self.dim_postag_emb*self.context_win_size))
                # self.postag_emb_val = self.postag_emb[postag_idx[:, 0]].reshape((postag_idx.shape[1], self.dim_postag_emb))
                self.postag_emb_val = self.postag_emb[postag_idx]
                x_f = theano.tensor.concatenate([x_f, self.postag_emb_val], axis=1)
                x_b = theano.tensor.concatenate([x_b, self.postag_emb_val[::-1]], axis=1)

            if self.entity_class == True:
                entity_class_idx = T.ivector()
                self.entity_class_emb_val = self.entity_class_emb[entity_class_idx]
                x_f = theano.tensor.concatenate([x_f, self.entity_class_emb_val], axis=1)
                x_b = theano.tensor.concatenate([x_b, self.entity_class_emb_val[::-1]], axis=1)

        else:
            x_f = T.matrix()
            x_b = T.matrix()

        if embedding == 'theano_word_embeddings':
            print('self.emb.shape:', self.emb.shape)

        if embedding == 'word2vec_update':
            print('self.emb.shape:', self.emb.shape)

        # define symbolic variables
        lr = T.scalar('lr', dtype=theano.config.floatX)
        rho = T.scalar('rho', dtype=theano.config.floatX)
        t = T.iscalar()
        mom = T.scalar('mom', dtype=theano.config.floatX)

        print('W_xi_f.shape:', self.W_xi_f.shape)
        print('W_xi_b.shape:', self.W_xi_b.shape)

        print('W_xf_f.shape:', self.W_xf_f.shape)
        print('W_xf_b.shape:', self.W_xf_b.shape)

        print('W_xc_f.shape:', self.W_xc_f.shape)
        print('W_xc_b.shape:', self.W_xc_b.shape)

        print('W_xo_f.shape:', self.W_xo_f.shape)
        print('W_xo_b.shape:', self.W_xo_b.shape)

        print('W_ci_f.shape:', self.W_ci_f.shape)
        print('W_cf_f.shape:', self.W_cf_f.shape)
        print('W_co_f.shape:', self.W_co_f.shape)

        print('W_ci_b.shape:', self.W_ci_b.shape)
        print('W_cf_b.shape:', self.W_cf_b.shape)
        print('W_co_b.shape:', self.W_co_b.shape)

        print('W_hi_bi.shape:', self.W_hi_bi.shape)
        print('W_hf_bi.shape:', self.W_hf_bi.shape)
        print('W_hc_bi.shape:', self.W_hc_bi.shape)
        print('W_ho_bi.shape:', self.W_ho_bi.shape)

        print('W_hy_f.shape:', self.W_hy_f.shape)

        print('b_hi_f.shape:', self.b_hi_f.shape)
        print('b_hf_f.shape:', self.b_hf_f.shape)
        print('b_hc_f.shape:', self.b_hc_f.shape)
        print('b_ho_f.shape:', self.b_ho_f.shape)

        print('b_hy_f.shape:', self.b_hy_f.shape)
        print('W_hy_b.shape:', self.W_hy_b.shape)


        print('b_hi_b.shape:', self.b_hi_b.shape)
        print('b_hf_b.shape:', self.b_hf_b.shape)
        print('b_hc_b.shape:', self.b_hc_b.shape)
        print('b_ho_b.shape:', self.b_ho_b.shape)


        print('b_hy_b.shape:', self.b_hy_b.shape)
        print('W_hy_bi.shape:', self.W_hy_bi.shape)
        print('b_hy_bi.shape:', self.b_hy_bi.shape)
        print('W_hy_bi.shape:', self.W_hy_bi.shape)

        if w_hh_type == 'shared' or self.w_hh_type == 'transpose':
            print('W_hi.shape:', self.W_hi.shape)
            self.W_hi = theano.shared(self.W_hi, 'W_hi')

            print('W_hf.shape:', self.W_hf.shape)
            self.W_hf = theano.shared(self.W_hf, 'W_hf')

            print('W_hc.shape:', self.W_hc.shape)
            self.W_hc = theano.shared(self.W_hc, 'W_hc')

            print('W_ho.shape:', self.W_ho.shape)
            self.W_ho = theano.shared(self.W_ho, 'W_ho')
        else:
            print('W_hi_f.shape:', self.W_hi_f.shape)
            print('W_hi_b.shape:', self.W_hi_b.shape)
            self.W_hi_f = theano.shared(self.W_hi_f, 'W_hi_f')
            self.W_hi_b = theano.shared(self.W_hi_b, 'W_hi_b')

            print('W_hf_f.shape:', self.W_hf_f.shape)
            print('W_hf_b.shape:', self.W_hf_b.shape)
            self.W_hf_f = theano.shared(self.W_hf_f, 'W_hf_f')
            self.W_hf_b = theano.shared(self.W_hf_b, 'W_hf_b')

            print('W_hc_f.shape:', self.W_hc_f.shape)
            print('W_hc_b.shape:', self.W_hc_b.shape)
            self.W_hc_f = theano.shared(self.W_hc_f, 'W_hc_f')
            self.W_hc_b = theano.shared(self.W_hc_b, 'W_hc_b')

            print('W_ho_f.shape:', self.W_ho_f.shape)
            print('W_ho_b.shape:', self.W_ho_b.shape)
            self.W_ho_f = theano.shared(self.W_ho_f, 'W_ho_f')
            self.W_ho_b = theano.shared(self.W_ho_b, 'W_ho_b')

        self.W_xi_f = theano.shared(self.W_xi_f, 'W_xi_f')
        self.W_xi_b = theano.shared(self.W_xi_b, 'W_xi_b')

        self.W_xf_f = theano.shared(self.W_xf_f, 'W_xf_f')
        self.W_xf_b = theano.shared(self.W_xf_b, 'W_xf_b')

        self.W_xc_f = theano.shared(self.W_xc_f, 'W_xc_f')
        self.W_xc_b = theano.shared(self.W_xc_b, 'W_xc_b')

        self.W_xo_f = theano.shared(self.W_xo_f, 'W_xo_f')
        self.W_xo_b = theano.shared(self.W_xo_b, 'W_xo_b')

        self.W_ci_f = theano.shared(self.W_ci_f, 'W_ci_f')
        self.W_cf_f = theano.shared(self.W_cf_f, 'W_cf_f')
        self.W_co_f = theano.shared(self.W_co_f, 'W_co_f')

        self.W_ci_b = theano.shared(self.W_ci_b, 'W_ci_b')
        self.W_cf_b = theano.shared(self.W_cf_b, 'W_cf_b')
        self.W_co_b = theano.shared(self.W_co_b, 'W_co_b')

        self.b_hi_f = theano.shared(self.b_hi_f, 'b_hi_f')
        self.b_hf_f = theano.shared(self.b_hf_f, 'b_hf_f')
        self.b_hc_f = theano.shared(self.b_hc_f, 'b_hc_f')
        self.b_ho_f = theano.shared(self.b_ho_f, 'b_ho_f')

        self.b_hi_b = theano.shared(self.b_hi_b, 'b_hi_b')
        self.b_hf_b = theano.shared(self.b_hf_b, 'b_hf_b')
        self.b_hc_b = theano.shared(self.b_hc_b, 'b_hc_b')
        self.b_ho_b = theano.shared(self.b_ho_b, 'b_ho_b')

        self.W_hy_bi = theano.shared(self.W_hy_bi, 'W_hy_bi')
        self.b_hy_bi = theano.shared(self.b_hy_bi, 'b_hy_bi')

        self.h0_tm1_f = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX), 'h0_tm1_f')
        self.h0_tm1_b = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX), 'h0_tm1_b')
        self.h0_bi_tm1 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX), 'h0_bi_tm1')

        self.c0_tm1 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX), 'c0_tm1')

        """ Need to check weather c0 should take 2 values(f and b), or only 1"""

        if w_hh_type == 'shared' or self.w_hh_type == 'transpose':
            self.params = [ self.W_xi_f,self.W_xf_f,self.W_xc_f,self.W_xo_f,
                            self.W_xi_b,self.W_xf_b,self.W_xc_b,self.W_xo_b,
                            self.W_ci_f, self.W_cf_f, self.W_co_f,
                            self.W_ci_b, self.W_cf_b, self.W_co_b,
                            self.W_hi,self.W_hf,self.W_hc,self.W_ho, self.W_hy_bi,
                            self.b_hi_f,self.b_hf_f,self.b_hc_f,self.b_ho_f, self.b_hy_bi, self.h0_tm1_f,
                            self.h0_tm1_b, self.b_hi_b,self.b_hf_b,self.b_hc_b,self.b_ho_b, self.h0_bi_tm1, self.c0_tm1]


            self.names  = ['W_xi_f','W_xf_f','W_xc_f','W_xo_f',
                            'W_xi_b','W_xf_b','W_xc_b','W_xo_b',
                           'W_ci_f', 'W_cf_f', 'W_co_f',
                           'W_ci_b', 'W_cf_b', 'W_co_b',
                           'W_hi','W_hf','W_hc','W_ho','W_hy_bi',
                            'b_hi_f','b_hf_f','b_hc_f','b_ho_f', 'b_hy_bi', 'h0_tm1_f',
                            'h0_tm1_b', 'b_hi_b','b_hf_b','b_hc_b','b_ho_b', 'h0_bi_tm1' 'c0_tm1' ]

        else:
            self.params = [self.W_xi_f,self.W_xf_f,self.W_xc_f,self.W_xo_f,
                           self.W_xi_b,self.W_xf_b,self.W_xc_b,self.W_xo_b,
                           self.W_ci_f, self.W_cf_f, self.W_co_f,
                           self.W_ci_b, self.W_cf_b, self.W_co_b,
                           self.W_hi_f, self.W_hf_f, self.W_hc_f, self.W_ho_f, self.W_hy_bi,
                           self.b_hi_f, self.b_hf_f, self.b_hc_f, self.b_ho_f,
                           self.b_hy_bi, self.h0_tm1_f,
                           self.W_hi_b, self.W_hf_b, self.W_hc_b,self.W_ho_b, self.h0_tm1_b,
                           self.b_hi_b, self.b_hf_b, self.b_hc_b, self.b_ho_b,
                           self.h0_bi_tm1, self.c0_tm1]
            self.names = ['W_xi_f','W_xf_f','W_xc_f','W_xo_f',
                          'W_xi_b','W_xf_b','W_xc_b','W_xo_b',
                          'W_ci_f', 'W_cf_f', 'W_co_f',
                          'W_ci_b', 'W_cf_b', 'W_co_b',
                          'W_hi_f', 'W_hf_f', 'W_hc_f', 'W_ho_f', 'W_hy_bi',
                          'b_hi_f', 'b_hf_f', 'b_hc_f', 'b_ho_f'
                          'b_hy_bi', 'h0_tm1_f',
                          'W_hi_b', 'W_hf_b', 'W_hc_b','W_ho_b', 'h0_tm1_b',
                          'b_hi_b', 'b_hf_b', 'b_hc_b', 'b_ho_b',
                          'h0_bi_tm1' 'c0_tm1']

        if embedding == 'theano_word_embeddings' or embedding == 'word2vec_update':
            self.params.append(self.emb)
            self.names.append('embeddings')

            if self.pos_feat_embedding == True:
                if self.pos_emb_type == 'COUPLED':
                    self.params.append(self.pos_emb_e1_e2)
                    self.names.append('pos_emb_e1_e2')
                elif self.pos_emb_type == 'DECOUPLED':
                    self.params.append(self.pos_emb_e1)
                    self.names.append('pos_emb_e1')
                    self.params.append(self.pos_emb_e2)
                    self.names.append('pos_emb_e2')

            elif self.ent_pres_feat_embedding == True:
                if self.ent_pres_emb_type == 'COUPLED':
                    self.params.append(self.ent_pres_emb_e1_e2)
                    self.names.append('ent_pres_emb_e1_e2')
                elif self.ent_pres_emb_type == 'DECOUPLED':
                    self.params.append(self.ent_pres_emb_e1)
                    self.names.append('ent_pres_emb_e1')
                    self.params.append(self.ent_pres_emb_e2)
                    self.names.append('ent_pres_emb_e2')

            if self.postag == True and self.update_pos_emb == True:
                self.params.append(self.postag_emb)
                self.names.append('postag_emb')

            if self.entity_class == True and self.update_ner_emb == True:
                self.params.append(self.entity_class_emb)
                self.names.append('entity_class_emb')

            if self.add_subtree_emb:
                # < to do >
                # add h0 of rnn and sub tree embedding to params
                if treernn_weights == 'independent':
                    self.W_i = np.asarray(rng.normal(size=(self.hidden_dim, self.emb_dim), scale=.01, loc=0.0), dtype=theano.config.floatX)
                    self.U_i = np.asarray(rng.normal(size=(self.hidden_dim, self.hidden_dim), scale=.01, loc=0.0),dtype=theano.config.floatX)
                    self.b_i = np.zeros(self.hidden_dim, dtype=theano.config.floatX)

                    self.W_f = np.asarray(rng.normal(size=(self.hidden_dim, self.emb_dim), scale=.01, loc=0.0),
                                          dtype=theano.config.floatX)
                    self.U_f = np.asarray(rng.normal(size=(self.hidden_dim, self.hidden_dim), scale=.01, loc=0.0),
                                          dtype=theano.config.floatX)
                    self.b_f = np.zeros(self.hidden_dim, dtype=theano.config.floatX)

                    self.W_o = np.asarray(rng.normal(size=(self.hidden_dim, self.emb_dim), scale=.01, loc=0.0),
                                          dtype=theano.config.floatX)
                    self.U_o = np.asarray(rng.normal(size=(self.hidden_dim, self.hidden_dim), scale=.01, loc=0.0),
                                          dtype=theano.config.floatX)
                    self.b_o = np.zeros(self.hidden_dim, dtype=theano.config.floatX)

                    self.W_u = np.asarray(rng.normal(size=(self.hidden_dim, self.emb_dim), scale=.01, loc=0.0),
                                          dtype=theano.config.floatX)
                    self.U_u = np.asarray(rng.normal(size=(self.hidden_dim, self.hidden_dim), scale=.01, loc=0.0),
                                          dtype=theano.config.floatX)
                    self.b_u = np.zeros(self.hidden_dim, dtype=theano.config.floatX)


                    print('W_i.shape:', self.W_i.shape)
                    print('U_i.shape:', self.U_i.shape)
                    print('b_i.shape:', self.b_i.shape)

                    print('W_f.shape:', self.W_i.shape)
                    print('U_f.shape:', self.U_i.shape)
                    print('b_f.shape:', self.b_i.shape)

                    print('W_o.shape:', self.W_i.shape)
                    print('U_o.shape:', self.U_i.shape)
                    print('b_o.shape:', self.b_i.shape)

                    print('W_u.shape:', self.W_i.shape)
                    print('U_u.shape:', self.U_i.shape)
                    print('b_u.shape:', self.b_i.shape)


                    self.W_i = theano.shared(self.W_i, 'W_i')
                    self.U_i = theano.shared(self.U_i, 'U_i')
                    self.b_i = theano.shared(self.b_i, 'b_i')

                    self.W_f = theano.shared(self.W_f, 'W_f')
                    self.U_f = theano.shared(self.U_f, 'U_f')
                    self.b_f = theano.shared(self.b_f, 'b_f')

                    self.W_o = theano.shared(self.W_o, 'W_o')
                    self.U_o = theano.shared(self.U_o, 'U_o')
                    self.b_o = theano.shared(self.b_o, 'b_o')

                    self.W_u = theano.shared(self.W_u, 'W_u')
                    self.U_u = theano.shared(self.U_u, 'U_u')
                    self.b_u = theano.shared(self.b_u, 'b_u')


                    self.params.extend([self.W_i,  self.U_i, self.b_i,self.W_f, self.U_f, self.b_f,self.W_o, self.U_o, self.b_o,self.W_u,self.U_u,self.b_u ]) # independent weights
                    self.names.extend(['W_i', 'U_i', 'b_i', 'W_f', 'U_f', 'b_f', 'W_o', 'U_o', 'b_o', 'W_u', 'U_u', 'b_u'])

                sdp_sent_aug_info_leaf_internal_x = T.ivector(name='sdp_sent_aug_info_leaf_internal_x')
                sdp_sent_aug_info_leaf_internal_x_cwords = T.imatrix(name='sdp_sent_aug_info_leaf_internal_x_cwords')  # word indices
                sdp_sent_aug_info_computation_tree_matrix = T.imatrix(name='sdp_sent_aug_info_computation_tree_matrix')  # shape [None, self.degree]
                sdp_sent_aug_info_output_tree_state_idx = T.ivector(name='sdp_sent_aug_info_output_tree_state_idx')
                self.num_words = sdp_sent_aug_info_leaf_internal_x.shape[0]  # total number of nodes (leaves + internal) in tree
                emb_x = self.emb[sdp_sent_aug_info_leaf_internal_x_cwords].reshape((
                    sdp_sent_aug_info_leaf_internal_x.shape[0], self.dim_emb*self.context_win_size))
                # emb_x = emb_x * T.neq(sdp_sent_aug_info_leaf_internal_x, -1).dimshuffle(0, 'x')  # zero-out non-existent embeddings
                emb_x = emb_x * T.neq(sdp_sent_aug_info_leaf_internal_x, -1).dimshuffle(0, 'x')
                if self.postag == True:
                    emb_x = theano.tensor.concatenate([emb_x, T.zeros((
                        self.num_words, self.dim_postag_emb),
                        dtype=theano.config.floatX)], axis=1)
                if self.entity_class == True:
                    emb_x = theano.tensor.concatenate([emb_x, T.zeros((
                        self.num_words, self.dim_entity_class_emb),
                        dtype=theano.config.floatX)], axis=1)
                self.tree_states = self.compute_tree(emb_x, sdp_sent_aug_info_computation_tree_matrix)
                self.final_state = self.tree_states[-1]
                #print(emb)
                # extract the values from output tree states
                aug_tree_emb_val = self.tree_states[sdp_sent_aug_info_output_tree_state_idx]
                x_f = theano.tensor.concatenate([x_f, aug_tree_emb_val], axis=1)
                x_b = theano.tensor.concatenate([x_b, aug_tree_emb_val[::-1]], axis=1)


        # network dynamics
        # use the scan operation, which allows to define loops
        # unchanging variables are passed into 'non-sequences', initialization occurs in 'outputs_info'
        # if we set outputs_info to None, this indicates to scan that it doesnt need to pass the prior result
        # to recurrent_fn.
        # The general order of function parameters to recurrent_fn is:
        # sequences (if any), prior result(s) (if needed), non-sequences (if any)
        # http://deeplearning.net/software/theano/library/scan.html

        if reload_model == True:
            self.load(reload_path)

        if w_hh_type == 'shared' or self.w_hh_type == 'transpose':
            h_f, _ = theano.scan(self.recurrent_fn, sequences=x_f, outputs_info=[self.h0_tm1_f],
                                 non_sequences=[self.c0_tm1, self.W_hi, self.W_xi_f, self.W_ci_f, self.b_hi_f, self.W_xf_f,
                                                self.W_hf, self.W_cf_f, self.b_hf_f, self. W_xc_f, self.W_hc, self.b_hc_f,
                                                self.W_xo_f, self.W_ho, self.W_co_f, self.b_ho_f])

            if self.w_hh_type == 'transpose':
                h_b, _ = theano.scan(self.recurrent_fn, sequences = x_b, outputs_info = [self.h0_tm1_b],
                                     non_sequences = [self.c0_tm1, theano.tensor.transpose(self.W_hi), self.W_xi_b, self.W_ci_b, self.b_hi_b, self.W_xf_b,
                                                      theano.tensor.transpose(self.W_hf), self.W_cf_b, self.b_hf_b, self. W_xc_b, theano.tensor.transpose(self.W_hc)
                                                  ,self.b_hc_b, self.W_xo_b, theano.tensor.transpose(self.W_ho), self.W_co_b, self.b_ho_b ])
            else:
                h_b, _ = theano.scan(self.recurrent_fn, sequences = x_b, outputs_info = [self.h0_tm1_b],
                                     non_sequences = [self.c0_tm1, self.W_hi, self.W_xi_b, self.W_ci_b, self.b_hi_b, self.W_xf_b,
                                                self.W_hf, self.W_cf_b, self.b_hf_b, self. W_xc_b, self.W_hc, self.b_hc_b,
                                                self.W_xo_b, self.W_ho, self.W_co_b, self.b_ho_b])


            def concat(h_f, h_b, h0_bi_tm1, W_hi_bi, W_hf_bi, W_hc_bi, W_ho_bi):
                h_t_bi = self.activ(h_f + h_b + T.dot(h0_bi_tm1, W_hi_bi)+ T.dot(h0_bi_tm1, W_hf_bi) + T.dot(h0_bi_tm1, W_hc_bi) + T.dot(h0_bi_tm1, W_ho_bi))
                return h_t_bi

            h_bi, _ = theano.scan(fn=concat, sequences=[h_f, h_b], outputs_info=[self.h0_bi_tm1],
                                  non_sequences=[self.W_hi_bi, self.W_hf_bi, self.W_hc_bi, self.W_ho_bi])


        else:
            h_f, _ = theano.scan(self.recurrent_fn, sequences=x_f, outputs_info=[self.h0_tm1_f],
                                 non_sequences=[self.c0_tm1, self.W_hi_f, self.W_xi_f, self.W_ci_f, self.b_hi_f, self.W_xf_f,
                                                self.W_hf_f, self.W_cf_f, self.b_hf_f, self. W_xc_f, self.W_hc_f, self.b_hc_f,
                                                self.W_xo_f, self.W_ho_f, self.W_co_f, self.b_ho_f])

            h_b, _ = theano.scan(self.recurrent_fn, sequences=x_b, outputs_info=[self.h0_tm1_b],
                                 non_sequences=[self.c0_tm1, self.W_hi_b, self.W_xi_b, self.W_ci_b, self.b_hi_b, self.W_xf_b,
                                                self.W_hf_b, self.W_cf_b, self.b_hf_b, self. W_xc_b, self.W_hc_b, self.b_hc_b,
                                                self.W_xo_b, self.W_ho_b, self.W_co_b, self.b_ho_b])

            def concat(h_f, h_b, h0_bi_tm1, W_hi_bi, W_hf_bi, W_hc_bi, W_ho_bi ):
                h_t_bi = self.activ(h_f + h_b + T.dot(h0_bi_tm1, W_hi_bi) + T.dot(h0_bi_tm1, W_hf_bi) + T.dot(h0_bi_tm1, W_hc_bi) + T.dot(h0_bi_tm1, W_ho_bi))
                return h_t_bi

            h_bi, _ = theano.scan(fn=concat, sequences=[h_f, h_b], outputs_info=[self.h0_bi_tm1],
                                  non_sequences=[self.W_hi_bi, self.W_hf_bi, self.W_hc_bi, self.W_ho_bi])

        if self.use_dropout == True:
            h_bi = self.dropout_layer(h_bi, use_noise, trng)

        # network output
        self.y = T.dot(h_bi[-1], self.W_hy_bi) + self.b_hy_bi
        self.p_y_given_x = T.nnet.softmax(self.y)

        # compute prediction as class whose probability is maximal
        y_pred = T.argmax(self.p_y_given_x, axis=-1)
        y_pred_prob = self.p_y_given_x
        # computinhg cost
        # cost = -T.mean(T.log(self.p_y_given_x)[T.arange(t.shape[0]), t ])


        cost = -T.mean(T.log(self.p_y_given_x)[:, t])

        if w_hh_type == 'shared' or self.w_hh_type == 'transpose':
            cost += self.L2_reg * (T.sum(self.W_xi_f ** 2)+ T.sum(self.W_xf_f ** 2)+ T.sum(self.W_xc_f ** 2)+ T.sum(self.W_xo_f ** 2)
                                   + T.sum(self.W_xi_b ** 2) + T.sum(self.W_xf_b ** 2) + T.sum(self.W_xc_b ** 2) +T.sum(self.W_xo_b ** 2)
                                   + T.sum(self.W_ci_f ** 2) + T.sum(self.W_cf_f ** 2) + T.sum(self.W_co_f ** 2)
                                   + T.sum(self.W_ci_b ** 2) + T.sum(self.W_cf_b ** 2) + T.sum(self.W_co_b ** 2)
                                   + T.sum(self.W_hi ** 2)+ T.sum(self.W_hf ** 2)+T.sum(self.W_hc ** 2)+T.sum(self.W_ho ** 2)
                                   + T.sum(self.W_hy_bi ** 2)
                                   + T.sum(self.h0_tm1_b ** 2)
                                   + T.sum(self.h0_tm1_f ** 2)
                                   + T.sum(self.h0_bi_tm1 ** 2))



        else:
            cost += self.L2_reg * (T.sum(self.W_xi_f ** 2)+ T.sum(self.W_xf_f ** 2)+ T.sum(self.W_xc_f ** 2)+ T.sum(self.W_xo_f ** 2)
                                   + T.sum(self.W_xi_b ** 2) + T.sum(self.W_xf_b ** 2) + T.sum(self.W_xc_b ** 2) +T.sum(self.W_xo_b ** 2)
                                   + T.sum(self.W_ci_f ** 2) + T.sum(self.W_cf_f ** 2) + T.sum(self.W_co_f ** 2)
                                   + T.sum(self.W_ci_b ** 2) + T.sum(self.W_cf_b ** 2) + T.sum(self.W_co_b ** 2)
                                   + T.sum(self.W_hi_f ** 2)+ T.sum(self.W_hf_f ** 2)+T.sum(self.W_hc_f ** 2)+T.sum(self.W_ho_f ** 2)
                                   + T.sum(self.W_hy_bi ** 2)
                                   + T.sum(self.W_hi_b ** 2)+ T.sum(self.W_hf_b ** 2)+T.sum(self.W_hc_b ** 2)+T.sum(self.W_ho_b ** 2)
                                   + T.sum(self.h0_tm1_f ** 2)
                                   + T.sum(self.h0_tm1_b ** 2)
                                   + T.sum(self.h0_bi_tm1 ** 2))

        # the actual gradient descent, we need to evaluate the derivative of the cost function w.r.t. the parameters
        # and update their values. In the case of a recurrent network, this procedure is known as backpropagation
        # through time. Luckily with theano we dont really need to worry about this. Because all the code has been
        # defined as symbolic operations we can just ask for the derivatives of the parameters and it will propagate
        # them through the scan operation automatically
        #print[type(p) for p in self.params]

        updates, _, _, _, _ = create_optimization_updates(cost=cost, params=self.params, names=self.names,
                                                          method=self.optimiser, gradients=None,
                                                          lr=lr, rho=rho, max_norm=max_norm,
                                                          mom=mom)


        if embedding == 'theano_word_embeddings' or embedding == 'word2vec_update':
            if position_feat == True and entity_presence_feat == True:
                if self.pos_feat_embedding == True and self.ent_pres_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat, ent_pres_feat, pos_feat_idx,
                                                         ent_pres_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat, ent_pres_feat, pos_feat_idx,
                                                     ent_pres_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis

                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, pos_feat, ent_pres_feat,
                                                       pos_feat_idx, ent_pres_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.pos_emb_type == 'COUPLED':
                        self.normalize_pos_emb = theano.function( inputs = [],
                                                                  updates = {self.pos_emb_e1_e2:
                                                                                 self.pos_emb_e1_e2/T.sqrt((self.pos_emb_e1_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                    elif self.pos_emb_type == 'DECOUPLED':
                        self.normalize_pos_emb_e1 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e1:
                                                                                    self.pos_emb_e1/T.sqrt((self.pos_emb_e1**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_pos_emb_e2 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e2:
                                                                                    self.pos_emb_e2/T.sqrt((self.pos_emb_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                    if self.ent_pres_emb_type == 'COUPLED':
                        self.normalize_ent_pres_emb = theano.function( inputs = [],
                                                                       updates = {self.ent_pres_emb_e1_e2:
                                                                                      self.ent_pres_emb_e1_e2/T.sqrt((self.ent_pres_emb_e1_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                    elif self.ent_pres_emb_type == 'DECOUPLED':
                        self.normalize_ent_pres_emb_e1 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e1:
                                                                                         self.ent_pres_emb_e1/T.sqrt((self.ent_pres_emb_e1**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_ent_pres_emb_e2 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e2:
                                                                                         self.ent_pres_emb_e2/T.sqrt((self.ent_pres_emb_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})

                elif self.pos_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat, ent_pres_feat, pos_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat, ent_pres_feat, pos_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, pos_feat, ent_pres_feat,
                                                       pos_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.pos_emb_type == 'COUPLED':
                        self.normalize_pos_emb = theano.function( inputs = [],
                                                                  updates = {self.pos_emb_e1_e2:
                                                                                 self.pos_emb_e1_e2/T.sqrt((self.pos_emb_e1_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                    elif self.pos_emb_type == 'DECOUPLED':
                        self.normalize_pos_emb_e1 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e1:
                                                                                    self.pos_emb_e1/T.sqrt((self.pos_emb_e1**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_pos_emb_e2 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e2:
                                                                                    self.pos_emb_e2/T.sqrt((self.pos_emb_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                elif self.ent_pres_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat, ent_pres_feat,
                                                         ent_pres_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat, ent_pres_feat,
                                                     ent_pres_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, pos_feat, ent_pres_feat,
                                                       ent_pres_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.ent_pres_emb_type == 'COUPLED':
                        self.normalize_ent_pres_emb = theano.function( inputs = [],
                                                                       updates = {self.ent_pres_emb_e1_e2:
                                                                                      self.ent_pres_emb_e1_e2/T.sqrt((self.ent_pres_emb_e1_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                    elif self.ent_pres_emb_type == 'DECOUPLED':
                        self.normalize_ent_pres_emb_e1 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e1:
                                                                                         self.ent_pres_emb_e1/T.sqrt((self.ent_pres_emb_e1**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_ent_pres_emb_e2 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e2:
                                                                                         self.ent_pres_emb_e2/T.sqrt((self.ent_pres_emb_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                else:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat, ent_pres_feat], y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat, ent_pres_feat], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, pos_feat, ent_pres_feat], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

            elif entity_presence_feat == True:
                if self.pos_feat_embedding == True and self.ent_pres_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, ent_pres_feat, pos_feat_idx,
                                                         ent_pres_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, ent_pres_feat, pos_feat_idx,
                                                     ent_pres_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, ent_pres_feat,
                                                       pos_feat_idx, ent_pres_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.pos_emb_type == 'COUPLED':
                        self.normalize_pos_emb = theano.function( inputs = [],
                                                                  updates = {self.pos_emb_e1_e2:
                                                                                 self.pos_emb_e1_e2/T.sqrt((self.pos_emb_e1_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                    elif self.pos_emb_type == 'DECOUPLED':
                        self.normalize_pos_emb_e1 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e1:
                                                                                    self.pos_emb_e1/T.sqrt((self.pos_emb_e1**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_pos_emb_e2 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e2:
                                                                                    self.pos_emb_e2/T.sqrt((self.pos_emb_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                    if self.ent_pres_emb_type == 'COUPLED':
                        self.normalize_ent_pres_emb = theano.function( inputs = [],
                                                                       updates = {self.ent_pres_emb_e1_e2:
                                                                                      self.ent_pres_emb_e1_e2/T.sqrt((self.ent_pres_emb_e1_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                    elif self.ent_pres_emb_type == 'DECOUPLED':
                        self.normalize_ent_pres_emb_e1 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e1:
                                                                                         self.ent_pres_emb_e1/T.sqrt((self.ent_pres_emb_e1**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_ent_pres_emb_e2 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e2:
                                                                                         self.ent_pres_emb_e2/T.sqrt((self.ent_pres_emb_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})

                elif self.pos_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, ent_pres_feat, pos_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, ent_pres_feat, pos_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, ent_pres_feat,
                                                       pos_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.pos_emb_type == 'COUPLED':
                        self.normalize_pos_emb = theano.function( inputs = [],
                                                                  updates = {self.pos_emb_e1_e2:
                                                                                 self.pos_emb_e1_e2/T.sqrt((self.pos_emb_e1_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                    elif self.pos_emb_type == 'DECOUPLED':
                        self.normalize_pos_emb_e1 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e1:
                                                                                    self.pos_emb_e1/T.sqrt((self.pos_emb_e1**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_pos_emb_e2 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e2:
                                                                                    self.pos_emb_e2/T.sqrt((self.pos_emb_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                elif self.ent_pres_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, ent_pres_feat,
                                                         ent_pres_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, ent_pres_feat,
                                                     ent_pres_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, ent_pres_feat,
                                                       ent_pres_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.ent_pres_emb_type == 'COUPLED':
                        self.normalize_ent_pres_emb = theano.function( inputs = [],
                                                                       updates = {self.ent_pres_emb_e1_e2:
                                                                                      self.ent_pres_emb_e1_e2/T.sqrt((self.ent_pres_emb_e1_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                    elif self.ent_pres_emb_type == 'DECOUPLED':
                        self.normalize_ent_pres_emb_e1 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e1:
                                                                                         self.ent_pres_emb_e1/T.sqrt((self.ent_pres_emb_e1**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_ent_pres_emb_e2 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e2:
                                                                                         self.ent_pres_emb_e2/T.sqrt((self.ent_pres_emb_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                else:
                    self.predict_prob = theano.function([idx_f, idx_b,ent_pres_feat], y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, ent_pres_feat], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, ent_pres_feat], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

            elif position_feat == True:
                if self.pos_feat_embedding == True and self.ent_pres_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat, pos_feat_idx,
                                                         ent_pres_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat, pos_feat_idx,
                                                     ent_pres_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, pos_feat,
                                                       pos_feat_idx, ent_pres_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.pos_emb_type == 'COUPLED':
                        self.normalize_pos_emb = theano.function( inputs = [],
                                                                  updates = {self.pos_emb_e1_e2:
                                                                                 self.pos_emb_e1_e2/T.sqrt((self.pos_emb_e1_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                    elif self.pos_emb_type == 'DECOUPLED':
                        self.normalize_pos_emb_e1 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e1:
                                                                                    self.pos_emb_e1/T.sqrt((self.pos_emb_e1**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_pos_emb_e2 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e2:
                                                                                    self.pos_emb_e2/T.sqrt((self.pos_emb_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                    if self.ent_pres_emb_type == 'COUPLED':
                        self.normalize_ent_pres_emb = theano.function( inputs = [],
                                                                       updates = {self.ent_pres_emb_e1_e2:
                                                                                      self.ent_pres_emb_e1_e2/T.sqrt((self.ent_pres_emb_e1_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                    elif self.ent_pres_emb_type == 'DECOUPLED':
                        self.normalize_ent_pres_emb_e1 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e1:
                                                                                         self.ent_pres_emb_e1/T.sqrt((self.ent_pres_emb_e1**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_ent_pres_emb_e2 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e2:
                                                                                         self.ent_pres_emb_e2/T.sqrt((self.ent_pres_emb_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})

                elif self.pos_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat, pos_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat, pos_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, pos_feat,
                                                       pos_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.pos_emb_type == 'COUPLED':
                        self.normalize_pos_emb = theano.function( inputs = [],
                                                                  updates = {self.pos_emb_e1_e2:
                                                                                 self.pos_emb_e1_e2/T.sqrt((self.pos_emb_e1_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                    elif self.pos_emb_type == 'DECOUPLED':
                        self.normalize_pos_emb_e1 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e1:
                                                                                    self.pos_emb_e1/T.sqrt((self.pos_emb_e1**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_pos_emb_e2 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e2:
                                                                                    self.pos_emb_e2/T.sqrt((self.pos_emb_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                elif self.ent_pres_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat,
                                                         ent_pres_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat,
                                                     ent_pres_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, pos_feat,
                                                       ent_pres_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.ent_pres_emb_type == 'COUPLED':
                        self.normalize_ent_pres_emb = theano.function( inputs = [],
                                                                       updates = {self.ent_pres_emb_e1_e2:
                                                                                      self.ent_pres_emb_e1_e2/T.sqrt((self.ent_pres_emb_e1_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                    elif self.ent_pres_emb_type == 'DECOUPLED':
                        self.normalize_ent_pres_emb_e1 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e1:
                                                                                         self.ent_pres_emb_e1/T.sqrt((self.ent_pres_emb_e1**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_ent_pres_emb_e2 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e2:
                                                                                         self.ent_pres_emb_e2/T.sqrt((self.ent_pres_emb_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                else:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat], y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom, pos_feat], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

            else:
                if self.pos_feat_embedding == True and self.ent_pres_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat_idx,
                                                         ent_pres_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat_idx,
                                                     ent_pres_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom,
                                                       pos_feat_idx, ent_pres_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.pos_emb_type == 'COUPLED':
                        self.normalize_pos_emb = theano.function( inputs = [],
                                                                  updates = {self.pos_emb_e1_e2:
                                                                                 self.pos_emb_e1_e2/T.sqrt((self.pos_emb_e1_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                    elif self.pos_emb_type == 'DECOUPLED':
                        self.normalize_pos_emb_e1 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e1:
                                                                                    self.pos_emb_e1/T.sqrt((self.pos_emb_e1**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_pos_emb_e2 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e2:
                                                                                    self.pos_emb_e2/T.sqrt((self.pos_emb_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                    if self.ent_pres_emb_type == 'COUPLED':
                        self.normalize_ent_pres_emb = theano.function( inputs = [],
                                                                       updates = {self.ent_pres_emb_e1_e2:
                                                                                      self.ent_pres_emb_e1_e2/T.sqrt((self.ent_pres_emb_e1_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                    elif self.ent_pres_emb_type == 'DECOUPLED':
                        self.normalize_ent_pres_emb_e1 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e1:
                                                                                         self.ent_pres_emb_e1/T.sqrt((self.ent_pres_emb_e1**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_ent_pres_emb_e2 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e2:
                                                                                         self.ent_pres_emb_e2/T.sqrt((self.ent_pres_emb_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})

                elif self.pos_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b, pos_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b, pos_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom,
                                                       pos_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.pos_emb_type == 'COUPLED':
                        self.normalize_pos_emb = theano.function( inputs = [],
                                                                  updates = {self.pos_emb_e1_e2:
                                                                                 self.pos_emb_e1_e2/T.sqrt((self.pos_emb_e1_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                    elif self.pos_emb_type == 'DECOUPLED':
                        self.normalize_pos_emb_e1 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e1:
                                                                                    self.pos_emb_e1/T.sqrt((self.pos_emb_e1**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_pos_emb_e2 = theano.function( inputs = [],
                                                                     updates = {self.pos_emb_e2:
                                                                                    self.pos_emb_e2/T.sqrt((self.pos_emb_e2**2). \
                                                                                                           sum(axis=1)).dimshuffle(0,'x')})

                elif self.ent_pres_feat_embedding == True:
                    self.predict_prob = theano.function([idx_f, idx_b,
                                                         ent_pres_feat_idx],
                                                        y_pred_prob,
                                                        on_unused_input='ignore',
                                                        allow_input_downcast=True)

                    self.classify = theano.function([idx_f, idx_b,
                                                     ent_pres_feat_idx], y_pred,
                                                    on_unused_input='warn',
                                                    allow_input_downcast=True)
                    #the update itself happens directly on the parameter variables as part of theano update mechanis
                    self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom,
                                                       ent_pres_feat_idx], cost,
                                                      on_unused_input='ignore',
                                                      updates=updates,
                                                      allow_input_downcast=True)

                    if self.ent_pres_emb_type == 'COUPLED':
                        self.normalize_ent_pres_emb = theano.function( inputs = [],
                                                                       updates = {self.ent_pres_emb_e1_e2:
                                                                                      self.ent_pres_emb_e1_e2/T.sqrt((self.ent_pres_emb_e1_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                    elif self.ent_pres_emb_type == 'DECOUPLED':
                        self.normalize_ent_pres_emb_e1 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e1:
                                                                                         self.ent_pres_emb_e1/T.sqrt((self.ent_pres_emb_e1**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})
                        self.normalize_ent_pres_emb_e2 = theano.function( inputs = [],
                                                                          updates = {self.ent_pres_emb_e2:
                                                                                         self.ent_pres_emb_e2/T.sqrt((self.ent_pres_emb_e2**2). \
                                                                                                                     sum(axis=1)).dimshuffle(0,'x')})

                elif self.add_subtree_emb == True:
                    if self.postag == True and self.entity_class == True:
                        self.predict_prob = theano.function([idx_f, idx_b,
                                                             sdp_sent_aug_info_leaf_internal_x,
                                                             sdp_sent_aug_info_leaf_internal_x_cwords,
                                                             sdp_sent_aug_info_computation_tree_matrix,
                                                             sdp_sent_aug_info_output_tree_state_idx,
                                                             postag_idx, entity_class_idx], y_pred_prob,
                                                            on_unused_input='ignore',
                                                            allow_input_downcast=True)

                        self.classify = theano.function([idx_f, idx_b,
                                                         sdp_sent_aug_info_leaf_internal_x,
                                                         sdp_sent_aug_info_leaf_internal_x_cwords,
                                                         sdp_sent_aug_info_computation_tree_matrix,
                                                         sdp_sent_aug_info_output_tree_state_idx,
                                                         postag_idx, entity_class_idx], y_pred,
                                                        on_unused_input='warn',
                                                        allow_input_downcast=True)
                        #the update itself happens directly on the parameter variables as part of theano update mechanis
                        self.train_step = theano.function([idx_f, idx_b,
                                                           sdp_sent_aug_info_leaf_internal_x,
                                                           sdp_sent_aug_info_leaf_internal_x_cwords,
                                                           sdp_sent_aug_info_computation_tree_matrix,
                                                           sdp_sent_aug_info_output_tree_state_idx,
                                                           postag_idx, entity_class_idx, t, lr, rho, mom], cost,
                                                          on_unused_input='ignore',
                                                          updates=updates,
                                                          allow_input_downcast=True)

                    elif self.postag == True:
                        self.predict_prob = theano.function([idx_f, idx_b,
                                                             sdp_sent_aug_info_leaf_internal_x,
                                                             sdp_sent_aug_info_leaf_internal_x_cwords,
                                                             sdp_sent_aug_info_computation_tree_matrix,
                                                             sdp_sent_aug_info_output_tree_state_idx,
                                                             postag_idx], y_pred_prob,
                                                            on_unused_input='ignore',
                                                            allow_input_downcast=True)

                        self.classify = theano.function([idx_f, idx_b,
                                                         sdp_sent_aug_info_leaf_internal_x,
                                                         sdp_sent_aug_info_leaf_internal_x_cwords,
                                                         sdp_sent_aug_info_computation_tree_matrix,
                                                         sdp_sent_aug_info_output_tree_state_idx,
                                                         postag_idx], y_pred,
                                                        on_unused_input='warn',
                                                        allow_input_downcast=True)
                        #the update itself happens directly on the parameter variables as part of theano update mechanis
                        self.train_step = theano.function([idx_f, idx_b,
                                                           sdp_sent_aug_info_leaf_internal_x,
                                                           sdp_sent_aug_info_leaf_internal_x_cwords,
                                                           sdp_sent_aug_info_computation_tree_matrix,
                                                           sdp_sent_aug_info_output_tree_state_idx,
                                                           postag_idx, t, lr, rho, mom], cost,
                                                          on_unused_input='ignore',
                                                          updates=updates,
                                                          allow_input_downcast=True)

                    elif self.entity_class == True:
                        self.predict_prob = theano.function([idx_f, idx_b,
                                                             sdp_sent_aug_info_leaf_internal_x,
                                                             sdp_sent_aug_info_leaf_internal_x_cwords,
                                                             sdp_sent_aug_info_computation_tree_matrix,
                                                             sdp_sent_aug_info_output_tree_state_idx,
                                                             entity_class_idx], y_pred_prob,
                                                            on_unused_input='ignore',
                                                            allow_input_downcast=True)

                        self.classify = theano.function([idx_f, idx_b,
                                                         sdp_sent_aug_info_leaf_internal_x,
                                                         sdp_sent_aug_info_leaf_internal_x_cwords,
                                                         sdp_sent_aug_info_computation_tree_matrix,
                                                         sdp_sent_aug_info_output_tree_state_idx,
                                                         entity_class_idx], y_pred,
                                                        on_unused_input='warn',
                                                        allow_input_downcast=True)
                        #the update itself happens directly on the parameter variables as part of theano update mechanis
                        self.train_step = theano.function([idx_f, idx_b,
                                                           sdp_sent_aug_info_leaf_internal_x,
                                                           sdp_sent_aug_info_leaf_internal_x_cwords,
                                                           sdp_sent_aug_info_computation_tree_matrix,
                                                           sdp_sent_aug_info_output_tree_state_idx,
                                                           entity_class_idx, t, lr, rho, mom], cost,
                                                          on_unused_input='ignore',
                                                          updates=updates,
                                                          allow_input_downcast=True)
                    else:
                        self.predict_prob = theano.function([idx_f, idx_b,
                                                             sdp_sent_aug_info_leaf_internal_x,
                                                             sdp_sent_aug_info_leaf_internal_x_cwords,
                                                             sdp_sent_aug_info_computation_tree_matrix,
                                                             sdp_sent_aug_info_output_tree_state_idx],
                                                            y_pred_prob,
                                                            on_unused_input='ignore',
                                                            allow_input_downcast=True)

                        self.classify = theano.function([idx_f, idx_b,
                                                         sdp_sent_aug_info_leaf_internal_x,
                                                         sdp_sent_aug_info_leaf_internal_x_cwords,
                                                         sdp_sent_aug_info_computation_tree_matrix,
                                                         sdp_sent_aug_info_output_tree_state_idx],
                                                        y_pred,
                                                        on_unused_input='warn',
                                                        allow_input_downcast=True)
                        #the update itself happens directly on the parameter variables as part of theano update mechanis
                        self.train_step = theano.function([idx_f, idx_b,
                                                           sdp_sent_aug_info_leaf_internal_x,
                                                           sdp_sent_aug_info_leaf_internal_x_cwords,
                                                           sdp_sent_aug_info_computation_tree_matrix,
                                                           sdp_sent_aug_info_output_tree_state_idx,
                                                           t, lr, rho, mom], cost,
                                                          on_unused_input='ignore',
                                                          updates=updates,
                                                          allow_input_downcast=True)
                else:
                    if self.postag == True and self.entity_class == True:
                        self.predict_prob = theano.function([idx_f, idx_b, postag_idx, entity_class_idx], y_pred_prob,
                                                            on_unused_input='ignore',
                                                            allow_input_downcast=True)

                        self.classify = theano.function([idx_f, idx_b, postag_idx, entity_class_idx], y_pred,
                                                        on_unused_input='warn',
                                                        allow_input_downcast=True)
                        #the update itself happens directly on the parameter variables as part of theano update mechanis
                        self.train_step = theano.function([idx_f, idx_b, postag_idx, entity_class_idx, t, lr, rho, mom], cost,
                                                          on_unused_input='ignore',
                                                          updates=updates,
                                                          allow_input_downcast=True)

                    elif self.postag == True:
                        self.predict_prob = theano.function([idx_f, idx_b, postag_idx], y_pred_prob,
                                                            on_unused_input='ignore',
                                                            allow_input_downcast=True)

                        self.classify = theano.function([idx_f, idx_b, postag_idx], y_pred,
                                                        on_unused_input='warn',
                                                        allow_input_downcast=True)
                        #the update itself happens directly on the parameter variables as part of theano update mechanis
                        self.train_step = theano.function([idx_f, idx_b, postag_idx, t, lr, rho, mom], cost,
                                                          on_unused_input='ignore',
                                                          updates=updates,
                                                          allow_input_downcast=True)

                    elif self.entity_class == True:
                        self.predict_prob = theano.function([idx_f, idx_b, entity_class_idx], y_pred_prob,
                                                            on_unused_input='ignore',
                                                            allow_input_downcast=True)

                        self.classify = theano.function([idx_f, idx_b, entity_class_idx], y_pred,
                                                        on_unused_input='warn',
                                                        allow_input_downcast=True)
                        #the update itself happens directly on the parameter variables as part of theano update mechanis
                        self.train_step = theano.function([idx_f, idx_b, entity_class_idx, t, lr, rho, mom], cost,
                                                          on_unused_input='ignore',
                                                          updates=updates,
                                                          allow_input_downcast=True)
                    else:
                        self.predict_prob = theano.function([idx_f, idx_b], y_pred_prob,
                                                            on_unused_input='ignore',
                                                            allow_input_downcast=True)

                        self.classify = theano.function([idx_f, idx_b], y_pred,
                                                        on_unused_input='warn',
                                                        allow_input_downcast=True)
                        #the update itself happens directly on the parameter variables as part of theano update mechanis
                        self.train_step = theano.function([idx_f, idx_b, t, lr, rho, mom], cost,
                                                          on_unused_input='ignore',
                                                          updates=updates,
                                                          allow_input_downcast=True)
                        print("...............................................................")
            self.normalize = theano.function( inputs = [],
                                              updates = {self.emb: \
                                                             self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

        else:
            self.predict_prob = theano.function([x_f, x_b], y_pred_prob,
                                on_unused_input='ignore',
                                allow_input_downcast=True)
            self.classify = theano.function([x_f, x_b], y_pred,
                                on_unused_input='ignore',
                                allow_input_downcast=True)
            #the update itself happens directly on the parameter variables as part of theano update mechanis
            self.train_step = theano.function([x_f, x_b, t, lr, rho, mom], cost,
                                on_unused_input='ignore',
                                updates=updates,
                                allow_input_downcast=True)

        # recurrent step
        # The general order of function parameters to recurrent_fn is:
        # sequences (if any), prior result(s) (if needed), non-sequences (if any)
        # 'h_tm1': is the prior and would contain the output also for the intermediate computation which will be used in
        # the following computation as prior.

    def recurrent_fn(self, x_t, h_tm1, c_tm1, W_hi, W_xi, W_ci, b_hi, W_xf, W_hf, W_cf, b_hf, W_xc, W_hc, b_hc,
                     W_xo, W_ho, W_co, b_ho):

        i_gate = T.nnet.sigmoid(T.dot(x_t, W_xi) + T.dot(h_tm1, W_hi) + T.dot(c_tm1, W_ci) + b_hi)
        f_gate = T.nnet.sigmoid(T.dot(x_t, W_xf) + T.dot(h_tm1, W_hf.T) + T.dot(c_tm1, W_cf) + b_hf)

        c_gate = f_gate * c_tm1 + i_gate * T.tanh(theano.dot(x_t, W_xc) + theano.dot(h_tm1, W_hc) + b_hc)

        o_gate = T.nnet.sigmoid(T.dot(x_t, W_xo) + T.dot(h_tm1, W_ho) + T.dot(c_gate, W_co) + b_ho)

        h_t = o_gate * T.tanh(c_gate)

        return h_t

    def params(self):
        return self.params

    def nll_multiclass(self, y):
        # negative log likelihood based on multiclass cross entropy error
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of time steps (call it T) in the sequence
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y ])

    def save(self, folder):
        for param in self.params:
            np.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(np.load(os.path.join(folder,
                            param.name + '.npy')))

    # to sclae down the gradient
    # to deal with 'exploding gradient' probelm in RNN optimization
    def clip_norm(self, grad, norm):
        grad = T.switch(T.ge(norm, self.clip_norm_cutoff), grad * self.clip_norm_cutoff / norm, grad)
        return grad

    def get_gredient_clip(self, grads):
        # get the norm
        if self.clip_norm_cutoff > 0:
            norm = T.sqrt(sum([T.sum(g ** 2) for g in grads]))
            grads = [self.clip_norm(g, norm) for g in grads]
        return grads

    def dropout_layer(self, state_before, use_noise, trng):
        proj = T.switch(use_noise,
                        (state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                                      dtype=state_before.dtype)),
                        state_before * 0.5)
        return proj

    def ortho_weight(self, ndim):
        W = np.random.randn(ndim, ndim)
        u, s, v = np.linalg.svd(W)
        return u.astype(config.floatX)

    def relu(x):
        return x * (x > 1e-6)

    def clip_relu(x, clip_lim=20):
        return x * (T.lt(x, 1e-6) and T.gt(x, clip_lim))

    def dropout(random_state, X, keep_prob=0.5):
        if keep_prob > 0. and keep_prob < 1.:
            seed = random_state.randint(2 ** 30)
            srng = RandomStreams(seed)
            mask = srng.binomial(n=1, p=keep_prob, size=X.shape,
                                 dtype=theano.config.floatX)
            return X * mask
        return X

    def fast_dropout(random_state, X):
        seed = random_state.randint(2 ** 30)
        srng = RandomStreams(seed)
        mask = srng.normal(size=X.shape, avg=1., dtype=theano.config.floatX)
        return X * mask

    def _norm_constraint(self, param, update_step, max_col_norm):
        stepped_param = param + update_step
        if param.get_value(borrow=True).ndim == 2:
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, max_col_norm)
            scale = desired_norms / (1e-7 + col_norms)
            new_param = param * scale
            new_update_step = update_step * scale
        else:
            new_param = param
            new_update_step = update_step
        return new_param, new_update_step

    # def init_matrix(self, shape):
    #     return np.random.normal(scale=0.1, size=shape).astype(theano.config.floatX)
    #
    # def init_vector(self, shape):
    #     return np.zeros(shape, dtype=theano.config.floatX)


#tree_LSTM code

    def create_recursive_unit(self):

        def unit(parent_x, child_h, child_c, child_exists):
            h_tilde = T.sum(child_h, axis=0)

            i = T.nnet.sigmoid(T.dot(self.W_xi_f.T,  T.concatenate([parent_x, h_tilde])) + T.dot(self.W_hi_f, h_tilde) + self.b_hi_f)
            o = T.nnet.sigmoid(T.dot(self.W_xo_f.T,  T.concatenate([parent_x, h_tilde])) + T.dot(self.W_ho_f, h_tilde) + self.b_ho_f)
            u = T.tanh(T.dot(self.W_xc_f.T,  T.concatenate([parent_x, h_tilde])) + T.dot(self.W_hc_f, h_tilde) + self.b_hc_f)

            #f = (T.nnet.sigmoid( T.dot(self.W_xf_f, parent_x).dimshuffle('x', 0) + T.dot(child_h, self.W_hf_f.T) +self.b_hf_f.dimshuffle('x', 0)) * child_exists.dimshuffle(0, 'x'))

            f = (T.nnet.sigmoid(T.dot(self.W_xf_f.T,  T.concatenate([parent_x, h_tilde])).dimshuffle('x', 0) + T.dot(child_h, self.W_hf_f.T) +
                                self.b_hf_f.dimshuffle('x', 0)) * child_exists.dimshuffle(0, 'x'))

            c = i * u + T.sum(f * child_c, axis=0)
            h = o * T.tanh(c)
            return h, c


        return unit

    def create_leaf_unit(self):
        rng = np.random.RandomState(1234)
        dummy = 0 * theano.shared(
            value=np.asarray(rng.normal(size=(self.max_degree, self.hidden_dim), scale=.01, loc=0.0),
                             dtype=theano.config.floatX))
        def unit(leaf_x):
            return self.recursive_unit(
                leaf_x,
                dummy,
                dummy,
                dummy.sum(axis=1))
        return unit

    def compute_tree(self, emb_x, tree):
        self.irregular_tree = False
        self.recursive_unit = self.create_recursive_unit()
        self.leaf_unit = self.create_leaf_unit()
        num_nodes = tree.shape[0]  # num internal nodes
        num_leaves = self.num_words - num_nodes

        # compute leaf hidden states
        (leaf_h, leaf_c), _ = theano.map(
            fn=self.leaf_unit,
            sequences=[emb_x[:num_leaves]])
        init_node_h = leaf_h
        init_node_c = leaf_c

        # use recurrence to compute internal node hidden states
        def _recurrence(cur_emb, node_info, t, node_h, node_c, last_h):
            child_exists = node_info > -1
            offset = num_leaves * int(self.irregular_tree) - child_exists * t
            child_h = node_h[node_info + offset] * child_exists.dimshuffle(0, 'x')
            child_c = node_c[node_info + offset] * child_exists.dimshuffle(0, 'x')
            parent_h, parent_c = self.recursive_unit(cur_emb, child_h, child_c, child_exists)
            node_h = T.concatenate([node_h,
                                    parent_h.reshape([1, self.hidden_dim])])
            node_c = T.concatenate([node_c,
                                    parent_c.reshape([1, self.hidden_dim])])
            return node_h[1:], node_c[1:], parent_h

        dummy = theano.shared(np.zeros(self.hidden_dim, dtype=theano.config.floatX))
        (_, _, parent_h), _ = theano.scan(
            fn=_recurrence,
            outputs_info=[init_node_h, init_node_c, dummy],
            sequences=[emb_x[num_leaves:], tree, T.arange(num_nodes)],
            n_steps=num_nodes)

        return T.concatenate([leaf_h, parent_h], axis=0)