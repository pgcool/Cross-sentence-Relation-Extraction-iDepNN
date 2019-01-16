import itertools
import importlib
import numpy as np
import os
imp_eb = importlib.import_module('elman-bidirection')

os.environ["THEANO_FLAGS"] = 'floatX=float64'
config_state = {
    'RNN_model':['EB_RNN'],
    'RANKING': [False],
    'SHUFFLE': [False],
    'dev_eval_split':[False],
    'EF_RNN_ARC':['s'],
    'EB_RNN_ARC':['EB_LSTM'],
    'n_hidden': [100],
    'n_classes':[2],
    'epoch':[500],
    'lr':[0.0025],
    'L2_penalty': [0.0001],
    'optimiser':['sgd'], # 'sgd', 'adagrad', 'adadelta'
    'learning_rate_decay':[0.99],
    'rho':[0.99], # for adadelta

    # 'shared', 'transpose' if EF_RNN_ARC is 's->s_rev'
    # 'independent','shared', 'transpose' if RNN_model is 'EB_RNN'
    # 'independent' if  if RNN_model is 'EF_RNN'
    'w_hh_type':['independent'],

    # learning_rate_decay_strategies:
    # strategy 0: do not update learning rate, for other than sgd optimiser
    # strategy 1: Learning rate decay as defined by the update_learning_rate_after_n_epochs
    # strategy 2: Learning rate decay at each epoch. Let at epoch t, the learning rate be lr then update : lr = lr / t
    # strategy 3: Update learning rate by 'learning_rate_decay', if validation score does not
    # increase in 'update_learning_rate_after_n_epochs' epochs
    'learning_rate_decay_strategy':[3],
    'update_learning_rate_after_n_epochs':[10],

    # exploding gradient params, if no gradient clipping then set 'clipstyle' to None
    'clipstyle':['None'], # 'rescale in case of vanilla RNN'
    'cutoff':[5.0, 10.0, 20.0],

    'augment_entity_presence' : [False],
    'position_features': [False],
    'normalise_position_features': [False],
    'pos_feat_embedding' : [False],
    'dim_pos_emb' : [5],

    # 'DECOUPLED', 'COUPLED': 'COUPLED' not evaluated
    # 'DECOUPLED': two embeddings, 1 for relative_distance_from_e1, and other is relative_distance_from_e2
    'pos_emb_type' : ['DECOUPLED'],

    'pos_indicator_embedding' : [True],
    'ent_pres_feat_embedding' : [False],
    'dim_ent_pres_emb' : [5],
    'ent_pres_emb_type' : ['DECOUPLED'],
    'filename_train_data' : ['../../../data/processed_input/BB2013/split_1/k_le_1/BB2013_train_data_k_le_1_sdp_sent.txt'],
    'filename_dev_data' : ['../../../data/processed_input/BB2013/split_1/k_le_3/BB2013_dev_data_k_le_3_sdp_sent.txt'],
    'filename_test_data' : ['../../../data/processed_input/BB2013/split_1/k_le_3/BB2013_dev_data_k_le_3_sdp_sent.txt'],
    'data_set_sent_corpus' : ['../../../data/resources/bb_all_sentences_list.txt'],
    'train_model' : [True],  # True: performs training  False: performs only prediction
    'verbose_print' : [False], # for troubleshooting
    'train_on_text_between_target_entities' : [False],
    'dev_on_text_between_target_entities' : [False],
    'test_on_text_between_target_entities' : [False],
    'exclude_entity_terms' : [False],
    'text_neighbourhood_to_entity': [False],
    'left_neighbourhood_size':[5],
    'right_neighbourhood_size':[5],

    # 'word2vec_update', 'word2vec_init', '1-hot-encoding', 'theano_word_embeddings'
    'embedding_type':['word2vec_update'],
    'dim_emb':[200],
    'word2vec_emb_file':['../../../data/resources/bionlp_embedding.txt'],
    'normalise_embeddings': [True],
    'context_window_usage':[True],
    'context_win_size':[1],
    'batch_size':[1],
    'word_minibatch_usage':[False],

    # strategy 'rand'       : random initialization
    # strategy 'identity'   : Identity
    # strategy 'ortho'      : orthogonal weights
    'w_hh_initialise_strategy':['ortho'],

    # 'tanh', 'sigmoid', 'relu', 'cappedrelu'
    'activation':['tanh'],
    # save the model in directory ./models/
    'savemodel':[True],
    'merge_multi_words_for_target_entities':[False],
    'remove_other_class':[False],
    'NEs_representation':['1-hot-encoding'],

    'reload_model': [False],
    'reload_path':['./models/EB_RNN_RC_word2vec_update_2018-02-20_6/'],

    # postag feature
    'postag': [False], # [True, False]
    'update_pos_emb': [True], # [True, False]

    # Entity class/iob feature
    'entity_class': [False], # [True, False]
    'update_ner_emb': [True], # [True, False]

    # subtree feature
    'add_subtree_emb': [False], # [True, False]
    'treernn_weights': ['shared']  # [shared, independent]
}

config_list = [[] for x in range(len(config_state.keys()))]

keyIndexInConfigList_map = {}
key_count = 0
for key in config_state.keys():
    for item in config_state[key]:
        config_list[key_count].append(item)
    keyIndexInConfigList_map[key_count] = key
    key_count += 1

import csv
import os.path

filename = './param_search_rslt_RNN_bb_2013_Sdp.csv'

# check if the file exists
if  os.path.isfile(filename):
    record = open(filename, 'a')
else:
    record = open(filename, 'a')
    writer = csv.writer(record, delimiter=',')
    writer.writerow('--------------------------------------------------------------------')
    writer.writerow([
                     'RNN_model',
                     'EF_RNN_ARC',
                     'RANKING',
                     'SHUFFLE',
                     'EB_RNN_ARC',
                     'w_hh_type',
                     'n_hidden',
                     'epoch',
                     'lr',
                     'lr_decay',
                     'lr_decay_strat',
                     'L2_penalty',
                     'upd_lr_after_n_epochs',
                     'optimiser',
                     'rho',
                     'clipstyle',
                     'cutoff',
                     'entPres',
                     'pos_feat',
                     'norm_pos_feat',
                     'pos_feat_embedding',
                     'pos_indicator_embedding',
                     'ent_pres_feat_embedding',
                     'dim_ent_pres_emb',
                     'dim_pos_emb',
                     'pos_emb_type',
                     'ent_pres_emb_type',
                     'text',
                     'exclude_entity_terms',
                     'emb_type',
                     'dim_emb',
                     'normalise_embeddings',
                     'cont_win_usage',
                     'cont_win_size',
                     'bt_size',
                     'word_minibt_usage',
                     'w_hh_init',
                     'act',
                     'dev_eval_split',
                     'update_pos_emb',
                     'update_ner_emb',
                     'update_pos_ner_emb',
                     'postag',
                     'entity_class',
                     'add_subtree_emb',
                     'treernn_weights',
                     'dev_F1',
                     'test_F1',
                     'model'])
record.close()

best_f1 = -np.inf
dev_F1 = None

for items in itertools.product(*config_list):
    config_state_tmp = {}
    for item_count in range(len(items)):
        key = keyIndexInConfigList_map[item_count]
        config_state_tmp[key] = items[item_count]
    print('configurations:')
    print(config_state_tmp)

    # invoke the required model with configuration parameters
    if config_state_tmp['RNN_model'] == 'EB_RNN':
        if config_state_tmp['dev_eval_split'] == True:
            dev_F1, F1, model = imp_eb.run(config_state_tmp)
        else:
            dev_F1, test_F1, model = imp_eb.run(config_state_tmp)

    if config_state_tmp['train_on_text_between_target_entities'] == True or\
                    config_state_tmp['test_on_text_between_target_entities'] == True:
        text = 'btw'
        if config_state_tmp['text_neighbourhood_to_entity'] == True:
            text += '+L-R='+str(config_state_tmp['left_neighbourhood_size'])
    elif config_state_tmp['text_neighbourhood_to_entity'] == True:
        text = 'btw+L-R='+str(config_state_tmp['left_neighbourhood_size'])
    elif config_state_tmp['text_neighbourhood_to_entity'] == False:
        text = 'complteS'

    record = open(filename, 'a')
    writer = csv.writer(record, delimiter=',')
    writer.writerow([
                    config_state_tmp['RNN_model'],
                    str(config_state_tmp['EF_RNN_ARC']),
                    str(config_state_tmp['RANKING']),
                    str(config_state_tmp['SHUFFLE']),
                    str(config_state_tmp['EB_RNN_ARC']),
                    config_state_tmp['w_hh_type'],
                    str(config_state_tmp['n_hidden']),
                    str(config_state_tmp['epoch']),
                    str(config_state_tmp['lr']),
                    str(config_state_tmp['learning_rate_decay']),
                    str(config_state_tmp['learning_rate_decay_strategy']),
                    str(config_state_tmp['L2_penalty']),
                    str(config_state_tmp['update_learning_rate_after_n_epochs']),
                    config_state_tmp['optimiser'],
                    str(config_state_tmp['rho']),
                    config_state_tmp['clipstyle'],
                    str(config_state_tmp['cutoff']),
                    str(config_state_tmp['augment_entity_presence']),
                    str(config_state_tmp['position_features']),
                    str(config_state_tmp['normalise_position_features']),
                    str(config_state_tmp['pos_feat_embedding']),
                    str(config_state_tmp['pos_indicator_embedding']),
                    str(config_state_tmp['ent_pres_feat_embedding']),
                    str(config_state_tmp['dim_ent_pres_emb']),
                    str(config_state_tmp['dim_pos_emb']),
                    str(config_state_tmp['pos_emb_type']),
                    str(config_state_tmp['ent_pres_emb_type']),
                    text,
                    str(config_state_tmp['exclude_entity_terms']),
                    config_state_tmp['embedding_type'],
                    str(config_state_tmp['dim_emb']),
                    str(config_state_tmp['normalise_embeddings']),
                    str(config_state_tmp['context_window_usage']),
                    str(config_state_tmp['context_win_size']),
                    str(config_state_tmp['batch_size']),
                    str(config_state_tmp['word_minibatch_usage']),
                    str(config_state_tmp['w_hh_initialise_strategy']),
                    config_state_tmp['activation'],
                    config_state['dev_eval_split'],
                    config_state['update_pos_emb'],
                    config_state['update_ner_emb'],
                    config_state['postag'],
                    config_state['entity_class'],
                    config_state['add_subtree_emb'],
                    config_state['treernn_weights'],
                    str(dev_F1),
                    str(test_F1),
                    str(model)]
                    )
    record.close()

    if config_state['dev_eval_split'] == True:
        if best_f1 < dev_F1:
            best_f1 = dev_F1
    else:
        if best_f1 < test_F1:
            best_f1 = test_F1

if config_state['dev_eval_split'] == True:
    print('best dev f1:', best_f1)
else:
    print('best test f1:', best_f1)
