import numpy as np
import time
import os
import datetime
from rnn.elman_bidirection_RC import EB_RNN_4
from rnn.elman_bidirection_RC_LSTM import EB_LSTM
from utils.getMacroFScore import getMacroFScore
from utils.features_bb2011_13 import *
from utils import load_save_pkl
from scipy.stats import itemfreq
from sklearn.metrics import precision_recall_fscore_support
from recursive_net_utils import data_utils, tree_rnn

def run(config_state):
    #--------------------- configuration Starts -------------------------------------------
    n_hidden = config_state['n_hidden']
    n_classes = config_state['n_classes']
    n_out = n_classes  # restricted to single softmax per time step
    epoch = config_state['epoch']
    learning_rate_decay=config_state['learning_rate_decay']  #if applcation of learning_rate_decay then set learning_rate_decay = 1.0
    lr = config_state['lr']
    optimiser = config_state['optimiser']   # sgd, adadelta, adagrad available
    rho = config_state['rho'] # for adadelta
    w_hh_type = config_state['w_hh_type']
    RANKING = config_state['RANKING']

    # learning_rate_decay_strategies:
    # strategy 0: do not update learning rate, for other than sgd optimiser
    # strategy 1: Learning rate decay as defined by the update_learning_rate_after_n_epochs
    # strategy 2: Learning rate decay at each epoch. Let at epoch t, the learning rate be lr then update : lr = lr / t
    # strategy 3: Update learning rate by 'learning_rate_decay', if validation score does not increase in 'update_learning_rate_after_n_epochs' epochs
    # strategy 4: new bob
    learning_rate_decay_strategy = config_state['learning_rate_decay_strategy']
    update_learning_rate_after_n_epochs = config_state['update_learning_rate_after_n_epochs'] # for sgd
    state={}
    state['clipstyle'] = config_state['clipstyle']
    state['cutoff'] = config_state['cutoff']

    augment_entity_presence = config_state['augment_entity_presence']
    position_features = config_state['position_features']
    normalise_position_features = config_state['normalise_position_features']
    train_on_text_between_target_entities = config_state['train_on_text_between_target_entities']
    dev_on_text_between_target_entities = config_state['dev_on_text_between_target_entities']
    test_on_text_between_target_entities = config_state['test_on_text_between_target_entities']
    exclude_entity_terms = config_state['exclude_entity_terms']
    remove_other_class = config_state['remove_other_class']
    # if text_neighbourhood_to_entity is not True then load complete sentence
    text_neighbourhood_to_entity = config_state['text_neighbourhood_to_entity']
    left_neighbourhood_size = config_state['left_neighbourhood_size']
    right_neighbourhood_size = config_state['right_neighbourhood_size']

    # For theano embedding and context window
    embedding_type = config_state['embedding_type'] # 'theano_word_embeddings' or 'word2vec_update' or 'word2vec_init' or '1-hot-encoding'
    dim_emb = config_state['dim_emb']  # word embedding dimension
    context_window_usage = config_state['context_window_usage']
    context_win_size = config_state['context_win_size']
    batch_size = config_state['batch_size']
    word_minibatch_usage = config_state['word_minibatch_usage']

    filename_train_data = config_state['filename_train_data']
    filename_dev_data = config_state['filename_dev_data']
    filename_test_data = config_state['filename_test_data']
    train_model = config_state['train_model']
    verbose_print = config_state['verbose_print']

    ent = ''
    pos = ''
    if augment_entity_presence == True:
        ent = '_ent'
    if position_features == True:
        pos = '_pos'

    count = 1
    log_file_name = './STRNN_RESULTS/EB_RNN_RC_'+ent+pos+'_'+embedding_type+'_'+str(datetime.date.today())+'_'+str(count)+'.cfg'
    log_file_name_prob_dev = './STRNN_RESULTS/EB_RNN_RC_'+ent+pos+'_'+embedding_type+'_'+str(datetime.date.today())+'_'+str(count)+'.prob_dev'
    log_file_name_prob = './STRNN_RESULTS/EB_RNN_RC_'+ent+pos+'_'+embedding_type+'_'+str(datetime.date.today())+'_'+str(count)+'.prob'
    log_file_name_prob_2nd_rank = './STRNN_RESULTS/EB_RNN_RC_'+ent+pos+'_'+embedding_type+'_'+str(datetime.date.today())+'_'+str(count)+'.prob2ndRank'

    saveto = "./models/EB_RNN_RC_"+ent+pos+'_'+embedding_type+'_'+str(datetime.date.today())+'_'+str(count) # The best model will be saved there
    log_file_name_scores_dev = './STRNN_RESULTS/EB_RNN_RC_'+ent+pos+'_'+embedding_type+'_'+str(datetime.date.today())+'_'+str(count)+'.decision_dev'
    log_file_name_scores = './STRNN_RESULTS/EB_RNN_RC_'+ent+pos+'_'+embedding_type+'_'+str(datetime.date.today())+'_'+str(count)+'.decision'
    log_file_name_scores_2nd_rank = './STRNN_RESULTS/EB_RNN_RC_'+ent+pos+'_'+embedding_type+'_'+str(datetime.date.today())+'_'+str(count)+'.decision2ndRank'
    # check if file exists
    found = 0
    while(os.path.isdir(saveto)):
        print(int(str(saveto).split('_')[-1]))
        count = int(str(saveto).split('_')[-1])
        # increase count
        count += 1
        found = 1
        saveto = "./models/EB_RNN_RC_"+ent+pos+'_'+embedding_type+'_'+str(datetime.date.today())+'_'+str(count)

    if found ==1:
        log_file_name = './STRNN_RESULTS/EB_RNN_RC_'+ent+pos+'_'+embedding_type+'_'+str(datetime.date.today())+'_'+str(count)+'.cfg'
        log_file_name_prob_dev = './STRNN_RESULTS/EB_RNN_RC_'+ent+pos+'_'+embedding_type+'_'+str(datetime.date.today())+'_'+str(count)+'.prob_dev'
        log_file_name_prob = './STRNN_RESULTS/EB_RNN_RC_'+ent+pos+'_'+embedding_type+'_'+str(datetime.date.today())+'_'+str(count)+'.prob'
        log_file_name_prob_2nd_rank = './STRNN_RESULTS/EB_RNN_RC_'+ent+pos+'_'+embedding_type+'_'+str(datetime.date.today())+'_'+str(count)+'.prob2ndRank'
        saveto = "./models/EB_RNN_RC_"+ent+pos+'_'+embedding_type+'_'+str(datetime.date.today())+'_'+str(count) # The best model will be saved there
        log_file_name_scores_dev = './STRNN_RESULTS/EB_RNN_RC_'+ent+pos+'_'+embedding_type+'_'+str(datetime.date.today())+'_'+str(count)+'.decision_dev'
        log_file_name_scores = './STRNN_RESULTS/EB_RNN_RC_'+ent+pos+'_'+embedding_type+'_'+str(datetime.date.today())+'_'+str(count)+'.decision'
        log_file_name_scores_2nd_rank = './STRNN_RESULTS/EB_RNN_RC_'+ent+pos+'_'+embedding_type+'_'+str(datetime.date.today())+'_'+str(count)+'.decision2ndRank'

    savemodel = config_state['savemodel']                       # The best model will be saved
    # strategy 'rand'       : random initialization
    # strategy 'identity'   : Identity
    # strategy 'ortho'      : orthogonal weights
    w_hh_initialise_strategy = config_state['w_hh_initialise_strategy']
    activation = config_state['activation']

    noise_std = 0.
    use_dropout = False   # if False slightly faster, but worst test error
                          # This frequently need a bigger model.
    reload_model = config_state['reload_model']  # Path to a saved model we want to start from
    reload_path = config_state['reload_path']
    verbose = True
    NER_for_target_entities = False
    word2vec_emb_file = config_state['word2vec_emb_file']
    merge_multi_words_for_target_entities=config_state['merge_multi_words_for_target_entities']
    eb_rnn_type = config_state['EB_RNN_ARC']

    pos_feat_embedding= config_state['pos_feat_embedding']
    pos_indicator_embedding = config_state['pos_indicator_embedding']
    ent_pres_feat_embedding = config_state['ent_pres_feat_embedding']
    dim_ent_pres_emb = int(config_state['dim_ent_pres_emb'])
    dim_pos_emb = int(config_state['dim_pos_emb'])
    pos_emb_type= config_state['pos_emb_type']
    ent_pres_emb_type= config_state['ent_pres_emb_type']
    L2_reg= config_state['L2_penalty']
    reload_model = config_state['reload_model']  # Path to a saved model we want to start from
    reload_path = config_state['reload_path']
    SHUFFLE = config_state['SHUFFLE']
    normalise_embeddings = config_state['normalise_embeddings']
    dev_eval_split = config_state['dev_eval_split']
    data_set_sent_corpus = config_state['data_set_sent_corpus']
    postag = config_state['postag']
    entity_class = config_state['entity_class']
    update_pos_emb = config_state['update_pos_emb']
    update_ner_emb = config_state['update_ner_emb']
    add_subtree_emb = config_state['add_subtree_emb']
    treernn_weights = config_state['treernn_weights']

#--------------------- configuration Ends -------------------------------------------

    if savemodel == True:
        folder_name = saveto
        folder = folder_name
        print(folder)
        if not os.path.exists(folder):
            os.mkdir(folder)

    postag_vocab_size = None
    entity_class_vocab_size = None
    if postag == True:
        postag_corpus = ['#', '$', "''", '(', ')', ',', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``', 'E_TAG']
        postag_vocab_size = len(postag_corpus)
        train_postag_seq = read_postag(filename_train_data, postag_corpus)
        dev_postag_seq = read_postag(filename_dev_data, postag_corpus)
        test_postag_seq = read_postag(filename_test_data, postag_corpus)

    if entity_class == True:
        entity_class_corpus = ['B-BACTERIA', 'I-BACTERIA', 'B-HABITAT', 'I-HABITAT', 'O']
        entity_class_vocab_size = len(entity_class_corpus)
        if verbose_print:
            print("entity_class_vocab_size %s" %entity_class_vocab_size)
        train_entity_class_seq = read_iob(filename_train_data, entity_class_corpus)
        dev_entity_class_seq = read_iob(filename_dev_data, entity_class_corpus)
        test_entity_class_seq = read_iob(filename_test_data, entity_class_corpus)

    if add_subtree_emb == True:
        train_sdp_dep_data = load_save_pkl.load_pickle_file(filename_train_data.replace('sdp_sent','sdp_dep').replace('.txt','.pkl'))
        dev_sdp_dep_data = load_save_pkl.load_pickle_file(filename_dev_data.replace('sdp_sent','sdp_dep').replace('.txt','.pkl'))
        test_sdp_dep_data = load_save_pkl.load_pickle_file(filename_test_data.replace('sdp_sent','sdp_dep').replace('.txt','.pkl'))

        train_sdp_sent_data = open(filename_train_data).readlines()
        dev_sdp_sent_data = open(filename_dev_data).readlines()
        test_sdp_sent_data = open(filename_test_data).readlines()

        assert len(train_sdp_dep_data) == len(train_sdp_sent_data), 'len(train_sdp_dep_data) %s, len(train_sdp_sent_data) %s' %(len(train_sdp_dep_data),len(train_sdp_sent_data))
        assert len(dev_sdp_dep_data) == len(dev_sdp_sent_data), 'len(dev_sdp_dep_data) %s, len(dev_sdp_sent_data) %s' %(len(dev_sdp_dep_data),len(dev_sdp_sent_data))
        assert len(test_sdp_dep_data) == len(test_sdp_sent_data), 'len(test_sdp_dep_data) %s, len(test_sdp_sent_data) %s' %(len(test_sdp_dep_data),len(test_sdp_sent_data))

    f = open(log_file_name, 'a')
    open_flag = True
    np.random.seed(0)

    first_entity_word = [1,0]
    second_entity_word = [0,1]
    not_entity_word = [0,0]
    both_entities_in_window = [1,1]

    # generate word2vec vocabulary/dictionary
    word2vec_dict = generate_dict_for_word2vec_emb(word2vec_file_path=word2vec_emb_file)

    get_unnorm_position_feat = False
    if pos_feat_embedding == True:
        get_unnorm_position_feat = True

    get_position_indicators = False
    if pos_indicator_embedding == True:
        get_position_indicators = True

    # load train
    print('load training...')
    [seq_train, targets_train, target_entities_train, _, unnormalised_pos_feat_seq_train] = \
        read_train(filename_train_data, get_text_btwn_entities=train_on_text_between_target_entities,
                          remove_other_class = remove_other_class, exclude_entity_terms = exclude_entity_terms,
                          text_neighbourhood_to_entity = text_neighbourhood_to_entity,
                          left_neighbourhood_size = left_neighbourhood_size,
                          right_neighbourhood_size = right_neighbourhood_size,
                          NER_for_target_entities=NER_for_target_entities,
                          merge_multi_words_for_target_entities=merge_multi_words_for_target_entities,
                          get_entity_types=False,
                          get_unnorm_position_feat=get_unnorm_position_feat,
                          get_position_indicators=get_position_indicators)

    # random split train and test here
    # load dev
    print('load dev data...')
    [seq_dev, targets_dev_labels, target_entities_dev, _, unnormalised_pos_feat_seq_dev] = \
        read_train(filename_dev_data, get_text_btwn_entities=train_on_text_between_target_entities,
                       remove_other_class = remove_other_class,
                       exclude_entity_terms = exclude_entity_terms,
                       text_neighbourhood_to_entity = text_neighbourhood_to_entity,
                       left_neighbourhood_size = left_neighbourhood_size,
                       right_neighbourhood_size = right_neighbourhood_size,
                       NER_for_target_entities = NER_for_target_entities,
                       merge_multi_words_for_target_entities=merge_multi_words_for_target_entities,
                       get_entity_types=False, get_unnorm_position_feat=get_unnorm_position_feat,
                       get_position_indicators=get_position_indicators)

    # load test
    print('load testing...')
    [seq_test, targets_test_labels, target_entities_test, _, unnormalised_pos_feat_seq_test] = \
        read_train(filename_test_data, get_text_btwn_entities=train_on_text_between_target_entities,
                          remove_other_class = remove_other_class,
                          exclude_entity_terms = exclude_entity_terms,
                          text_neighbourhood_to_entity = text_neighbourhood_to_entity,
                          left_neighbourhood_size = left_neighbourhood_size,
                          right_neighbourhood_size = right_neighbourhood_size,
                          NER_for_target_entities = NER_for_target_entities,
                          merge_multi_words_for_target_entities=merge_multi_words_for_target_entities,
                          get_entity_types=False, get_unnorm_position_feat=get_unnorm_position_feat,
                          get_position_indicators=get_position_indicators)

    if np.array(seq_train).shape[0] != np.array(targets_train).shape[0]:
        raise ValueError("Number of samples in X_train and Y_train are not equal !! ")

    if np.array(seq_dev).shape[0] != np.array(targets_dev_labels).shape[0]:
        raise ValueError("Number of samples in X_dev and Y_dev are not equal !! ")

    if np.array(seq_test).shape[0] != np.array(targets_test_labels).shape[0]:
        raise ValueError("Number of samples in X_test and Y_test are not equal !! ")

    if get_unnorm_position_feat == True:
        if np.array(unnormalised_pos_feat_seq_train).shape[0] != np.array(seq_train).shape[0]:
            raise ValueError("Number of samples in unnormalised_pos_feat_seq_train and seq_train are not equal !! ")

        if np.array(unnormalised_pos_feat_seq_test).shape[0] != np.array(seq_test).shape[0]:
            raise ValueError("Number of samples in unnormalised_pos_feat_seq_test and seq_test are not equal !! ")

    if verbose_print:
        print('seq_train.shape[0]:', np.array(seq_train).shape[0])
        print('targets_train.shape[0]:', np.array(targets_train).shape[0])
        print('seq_dev.shape[0]:', np.array(seq_dev).shape[0])
        print('targets_dev.shape[0]:', np.array(targets_dev_labels).shape[0])
        print('seq_test.shape[0]:', np.array(seq_test).shape[0])
        print('targets_test.shape[0]:', np.array(targets_test_labels).shape[0])

        print('seq_train[0]:', seq_train[0])
        print('targets_train[0]:', targets_train[0])
        print('seq_dev[0]:', seq_dev[0])
        print('targets_dev[0]:', targets_dev_labels[0])
        print('seq_test[0]:', seq_test[0])
        print('targets_test[0]:', targets_test_labels[0])

    pos_vocab_size = 0
    if pos_feat_embedding == True:
        min = np.inf
        max = - np.inf
        for s_train in unnormalised_pos_feat_seq_train:
            if np.amin(s_train) < min:
                min = np.amin(s_train)
            if np.amax(s_train) > max:
                max = np.amax(s_train)

        for s_dev in unnormalised_pos_feat_seq_dev:
            if np.amin(s_dev) < min:
                min = np.amin(s_dev)
            if np.amax(s_dev) > max:
                max = np.amax(s_dev)

        for s_test in unnormalised_pos_feat_seq_test:
            if np.amin(s_test) < min:
                min = np.amin(s_test)
            if np.amax(s_test) > max:
                max = np.amax(s_test)

        min_position_val = min
        max_position_val = max

        # negative values are mapped to positive ones, by shifting the negative values to positive side scale
        # e.g. minimum distance is -10 and maximum distance value is 20. So, pos_vocab_size = 20+ |-10| = 30 distance values
        # so, for each position value, add 10 for the shift. So, position value '-10' is represented as '0'.
        pos_vocab_size = int(max_position_val + abs(min_position_val))

        # shift all postion values by min_position_val
        for seq_num in range(np.array(unnormalised_pos_feat_seq_train).shape[0]):
            unnormalised_pos_feat_seq_train[seq_num] += [abs(min_position_val), abs(min_position_val)]

        for seq_num in range(np.array(unnormalised_pos_feat_seq_dev).shape[0]):
            unnormalised_pos_feat_seq_dev[seq_num] += [abs(min_position_val), abs(min_position_val)]

        for seq_num in range(np.array(unnormalised_pos_feat_seq_test).shape[0]):
            unnormalised_pos_feat_seq_test[seq_num] += [abs(min_position_val), abs(min_position_val)]

    targets_dev = targets_dev_labels
    targets_test = targets_test_labels
    vocab_size = 0
    if embedding_type == '1-hot-encoding' or embedding_type == 'theano_word_embeddings':
        # data includes training and test sentences to generate vocab for all words occurring in train and test
        # merge test and train to generate vocabulary
        data = []
        for s_train in seq_train:
            data.append(s_train)
        for s_test in seq_test:
            data.append(s_test)
        for s_dev in seq_dev:
            data.append(s_dev)

        vectorizer, vocab_size = get_word_vectoriser_vocab_size(data)
        print('vocab_size:', vocab_size)

    n_train = np.array(seq_train).shape[0]
    n_dev = np.array(seq_dev).shape[0]
    n_test = np.array(seq_test).shape[0]

    if embedding_type == 'theano_word_embeddings':
        seq_train_indx, target_entities_train_indx, seq_test_indx , target_entities_test_indx = \
            get_train_test_word_indices(vectorizer, seq_train, target_entities_train, seq_test, target_entities_test)

        seq_train = seq_train_indx
        seq_test = seq_test_indx
        target_entities_train = target_entities_train_indx
        target_entities_test = target_entities_test_indx

    if embedding_type == 'word2vec_update':
        # get vocab size using the pre-trained models, filters words from data which are not present in word2vec vocab
        # Using the full vocab
        data = [line.rstrip('\n') for line in open(data_set_sent_corpus, 'rb')]
        if verbose_print:
            print("len(data_set_sent_corpus) %s" %len(data))
            print(data[0])

        embedding, emb_dict, dict_word_indices_for_emb, vocab_size = get_w2v_emb_dict_vocab(word2vec_emb_file,
                                                                                            data=data,
                                                                                            data_type='SemVal',
                                                                                            get_position_indicators=get_position_indicators)

        print('vocab_size, words macthing in w2v dict:', vocab_size)

        if not (len(dict_word_indices_for_emb) == len(emb_dict) and  len(dict_word_indices_for_emb) == len(dict_word_indices_for_emb)):
            raise ValueError('Embeddings mismatch')
        # convert words to indices using emb_dict and skip words not in emb_dict
        seq_train_indx, target_entities_train_indx, seq_dev_indx, target_entities_dev_indx, seq_test_indx, target_entities_test_indx , targets_train_NEs, targets_dev_NEs, targets_test_NEs  = \
            get_train_dev_test_word_indices_from_emb_dict(emb_dict, dict_word_indices_for_emb,
                                                      seq_train, target_entities_train, seq_dev,target_entities_dev, seq_test,target_entities_test, slot_fill=False, get_NEs = False)
        seq_train = seq_train_indx
        seq_test = seq_test_indx
        seq_dev = seq_dev_indx
        target_entities_train = target_entities_train_indx
        target_entities_dev = target_entities_dev_indx
        target_entities_test = target_entities_test_indx

    n_in = 0
    if embedding_type == '1-hot-encoding':
        n_in = vocab_size # for 1-hot encoding on words
        print('vocab_size:', vocab_size)
    elif embedding_type == 'word2vec_update' or embedding_type == 'theano_word_embeddings':
        if pos_feat_embedding == True and ent_pres_feat_embedding == True:
            n_in = dim_pos_emb * context_win_size * 2 + dim_ent_pres_emb * context_win_size * 2
        elif pos_feat_embedding == True:
            n_in = dim_pos_emb * context_win_size * 2
        elif ent_pres_feat_embedding == True:
            n_in = dim_ent_pres_emb * context_win_size * 2

        n_in += dim_emb * context_win_size
    else:
        n_in = dim_emb

    if augment_entity_presence == True:
        n_in += 2   #entity presence: first, second or both for window learning

    if position_features == True:
        n_in += 2

    if postag == True:
        n_in += 5

    if entity_class == True:
        n_in += 5

    dim_subtree_emb = None
    overall_max_degree = 0
    if add_subtree_emb == True:
        if treernn_weights == 'shared':
            dim_subtree_emb = n_hidden
            n_in += dim_subtree_emb
        elif treernn_weights == 'independent':
            dim_subtree_emb = 200
            n_in += dim_subtree_emb

        train_tree_data, train_dataset_sdp_sent_aug_info_combined_sent_idx, train_max_degree = data_utils.recnn_read_dataset_inter(
            sdp_dep_data=train_sdp_dep_data, sdp_sent_data=train_sdp_sent_data,
            dict_word_indices_for_emb=dict_word_indices_for_emb)

        overall_max_degree = overall_max_degree if overall_max_degree > train_max_degree else train_max_degree

        dev_tree_data, dev_dataset_sdp_sent_aug_info_combined_sent_idx, dev_max_degree = data_utils.recnn_read_dataset_inter(
            sdp_dep_data=dev_sdp_dep_data, sdp_sent_data=dev_sdp_sent_data,
            dict_word_indices_for_emb=dict_word_indices_for_emb)

        overall_max_degree = overall_max_degree if overall_max_degree > dev_max_degree else dev_max_degree

        test_tree_data, test_dataset_sdp_sent_aug_info_combined_sent_idx, test_max_degree =\
            data_utils.recnn_read_dataset_inter(
            sdp_dep_data=test_sdp_dep_data, sdp_sent_data=test_sdp_sent_data,
            dict_word_indices_for_emb=dict_word_indices_for_emb)
        overall_max_degree = overall_max_degree if overall_max_degree > test_max_degree else test_max_degree


    if verbose_print:
        print("n_in %s" %n_in)
        print('training data shape:', np.array(seq_train).shape)
        print('training data labels shape:', np.array(targets_train).shape)
        print('target_entities_train shape:', np.array(target_entities_train).shape)
        print('dev data shape:', np.array(seq_dev).shape)
        print('dev data labels shape:', np.array(targets_dev).shape)
        print('target_entities_dev shape:', np.array(target_entities_dev).shape)
        print('testing data shape:', np.array(seq_test).shape)
        print('testing data labels shape:', np.array(targets_test).shape)
        print('target_entities_test shape:', np.array(target_entities_test).shape)
        print('Class frequency for training data:\n',itemfreq(targets_train))
        print('Class frequency for dev data:\n',itemfreq(targets_dev))
        print('Class frequency for testing data:\n',itemfreq(targets_test))
        print('seq_train[0]:', seq_train[0])
        print('target_entities_train[0]:', target_entities_train[0])
        print('seq_dev[0]:', seq_dev[0])
        print('seq_test[0]:', seq_test[0])

    if eb_rnn_type == 'EB_RNN_4':
        print('EB_RNN_4 bi-directinal RNN')
        # Elman-bidirectional type RNN for relation classification
        if embedding_type == 'theano_word_embeddings':
            rnn = EB_RNN_4(n_in, n_hidden=n_hidden, nout=n_out, learning_rate_decay=learning_rate_decay,
                         activation=activation, state=state, vocab_size=vocab_size, dim_emb=dim_emb,
                         context_win_size=context_win_size, embedding=embedding_type, w_hh_type = w_hh_type,
                         position_feat=position_features, entity_presence_feat=augment_entity_presence,
                         pos_feat_embedding=pos_feat_embedding, pos_indicator_embedding=pos_indicator_embedding,
                         ent_pres_feat_embedding=ent_pres_feat_embedding,
                         dim_ent_pres_emb = dim_ent_pres_emb, dim_pos_emb = dim_pos_emb, pos_vocab_size = pos_vocab_size,
                         pos_emb_type=pos_emb_type, ent_pres_emb_type=ent_pres_emb_type,
                         reload_model=reload_model, reload_path=reload_path, L2_reg=L2_reg,
                         context_window_usage=context_window_usage)

        elif embedding_type == 'word2vec_update':
            rnn = EB_RNN_4(n_in, n_hidden=n_hidden, nout=n_out, learning_rate_decay=learning_rate_decay,
                         activation=activation, state=state, dim_emb=dim_emb, context_win_size=context_win_size,
                         embedding=embedding_type, use_dropout=use_dropout, optimiser=optimiser,
                         w_hh_initialise_strategy=w_hh_initialise_strategy, w_hh_type = w_hh_type,
                         w2v_embedding=embedding, position_feat=position_features,
                         entity_presence_feat=augment_entity_presence,
                         pos_feat_embedding=pos_feat_embedding, pos_indicator_embedding=pos_indicator_embedding,
                         ent_pres_feat_embedding=ent_pres_feat_embedding,
                         dim_ent_pres_emb = dim_ent_pres_emb, dim_pos_emb = dim_pos_emb,
                         pos_vocab_size = pos_vocab_size,
                         pos_emb_type=pos_emb_type, ent_pres_emb_type=ent_pres_emb_type,
                         reload_model=reload_model, reload_path=reload_path, L2_reg=L2_reg,
                         context_window_usage=context_window_usage, postag = postag,
                         postag_vocab_size = postag_vocab_size, entity_class = entity_class,
                         entity_class_vocab_size = entity_class_vocab_size, dim_postag_emb = 5,
                         dim_entity_class_emb = 5, update_pos_emb = update_pos_emb,
                           update_ner_emb = update_ner_emb,
                         add_subtree_emb = add_subtree_emb, dim_subtree_emb=dim_subtree_emb, max_degree=overall_max_degree, treernn_weights = treernn_weights)

        elif embedding_type == 'word2vec_init':
            rnn = EB_RNN_4(n_in, n_hidden=n_hidden, nout=n_out, learning_rate_decay=learning_rate_decay,
                         activation=activation, state=state, dim_emb=dim_emb, context_win_size=context_win_size,
                         embedding=embedding_type, use_dropout=use_dropout, optimiser=optimiser,
                         w_hh_initialise_strategy=w_hh_initialise_strategy, w_hh_type = w_hh_type,
                         position_feat=position_features, entity_presence_feat=augment_entity_presence,
                         pos_feat_embedding=pos_feat_embedding, pos_indicator_embedding=pos_indicator_embedding,
                         ent_pres_feat_embedding=ent_pres_feat_embedding,
                         dim_ent_pres_emb = dim_ent_pres_emb, dim_pos_emb = dim_pos_emb, pos_vocab_size = pos_vocab_size,
                         pos_emb_type=pos_emb_type, ent_pres_emb_type=ent_pres_emb_type,
                         reload_model=reload_model, reload_path=reload_path, L2_reg=L2_reg,
                         context_window_usage=context_window_usage)
        else:
            rnn = EB_RNN_4(n_in, n_hidden, n_out, learning_rate_decay, activation=activation, state=state,
                         use_dropout=use_dropout, optimiser=optimiser, w_hh_initialise_strategy=w_hh_initialise_strategy,
                         w_hh_type = w_hh_type, position_feat=position_features, entity_presence_feat=augment_entity_presence,
                         pos_feat_embedding=pos_feat_embedding, pos_indicator_embedding=pos_indicator_embedding,
                         ent_pres_feat_embedding=ent_pres_feat_embedding,
                         dim_ent_pres_emb = dim_ent_pres_emb, dim_pos_emb = dim_pos_emb, pos_vocab_size = pos_vocab_size,
                         pos_emb_type=pos_emb_type, ent_pres_emb_type=ent_pres_emb_type,
                         reload_model=reload_model, reload_path=reload_path, L2_reg=L2_reg,
                         context_window_usage=context_window_usage)

    if eb_rnn_type == 'EB_LSTM':
        print('EB_LSTM bi-directinal LSTM')
        # Elman-bidirectional type RNN for relation classification
        if embedding_type == 'theano_word_embeddings':
            rnn = EB_LSTM(n_in, n_hidden=n_hidden, nout=n_out, learning_rate_decay=learning_rate_decay,
                         activation=activation, state=state, vocab_size=vocab_size, dim_emb=dim_emb,
                         context_win_size=context_win_size, embedding=embedding_type, w_hh_type = w_hh_type,
                         position_feat=position_features, entity_presence_feat=augment_entity_presence,
                         pos_feat_embedding=pos_feat_embedding, pos_indicator_embedding=pos_indicator_embedding,
                         ent_pres_feat_embedding=ent_pres_feat_embedding,
                         dim_ent_pres_emb = dim_ent_pres_emb, dim_pos_emb = dim_pos_emb, pos_vocab_size = pos_vocab_size,
                         pos_emb_type=pos_emb_type, ent_pres_emb_type=ent_pres_emb_type,
                         reload_model=reload_model, reload_path=reload_path, L2_reg=L2_reg,
                         context_window_usage=context_window_usage)

        elif embedding_type == 'word2vec_update':
            rnn = EB_LSTM(n_in, n_hidden=n_hidden, nout=n_out, learning_rate_decay=learning_rate_decay,
                         activation=activation, state=state, dim_emb=dim_emb, context_win_size=context_win_size,
                         embedding=embedding_type, use_dropout=use_dropout, optimiser=optimiser,
                         w_hh_initialise_strategy=w_hh_initialise_strategy, w_hh_type = w_hh_type,
                         w2v_embedding=embedding, position_feat=position_features,
                         entity_presence_feat=augment_entity_presence,
                         pos_feat_embedding=pos_feat_embedding, pos_indicator_embedding=pos_indicator_embedding,
                         ent_pres_feat_embedding=ent_pres_feat_embedding,
                         dim_ent_pres_emb = dim_ent_pres_emb, dim_pos_emb = dim_pos_emb,
                         pos_vocab_size = pos_vocab_size,
                         pos_emb_type=pos_emb_type, ent_pres_emb_type=ent_pres_emb_type,
                         reload_model=reload_model, reload_path=reload_path, L2_reg=L2_reg,
                         context_window_usage=context_window_usage, postag = postag,
                         postag_vocab_size = postag_vocab_size, entity_class = entity_class,
                         entity_class_vocab_size = entity_class_vocab_size, dim_postag_emb = 5,
                         dim_entity_class_emb = 5, update_pos_emb = update_pos_emb,
                           update_ner_emb = update_ner_emb,
                         add_subtree_emb = add_subtree_emb, dim_subtree_emb=dim_subtree_emb, max_degree=overall_max_degree, treernn_weights = treernn_weights)

        elif embedding_type == 'word2vec_init':
            rnn = EB_LSTM(n_in, n_hidden=n_hidden, nout=n_out, learning_rate_decay=learning_rate_decay,
                         activation=activation, state=state, dim_emb=dim_emb, context_win_size=context_win_size,
                         embedding=embedding_type, use_dropout=use_dropout, optimiser=optimiser,
                         w_hh_initialise_strategy=w_hh_initialise_strategy, w_hh_type = w_hh_type,
                         position_feat=position_features, entity_presence_feat=augment_entity_presence,
                         pos_feat_embedding=pos_feat_embedding, pos_indicator_embedding=pos_indicator_embedding,
                         ent_pres_feat_embedding=ent_pres_feat_embedding,
                         dim_ent_pres_emb = dim_ent_pres_emb, dim_pos_emb = dim_pos_emb, pos_vocab_size = pos_vocab_size,
                         pos_emb_type=pos_emb_type, ent_pres_emb_type=ent_pres_emb_type,
                         reload_model=reload_model, reload_path=reload_path, L2_reg=L2_reg,
                         context_window_usage=context_window_usage)
        else:
            rnn = EB_LSTM(n_in, n_hidden, n_out, learning_rate_decay, activation=activation, state=state,
                         use_dropout=use_dropout, optimiser=optimiser, w_hh_initialise_strategy=w_hh_initialise_strategy,
                         w_hh_type = w_hh_type, position_feat=position_features, entity_presence_feat=augment_entity_presence,
                         pos_feat_embedding=pos_feat_embedding, pos_indicator_embedding=pos_indicator_embedding,
                         ent_pres_feat_embedding=ent_pres_feat_embedding,
                         dim_ent_pres_emb = dim_ent_pres_emb, dim_pos_emb = dim_pos_emb, pos_vocab_size = pos_vocab_size,
                         pos_emb_type=pos_emb_type, ent_pres_emb_type=ent_pres_emb_type,
                         reload_model=reload_model, reload_path=reload_path, L2_reg=L2_reg,
                         context_window_usage=context_window_usage)

    f.write('\nn_hidden: '+str(n_hidden)
            +'\nlr: '+str(lr)
            +'\nlr_decay: '+str(learning_rate_decay)
            +'\nepochs: '+str(epoch)
            +'\nclipstype: '+str(state['clipstyle'])
            +'\ncutoff: '+str(state['cutoff'])
            +'\naugment_entity_presence: '+str(augment_entity_presence)
            +'\nposition_features: '+ str(position_features)
            +'\nupdate_learning_rate_after_n_epochs: '+ str(update_learning_rate_after_n_epochs)
            +'\nvocab_size:'+str(vocab_size)
            +'\ntrain_on_text_between_target_entities:'+str(train_on_text_between_target_entities)
            +'\ndev_on_text_between_target_entities:'+str(dev_on_text_between_target_entities)
            +'\ntest_on_text_between_target_entities:'+str(test_on_text_between_target_entities)
            +'\nembedding_type:'+str(embedding_type)
            +'\ncontext_win_size:'+str(context_win_size)
            +'\nuse_dropout:'+str(use_dropout)
            +'\noptimiser:'+str(optimiser) +
            '\nlearning_rate_decay_strategy:'+str(learning_rate_decay_strategy)
            +'\nw_hh_initialise_strategy:'+str(w_hh_initialise_strategy)
            +'\nactivation:'+str(activation)
			+'\nexclude_entity_terms:'+str(exclude_entity_terms)
			+'\ntext_neighbourhood_to_entity:'+str(text_neighbourhood_to_entity)
			+'\nleft_neighbourhood_size:'+str(left_neighbourhood_size)
			+'\nright_neighbourhood_size:'+str(right_neighbourhood_size)
            +'\nw_hh_type:'+str(w_hh_type)
            +'\ncontext_window_usage:'+str(context_window_usage)
            +'\nbatch_size:'+str(batch_size)
            +'\nword_minibatch_usage:'+str(word_minibatch_usage)
            + '\neb_rnn_type:'+str(eb_rnn_type)
            + '\npos_feat_embedding:'+str(pos_feat_embedding)
            + '\npos_indicator_embedding:'+str(pos_indicator_embedding)
            + '\nent_pres_feat_embedding:'+str(ent_pres_feat_embedding)
            + '\ndim_ent_pres_emb:'+str(dim_ent_pres_emb)
            + '\ndim_pos_emb:'+str(dim_pos_emb)
            + '\npos_vocab_size:'+str(pos_vocab_size)
            + '\npos_emb_type:'+str(pos_emb_type)
            + '\nent_pres_emb_type:'+str(ent_pres_emb_type)
            + '\nL2_reg:'+str(L2_reg)
            + '\nRANKING:'+str(RANKING)
            +'\n\n')

    if embedding_type == 'word2vec_update':
        if position_features == True:
            pos_feat_train = [[] for p in range(n_train)]
            pos_feat_dev = [[] for p in range(n_dev)]
            pos_feat_test = [[] for p in range(n_test)]
        if augment_entity_presence == True:
            ent_pres_feat_train = [[] for p in range(n_train)]
            ent_pres_feat_dev = [[] for p in range(n_dev)]
            ent_pres_feat_test = [[] for p in range(n_test)]

    # compute number of minibatches for training and testing
    n_train_batches = n_train/batch_size
    n_dev_batches = n_dev/batch_size
    n_test_batches = n_test/batch_size

    # prepare training data sequence for RNN training
    if verbose_print:
        print('preparing training data sequence for RNN training')

    for seq_num in range(np.array(seq_train).shape[0]):
        # print('seq_train[0].shape:', seq_train[0].shape)
        if embedding_type == 'theano_word_embeddings':
            s = seq_train[seq_num]
            # word window and embedding here
            cwords = contextwin(s, win=context_win_size)
            # check for entity presence and position_features of each word in the context window
            if augment_entity_presence == True:
                # augment_entity_presence within the context window of each word
                cwords = get_entity_presence_in_context_window(vectorizer=vectorizer, cwords=cwords,
                                                                target_entities=target_entities_train[seq_num],
                                                                first_entity_word=first_entity_word,
                                                                second_entity_word=second_entity_word,
                                                                not_entity_word=not_entity_word,
                                                                both_entities_in_window=both_entities_in_window)
            seq_train[seq_num] = cwords

        elif embedding_type == 'word2vec_update':
            s = seq_train[seq_num]
            # word window and embedding here
            if context_window_usage == True:
                #print('s:', s)
                cwords = contextwin(s, win=context_win_size)

                # to do : augment_entity_presence == True ??
                seq_train[seq_num] = cwords

            pos_feat, ent_pres_feat, _ = get_entity_presence_position_features_NEs(tokenised_sentence_indices=seq_train[seq_num],
                                                target_entities=target_entities_train[seq_num],
                                                position_features=position_features,
                                                augment_entity_presence=augment_entity_presence,
                                                first_entity_word=first_entity_word,
                                                second_entity_word=second_entity_word,
                                                not_entity_word=not_entity_word,
                                                normalise_pos_feat=normalise_position_features, get_NEs=False)

            if position_features == True:
                pos_feat_train[seq_num] = pos_feat

            if augment_entity_presence == True:
                ent_pres_feat_train[seq_num] = ent_pres_feat

            s = np.array(seq_train[seq_num])
            if s.shape[0] == 0:
                continue

            seq_train[seq_num]  = s

        elif embedding_type == 'word2vec_init':
            s = [word for word in seq_train[seq_num].lower().split()]
            if len(s) == 0:
                continue
            x = get_word2vec_emb_entity_presence_position_features(word2vec_dict=word2vec_dict,
                                                                    tokenised_sentence=s,
                                                target_entities=target_entities_train[seq_num],
                                                position_features=position_features,
                                                augment_entity_presence=augment_entity_presence,
                                                first_entity_word=first_entity_word,
                                                second_entity_word=second_entity_word,
                                                not_entity_word=not_entity_word,
                                                normalise_pos_feat=normalise_position_features, get_NEs=False)
            x = np.array(x)
            if x.shape[0] == 0:
                continue

            seq_train[seq_num]  = x
        else:
            s = [word for word in seq_train[seq_num].lower().split() if word not in stoplist]
            if len(s) == 0:
                continue
            x = get_one_hot_word_representation(vectorizer=vectorizer, s=s,
                                                target_entities=target_entities_train[seq_num],
                                                position_features=position_features,
                                                augment_entity_presence=augment_entity_presence,
                                                first_entity_word=first_entity_word,
                                                second_entity_word=second_entity_word,
                                                not_entity_word=not_entity_word,
                                                normalise_pos_feat=normalise_position_features, get_NEs=False)
            x = np.array(x)
            if x.shape[0] == 0:
                continue
            seq_train[seq_num]  = x

    # prepare dev data
    for seq_num in range(np.array(seq_dev).shape[0]):
        # if embedding used, then work with word indices, else work with words
        if embedding_type == 'theano_word_embeddings':
            s = seq_dev[seq_num]
            # word window and embedding here
            cwords = contextwin(s, win=context_win_size)
            cwords = np.asarray(cwords).astype('int32')
            seq_dev[seq_num] = cwords

        elif embedding_type == 'word2vec_update':
            s = seq_dev[seq_num]
            # word window and embedding here
            if context_window_usage == True:
                cwords = contextwin(s, win=context_win_size)
                # to do : augment_entity_presence == True ??
                seq_dev[seq_num] = cwords

            pos_feat, ent_pres_feat, _ = get_entity_presence_position_features_NEs(tokenised_sentence_indices=seq_dev[seq_num],
                                                                                   target_entities=target_entities_dev[seq_num],
                                                                                   position_features=position_features,
                                                                                   augment_entity_presence=augment_entity_presence,
                                                                                   first_entity_word=first_entity_word,
                                                                                   second_entity_word=second_entity_word,
                                                                                   not_entity_word=not_entity_word,
                                                                                   normalise_pos_feat=normalise_position_features, get_NEs=False)
            if position_features == True:
                pos_feat_dev[seq_num] = pos_feat

            if augment_entity_presence == True:
                ent_pres_feat_dev[seq_num] = ent_pres_feat

        elif embedding_type == 'word2vec_init':
            s = [word for word in seq_dev[seq_num].lower().split()]
            if len(s) == 0:
                continue
            x = get_word2vec_emb_entity_presence_position_features(word2vec_dict=word2vec_dict,
                                                                   tokenised_sentence=s,
                                                                   target_entities=target_entities_dev[seq_num],
                                                                   position_features=position_features,
                                                                   augment_entity_presence=augment_entity_presence,
                                                                   first_entity_word=first_entity_word,
                                                                   second_entity_word=second_entity_word,
                                                                   not_entity_word=not_entity_word,
                                                                   normalise_pos_feat=normalise_position_features)
            x = np.array(x)
            if x.shape[0] == 0:
                continue
            seq_dev[seq_num] = x
        else:
            s = [word for word in seq_dev[seq_num].lower().split()]
            if len(s) == 0:
                continue
            # s is list of words in the sentence
            x = get_one_hot_word_representation(vectorizer=vectorizer, s=s,
                                                target_entities = target_entities_dev[seq_num],
                                                position_features = position_features,
                                                augment_entity_presence = augment_entity_presence,
                                                first_entity_word=first_entity_word,
                                                second_entity_word=second_entity_word,
                                                not_entity_word=not_entity_word,
                                                normalise_pos_feat=normalise_position_features)
            x = np.array(x)
            if x.shape[0] == 0:
                continue
            seq_dev[seq_num] = x

    # prepare test data
    for seq_num in range(np.array(seq_test).shape[0]):
        # if embedding used, then work with word indices, else work with words
        if embedding_type == 'theano_word_embeddings':
            s = seq_test[seq_num]
            # word window and embedding here
            cwords = contextwin(s, win=context_win_size)
            cwords = np.asarray(cwords).astype('int32')
            seq_test[seq_num] = cwords

        elif embedding_type == 'word2vec_update':
            s = seq_test[seq_num]
            # word window and embedding here
            if context_window_usage == True:
                cwords = contextwin(s, win=context_win_size)
                seq_test[seq_num] = cwords

            pos_feat, ent_pres_feat, _ = get_entity_presence_position_features_NEs(tokenised_sentence_indices=seq_test[seq_num],
                                                target_entities=target_entities_test[seq_num],
                                                position_features=position_features,
                                                augment_entity_presence=augment_entity_presence,
                                                first_entity_word=first_entity_word,
                                                second_entity_word=second_entity_word,
                                                not_entity_word=not_entity_word,
                                                normalise_pos_feat=normalise_position_features, get_NEs=False)
            if position_features == True:
                pos_feat_test[seq_num] = pos_feat

            if augment_entity_presence == True:
                ent_pres_feat_test[seq_num] = ent_pres_feat

        elif embedding_type == 'word2vec_init':
            s = [word for word in seq_test[seq_num].lower().split()]
            if len(s) == 0:
                continue
            x = get_word2vec_emb_entity_presence_position_features(word2vec_dict=word2vec_dict,
                                                                    tokenised_sentence=s,
                                                target_entities=target_entities_test[seq_num],
                                                position_features=position_features,
                                                augment_entity_presence=augment_entity_presence,
                                                first_entity_word=first_entity_word,
                                                second_entity_word=second_entity_word,
                                                not_entity_word=not_entity_word,
                                                normalise_pos_feat=normalise_position_features)
            x = np.array(x)
            if x.shape[0] == 0:
                continue
            seq_test[seq_num] = x
        else:
            s = [word for word in seq_test[seq_num].lower().split()]
            if len(s) == 0:
                continue
            # s is list of words in the sentence
            x = get_one_hot_word_representation(vectorizer=vectorizer, s=s,
                                                target_entities = target_entities_test[seq_num],
                                                position_features = position_features,
                                                augment_entity_presence = augment_entity_presence,
                                                first_entity_word=first_entity_word,
                                                second_entity_word=second_entity_word,
                                                not_entity_word=not_entity_word,
                                                normalise_pos_feat=normalise_position_features)
            x = np.array(x)
            if x.shape[0] == 0:
                continue
            seq_test[seq_num] = x

    if verbose_print:
        print('seq_train.shape:', np.array(seq_train).shape)
        print('seq_dev.shape:', np.array(seq_dev).shape)
        print('seq_test.shape:', np.array(seq_test).shape)
        print('seq_train[0].shape:', seq_train[0].shape)


    best_dev_f1 = -np.inf

    if SHUFFLE == True:
        seq_train_all = seq_train[:]
        targets_train_all = targets_train[:]

    #epoch=1
    y_pred_probilities = []
    y_predictions = []

    for i in range(epoch):
        training_prections = []
        training_true_labels = []
        dev_predictions = []
        dev_pred_prob = []
        dev_true_labels = []
        dev_decision = []
        testing_predictions = []

        testing_pred_prob = []
        testing_true_labels = []
        testing_decision = []
        if verbose_print:
            print('Running epoch %s' %i)
        if dev_eval_split == True:
            dev_pred_prob = []
            dev_true_labels = []
            dev_decision = []
            dev_predictions = []

        cost = 0
        # not_in_word2vec = 0
        tic = time.time()
        '''
        rnn.effective_momentum = rnn.final_momentum \
                               if i > rnn.momentum_switchover \
                               else rnn.initial_momentum
        '''

        if SHUFFLE == True:
            #first selection of subset
            seq_train = []
            targets_train = []
            random_indices = np.random.permutation(np.array(seq_train_all).shape[0])
            random_indices_this = random_indices[0:np.array(seq_train_all).shape[0]]
            for j in random_indices_this:
                seq_train.append(seq_train_all[j])
                targets_train.append(targets_train_all[j])

            #first selection of subset
            seq_train = []
            targets_train = []
            random_indices = np.random.permutation(np.array(seq_train_all).shape[0])
            #random_indices_this = random_indices[0:8000]
            random_indices_this = random_indices[0:np.array(seq_train_all).shape[0]]
            for j in random_indices_this:
                seq_train.append(seq_train_all[j])
                targets_train.append(targets_train_all[j])

        rnn.effective_momentum = 1.0

        for seq_num in range(np.array(seq_train).shape[0]):
            if train_model == False:
                break
            # if embedding used, then work with word indices, else work with words
            if embedding_type == 'theano_word_embeddings':
                cwords = seq_train[seq_num]
                t = targets_train[seq_num]

                if word_minibatch_usage == True:
                    print 'Dont support word minibatch'
                    exit()
                else:
                    x_rev = cwords[::-1]
                    train_confidences = rnn.score(cwords, x_rev, t, lr, rho, rnn.effective_momentum, neg_classes)
                    neg_classes = np.argmax(train_confidences)
                    if neg_classes == targets_train[seq_num]:
                       neg_classes = np.argsort(train_confidences)[17]
                    cost = rnn.train_step(cwords, x_rev, t, lr, rho, rnn.effective_momentum, neg_classes)

                    rnn.normalize()

            elif embedding_type == 'word2vec_update':
                x = seq_train[seq_num]
                x_rev = x[::-1]
                t = targets_train[seq_num]

                if postag:
                    postag_idx = train_postag_seq[seq_num]
                    if x.shape[0] != len(postag_idx):
                        print('Error postag_idx Dimension mismatch line_num %s' %seq_num)
                        print('x.shape[0] %s' %x.shape[0])
                        print('len(postag_idx) %s' %len(postag_idx))

                if entity_class:
                    entity_class_idx = train_entity_class_seq[seq_num]
                    if x.shape[0] != len(entity_class_idx):
                        print('Error entity_class_idx Dimension mismatch line_num %s' %seq_num)
                        print('x.shape[0] %s' %x.shape[0])
                        print('len(entity_class_idx) %s' %len(entity_class_idx))

                if add_subtree_emb == True:
                    tree_data = train_tree_data[seq_num]
                    sdp_sent_aug_info_combined_sent_idx = \
                        train_dataset_sdp_sent_aug_info_combined_sent_idx[seq_num]
                    sdp_sent_aug_info_leaf_internal_x, sdp_sent_aug_info_computation_tree_matrix,\
                    sdp_sent_aug_info_tree_states_order = tree_rnn.gen_nn_inputs(tree_data,
                                                                            max_degree=overall_max_degree,
                                                     only_leaves_have_vals=False,
                                                     with_labels=False)
                    sdp_sent_aug_info_output_tree_state_idx = []

                    if context_window_usage == True:
                        cwords_leaf_interal = contextwin(sdp_sent_aug_info_leaf_internal_x,
                                                         win=context_win_size)
                        sdp_sent_aug_info_leaf_internal_x_cwords = cwords_leaf_interal
                    else:
                        sdp_sent_aug_info_leaf_internal_x_cwords = sdp_sent_aug_info_leaf_internal_x

                    for sdp_tok_idx, combined_sent_idx in enumerate(sdp_sent_aug_info_combined_sent_idx):
                        if combined_sent_idx==None:
                            sdp_sent_aug_info_output_tree_state_idx.append(0) # for entity tags use the embedding from the first leaf node in the leaf_computation order
                        else:
                            orig_sent_idx = combined_sent_idx
                            output_tree_state_idx = sdp_sent_aug_info_tree_states_order.index(orig_sent_idx)
                            sdp_sent_aug_info_output_tree_state_idx.append(output_tree_state_idx)

                if word_minibatch_usage == True:
                    print 'Dont support yet'
                    exit()
                elif position_features == True and augment_entity_presence == True:
                    if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                        neg_classes = rnn.score(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                t, lr, rho, rnn.effective_momentum,
                                                pos_feat_train[seq_num], ent_pres_feat_train[seq_num],
                                                unnormalised_pos_feat_seq_train[seq_num],
                                                ent_pres_feat_train[seq_num])
                        if neg_classes == targets_train[seq_num]:
                            neg_classes = np.argsort(train_confidences)[17]
                        cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                t, lr, rho, rnn.effective_momentum,
                                                pos_feat_train[seq_num], ent_pres_feat_train[seq_num],
                                                unnormalised_pos_feat_seq_train[seq_num],
                                                ent_pres_feat_train[seq_num], neg_classes)

                        if pos_emb_type == 'DECOUPLED':
                            rnn.normalize_pos_emb_e1()
                            rnn.normalize_pos_emb_e2()
                            rnn.normalize_ent_pres_emb_e1()
                            rnn.normalize_pos_emb_e2()
                        elif pos_emb_type == 'COUPLED':
                            rnn.normalize_pos_emb()
                            rnn.normalize_ent_pres_emb()

                    elif pos_feat_embedding == True:
                        neg_classes = rnn.score(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                t, lr, rho, rnn.effective_momentum,
                                                pos_feat_train[seq_num], ent_pres_feat_train[seq_num],
                                                np.array(unnormalised_pos_feat_seq_train[seq_num]).
                                                reshape(np.array(unnormalised_pos_feat_seq_train[seq_num]).shape[0], 2))
                        if neg_classes == targets_train[seq_num]:
                            neg_classes = np.argsort(train_confidences)[17]
                        cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                t, lr, rho, rnn.effective_momentum,
                                                pos_feat_train[seq_num], ent_pres_feat_train[seq_num],
                                                np.array(unnormalised_pos_feat_seq_train[seq_num]).
                                                reshape(np.array(unnormalised_pos_feat_seq_train[seq_num]).shape[0], 2), neg_classes)

                        if pos_emb_type == 'DECOUPLED':
                            rnn.normalize_pos_emb_e1()
                            rnn.normalize_pos_emb_e2()
                        elif pos_emb_type == 'COUPLED':
                            rnn.normalize_pos_emb()

                    elif ent_pres_emb_type == True:
                        neg_classes = rnn.score(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                pos_feat_train[seq_num], ent_pres_feat_train[seq_num],
                                                t, lr, rho, rnn.effective_momentum,
                                                ent_pres_feat_train[seq_num])
                        if neg_classes == targets_train[seq_num]:
                            neg_classes = np.argsort(train_confidences)[17]
                        cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                pos_feat_train[seq_num], ent_pres_feat_train[seq_num],
                                                t, lr, rho, rnn.effective_momentum,
                                                ent_pres_feat_train[seq_num], neg_classes)

                        if pos_emb_type == 'DECOUPLED':
                            rnn.normalize_ent_pres_emb_e1()
                            rnn.normalize_pos_emb_e2()
                        elif pos_emb_type == 'COUPLED':
                            rnn.normalize_ent_pres_emb()
                    else:
                        neg_classes = rnn.score(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                t, lr, rho, rnn.effective_momentum,
                                                pos_feat_train[seq_num], ent_pres_feat_train[seq_num])
                        if neg_classes == targets_train[seq_num]:
                            neg_classes = np.argsort(train_confidences)[17]
                        cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                t, lr, rho, rnn.effective_momentum,
                                                pos_feat_train[seq_num], ent_pres_feat_train[seq_num], neg_classes)
                    rnn.normalize()

                elif position_features == True:
                    if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                        cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                t, lr, rho, rnn.effective_momentum,
                                                pos_feat_train[seq_num],
                                                unnormalised_pos_feat_seq_train[seq_num],
                                                ent_pres_feat_train[seq_num], neg_classes)

                        if pos_emb_type == 'DECOUPLED':
                            rnn.normalize_pos_emb_e1()
                            rnn.normalize_pos_emb_e2()
                            rnn.normalize_ent_pres_emb_e1()
                            rnn.normalize_pos_emb_e2()
                        elif pos_emb_type == 'COUPLED':
                            rnn.normalize_pos_emb()
                            rnn.normalize_ent_pres_emb()

                    elif pos_feat_embedding == True:
                        cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                t, lr, rho, rnn.effective_momentum, pos_feat_train[seq_num],
                                                np.array(unnormalised_pos_feat_seq_train[seq_num]).
                                                reshape(np.array(unnormalised_pos_feat_seq_train[seq_num]).shape[0], 2), neg_classes)

                        if pos_emb_type == 'DECOUPLED':
                            rnn.normalize_pos_emb_e1()
                            rnn.normalize_pos_emb_e2()
                        elif pos_emb_type == 'COUPLED':
                            rnn.normalize_pos_emb()

                    elif ent_pres_emb_type == True:
                        cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                t, lr, rho, rnn.effective_momentum, pos_feat_train[seq_num],
                                                ent_pres_feat_train[seq_num], neg_classes)

                        if pos_emb_type == 'DECOUPLED':
                            rnn.normalize_ent_pres_emb_e1()
                            rnn.normalize_pos_emb_e2()
                        elif pos_emb_type == 'COUPLED':
                            rnn.normalize_ent_pres_emb()
                    else:
                        cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                t, lr, rho, rnn.effective_momentum, pos_feat_train[seq_num], neg_classes)
                    rnn.normalize()

                elif augment_entity_presence == True:
                    if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                        cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                t, lr, rho, rnn.effective_momentum, ent_pres_feat_train[seq_num],
                                                unnormalised_pos_feat_seq_train[seq_num],
                                                ent_pres_feat_train[seq_num], neg_classes)

                        if pos_emb_type == 'DECOUPLED':
                            rnn.normalize_pos_emb_e1()
                            rnn.normalize_pos_emb_e2()
                            rnn.normalize_ent_pres_emb_e1()
                            rnn.normalize_pos_emb_e2()
                        elif pos_emb_type == 'COUPLED':
                            rnn.normalize_pos_emb()
                            rnn.normalize_ent_pres_emb()

                    elif pos_feat_embedding == True:
                        cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                t, lr, rho, rnn.effective_momentum, ent_pres_feat_train[seq_num],
                                                np.array(unnormalised_pos_feat_seq_train[seq_num]).
                                                reshape(np.array(unnormalised_pos_feat_seq_train[seq_num]).shape[0], 2), neg_classes)

                        if pos_emb_type == 'DECOUPLED':
                            rnn.normalize_pos_emb_e1()
                            rnn.normalize_pos_emb_e2()
                        elif pos_emb_type == 'COUPLED':
                            rnn.normalize_pos_emb()

                    elif ent_pres_emb_type == True:
                        cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                t, lr, rho, rnn.effective_momentum, ent_pres_feat_train[seq_num],
                                                ent_pres_feat_train[seq_num], neg_classes)

                        if pos_emb_type == 'DECOUPLED':
                            rnn.normalize_ent_pres_emb_e1()
                            rnn.normalize_pos_emb_e2()
                        elif pos_emb_type == 'COUPLED':
                            rnn.normalize_ent_pres_emb()
                    else:
                        cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                t, lr, rho, rnn.effective_momentum, ent_pres_feat_train[seq_num], neg_classes)
                    rnn.normalize()
                else:
                    if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                        cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                t, lr, rho, rnn.effective_momentum,
                                                unnormalised_pos_feat_seq_train[seq_num],
                                                ent_pres_feat_train[seq_num], neg_classes)

                        if pos_emb_type == 'DECOUPLED':
                            rnn.normalize_pos_emb_e1()
                            rnn.normalize_pos_emb_e2()
                            rnn.normalize_ent_pres_emb_e1()
                            rnn.normalize_pos_emb_e2()
                        elif pos_emb_type == 'COUPLED':
                            rnn.normalize_pos_emb()
                            rnn.normalize_ent_pres_emb()

                    elif pos_feat_embedding == True:
                        cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                t, lr, rho, rnn.effective_momentum,
                                                np.array(unnormalised_pos_feat_seq_train[seq_num]).
                                                reshape(np.array(unnormalised_pos_feat_seq_train[seq_num]).shape[0], 2), neg_classes)

                        if pos_emb_type == 'DECOUPLED':
                            rnn.normalize_pos_emb_e1()
                            rnn.normalize_pos_emb_e2()
                        elif pos_emb_type == 'COUPLED':
                            rnn.normalize_pos_emb()

                    elif ent_pres_emb_type == True:
                        cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                t, lr, rho, rnn.effective_momentum,
                                                ent_pres_feat_train[seq_num], neg_classes)

                        if pos_emb_type == 'DECOUPLED':
                            rnn.normalize_ent_pres_emb_e1()
                            rnn.normalize_pos_emb_e2()
                        elif pos_emb_type == 'COUPLED':
                            rnn.normalize_ent_pres_emb()
                    else:
                        if RANKING == True:
                            if context_window_usage == True:
                                train_confidences = rnn.score(np.array(x),
                                                    np.array(x_rev),
                                                    t, lr, rho, rnn.effective_momentum)
                                neg_classes = np.argmax(train_confidences)
                                if neg_classes == targets_train[seq_num]:
                                   neg_classes = np.argsort(train_confidences)[17]

                                cost = rnn.train_step(np.array(x),
                                                    np.array(x_rev),
                                                    t, lr, rho, rnn.effective_momentum, neg_classes)
                            else:
                                train_confidences = rnn.score(np.array(x).reshape(1, np.array(x).shape[0]),
                                                        np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                        t, lr, rho, rnn.effective_momentum)
                                neg_classes = np.argmax(train_confidences)
                                if neg_classes == targets_train[seq_num]:
                                   neg_classes = np.argsort(train_confidences)[17]

                                cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                        np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                        t, lr, rho, rnn.effective_momentum, neg_classes)
                        else:
                            if context_window_usage == True:
                                if add_subtree_emb == True:
                                    if postag == True and entity_class == True:
                                        cost = rnn.train_step(np.array(x),
                                                              np.array(x_rev),
                                                              np.array(sdp_sent_aug_info_leaf_internal_x),
                                                              np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                              np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                              np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                              np.array(postag_idx),
                                                              np.array(entity_class_idx),
                                                              t, lr, rho, rnn.effective_momentum)
                                    elif postag == True:
                                        cost = rnn.train_step(np.array(x),
                                                              np.array(x_rev),
                                                              np.array(sdp_sent_aug_info_leaf_internal_x),
                                                              np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                              np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                              np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                              np.array(postag_idx),
                                                              t, lr, rho, rnn.effective_momentum)
                                    elif entity_class == True:
                                        cost = rnn.train_step(np.array(x),
                                                              np.array(x_rev),
                                                              np.array(sdp_sent_aug_info_leaf_internal_x),
                                                              np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                              np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                              np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                              np.array(entity_class_idx),
                                                              t, lr, rho, rnn.effective_momentum)
                                    else:
                                        cost = rnn.train_step(np.array(x),
                                                              np.array(x_rev),
                                                              np.array(sdp_sent_aug_info_leaf_internal_x),
                                                              np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                              np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                              np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                            t, lr, rho, rnn.effective_momentum)
                                else:
                                    if postag == True and entity_class == True:
                                        cost = rnn.train_step(np.array(x),
                                                              np.array(x_rev),
                                                              np.array(postag_idx),
                                                              np.array(entity_class_idx),
                                                              t, lr, rho, rnn.effective_momentum)
                                    elif postag == True:
                                        cost = rnn.train_step(np.array(x),
                                                              np.array(x_rev),
                                                              np.array(postag_idx),
                                                              t, lr, rho, rnn.effective_momentum)
                                    elif entity_class == True:
                                        cost = rnn.train_step(np.array(x),
                                                              np.array(x_rev),
                                                              np.array(entity_class_idx),
                                                              t, lr, rho, rnn.effective_momentum)
                                    else:
                                        cost = rnn.train_step(np.array(x),
                                                              np.array(x_rev),
                                                              t, lr, rho, rnn.effective_momentum)
                            else:
                                if postag == True and entity_class == True:
                                    cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                          np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                          np.array(postag_idx),
                                                          np.array(entity_class_idx),
                                                          t, lr, rho, rnn.effective_momentum)
                                elif postag == True:
                                    cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                          np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                          np.array(postag_idx),
                                                          t, lr, rho, rnn.effective_momentum)
                                elif entity_class == True:
                                    cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                          np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                          np.array(entity_class_idx),
                                                          t, lr, rho, rnn.effective_momentum)
                                else:
                                    cost = rnn.train_step(np.array(x).reshape(1, np.array(x).shape[0]),
                                                          np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                          t, lr, rho, rnn.effective_momentum)

                    rnn.normalize()

            elif embedding_type == 'word2vec_init':
                x = seq_train[seq_num]
                x_rev = x[::-1]
                t = targets_train[seq_num]
                train_confidences = rnn.score(x, x_rev, t, lr, rho, rnn.effective_momentum)
                neg_classes = np.argmax(train_confidences)
                if neg_classes == targets_train[seq_num]:
                   neg_classes = np.argsort(train_confidences)[17]

                cost = rnn.train_step(x, x_rev, t, lr, rho, rnn.effective_momentum, neg_classes)

            else:
                x = seq_train[seq_num]
                x_rev = x[::-1]
                t = targets_train[seq_num]
                train_confidences = rnn.score(x, x_rev, t, lr, rho, rnn.effective_momentum)
                neg_classes = np.argmax(train_confidences)
                if neg_classes == targets_train[seq_num]:
                   neg_classes = np.argsort(train_confidences)[17]

                cost = rnn.train_step(x, x_rev, t, lr, rho, rnn.effective_momentum, neg_classes)

        # compute training error
        if verbose_print:
            print("compute training error")
        for seq_num in range(np.array(seq_train).shape[0]):
            if train_model == False:
                break
            # if embedding used, then work with word indices, else work with words
            if embedding_type == 'theano_word_embeddings':
                cwords = seq_train[seq_num]
                t = targets_train[seq_num]
                x_rev = cwords[::-1]
                y_pred = rnn.classify(cwords, x_rev)
                y_pred = y_pred[0]
                training_prections.append(y_pred)
                training_true_labels.append(t)

            elif embedding_type == 'word2vec_update':
                x = seq_train[seq_num]
                x_rev = x[::-1]
                t = targets_train[seq_num]
                if postag:
                    postag_idx = train_postag_seq[seq_num]
                    if x.shape[0] != len(postag_idx):
                        print('Error postag_idx Dimension mismatch line_num %s' %seq_num)
                        print('x.shape[0] %s' %x.shape[0])
                        print('len(postag_idx) %s' %len(postag_idx))
                        # postag_idx = np.array(postag_idx).reshape(np.array(postag_idx).shape[0], 1)

                if entity_class:
                    entity_class_idx = train_entity_class_seq[seq_num]
                    if x.shape[0] != len(entity_class_idx):
                        print('Error entity_class_idx Dimension mismatch line_num %s' %seq_num)
                        print('x.shape[0] %s' %x.shape[0])
                        print('len(entity_class_idx) %s' %len(entity_class_idx))
                if add_subtree_emb:
                    tree_data = train_tree_data[seq_num]
                    sdp_sent_aug_info_combined_sent_idx = \
                        train_dataset_sdp_sent_aug_info_combined_sent_idx[seq_num]

                    sdp_sent_aug_info_leaf_internal_x, sdp_sent_aug_info_computation_tree_matrix, sdp_sent_aug_info_tree_states_order = tree_rnn.gen_nn_inputs(tree_data, max_degree=overall_max_degree,
                                    only_leaves_have_vals=False,
                                    with_labels=False)
                    if context_window_usage == True:
                        #  print('s:', s)
                        cwords_leaf_interal = contextwin(sdp_sent_aug_info_leaf_internal_x,
                                                         win=context_win_size)
                        sdp_sent_aug_info_leaf_internal_x_cwords = cwords_leaf_interal
                    else:
                        sdp_sent_aug_info_leaf_internal_x_cwords = sdp_sent_aug_info_leaf_internal_x
                    sdp_sent_aug_info_output_tree_state_idx = []
                    for sdp_tok_idx, combined_sent_idx in enumerate(sdp_sent_aug_info_combined_sent_idx):
                        if combined_sent_idx==None:
                            sdp_sent_aug_info_output_tree_state_idx.append(0) # for entity tags use the embedding from the first leaf node in the leaf_computation order
                        else:
                            orig_sent_idx = combined_sent_idx
                            output_tree_state_idx = sdp_sent_aug_info_tree_states_order.index(orig_sent_idx)
                            sdp_sent_aug_info_output_tree_state_idx.append(output_tree_state_idx)

                if position_features == True and augment_entity_presence == True:
                    if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_train[seq_num], ent_pres_feat_train[seq_num],
                                              unnormalised_pos_feat_seq_train[seq_num],
                                              ent_pres_feat_train[seq_num])

                    elif pos_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_train[seq_num], ent_pres_feat_train[seq_num],
                                              unnormalised_pos_feat_seq_train[seq_num])
                    elif ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_train[seq_num], ent_pres_feat_train[seq_num],
                                              ent_pres_feat_train[seq_num])
                    else:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_train[seq_num], ent_pres_feat_train[seq_num])

                elif position_features == True:
                    if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_train[seq_num],
                                              unnormalised_pos_feat_seq_train[seq_num],
                                              ent_pres_feat_train[seq_num])
                    elif pos_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_train[seq_num],
                                              unnormalised_pos_feat_seq_train[seq_num])
                    elif ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_train[seq_num],
                                              ent_pres_feat_train[seq_num])
                    else:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_train[seq_num])

                elif augment_entity_presence == True:
                    if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_train[seq_num],
                                              unnormalised_pos_feat_seq_train[seq_num],
                                              ent_pres_feat_train[seq_num])

                    elif pos_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_train[seq_num],
                                              unnormalised_pos_feat_seq_train[seq_num])
                    elif ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_train[seq_num],
                                              ent_pres_feat_train[seq_num])
                    else:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_train[seq_num])
                else:
                    if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                          np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                          unnormalised_pos_feat_seq_train[seq_num],ent_pres_feat_train[seq_num])
                    elif pos_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                          np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                          unnormalised_pos_feat_seq_train[seq_num])
                    elif ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                          np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                          ent_pres_feat_train[seq_num])
                    else:
                        if context_window_usage == True:
                            if add_subtree_emb == True:
                                if postag == True and entity_class == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                          np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                          np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                          np.array(postag_idx), np.array(entity_class_idx))
                                elif postag == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                          np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                          np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                          np.array(postag_idx))
                                elif entity_class == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                          np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                          np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                          np.array(entity_class_idx))
                                else:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                          np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                          np.array(sdp_sent_aug_info_output_tree_state_idx))
                            else:
                                if postag == True and entity_class == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(postag_idx), np.array(entity_class_idx))
                                elif postag == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(postag_idx))
                                elif entity_class == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(entity_class_idx))
                                else:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev))
                        else:
                            if postag == True and entity_class == True:
                                y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                      np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                      np.array(postag_idx), np.array(entity_class_idx))

                            elif postag == True:
                                y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                      np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                      np.array(postag_idx))

                            elif entity_class == True:
                                y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                      np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                      np.array(entity_class_idx))

                            else:
                                y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                      np.array(x_rev).reshape(1, np.array(x_rev).shape[0]))


                y_pred = y_pred[0]
                training_prections.append(y_pred)
                training_true_labels.append(t)

            elif embedding_type == 'word2vec_init':
                x = seq_train[seq_num]
                x_rev = x[::-1]
                t = targets_train[seq_num]
                y_pred = rnn.classify(x, x_rev)
                y_pred = y_pred[0]
                training_prections.append(y_pred)
                training_true_labels.append(t)
            else:
                x = seq_train[seq_num]
                x_rev = x[::-1]
                t = targets_train[seq_num]
                y_pred = rnn.classify(x, x_rev)
                y_pred = y_pred[0]
                training_prections.append(y_pred)
                training_true_labels.append(t)

        if dev_eval_split == True:
            # compute development error
            for seq_num in range(np.array(seq_dev).shape[0]):
                # if embedding used, then work with word indices, else work with words
                if embedding_type == 'theano_word_embeddings':
                    cwords = seq_dev[seq_num]
                    t = targets_dev[seq_num]
                    x_rev = cwords[::-1]
                    y_pred = rnn.classify(cwords, x_rev)
                    y_pred = y_pred[0]
                    training_prections.append(y_pred)
                    training_true_labels.append(t)

                elif embedding_type == 'word2vec_update':
                    x = seq_dev[seq_num]
                    x_rev = x[::-1]
                    t = targets_dev[seq_num]
                    # supply entity presence and position features from outside: INPUT:[emb, d_e1, d_e2, p_e1, p_e2, ]
                    if position_features == True and augment_entity_presence == True:
                        if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                            y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  pos_feat_train[seq_num], ent_pres_feat_train[seq_num],
                                                  unnormalised_pos_feat_seq_dev[seq_num],
                                                  ent_pres_feat_train[seq_num])

                            y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  pos_feat_train[seq_num], ent_pres_feat_train[seq_num],
                                                  unnormalised_pos_feat_seq_dev[seq_num],
                                                  ent_pres_feat_train[seq_num])

                        elif pos_feat_embedding == True:
                            y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  pos_feat_train[seq_num], ent_pres_feat_train[seq_num],
                                                  unnormalised_pos_feat_seq_dev[seq_num])

                            y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  pos_feat_train[seq_num], ent_pres_feat_train[seq_num],
                                                  unnormalised_pos_feat_seq_dev[seq_num])

                        elif ent_pres_feat_embedding == True:
                            y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  pos_feat_train[seq_num], ent_pres_feat_train[seq_num],
                                                  ent_pres_feat_train[seq_num])

                            y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  pos_feat_train[seq_num], ent_pres_feat_train[seq_num],
                                                  ent_pres_feat_train[seq_num])
                        else:
                            y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  pos_feat_train[seq_num], ent_pres_feat_train[seq_num])

                            y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  pos_feat_train[seq_num], ent_pres_feat_train[seq_num])

                    elif position_features == True:
                        if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                            y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  pos_feat_train[seq_num],
                                                  unnormalised_pos_feat_seq_dev[seq_num],
                                                  ent_pres_feat_train[seq_num])

                            y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  pos_feat_train[seq_num],
                                                  unnormalised_pos_feat_seq_dev[seq_num],
                                                  ent_pres_feat_train[seq_num])

                        elif pos_feat_embedding == True:
                            y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  pos_feat_train[seq_num],
                                                  unnormalised_pos_feat_seq_dev[seq_num])

                            y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  pos_feat_train[seq_num],
                                                  unnormalised_pos_feat_seq_dev[seq_num])

                        elif ent_pres_feat_embedding == True:
                            y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  pos_feat_train[seq_num],
                                                  ent_pres_feat_train[seq_num])

                            y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  pos_feat_train[seq_num],
                                                  ent_pres_feat_train[seq_num])

                        else:
                            y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  pos_feat_train[seq_num])

                            y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  pos_feat_train[seq_num])


                    elif augment_entity_presence == True:
                        if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                            y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  ent_pres_feat_train[seq_num],
                                                  unnormalised_pos_feat_seq_dev[seq_num],
                                                  ent_pres_feat_train[seq_num])

                            y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  ent_pres_feat_train[seq_num],
                                                  unnormalised_pos_feat_seq_dev[seq_num],
                                                  ent_pres_feat_train[seq_num])

                        elif pos_feat_embedding == True:
                            y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  ent_pres_feat_train[seq_num],
                                                  unnormalised_pos_feat_seq_dev[seq_num])

                            y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  ent_pres_feat_train[seq_num],
                                                  unnormalised_pos_feat_seq_dev[seq_num])
                        elif ent_pres_feat_embedding == True:
                            y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  ent_pres_feat_train[seq_num],
                                                  ent_pres_feat_train[seq_num])

                            y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  ent_pres_feat_train[seq_num],
                                                  ent_pres_feat_train[seq_num])

                        else:
                            y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  ent_pres_feat_train[seq_num])
                            y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                  ent_pres_feat_train[seq_num])

                    else:
                        if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                            y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              unnormalised_pos_feat_seq_dev[seq_num],ent_pres_feat_train[seq_num])

                            y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              unnormalised_pos_feat_seq_dev[seq_num],ent_pres_feat_train[seq_num])
                        elif pos_feat_embedding == True:
                            y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              unnormalised_pos_feat_seq_dev[seq_num])

                            y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              unnormalised_pos_feat_seq_dev[seq_num])

                        elif ent_pres_feat_embedding == True:
                            y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_train[seq_num])

                            y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_train[seq_num])
                        else:
                            if context_window_usage == True:
                                y_pred = rnn.classify(np.array(x), np.array(x_rev))
                                y_pred_prob = rnn.predict_prob(np.array(x), np.array(x_rev))
                            else:
                                y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]))

                                y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                  np.array(x_rev).reshape(1, np.array(x_rev).shape[0]))

                    y_pred = y_pred[0]
                    dev_pred_prob.append(str(y_pred_prob[:, 1][0]))
                    dev_predictions.append(y_pred)
                    dev_decision.append(str(y_pred))
                    dev_true_labels.append(t)

                elif embedding_type == 'word2vec_init':
                    x = seq_dev[seq_num]
                    x_rev = x[::-1]
                    t = targets_dev[seq_num]
                    y_pred = rnn.classify(x, x_rev)
                    y_pred_prob = rnn.predict_prob(x, x_rev)
                    y_pred = y_pred[0]
                    dev_pred_prob.append(str(y_pred_prob[:, 1][0]))
                    dev_predictions.append(y_pred)
                    dev_decision.append(str(y_pred))
                    dev_true_labels.append(t)

                else:
                    x = seq_dev[seq_num]
                    x_rev = x[::-1]
                    t = targets_dev[seq_num]
                    y_pred = rnn.classify(x, x_rev)
                    y_pred_prob = rnn.predict_prob(x, x_rev)
                    y_pred = y_pred[0]
                    dev_pred_prob.append(str(y_pred_prob[:, 1][0]))
                    dev_predictions.append(y_pred)
                    dev_decision.append(str(y_pred))
                    dev_true_labels.append(t)

        # compute dev error
        if verbose_print:
            print("compute dev error")
        for seq_num in range(np.array(seq_dev).shape[0]):
            # if embedding used, then work with word indices, else work with words
            if embedding_type == 'theano_word_embeddings':
                cwords = seq_dev[seq_num]
                x_rev = cwords[::-1]
                t = targets_dev[seq_num]
                y_pred = rnn.classify(cwords, x_rev)
                y_pred_prob = rnn.predict_prob(cwords, x_rev)
                y_pred = y_pred[0]
                dev_predictions.append(y_pred)
                dev_decision.append(str(y_pred))
                dev_true_labels.append(t)
                dev_pred_prob.append(str(y_pred_prob[:, 1][0]))

            elif embedding_type == 'word2vec_update':
                x = seq_dev[seq_num]
                x_rev = x[::-1]
                t = targets_dev[seq_num]

                if postag:
                    postag_idx = dev_postag_seq[seq_num]
                    if np.array(x).shape[0] != len(postag_idx):
                        print('Error postag_idx Dimension mismatch line_num %s' %seq_num)
                        print('x.shape[0] %s' %np.array(x).shape[0])
                        print('len(postag_idx) %s' %len(postag_idx))
                if entity_class:
                    entity_class_idx = dev_entity_class_seq[seq_num]
                    if np.array(x).shape[0] != len(entity_class_idx):
                        print('Error entity_class_idx Dimension mismatch line_num %s' %seq_num)
                        print('x.shape[0] %s' %np.array(x).shape[0])
                        print('len(entity_class_idx) %s' %len(entity_class_idx))

                if add_subtree_emb:
                    tree_data = dev_tree_data[seq_num]
                    sdp_sent_aug_info_combined_sent_idx = \
                        dev_dataset_sdp_sent_aug_info_combined_sent_idx[seq_num]

                    sdp_sent_aug_info_leaf_internal_x, sdp_sent_aug_info_computation_tree_matrix, sdp_sent_aug_info_tree_states_order = tree_rnn.gen_nn_inputs(tree_data, max_degree=overall_max_degree,
                                                                                                                                                               only_leaves_have_vals=False,
                                                                                                                                                               with_labels=False)
                    if context_window_usage == True:
                        cwords_leaf_interal = contextwin(sdp_sent_aug_info_leaf_internal_x,
                                                         win=context_win_size)
                        sdp_sent_aug_info_leaf_internal_x_cwords = cwords_leaf_interal
                    else:
                        sdp_sent_aug_info_leaf_internal_x_cwords = sdp_sent_aug_info_leaf_internal_x
                    sdp_sent_aug_info_output_tree_state_idx = []
                    for sdp_tok_idx, combined_sent_idx in enumerate(sdp_sent_aug_info_combined_sent_idx):
                        if combined_sent_idx==None:
                            # sdp_sent_aug_info_output_tree_state_idx.append(None)
                            sdp_sent_aug_info_output_tree_state_idx.append(0) # for entity tags use the embedding from the first leaf node in the leaf_computation order
                        else:
                            orig_sent_idx = combined_sent_idx
                            output_tree_state_idx = sdp_sent_aug_info_tree_states_order.index(orig_sent_idx)
                            sdp_sent_aug_info_output_tree_state_idx.append(output_tree_state_idx)

                if position_features == True and augment_entity_presence == True:
                    if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_dev[seq_num], ent_pres_feat_dev[seq_num],
                                              unnormalised_pos_feat_seq_dev[seq_num],
                                              ent_pres_feat_dev[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                       np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                       pos_feat_dev[seq_num], ent_pres_feat_dev[seq_num],
                                                       unnormalised_pos_feat_seq_dev[seq_num],
                                                       ent_pres_feat_dev[seq_num])

                    elif pos_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_dev[seq_num], ent_pres_feat_dev[seq_num],
                                              unnormalised_pos_feat_seq_dev[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                       np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                       pos_feat_dev[seq_num], ent_pres_feat_dev[seq_num],
                                                       unnormalised_pos_feat_seq_dev[seq_num])

                    elif ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_dev[seq_num], ent_pres_feat_dev[seq_num],
                                              ent_pres_feat_dev[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                       np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                       pos_feat_dev[seq_num], ent_pres_feat_dev[seq_num],
                                                       ent_pres_feat_dev[seq_num])
                    else:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_dev[seq_num], ent_pres_feat_dev[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                       np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                       pos_feat_dev[seq_num], ent_pres_feat_dev[seq_num])

                elif position_features == True:
                    if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_dev[seq_num],
                                              unnormalised_pos_feat_seq_dev[seq_num],
                                              ent_pres_feat_dev[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                       np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                       pos_feat_dev[seq_num],
                                                       unnormalised_pos_feat_seq_dev[seq_num],
                                                       ent_pres_feat_dev[seq_num])

                    elif pos_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_dev[seq_num],
                                              unnormalised_pos_feat_seq_dev[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                       np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                       pos_feat_dev[seq_num],
                                                       unnormalised_pos_feat_seq_dev[seq_num])

                    elif ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_dev[seq_num],
                                              ent_pres_feat_dev[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                       np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                       pos_feat_dev[seq_num],
                                                       ent_pres_feat_dev[seq_num])
                    else:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_dev[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                       np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                       pos_feat_dev[seq_num])

                elif augment_entity_presence == True:
                    if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_dev[seq_num],
                                              unnormalised_pos_feat_seq_dev[seq_num],
                                              ent_pres_feat_dev[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                       np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                       ent_pres_feat_dev[seq_num],
                                                       unnormalised_pos_feat_seq_dev[seq_num],
                                                       ent_pres_feat_dev[seq_num])

                    elif pos_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_dev[seq_num],
                                              unnormalised_pos_feat_seq_dev[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                       np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                       ent_pres_feat_dev[seq_num],
                                                       unnormalised_pos_feat_seq_dev[seq_num])

                    elif ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_dev[seq_num],
                                              ent_pres_feat_dev[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                       np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                       ent_pres_feat_dev[seq_num],
                                                       ent_pres_feat_dev[seq_num])

                    else:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_dev[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                       np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                       ent_pres_feat_dev[seq_num])

                else:
                    if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              unnormalised_pos_feat_seq_dev[seq_num],
                                              ent_pres_feat_dev[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                       np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                       unnormalised_pos_feat_seq_dev[seq_num],
                                                       ent_pres_feat_dev[seq_num])

                    elif pos_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              unnormalised_pos_feat_seq_dev[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                       np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                       unnormalised_pos_feat_seq_dev[seq_num])

                    elif ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_dev[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                       np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                       ent_pres_feat_dev[seq_num])
                    else:
                        if context_window_usage == True:
                            if add_subtree_emb == True:
                                if postag == True and entity_class == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                          np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                          np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                          np.array(postag_idx),
                                                          np.array(entity_class_idx))[0]
                                    y_pred_prob = rnn.predict_prob(np.array(x), np.array(x_rev),
                                                                   np.array(sdp_sent_aug_info_leaf_internal_x),
                                                                   np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                                   np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                                   np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                                   np.array(postag_idx),
                                                                   np.array(entity_class_idx))
                                elif postag == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                          np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                          np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                          np.array(postag_idx))[0]
                                    y_pred_prob = rnn.predict_prob(np.array(x), np.array(x_rev),
                                                                   np.array(sdp_sent_aug_info_leaf_internal_x),
                                                                   np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                                   np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                                   np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                                   np.array(postag_idx))
                                elif entity_class == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                          np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                          np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                          np.array(entity_class_idx))[0]
                                    y_pred_prob = rnn.predict_prob(np.array(x), np.array(x_rev),
                                                                   np.array(sdp_sent_aug_info_leaf_internal_x),
                                                                   np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                                   np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                                   np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                                   np.array(entity_class_idx))
                                else:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                          np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                          np.array(sdp_sent_aug_info_output_tree_state_idx))[0]
                                    y_pred_prob = rnn.predict_prob(np.array(x), np.array(x_rev),
                                                                   np.array(sdp_sent_aug_info_leaf_internal_x),
                                                                   np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                                   np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                                   np.array(sdp_sent_aug_info_output_tree_state_idx))
                            else:
                                if postag == True and entity_class == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(postag_idx),
                                                          np.array(entity_class_idx))[0]
                                    y_pred_prob = rnn.predict_prob(np.array(x), np.array(x_rev),
                                                                   np.array(postag_idx),
                                                                   np.array(entity_class_idx))
                                elif postag == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(postag_idx))[0]
                                    y_pred_prob = rnn.predict_prob(np.array(x), np.array(x_rev),
                                                                   np.array(postag_idx))
                                elif entity_class == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(entity_class_idx))[0]
                                    y_pred_prob = rnn.predict_prob(np.array(x), np.array(x_rev),
                                                                   np.array(entity_class_idx))
                                else:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev))[0]
                                    y_pred_prob = rnn.predict_prob(np.array(x), np.array(x_rev))
                        else:
                            if postag == True and entity_class == True:
                                y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                      np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                      np.array(postag_idx),
                                                      np.array(entity_class_idx))[0]

                                y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                               np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                               np.array(postag_idx),
                                                               np.array(entity_class_idx))
                            elif postag == True:
                                y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                      np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                      np.array(postag_idx))[0]

                                y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                               np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                               np.array(postag_idx))

                            elif entity_class == True:
                                y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                      np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                      np.array(entity_class_idx))[0]

                                y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                               np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                               np.array(entity_class_idx))

                            else:
                                y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                      np.array(x_rev).reshape(1, np.array(x_rev).shape[0]))[0]

                                y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                           np.array(x_rev).reshape(1, np.array(x_rev).shape[0]))

                dev_decision.append(str(y_pred))
                dev_predictions.append(y_pred)
                dev_pred_prob.append(str(y_pred_prob[:, y_pred][0]))
                dev_true_labels.append(t)

            elif embedding_type == 'word2vec_init':
                x = seq_dev[seq_num]
                x_rev = x[::-1]
                t = targets_dev[seq_num]
                y_pred = rnn.classify(x, x_rev)
                y_pred_prob = rnn.predict_prob(x, x_rev)
                y_pred = y_pred[0]
                dev_pred_prob.append(str(y_pred_prob[:, 1][0]))
                dev_predictions.append(y_pred)
                dev_decision.append(str(y_pred))
                dev_true_labels.append(t)
            else:
                x = seq_dev[seq_num]
                x_rev = x[::-1]
                t = targets_dev[seq_num]
                y_pred = rnn.classify(x, x_rev)
                y_pred = y_pred[0]
                dev_predictions.append(y_pred)
                dev_true_labels.append(t)
                dev_decision.append(str(y_pred))
                y_pred_prob = rnn.predict_prob(x, x_rev)
                dev_pred_prob.append(str(y_pred_prob[:, 1][0]))

        # compute testing error
        if verbose_print:
            print('computing testing error')
        for seq_num in range(np.array(seq_test).shape[0]):

            # if embedding used, then work with word indices, else work with words
            if embedding_type == 'theano_word_embeddings':
                cwords = seq_test[seq_num]
                x_rev = cwords[::-1]
                t = targets_test[seq_num]
                y_pred = rnn.classify(cwords, x_rev)
                y_pred_prob = rnn.predict_prob(cwords, x_rev)
                y_pred = y_pred[0]
                testing_predictions.append(y_pred)
                testing_decision.append(str(y_pred))
                testing_true_labels.append(t)
                testing_pred_prob.append(str(y_pred_prob[:, 1][0]))

            elif embedding_type == 'word2vec_update':
                x = seq_test[seq_num]
                x_rev = x[::-1]
                t = targets_test[seq_num]
                # supply entity presence and position features from outside: INPUT:[emb, d_e1, d_e2, p_e1, p_e2, ]
                if postag:
                    postag_idx = test_postag_seq[seq_num]
                    if np.array(x).shape[0] != len(postag_idx):
                        print('Error postag_idx Dimension mismatch line_num %s' %seq_num)
                        print('x.shape[0] %s' %np.array(x).shape[0])
                        print('len(postag_idx) %s' %len(postag_idx))
                if entity_class:
                    entity_class_idx = test_entity_class_seq[seq_num]
                    if np.array(x).shape[0] != len(entity_class_idx):
                        print('Error entity_class_idx Dimension mismatch line_num %s' %seq_num)
                        print('x.shape[0] %s' %np.array(x).shape[0])
                        print('len(entity_class_idx) %s' %len(entity_class_idx))
                if add_subtree_emb:
                    tree_data = test_tree_data[seq_num]
                    sdp_sent_aug_info_combined_sent_idx = \
                        test_dataset_sdp_sent_aug_info_combined_sent_idx[seq_num]

                    sdp_sent_aug_info_leaf_internal_x, sdp_sent_aug_info_computation_tree_matrix, sdp_sent_aug_info_tree_states_order = tree_rnn.gen_nn_inputs(tree_data, max_degree=overall_max_degree,
                                                     only_leaves_have_vals=False,
                                                     with_labels=False)
                    if context_window_usage == True:
                        cwords_leaf_interal = contextwin(sdp_sent_aug_info_leaf_internal_x,
                                                         win=context_win_size)
                        sdp_sent_aug_info_leaf_internal_x_cwords = cwords_leaf_interal
                    else:
                        sdp_sent_aug_info_leaf_internal_x_cwords = sdp_sent_aug_info_leaf_internal_x
                    sdp_sent_aug_info_output_tree_state_idx = []
                    for sdp_tok_idx, combined_sent_idx in enumerate(sdp_sent_aug_info_combined_sent_idx):
                        if combined_sent_idx==None:
                            # sdp_sent_aug_info_output_tree_state_idx.append(None)
                            sdp_sent_aug_info_output_tree_state_idx.append(0) # for entity tags use the embedding from the first leaf node in the leaf_computation order
                        else:
                            orig_sent_idx = combined_sent_idx
                            output_tree_state_idx = sdp_sent_aug_info_tree_states_order.index(orig_sent_idx)
                            sdp_sent_aug_info_output_tree_state_idx.append(output_tree_state_idx)

                if position_features == True and augment_entity_presence == True:
                    if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_test[seq_num], ent_pres_feat_test[seq_num],
                                              unnormalised_pos_feat_seq_test[seq_num],
                                              ent_pres_feat_test[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_test[seq_num], ent_pres_feat_test[seq_num],
                                              unnormalised_pos_feat_seq_test[seq_num],
                                              ent_pres_feat_test[seq_num])

                    elif pos_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_test[seq_num], ent_pres_feat_test[seq_num],
                                              unnormalised_pos_feat_seq_test[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_test[seq_num], ent_pres_feat_test[seq_num],
                                              unnormalised_pos_feat_seq_test[seq_num])

                    elif ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_test[seq_num], ent_pres_feat_test[seq_num],
                                              ent_pres_feat_test[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_test[seq_num], ent_pres_feat_test[seq_num],
                                              ent_pres_feat_test[seq_num])
                    else:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_test[seq_num], ent_pres_feat_test[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_test[seq_num], ent_pres_feat_test[seq_num])

                elif position_features == True:
                    if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_test[seq_num],
                                              unnormalised_pos_feat_seq_test[seq_num],
                                              ent_pres_feat_test[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_test[seq_num],
                                              unnormalised_pos_feat_seq_test[seq_num],
                                              ent_pres_feat_test[seq_num])

                    elif pos_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_test[seq_num],
                                              unnormalised_pos_feat_seq_test[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_test[seq_num],
                                              unnormalised_pos_feat_seq_test[seq_num])

                    elif ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_test[seq_num],
                                              ent_pres_feat_test[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_test[seq_num],
                                              ent_pres_feat_test[seq_num])
                    else:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_test[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              pos_feat_test[seq_num])

                elif augment_entity_presence == True:
                    if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_test[seq_num],
                                              unnormalised_pos_feat_seq_test[seq_num],
                                              ent_pres_feat_test[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_test[seq_num],
                                              unnormalised_pos_feat_seq_test[seq_num],
                                              ent_pres_feat_test[seq_num])

                    elif pos_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                            ent_pres_feat_test[seq_num],
                                              unnormalised_pos_feat_seq_test[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                            ent_pres_feat_test[seq_num],
                                              unnormalised_pos_feat_seq_test[seq_num])

                    elif ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_test[seq_num],
                                              ent_pres_feat_test[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_test[seq_num],
                                              ent_pres_feat_test[seq_num])

                    else:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_test[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_test[seq_num])

                else:
                    if pos_feat_embedding == True and ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              unnormalised_pos_feat_seq_test[seq_num],
                                              ent_pres_feat_test[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              unnormalised_pos_feat_seq_test[seq_num],
                                              ent_pres_feat_test[seq_num])

                    elif pos_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              unnormalised_pos_feat_seq_test[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              unnormalised_pos_feat_seq_test[seq_num])

                    elif ent_pres_feat_embedding == True:
                        y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_test[seq_num])

                        y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                              np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                              ent_pres_feat_test[seq_num])
                    else:
                        if context_window_usage == True:
                            if add_subtree_emb == True:
                                if postag == True and entity_class == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                          np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                          np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                          np.array(postag_idx),
                                                          np.array(entity_class_idx))[0]
                                    y_pred_prob = rnn.predict_prob(np.array(x), np.array(x_rev),
                                                                   np.array(sdp_sent_aug_info_leaf_internal_x),
                                                                   np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                                   np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                                   np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                                   np.array(postag_idx),
                                                                   np.array(entity_class_idx))
                                elif postag == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                          np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                          np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                          np.array(postag_idx))[0]
                                    y_pred_prob = rnn.predict_prob(np.array(x), np.array(x_rev),
                                                                   np.array(sdp_sent_aug_info_leaf_internal_x),
                                                                   np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                                   np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                                   np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                                   np.array(postag_idx))
                                elif entity_class == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                          np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                          np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                          np.array(entity_class_idx))[0]
                                    y_pred_prob = rnn.predict_prob(np.array(x), np.array(x_rev),
                                                                   np.array(sdp_sent_aug_info_leaf_internal_x),
                                                                   np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                                   np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                                   np.array(sdp_sent_aug_info_output_tree_state_idx),
                                                                   np.array(entity_class_idx))
                                else:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x),
                                                          np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                          np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                          np.array(sdp_sent_aug_info_output_tree_state_idx))[0]
                                    y_pred_prob = rnn.predict_prob(np.array(x), np.array(x_rev),
                                                                   np.array(sdp_sent_aug_info_leaf_internal_x),
                                                                   np.array(sdp_sent_aug_info_leaf_internal_x_cwords),
                                                                   np.array(sdp_sent_aug_info_computation_tree_matrix[:, :-1]),
                                                                   np.array(sdp_sent_aug_info_output_tree_state_idx))
                            else:
                                if postag == True and entity_class == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(postag_idx),
                                                          np.array(entity_class_idx))[0]
                                    y_pred_prob = rnn.predict_prob(np.array(x), np.array(x_rev),
                                                                   np.array(postag_idx),
                                                                   np.array(entity_class_idx))
                                elif postag == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(postag_idx))[0]
                                    y_pred_prob = rnn.predict_prob(np.array(x), np.array(x_rev),
                                                                   np.array(postag_idx))
                                elif entity_class == True:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev),
                                                          np.array(entity_class_idx))[0]
                                    y_pred_prob = rnn.predict_prob(np.array(x), np.array(x_rev),
                                                                   np.array(entity_class_idx))
                                else:
                                    y_pred = rnn.classify(np.array(x), np.array(x_rev))[0]
                                    y_pred_prob = rnn.predict_prob(np.array(x), np.array(x_rev))
                        else:
                            if postag == True and entity_class == True:
                                y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                      np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                      np.array(postag_idx),
                                                      np.array(entity_class_idx))[0]

                                y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                               np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                               np.array(postag_idx),
                                                               np.array(entity_class_idx))
                            elif postag == True:
                                y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                      np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                      np.array(postag_idx))[0]

                                y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                               np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                               np.array(postag_idx))

                            elif entity_class == True:
                                y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                      np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                      np.array(entity_class_idx))[0]

                                y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                               np.array(x_rev).reshape(1, np.array(x_rev).shape[0]),
                                                               np.array(entity_class_idx))

                            else:
                                y_pred = rnn.classify(np.array(x).reshape(1, np.array(x).shape[0]),
                                                      np.array(x_rev).reshape(1, np.array(x_rev).shape[0]))[0]

                                y_pred_prob = rnn.predict_prob(np.array(x).reshape(1, np.array(x).shape[0]),
                                                               np.array(x_rev).reshape(1, np.array(x_rev).shape[0]))

                testing_decision.append(str(y_pred))
                testing_predictions.append(y_pred)
                testing_pred_prob.append(str(y_pred_prob[:, y_pred][0]))
                testing_true_labels.append(t)

            elif embedding_type == 'word2vec_init':
                x = seq_test[seq_num]
                x_rev = x[::-1]
                t = targets_test[seq_num]
                y_pred = rnn.classify(x, x_rev)
                y_pred_prob = rnn.predict_prob(x, x_rev)
                y_pred = y_pred[0]
                testing_pred_prob.append(str(y_pred_prob[:, 1][0]))
                testing_predictions.append(y_pred)
                testing_decision.append(str(y_pred))
                testing_true_labels.append(t)
            else:
                x = seq_test[seq_num]
                x_rev = x[::-1]
                t = targets_test[seq_num]
                y_pred = rnn.classify(x, x_rev)
                y_pred = y_pred[0]
                testing_predictions.append(y_pred)
                testing_true_labels.append(t)
                testing_decision.append(str(y_pred))
                y_pred_prob = rnn.predict_prob(x, x_rev)
                testing_pred_prob.append(str(y_pred_prob[:, 1][0]))

        if train_model == False:
            training_f1_score = 0.0
        if train_model:
            training_P_score, training_R_score, training_f1_score, _ = precision_recall_fscore_support(training_true_labels,
                                                                                                       training_prections,
                                                                                                       average='binary'
                                                                                                        )


        dev_P_score, dev_R_score, dev_f1_score, _ = precision_recall_fscore_support(dev_true_labels,
                                                                                    dev_predictions,
                                                                                    average='binary'
                                                                                    )

        if verbose_print:
            print "dev_true_labels"
            print dev_true_labels
            print "dev_predictions"
            print dev_predictions
            print "dev_pred_prob"
            print dev_pred_prob
            print "dev_decision"
            print dev_decision

        testing_P_score, testing_R_score, testing_f1_score, _ = precision_recall_fscore_support(testing_true_labels,
                                                                                                testing_predictions,
                                                                                                average='binary'
                                                                                                 )

        if verbose_print:
            print "testing_true_labels"
            print testing_true_labels
            print "testing_predictions"
            print testing_predictions
            print "testing_pred_prob"
            print testing_pred_prob
            print "testing_decision"
            print testing_decision

        dev_f1_score = getMacroFScore(dev_predictions, dev_true_labels)
        testing_f1_score = getMacroFScore(testing_predictions, testing_true_labels)
        if dev_f1_score > best_dev_f1:
            best_dev_f1 = dev_f1_score
            test_f1_for_best_dev_f1 = testing_f1_score
            stop_incc = 0

            if savemodel == True:
                rnn.save(folder)

            if verbose == True:
                print("NEW BEST: iter {0} lr: {1} time: {2} train cost: {3}  train F1: {4} dev F1: {5} test F1: {6}"
                      .format(i, lr, time.time()-tic, np.sqrt(np.round(cost, 3)),
                              np.round(training_f1_score, 3),
                              np.round(dev_f1_score, 3),
                              np.round(test_f1_for_best_dev_f1, 3)))

                f.write('\nNEW BEST: iter '+str(i)+' lr:'+str(lr)+' train cost: '+str(np.sqrt(np.round(cost, 3)))+
                        ' train F1: '+str(np.round(training_f1_score, 3))+
                        ' dev F1: ' +str(np.round(dev_f1_score, 3))+
                        ' test F1: ' + str(np.round(test_f1_for_best_dev_f1, 3)))

            f_prob_dev = open(log_file_name_prob_dev, 'w')
            f_prob_dev.write("\n".join(dev_pred_prob))
            f_prob = open(log_file_name_prob, 'w')
            f_prob.write("\n".join(testing_pred_prob))
            f_prob_dev.close()
            f_prob.close()

            f_prob_dev_scores = open(log_file_name_scores_dev, 'w')
            f_prob_scores = open(log_file_name_scores, 'w')
            for td in range(len(dev_decision)):
                f_prob_dev_scores.write(str(dev_decision[td])+'\n')

            for td in range(len(testing_decision)):
                f_prob_scores.write(str(testing_decision[td])+'\n')
            f_prob_dev_scores.close()
            f_prob_scores.close()

            best_epoch = i
        else:
            print(" iter {0} lr: {1} time: {2} train cost: {3}  train F1: {4} dev F1: {5} test F1: {6}"
                  .format(i, lr, time.time()-tic, np.sqrt(np.round(cost, 3)),
                          np.round(training_f1_score, 3),
                          np.round(dev_f1_score, 3),
                          np.round(testing_f1_score, 3)))

            f.write('\n iter '+str(i)+' lr:'+str(lr)+' train cost: '+str(np.sqrt(np.round(cost, 3)))+
                        ' train F1: '+str(np.round(training_f1_score, 3))+
                        ' dev F1: ' +str(np.round(dev_f1_score, 3))+
                        ' test F1: ' + str(np.round(testing_f1_score, 3)))
            stop_incc+=1

        if learning_rate_decay_strategy == 1 and i%update_learning_rate_after_n_epochs==0 and i!=0:
            lr *= learning_rate_decay

        if learning_rate_decay_strategy == 2 and i!=0:
            lr /= (i+1)

        if learning_rate_decay_strategy == 3 and i>3 and lr > 1e-5:
            # learning rate decay if no improvement in 10 epochs
            if learning_rate_decay and abs(best_epoch-i) >=update_learning_rate_after_n_epochs:
                lr *= learning_rate_decay
                if stop_incc >= 100:
                    return best_dev_f1, test_f1_for_best_dev_f1, saveto

        if learning_rate_decay_strategy == 4:
            if i > 3 and lr > 1e-6:
                lr = float(lr)/2

        if learning_rate_decay_strategy == 5:
            if abs(best_epoch-i) >=update_learning_rate_after_n_epochs:
                # reload RNN model to the best model and optimise by learning rate from the last best
                print('load model')
                rnn.load(folder)
                if lr > 1e-6:
                    lr = float(lr)/2
                else:
                    lr *= 100

        if learning_rate_decay_strategy == 6:
            if abs(best_epoch-i) >=update_learning_rate_after_n_epochs:
                # reload RNN model to the best model and optimise by learning rate from the last best
                print('load model')
                rnn.load(folder)
                if lr > 1e-5:
                    lr = float(lr)/2
                elif lr > 1e-6:
                    lr = float(lr) * 0.9
                else:
                    lr *= 100

        #open and close file after every 10 epochs
        if(i %10==0):
            f.close()
            open_flag = False

        if open_flag == False:
            f = open(log_file_name, 'a')
            open_flag=True

        if train_model == False:
            break

    return best_dev_f1, test_f1_for_best_dev_f1, saveto