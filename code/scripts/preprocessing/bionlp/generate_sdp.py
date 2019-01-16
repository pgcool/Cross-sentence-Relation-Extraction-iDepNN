
from __future__ import print_function
import os
import load_save_pkl
from collections import defaultdict
import networkx as nx
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tokenize import RegexpTokenizer
import sys

reload(sys)
sys.setdefaultencoding('utf8')

# Input: data ie training sentences
# Output: Sentence id: [dep_parsed_doc_list, sdp_dep_parsed_list]
training_data_path = './BB2016_test_final/'
dataset_type = "k_all_test_final"
training_data_file = os.path.join(training_data_path, "BB2016_" + dataset_type + "_full_sent.txt")
training_data_svm_dep_sdp_file = os.path.join(training_data_path, "BB2016_" + dataset_type + "_train_data_svm_dep_sdp.pkl")

# Packages declaration
java_path = "/analytics/data/rajaram_s/thesis_stuff/software/jdk1.8.0_131/bin/java"
os.environ['JAVAHOME'] = java_path
path_to_jar = '/analytics/data/rajaram_s/thesis_stuff/software/stanford-parser-full-2016-10-31/stanford-parser.jar'
path_to_models_jar = '/analytics/data/rajaram_s/thesis_stuff/software/stanford-parser-full-2016-10-31/stanford-parser-3.7.0-models.jar'
dep_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
tokenizer = RegexpTokenizer(r'\w+')

def get_entity_index(entity, token_dict, entity_priority):
    entity_tokenized = word_tokenize(entity)
    entity_idx_list = []
    for entity_token in entity_tokenized:
        entity_idx_list.append([item[0] for item in token_dict.items() if item[1] == entity_token])

    if not any(entity_idx_list):
        entity_idx_list = []
        entity_idx_list.append([item[0] for item in token_dict.items() if entity_token in item[1]])

    if entity_priority:
        try:
            entity_index = entity_idx_list[0][0]
        except:
            entity_index = entity_idx_list[1][0]
        for entity_idx_iterator in range(len(entity_idx_list)):
            if entity_idx_iterator > 2:
                break
            if len(entity_idx_list[entity_idx_iterator]) == 1:
                entity_index = entity_idx_list[entity_idx_iterator][0] - entity_idx_iterator
    else:
        try:
            entity_index = entity_idx_list[0][-1]
        except:
            entity_index = entity_idx_list[1][-1]
        for entity_idx_iterator in range(len(entity_idx_list)):
            if entity_idx_iterator > 2:
                break
            if len(entity_idx_list[entity_idx_iterator]) == 1:
                entity_index = entity_idx_list[entity_idx_iterator][0] - entity_idx_iterator
    return entity_index

def generate_sdp(document, e1, e2, e1_entity_priority):
    dep_parsed_doc_list = []
    training_sent_tokenized = []
    training_sent_tokenized_dict = {}
    training_sent_dep_parsed_dict = {}
    edges = []
    doc_sent_tokenize_list = sent_tokenize(document.replace("\n"," ").strip())
    for sent in doc_sent_tokenize_list:
        tokenized_sent = word_tokenize(sent)
        result = dep_parser.raw_parse(sent)
        output = next(result)
        dep_parsed_sent_list = []
        for i, node in sorted(output.nodes.items()):
            if (node['tag'] != 'TOP'):
                dep_parsed_list = ''.join('{address},{word},{tag},{head},{rel}'.format(i=i, **node)).split(",")
                dep_parsed_sent_list.append(dep_parsed_list)
        dep_parsed_doc_list.append(dep_parsed_sent_list)
        training_sent_tokenized.append(tokenized_sent)

    sentence_num = 0
    sentences_offset = 0
    root_tokens = []
    num_sentences = len(training_sent_tokenized)
    for sentence_idx in range(num_sentences):
        current_sentence_dep_list = dep_parsed_doc_list[sentence_idx]
        current_sentence_tokenized_list = training_sent_tokenized[sentence_idx]
        sentence_len = len(current_sentence_tokenized_list)
        for token_idx in range(sentence_len):
            try:
                current_token_dep_list = current_sentence_dep_list[token_idx]
                dependent_word_index = int(current_token_dep_list[0]) + sentences_offset
                training_sent_dep_parsed_dict[dependent_word_index] = current_token_dep_list
                if current_token_dep_list[4] == "root" and num_sentences > 1:
                    root_tokens.append(dependent_word_index)
                else:
                    edges.append((dependent_word_index, int(current_token_dep_list[3]) + sentences_offset))
            except:
                pass
            training_sent_tokenized_dict[token_idx + 1 + sentences_offset] = training_sent_tokenized[sentence_idx][token_idx]

        sentence_num += 1
        sentences_offset += sentence_len
    if num_sentences > 1:
        edges.extend([(x, root_tokens[i+1]) for i,x in enumerate(root_tokens) if i < len(root_tokens)-1])
    graph = nx.Graph(edges)
    e1_idx = get_entity_index(e1, training_sent_tokenized_dict, e1_entity_priority)
    e1_entity_priority= not e1_entity_priority
    e2_idx = get_entity_index(e2, training_sent_tokenized_dict, e1_entity_priority)
    try:
        path = nx.shortest_path(graph, source=e1_idx, target=e2_idx)
    except:
        return None, None

    sdp_dep_parsed_list = []
    for item in path:
        sdp_dep_parsed_list.append([item,training_sent_dep_parsed_dict[item]])
    return dep_parsed_doc_list, sdp_dep_parsed_list


def prepare_svm_dep_sdp(training_data_file):
    trainin_data_svm_dep_sdp = defaultdict(list)
    for i, line in enumerate(open(training_data_file, 'rb')):
        e1_entity_priority = True
        training_example_list = line.split("::")
        e1 = training_example_list[3]
        e2 = training_example_list[4]
        sent = training_example_list[5]
        entity_tags = ['<e1>','</e1>','<e2>','</e2>']
        if not all(x in sent for x in entity_tags):
            trainin_data_svm_dep_sdp[i] = [None, None]
        else:
            if sent.find('<e1>') > sent.find('<e2>'):
                e1_entity_priority = False
            for tag in entity_tags:
                sent = sent.replace(tag,"")
            dep_parsed_doc_list, sdp_dep_parsed_list  = generate_sdp(sent, e1, e2, e1_entity_priority)
            trainin_data_svm_dep_sdp[i] = [dep_parsed_doc_list, sdp_dep_parsed_list]
    return trainin_data_svm_dep_sdp
trainin_data_svm_dep_sdp = prepare_svm_dep_sdp(training_data_file)
load_save_pkl.save_as_pkl(trainin_data_svm_dep_sdp,training_data_svm_dep_sdp_file)