__doc__ = """Utilities for loading language datasets.

Basically porting http://github.com/stanfordnlp/treelstm/tree/master/util to Python.

"""

import tree_rnn

import numpy as np
import os


def combine_parents_list(parents_list):
    data_sample_root_list = []
    data_sample_sent_boundary = []
    end_offset = 0
    for p in parents_list:
        end_offset+=len(p)
        data_sample_sent_boundary.append(end_offset)
        data_sample_root_list.append(p.index(0))
    combined_parents = []
    for i, p in enumerate(parents_list):
        dummy_root = data_sample_sent_boundary[-1] + 1
        if i == 0:
            offset = 0
        else:
            offset = data_sample_sent_boundary[i-1]
        new_parents = [p_elem+offset if p_elem !=0 else dummy_root for p_elem in p]
        combined_parents.extend(new_parents)
    combined_parents.insert(dummy_root-1,0)
    return combined_parents

def combine_data_samples(data_sample_token_idx_list,
                         data_sample_dep_rels):

    data_sample_token_idx_combined = [item for sublist in data_sample_token_idx_list
                                      for item in sublist]
    data_sample_dep_rels_combined = [item for sublist in data_sample_dep_rels
                                     for item in sublist]
    return data_sample_token_idx_combined, data_sample_dep_rels_combined


def recnn_read_dataset_inter(sdp_dep_data, sdp_sent_data, dict_word_indices_for_emb):
    overall_max_degree = 0
    dataset_trees = []
    dataset_sdp_sent_aug_info_parse_tree = []
    dataset_sdp_sent_aug_info_sent_idx = []
    dataset_sdp_sent_aug_info_combined_sent_idx = []

    for line_num, dataline in enumerate(sdp_dep_data):
        sdp_sent = sdp_sent_data[line_num].strip().split("::")[5]
        k_val = int(dataline[0][0].strip().split("::")[1])
        max_degree = 0
        num_sub_trees = k_val + 1
        dep_parse_trees = dataline[1][0]
        sdp_dep_parse_trees = dataline[1][1]

        if num_sub_trees != len(dep_parse_trees):
            print 'line_num %s AssertionError num_sub_trees %s, len(dep_parse_trees) %s' %(line_num, num_sub_trees, len(dep_parse_trees))
        data_sample_parent_idx = []
        data_sample_token_dict_list = []
        data_sample_token_idx_list = []
        data_sample_dep_rels = []
        data_sample_sent_boundary = []
        for sub_tree in dep_parse_trees:
            max_token = int(sub_tree[-1][0])
            if len(data_sample_sent_boundary) > 0:
                curr_sent_boundary = data_sample_sent_boundary[-1] + max_token
            else:
                curr_sent_boundary = max_token
            data_sample_sent_boundary.append(curr_sent_boundary)
            subtree_token_dict={}
            for elem in sub_tree:
                if elem[1].lower() in dict_word_indices_for_emb.keys():
                    subtree_token_dict[int(elem[0])]=dict_word_indices_for_emb[elem[1].lower()]
                else:
                    subtree_token_dict[int(elem[0])]=dict_word_indices_for_emb['<unk>']
            subtree_parent_idx = [int(elem[-2]) for elem in sub_tree]
            subtree_dep_rels = [elem[4] for elem in sub_tree]
            leaf_token_idx = 1
            for token_idx in range(1, max_token+1):
                if token_idx not in subtree_parent_idx and token_idx in subtree_token_dict:
                    leaf_token_idx = token_idx
                    break
            for token_idx in range(1, max_token+1):
                if token_idx not in subtree_token_dict.keys():
                    subtree_parent_idx.insert(token_idx-1, leaf_token_idx)
                    subtree_token_dict[token_idx] = dict_word_indices_for_emb['<unk>']
                    subtree_dep_rels.insert(token_idx, 'punct')

            data_sample_parent_idx.append(subtree_parent_idx)
            data_sample_token_dict_list.append(subtree_token_dict)
            data_sample_dep_rels.append(subtree_dep_rels)
            data_sample_token_idx_list.append([v for k,v in subtree_token_dict.iteritems()])
            assert max_token == len(subtree_parent_idx), 'max_token %s, len(subtree_parent_idx) %s' %(max_token, len(subtree_parent_idx))

        data_sample_root_idx = [sent.index(0)for sent in data_sample_parent_idx]
        data_sample_parent_idx_combined=combine_parents_list(data_sample_parent_idx)
        data_sample_token_idx_combined, data_sample_dep_rels_combined = combine_data_samples(data_sample_token_idx_list, data_sample_dep_rels)
        tags = ['<e1>', '</e1>', '<e2>', '</e2>']

        sdp_sent_aug_info_parse_tree = []
        sdp_sent_aug_info_sent_idx = []
        sdp_sent_aug_info_combined_sent_idx = []
        for tag in tags:
            sdp_sent = sdp_sent.replace(tag, ' ' + tag + ' ')
        sdp_sent_word_list = [word.lower() for word in sdp_sent.split()]
        curr_parse_tree_idx = 0

        # add content to parse trees
        for sdp_tok_idx, sdp_tok in enumerate(sdp_sent_word_list):
            sdp_tok_found = False
            if sdp_tok in tags:
                sdp_sent_aug_info_parse_tree.append(None)
                sdp_sent_aug_info_sent_idx.append(None)
                sdp_sent_aug_info_combined_sent_idx.append(None)
                continue
            curr_parse_tree = dep_parse_trees[curr_parse_tree_idx]
            for elem in curr_parse_tree:
                if sdp_tok == elem[1].lower():
                    sdp_sent_aug_info_parse_tree.append(curr_parse_tree_idx)
                    sdp_sent_aug_info_sent_idx.append(int(elem[0])-1)
                    sent_offset = 0 if curr_parse_tree_idx == 0 else data_sample_sent_boundary[
                        curr_parse_tree_idx-1]
                    sdp_sent_aug_info_combined_sent_idx.append(int(elem[0])-1 + sent_offset)
                    sdp_tok_found = True
                    if elem[-1] == 'root' and curr_parse_tree_idx < len(dep_parse_trees)-1:
                        curr_parse_tree_idx+=1
                    break
            if sdp_tok_found == False:
                sdp_sent_aug_info_parse_tree.append(None)
                sdp_sent_aug_info_sent_idx.append(None)
                sdp_sent_aug_info_combined_sent_idx.append(None)

        assert len(sdp_sent_word_list) == len(sdp_sent_aug_info_parse_tree) and len(sdp_sent_word_list) == len(sdp_sent_aug_info_sent_idx), 'sdp_sent_aug_info_parse_tree %s sdp_sent_aug_info_sent_idx %s dep_parse_trees %s' %(sdp_sent_aug_info_parse_tree, sdp_sent_aug_info_sent_idx,dep_parse_trees)
        dataset_sdp_sent_aug_info_parse_tree.append(sdp_sent_aug_info_parse_tree)
        dataset_sdp_sent_aug_info_sent_idx.append(sdp_sent_aug_info_sent_idx)
        dataset_sdp_sent_aug_info_combined_sent_idx.append(sdp_sent_aug_info_combined_sent_idx)
        data_sample_max_degree, data_sample_tree = read_sent_tree(data_sample_parent_idx_combined)
        max_degree = max(max_degree, data_sample_max_degree)

        _remap_tokens_and_labels(data_sample_tree, data_sample_token_idx_combined)

        dataset_trees.append(data_sample_tree)

        overall_max_degree = max(overall_max_degree, max_degree)

    assert len(dataset_trees) == len(dataset_sdp_sent_aug_info_combined_sent_idx), 'len(dataset_trees) %s, len(dataset_sdp_sent_aug_info_combined_sent_idx) %s' %(len(dataset_trees), len(dataset_sdp_sent_aug_info_combined_sent_idx))
    return dataset_trees, dataset_sdp_sent_aug_info_combined_sent_idx, overall_max_degree


def recnn_read_muc6_dataset(sdp_dep_data, sdp_sent_data, dict_word_indices_for_emb):
    overall_max_degree = 0
    dataset_trees = []
    dataset_sdp_sent_aug_info_parse_tree = []
    dataset_sdp_sent_aug_info_sent_idx = []

    for line_num, dataline in enumerate(sdp_dep_data):
        sdp_sent = sdp_sent_data[line_num].strip().split("::")[5]
        k_val = int(dataline[0][0].strip().split("::")[1])
        data_sample_trees = []
        max_degree = 0
        num_sub_trees = k_val + 1
        dep_parse_trees = dataline[1][0]
        sdp_dep_parse_trees = dataline[1][1]

        if num_sub_trees != len(dep_parse_trees):
            print 'line_num %s AssertionError num_sub_trees %s, len(dep_parse_trees) %s' %(line_num, num_sub_trees, len(dep_parse_trees))
        data_sample_parent_idx = []
        data_sample_token_dict_list = []
        data_sample_token_idx_list = []
        data_sample_dep_rels = []
        data_sample_sent_boundary = []
        for sub_tree in dep_parse_trees:
            max_token = int(sub_tree[-1][0])
            if len(data_sample_sent_boundary) > 0:
                curr_sent_boundary = data_sample_sent_boundary[-1] + max_token
            else:
                curr_sent_boundary = max_token
            data_sample_sent_boundary.append(curr_sent_boundary)
            subtree_token_dict={}
            for elem in sub_tree:
                if elem[1].lower() in dict_word_indices_for_emb.keys():
                    subtree_token_dict[int(elem[0])]=dict_word_indices_for_emb[elem[1].lower()]
                else:
                    subtree_token_dict[int(elem[0])]=dict_word_indices_for_emb['<unk>']

            subtree_parent_idx = [int(elem[-2]) for elem in sub_tree]
            subtree_dep_rels = [elem[4] for elem in sub_tree]
            leaf_token_idx = 1
            for token_idx in range(1, max_token+1):
                if token_idx not in subtree_parent_idx and token_idx in subtree_token_dict:
                    leaf_token_idx = token_idx
                    break
            for token_idx in range(1, max_token+1):
                if token_idx not in subtree_token_dict.keys():
                    subtree_parent_idx.insert(token_idx-1, leaf_token_idx)
                    subtree_token_dict[token_idx] = dict_word_indices_for_emb['<unk>']
                    subtree_dep_rels.insert(token_idx, 'punct')

            data_sample_parent_idx.append(subtree_parent_idx)
            data_sample_token_dict_list.append(subtree_token_dict)
            data_sample_dep_rels.append(subtree_dep_rels)
            data_sample_token_idx_list.append([v for k,v in subtree_token_dict.iteritems()])
            assert max_token == len(subtree_parent_idx), 'max_token %s, len(subtree_parent_idx) %s' %(max_token, len(subtree_parent_idx))

        tags = ['<e1>', '</e1>', '<e2>', '</e2>']

        sdp_sent_aug_info_parse_tree = []
        sdp_sent_aug_info_sent_idx = []
        for tag in tags:
            sdp_sent = sdp_sent.replace(tag, ' ' + tag + ' ')
        sdp_sent_word_list = [word.lower() for word in sdp_sent.split()]
        curr_parse_tree_idx = 0

        for sdp_tok_idx, sdp_tok in enumerate(sdp_sent_word_list):
            sdp_tok_found = False
            if sdp_tok in tags:
                sdp_sent_aug_info_parse_tree.append(-1)
                sdp_sent_aug_info_sent_idx.append(-1)
                continue
            curr_parse_tree = dep_parse_trees[curr_parse_tree_idx]
            for elem in curr_parse_tree:
                if sdp_tok == elem[1].lower():
                    sdp_sent_aug_info_parse_tree.append(curr_parse_tree_idx)
                    sdp_sent_aug_info_sent_idx.append(int(elem[0]) -1)
                    sdp_tok_found = True
                    if elem[-1] == 'root' and curr_parse_tree_idx < len(dep_parse_trees)-1:
                        curr_parse_tree_idx+=1
                    break
            if sdp_tok_found == False:
                sdp_sent_aug_info_parse_tree.append(-1)
                sdp_sent_aug_info_sent_idx.append(-1)

        assert len(sdp_sent_word_list) == len(sdp_sent_aug_info_parse_tree) and len(sdp_sent_word_list) == len(sdp_sent_aug_info_sent_idx), 'sdp_sent_aug_info_parse_tree %s sdp_sent_aug_info_sent_idx %s dep_parse_trees %s' %(sdp_sent_aug_info_parse_tree, sdp_sent_aug_info_sent_idx,dep_parse_trees)
        dataset_sdp_sent_aug_info_parse_tree.append(sdp_sent_aug_info_parse_tree)
        dataset_sdp_sent_aug_info_sent_idx.append(sdp_sent_aug_info_sent_idx)

        for cur_parents in data_sample_parent_idx:
            cur_max_degree, cur_tree = read_sent_tree(cur_parents)
            max_degree = max(max_degree, cur_max_degree)

            data_sample_trees.append(cur_tree)

        this_dataset = zip(data_sample_trees, data_sample_token_idx_list)

        for tree, sentence in this_dataset:
            _remap_tokens_and_labels(tree, sentence)

        tree_data = [(tree) for tree, _ in this_dataset]
        dataset_trees.append(tree_data)

        overall_max_degree = max(overall_max_degree, max_degree)

    assert len(dataset_trees) == len(dataset_sdp_sent_aug_info_parse_tree) and len(dataset_trees) == len(dataset_sdp_sent_aug_info_sent_idx), 'len(dataset_trees) %s, len(dataset_sdp_sent_aug_info_parse_tree) %s len(dataset_sdp_sent_aug_info_sent_idx %s' %(len(dataset_trees), len(dataset_sdp_sent_aug_info_parse_tree), len(dataset_sdp_sent_aug_info_sent_idx))
    return dataset_trees, dataset_sdp_sent_aug_info_parse_tree, dataset_sdp_sent_aug_info_sent_idx, overall_max_degree

class Vocab(object):

    def __init__(self):
        self.words = []
        self.word2idx = {}
        self.unk_index = None
        self.start_index = None
        self.end_index = None
        self.unk_token = None
        self.start_token = None
        self.end_token = None

    def load(self, path):
        with open(path, 'r') as in_file:
            for line in in_file:
                word = line.strip()
                assert word not in self.word2idx
                self.word2idx[word] = len(self.words)
                self.words.append(word)

        for unk in ['<unk>', '<UNK>', 'UUUNKKK']:
            self.unk_index = self.unk_index or self.word2idx.get(unk, None)
            if self.unk_index is not None:
                self.unk_token = unk
                break

        for start in ['<s>', '<S>']:
            self.start_index = self.start_index or self.word2idx.get(start, None)
            if self.start_index is not None:
                self.start_token = start
                break

        for end in ['</s>', '</S>']:
            self.end_index = self.end_index or self.word2idx.get(end, None)
            if self.end_index is not None:
                self.end_token = end
                break

    def index(self, word):
        if self.unk_index is None:
            assert word in self.word2idx
        return self.word2idx.get(word, self.unk_index)

    def size(self):
        return len(self.words)

def read_trees(parents_file, labels_file):
    trees = []
    max_degree = 0
    with open(parents_file, 'r') as parents_f:
        with open(labels_file, 'r') as labels_f:
            while True:
                cur_parents = parents_f.readline()
                cur_labels = labels_f.readline()
                if not cur_parents or not cur_labels:
                    break
                cur_parents = [int(p) for p in cur_parents.strip().split()]
                cur_labels = [int(l) if l != '#' else None for l in cur_labels.strip().split()]
                cur_max_degree, cur_tree = read_tree(cur_parents, cur_labels)
                max_degree = max(max_degree, cur_max_degree)
                trees.append(cur_tree)
    return max_degree, trees

# Input: a list [parents] such as [6, 3, 6, 6, 6, 0, 6, 6]
# output: max_degree: maximum number of children for a node
#       and root: the root node as node class object
def read_tree(parents, labels):
    nodes = {}
    parents = [p - 1 for p in parents]  # 1-indexed

    for i in xrange(len(parents)):
        if i not in nodes:
            idx = i
            prev = None
            while True:
                node = tree_rnn.Node(val=idx)  # for now, val is just idx
                if prev is not None:
                    assert prev.val != node.val
                    node.add_child(prev)

                node.label = labels[idx]
                nodes[idx] = node

                parent = parents[idx]
                if parent in nodes:
                    nodes[parent].add_child(node)
                    break
                elif parent == -1:
                    root = node
                    break

                prev = node
                idx = parent

    # ensure tree is connected: by confirming that there is only one root
    num_roots = sum(node.parent is None for node in nodes.itervalues())
    assert num_roots == 1, num_roots

    # overwrite vals to match sentence indices -
    # only leaves correspond to sentence tokens
    leaf_idx = 0
    for node in nodes.itervalues():
        if node.children:
            node.val = None
        else:
            node.val = leaf_idx
            leaf_idx += 1

    max_degree = max(len(node.children) for node in nodes.itervalues())
    # max_degree is the maximum number of children of a node

    return max_degree, root

def read_sent_tree(parents):
    nodes = {}
    parents = [p - 1 for p in parents]  # 1-indexed

    for i in xrange(len(parents)):
        if i not in nodes:
            idx = i
            prev = None
            while True:
                node = tree_rnn.Node(val=idx, sent_idx=idx)  # for now, val is just idx
                if prev is not None:
                    assert prev.val != node.val
                    node.add_child(prev)

                nodes[idx] = node

                parent = parents[idx]
                if parent in nodes:
                    nodes[parent].add_child(node)
                    break
                elif parent == -1:
                    root = node
                    break

                prev = node
                idx = parent

    # ensure tree is connected: by confirming that there is only one root
    num_roots = sum(node.parent is None for node in nodes.itervalues())
    assert num_roots == 1, num_roots

    # overwrite vals to match sentence indices -
    # only leaves correspond to sentence tokens
    leaf_idx = 0
    for node in nodes.itervalues():
        if node.children:
            node.val = None
        else:
            node.val = leaf_idx
            leaf_idx += 1

    max_degree = max(len(node.children) for node in nodes.itervalues())
    # max_degree is the maximum number of children of a node

    return max_degree, root

def read_sentences(path, vocab):
    sentences = []
    with open(path, 'r') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            sentences.append([vocab.index(tok) for tok in tokens])
    return sentences


def _remap_tokens_and_labels(tree, sentence, fine_grained=None):
    # map leaf idx to word idx
    if tree.val is not None:
        tree.val = sentence[tree.val]

    # map label to suitable range
    if tree.label is not None:
        if fine_grained:
            tree.label += 2
        else:
            if tree.label < 0:
                tree.label = 0
            elif tree.label == 0:
                tree.label = 1
            else:
                tree.label = 2

    [_remap_tokens_and_labels(child, sentence, fine_grained)
     for child in tree.children
     if child is not None]


def read_embeddings_into_numpy(file_name, vocab=None):
    """Reads Glove vector files and returns numpy arrays.

    If vocab is given, only intersection of vocab and words is used.

    """
    words = []
    array = []
    with open(file_name, 'r') as in_file:
        for line in in_file:
            fields = line.strip().split()
            word = fields[0]
            if vocab and word not in vocab.word2idx:
                continue
            embedding = np.array([float(f) for f in fields[1:]])
            words.append(word)
            array.append(embedding)

    return np.array(words), np.array(array)