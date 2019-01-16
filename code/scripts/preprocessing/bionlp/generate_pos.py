import os
import nltk

data_dir = '../data/bionlp/'
dataset_year = '2013'
data_type = "dev"
dataset_type = "full"
file_name = os.path.join(data_dir, dataset_year + '/BB' + dataset_year + '_' + data_type + '_data_rnn_' + dataset_type + '_sent_rnn_input.txt')

def write_pos(file):
    file_pos_list = []
    tags = ['<e1>', '</e1>', '<e2>', '</e2>']
    for i, line in enumerate(open(file, 'rb')):
        sent = line.split('::')[5]
        for tag in tags:
            sent = sent.replace(tag, ' ' + tag + ' ')
        word_list = [word.lower() for word in sent.split()]
        pos_tuple_list = nltk.pos_tag(word_list)
        tag_index_dict = {}
        for tag in tags:
             tag_index_dict[tag] = word_list.index(tag)
        pos_list =[pos_tuple[1] for pos_tuple in pos_tuple_list]
        for tag in tags:
            pos_list[tag_index_dict[tag]] = 'E_TAG'
        file_pos_list.append(pos_list)
    return file_pos_list
file_pos_list = write_pos(file_name)

with open(str(file_name).split('.txt')[0] + '.pos' , 'wb') as pos_file:
    for pos_list in file_pos_list:
        pos_file.write(' '.join(pos_list))
        pos_file.write('\n')

