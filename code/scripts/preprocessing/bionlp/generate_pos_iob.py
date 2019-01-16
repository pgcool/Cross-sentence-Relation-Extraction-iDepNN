import os
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

k_val = "k_le_2"
data_dir = '../data/bionlp/BB2016_test_final/' + k_val +  '/'
data_type = "test"
dataset_sent_type = "full" # [full, sdp]

# file_name = os.path.join(data_dir, 'BB2016_others_k_0_test_expt_input_full_sent.txt')
file_name = os.path.join(data_dir, 'BB2016_' + k_val + '_test_final_' + dataset_sent_type + '_sent.txt')

def write_pos_iob(file):
    file_pos_list = []
    file_iob_list = []
    tags = ['<e1>', '</e1>', '<e2>', '</e2>']
    count=0
    for i, line in enumerate(open(file, 'rb')):
        # print(i)
        sent = line.split('::')[5]
        e2_type = 'BACTERIA' if line.split('::')[7]=='BACTERIA' else 'HABITAT'

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

        # IOB encoding
        iob_list = ['O' for i in range(len(word_list))]
        iob_list[tag_index_dict['<e1>']+1] = 'B-BACTERIA'
        if tag_index_dict['</e1>'] > tag_index_dict['<e1>']+2:
            for i in range(tag_index_dict['<e1>']+2,tag_index_dict['</e1>']):
                iob_list[i] = 'I-BACTERIA'
        iob_list[tag_index_dict['<e2>']+1] = 'B-' + e2_type
        if tag_index_dict['</e2>'] > tag_index_dict['<e2>']+2:
            for i in range(tag_index_dict['<e2>']+2,tag_index_dict['</e2>']):
                iob_list[i] = 'I-' + e2_type
        file_iob_list.append(iob_list)
    print count
    return file_pos_list, file_iob_list
file_pos_list, file_iob_list = write_pos_iob(file_name)

with open(str(file_name).split('.txt')[0] + '.pos1' , 'wb') as pos_file:
    for pos_list in file_pos_list:
        pos_file.write(' '.join(pos_list))
        pos_file.write('\n')

with open(str(file_name).split('.txt')[0] + '.iob1' , 'wb') as iob_file:
    for iob_list in file_iob_list:
        iob_file.write(' '.join(iob_list))
        iob_file.write('\n')


