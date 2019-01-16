from os import listdir
from os.path import isfile, join
import os, codecs
import json
import errno
import cPickle
import os.path
import nltk
import nltk.data
import sys
import codecs
import load_save_pkl
import re

reload(sys)
sys.setdefaultencoding('utf8')

others_datatype = "dev"
annotations_path = "BioNLP-ST-2016_BB-event_" + others_datatype + "/"
train_file = "BB2016_others_" + others_datatype + "_train_data.txt"
containment_file = "BB2016_others_" + others_datatype + "_containment_data.txt"
error_file = "BB2016_others_" + others_datatype + "_error_data.txt"
target_sent_list_dict_file = "BB2016_others_" + others_datatype + "_train_data_target_sent_list_dict.pkl"

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def get_filepaths(directory):
    """
    This function will generate the file names in a directory
    tree by walking the tree either top-down or bottom-up. For each
    directory in the tree rooted at directory top (including top itself),
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

def gen_annotation_dict_doc_list_dict(basepath, data_dir, output_path):
    annotation_dict = {}
    doc_list_dict = {}
    star_count = 0
    mis_match_mention_text_but_within_sent_offsets = 0

    for dir in data_dir:
        list_filepaths = get_filepaths(os.path.join(basepath, dir))

        print('#doc files:%s in %s' %(len(list_filepaths), dir))
        # read data file
        for filename in list_filepaths:
            ent_dict = dict()
            rel_dict = dict()

            # read corresponding annotation file
            if filename.endswith(".a2"):
                key_file = str(str(filename).split('.a2')[0])
                print('filename:', filename)
                print('key_file:', key_file)
                for line in open(filename).readlines():
                    line = str(line).strip('\n')
                    if os.path.basename(filename).startswith("PMC-"): # file from GE4 dataset
                        if line.strip().startswith('T') or line.strip().startswith('E'):
                            ent_id = line.split()[0]
                            if ';' in str(line.split()[3]):
                                ent_dict[str(ent_id).strip()] = [line.split()[1], line.split()[2], str(line.split()[3]).split(';')[0],
                                                                 str(' '.join(line.split()[5:])).strip()]
                                # NER tag, start indx, end indx, mention
                            else:
                                ent_dict[str(ent_id).strip()] = [line.split()[1], line.split()[2], line.split()[3], str(' '.join(line.split()[4:])).strip()]
                                # NER tag, start indx, end indx, mention
                        elif line.strip().startswith('R'):# or line.strip().startswith('*'):
                            rel_id = line.split()[0]
                            rel_dict[str(rel_id).strip()] = [line.split()[1], line.split()[2], line.split()[3]]
                        elif line.strip().startswith('*'):
                            star_count+=1
                        else:
                            print('line:', line)
                            print('Error in the file annotation:', filename)
                            exit()
                    else:
                        if line.strip().startswith('T'):
                            ent_id = line.split()[0]
                            if ';' in str(line.split()[3]):
                                ent_dict[str(ent_id).strip()] = [line.split()[1], line.split()[2], str(line.split()[3]).split(';')[0],
                                                                 str(' '.join(line.split()[5:])).strip()]
                                # NER tag, start indx, end indx, mention
                            else:
                                ent_dict[str(ent_id).strip()] = [line.split()[1], line.split()[2], line.split()[3], str(' '.join(line.split()[4:])).strip()]
                                # NER tag, start indx, end indx, mention
                        elif line.strip().startswith('R') or line.strip().startswith('E'):# or line.strip().startswith('*'):
                            rel_id = line.split()[0]
                            rel_dict[str(rel_id).strip()] = [line.split()[1], line.split()[2], line.split()[3]]
                        elif line.strip().startswith('*'):
                            star_count+=1
                        else:
                            print('line:', line)
                            print('Error in the file annotation:', filename)
                            exit()

                if os.path.basename(filename).startswith("SeeDev"):
                    # read entities from .a1 file for entities
                    for line in open(str(str(filename).split(".a2")[0]) + ".a1").readlines():
                        line = str(line).strip('\n')
                        if line.strip().startswith('T'):
                            ent_id = line.split()[0]
                            if ';' in str(line.split()[3]):
                                ent_dict[str(ent_id).strip()] = [line.split()[1], line.split()[2],
                                                                 str(line.split()[3]).split(';')[0],
                                                                 str(' '.join(line.split()[5:])).strip()]
                                # NER tag, start indx, end indx, mention
                            else:
                                ent_dict[str(ent_id).strip()] = [line.split()[1], line.split()[2], line.split()[3],
                                                                 str(' '.join(line.split()[4:])).strip()]
                                # NER tag, start indx, end indx, mention
                        else:
                            print('line:', line)
                            print('Error in the file annotation:', filename)
                            exit()

                if annotation_dict.has_key(key_file):
                    print('Error: Annotation file already exists...')
                    exit()

                annotation_dict[key_file] = [ent_dict, rel_dict]
                print('annotation_dict[key_file]:', annotation_dict[key_file])
                # exit(0)

            # read corresponding annotation file
            if filename.endswith(".rel") and os.path.basename(filename).startswith("PMID"):
                key_file = str(str(filename).split('.rel')[0])
                print('filename:', filename)
                print('key_file:', key_file)
                for line in open(filename).readlines():
                    line = str(line).strip('\n')
                    if line.strip().startswith('T'):
                        ent_id = line.split()[0]
                        if ';' in str(line.split()[3]):
                            ent_dict[str(ent_id).strip()] = [line.split()[1], line.split()[2],
                                                             str(line.split()[3]).split(';')[0],
                                                             str(' '.join(line.split()[5:])).strip()]
                            # NER tag, start indx, end indx, mention
                        else:
                            ent_dict[str(ent_id).strip()] = [line.split()[1], line.split()[2], line.split()[3],
                                                             str(' '.join(line.split()[4:])).strip()]
                            # NER tag, start indx, end indx, mention
                    elif line.strip().startswith('R'):  # or line.strip().startswith('*'):
                        rel_id = line.split()[0]
                        rel_dict[str(rel_id).strip()] = [line.split()[1], line.split()[2], line.split()[3]]
                    elif line.strip().startswith('*'):
                        star_count += 1
                    else:
                        print('line:', line)
                        print('Error in the file annotation:', filename)
                        exit()

                # read entities from .a1 file
                for line in open(str(str(filename).split('.rel')[0])+".a1").readlines():
                    line = str(line).strip('\n')
                    if line.strip().startswith('T'):
                        ent_id = line.split()[0]
                        if ';' in str(line.split()[3]):
                            ent_dict[str(ent_id).strip()] = [line.split()[1], line.split()[2],
                                                             str(line.split()[3]).split(';')[0],
                                                             str(' '.join(line.split()[5:])).strip()]
                            # NER tag, start indx, end indx, mention
                        else:
                            ent_dict[str(ent_id).strip()] = [line.split()[1], line.split()[2], line.split()[3],
                                                             str(' '.join(line.split()[4:])).strip()]
                            # NER tag, start indx, end indx, mention
                    else:
                        print('line:', line)
                        print('Error in the file annotation:', filename)
                        exit()

                if annotation_dict.has_key(key_file):
                    print('Error: Annotation file already exists...')
                    exit()

                annotation_dict[key_file] = [ent_dict, rel_dict]
                print('annotation_dict[key_file]:', annotation_dict[key_file])
                # exit(0)

            # read corresponding data/text file
            if filename.endswith(".txt"):
                doc_key_file = str(str(filename).split('.txt')[0])
                # read text file
                print('Text filename:', filename)
                text = " "
                # text = open(filename).read().replace('\n', ' ')
                text = codecs.open(filename, "r", encoding='utf-8', errors='ignore').read().replace('\n', ' ')
                # text = open(filename).read().replace('\n', ' ')
                # for line in open(filename).readlines():
                #    line = str(line).strip('\n')
                #    text = " "+ text + " " + str(line).strip()

                if doc_list_dict.has_key(doc_key_file):
                    print('Error: doc file already exists...')
                    exit()
                else:
                    doc_list_dict[doc_key_file] = []

                # split text into sentences
                tokenised_text = tokenizer.tokenize(text)
                start = 0
                end = start
                for tok_text in tokenised_text:
                    end = start + len(tok_text)
                    doc_list_dict[doc_key_file].append([tok_text, start, end-1]) # text, begin, end
                    start = end + 1

                print('filename:%s #sentences=%s' %(filename, len(doc_list_dict[doc_key_file])))
                print('filename:%s #doc_list_dict=%s' %(filename, doc_list_dict[doc_key_file]))
                print('-------------------------------------------------------------------------------------')
                # print('text:', text)
                print('tokenised text:', tokenizer.tokenize(text))

    print('star_count:', star_count)

    # load_save_pkl.save_as_pkl(annotation_dict, os.path.join(output_path, "BB2016_annotation_dict.pkl"))
    # load_save_pkl.save_as_pkl(doc_list_dict, os.path.join(output_path, "BB2016_doc_list_dict.pkl"))
    return annotation_dict, doc_list_dict

# data_type = "NER+REL" # NER_ONLY, REL_ONLY, NER+REL

data_type = "REL_ONLY"

if __name__ == '__main__':

    # output_path = 'D:/thesis_stuff/code/master_thesis/BioNLP_extraction/data/input/'
    output_path = 'D:/thesis_stuff/code/master_thesis/BioNLP_extraction/data/output_2016/'

    if data_type == "NER+REL":
        '''
        basepath = "../../data/BioNLPdata/BB3-data/BB3-event-rel+ner/"
        data_dir = ["BioNLP-ST-2016_BB-event+ner_train",
                    "BioNLP-ST-2016_BB-event+ner_dev",
                    "BioNLP-ST-2016_BB-event+ner_test"]

        basepath = "../../data/BioNLPdata/BB2013/"
        data_dir = ["BioNLP-ST-2013_Bacteria_Biotopes_train/task_3",
                    "BioNLP-ST-2013_Bacteria_Biotopes_dev/task_3",
                    "BioNLP-ST-2013_Bacteria_Biotopes_test/task_3"]

        basepath = "../../data/BioNLPdata/BB2011/"
        data_dir = ["BioNLP-ST_2011_Bacteria_Biotopes_train_data_rev1",
                    "BioNLP-ST_2011_Bacteria_Biotopes_dev_data_rev1",
                    "BioNLP-ST_2011_Bacteria_Biotopes_test_data"]

        basepath = "../../data/BioNLPdata/BB2011/"
        data_dir = ["BioNLP-ST_2011_Entity_Relations_training_data",
                    "BioNLP-ST_2011_Entity_Relations_development_data"]

        basepath = "../../data/BioNLPdata/SeeDev/SeeDev-binary/"
        data_dir = ["BioNLP-ST-2016_SeeDev-binary_train"]
                    # "BioNLP-ST-2016_SeeDev-binary_dev"]
        '''

        # basepath = "../../data/BioNLPdata/GE4/bionlp-st-ge-2016-reference/"
        # data_dir = ["jsonparsed"]

        basepath = 'D:/thesis_stuff/code/master_thesis/datasets/BioNLP/BB2011/'
        data_dir = ["BioNLP-ST_2011_Bacteria_Biotopes_train_data_rev1/",
                    "BioNLP-ST_2011_Bacteria_Biotopes_dev_data_rev1/",
                    "BioNLP-ST_2011_Bacteria_Biotopes_test_data/"]


        annotation_dict, doc_list_dict = gen_annotation_dict_doc_list_dict(basepath, data_dir, output_path)
        # exit(0)

        annotation_dict = load_save_pkl.load_pickle_file(os.path.join(output_path,"BB2016_others" + others_datatype + "_annotation_dict.pkl"))
        doc_list_dict = load_save_pkl.load_pickle_file(os.path.join(output_path, "BB2016_others" + others_datatype + "doc_list_dict.pkl"))
        # print(annotation_dict)
        # print(doc_list_dict)
        inter_sent_rel_count = 0
        intra_sent_rel_count = 0
        intra_rel_count_dict = dict()
        inter_rel_count_dict = dict()
        k_val = 0
        mis_match_mention_text_but_within_sent_offsets = 0
        # compute inter-sentence and intra-sentence relationships
        for filename in annotation_dict.keys():
            ent_dict, rel_dict = annotation_dict[filename]
            print("Processing filename %s" %filename)
            # print('rel_dict:', rel_dict)
            for rel_key in rel_dict.keys():
                relation = rel_dict[rel_key][0]
                print(relation)
                if not intra_rel_count_dict.has_key(relation):
                    intra_rel_count_dict[relation] = 0

                if not inter_rel_count_dict.has_key(relation):
                    inter_rel_count_dict[relation] = 0

                relation_e1 = str(rel_dict[rel_key][1]).split(':')[-1]
                relation_e2 = str(rel_dict[rel_key][2]).split(':')[-1]
                relation_e1_type = str(rel_dict[rel_key][1]).split(':')[0]
                relation_e2_type = str(rel_dict[rel_key][2]).split(':')[0]
                start_e1_indx = int(ent_dict[relation_e1][1])
                end_e1_indx = int(ent_dict[relation_e1][2])
                start_e2_indx = int(ent_dict[relation_e2][1])
                end_e2_indx = int(ent_dict[relation_e2][2])
                # check in document the range of the entities, if they lie within or across the sentence boundaries
                sent_begin_end_indx_list = doc_list_dict[filename]

                sentence_number_for_e1 = -1
                sentence_number_for_e2 = -1
                e1_found = False
                e2_found = False
                print(relation_e1,relation_e2, relation_e1_type, relation_e2_type, start_e1_indx, start_e2_indx, sent_begin_end_indx_list)
                # exit(0)

                for sent_begin_end_indx, sent_indx in zip(sent_begin_end_indx_list, range(len(sent_begin_end_indx_list))):
                    sent = sent_begin_end_indx[0]
                    begin_indx = int(sent_begin_end_indx[1])
                    end_indx  = int(sent_begin_end_indx[2])

                    '''
                    if sent_indx != 0:
                        previous_sent_end = int(sent_begin_end_indx_list[sent_indx][2]) + 1
                    else:
                        previous_sent_end = 1
                    '''

                    # check if the two entities lie in the same sentences or across the sentence boundaries
                    if start_e1_indx >= begin_indx and start_e1_indx < end_indx and \
                                    start_e2_indx >= begin_indx and start_e2_indx < end_indx:
                        if str(ent_dict[relation_e1][-1]).lower().strip() == str(str(sent_begin_end_indx_list[sent_indx][0])[
                                                                                 start_e1_indx - sent_begin_end_indx_list[sent_indx][1]:
                                                                                         (start_e1_indx - sent_begin_end_indx_list[sent_indx][1]) + len(
                                                                                     str(ent_dict[relation_e1][-1]).strip())]).lower().strip() and \
                                        str(ent_dict[relation_e2][-1]).lower().strip() == str(str(sent_begin_end_indx_list[sent_indx][0])[
                                                                                              start_e2_indx - sent_begin_end_indx_list[sent_indx][1]:
                                                                                                      (start_e2_indx - sent_begin_end_indx_list[sent_indx][1]) + len(
                                                                                                  str(ent_dict[relation_e2][-1]).strip())]).lower().strip():
                            # intra-sentence relationship
                            sentence_number_for_e1 = sent_indx
                            sentence_number_for_e2 = sent_indx
                            e1_found = True
                            e2_found = True

                            # checks if the e1 word is present based on begin and end idx
                            # print(e1_found)
                            # print(str(ent_dict[relation_e1][-1]).lower().strip())
                            # print(str(str(sent_begin_end_indx_list[sent_indx][0])[start_e1_indx - sent_begin_end_indx_list[sent_indx][1]:(start_e1_indx - sent_begin_end_indx_list[sent_indx][1]) + len(str(ent_dict[relation_e1][-1]).strip())]).lower().strip())
                            # exit(0)
                            # print('------------------------------------------------------------------------')
                            # print('Intra-sentence Relation | e1: %s e2: %s Texts: %s' % (ent_dict[relation_e1][-1], ent_dict[relation_e2][-1], sent_begin_end_indx_list))
                            # print('------------------------------------------------------------------------')
                        else:
                            # checks if the e1 word is present within the sentence string
                            if str(ent_dict[relation_e1][-1]).lower().strip() in str(sent_begin_end_indx_list[sent_indx][0]).lower() and \
                                            str(ent_dict[relation_e2][-1]).lower().strip() in str(sent_begin_end_indx_list[sent_indx][0]).lower():
                                # intra-sentence relationship
                                sentence_number_for_e1 = sent_indx
                                sentence_number_for_e2 = sent_indx
                                e1_found = True
                                e2_found = True
                                # print('------------------------------------------------------------------------')
                                # print('Intra-sentence Relation | e1: %s e2: %s Texts: %s' % (ent_dict[relation_e1][-1], ent_dict[relation_e2][-1], sent_begin_end_indx_list))
                                # print('------------------------------------------------------------------------')
                            else:
                                print('------------------------------------------------------------------------')
                                print('Intra-sentence Relation | e1: %s e2: %s Texts: %s' % (
                                    ent_dict[relation_e1][-1], ent_dict[relation_e2][-1], sent_begin_end_indx_list))
                                print('filename:', filename)
                                print('start_e1_indx=%s begin_indx=%s end_indx=%s' %(start_e1_indx, begin_indx, end_indx))
                                print('for e1:', str(ent_dict[relation_e1][-1]).lower().strip())
                                print('for e1:', str(str(sent_begin_end_indx_list[sentence_number_for_e1][0])[
                                                     start_e1_indx - begin_indx:(start_e1_indx - begin_indx) + len(
                                                         str(ent_dict[relation_e1][-1]).strip())]).lower().strip())
                                print('for e2:', str(ent_dict[relation_e2][-1]).lower().strip())
                                print('for e2:', str(str(sent_begin_end_indx_list[sentence_number_for_e2][0])[
                                                     start_e2_indx - begin_indx:(start_e2_indx - begin_indx) + len(
                                                         str(ent_dict[relation_e2][-1]).strip())]).lower().strip())
                                print('------------------------------------------------------------------------')
                                mis_match_mention_text_but_within_sent_offsets+=1
                        break
                    else:
                        # check for intern-sentence relationship
                        if e1_found is not True and start_e1_indx >= begin_indx and start_e1_indx < end_indx:

                            if str(ent_dict[relation_e1][-1]).lower().strip() ==  str(str(sent_begin_end_indx_list[sent_indx][0])[
                                                                                      start_e1_indx - sent_begin_end_indx_list[sent_indx][1]:
                                                                                              (start_e1_indx - sent_begin_end_indx_list[sent_indx][1]) + len(
                                                                                          str(ent_dict[relation_e1][-1]).strip())]).lower().strip():

                                e1_found = True
                                sentence_number_for_e1 = sent_indx
                                print('--------------------------------------------------------------------')
                                print(str(ent_dict[relation_e1][-1]).lower().strip())
                                print(str(sent_begin_end_indx_list[sent_indx][0][start_e1_indx:len(
                                    str(ent_dict[relation_e1][-1]).strip())]).strip().lower())
                                print('--------------------------------------------------------------------')
                            else:
                                if str(ent_dict[relation_e1][-1]).lower().strip() in str(
                                        sent_begin_end_indx_list[sent_indx][0]).lower():
                                    e1_found = True
                                    sentence_number_for_e1 = sent_indx
                                    print('--------------------------------------------------------------------')
                                    print(str(ent_dict[relation_e1][-1]).lower().strip())
                                    print(str(sent_begin_end_indx_list[sent_indx][0][start_e1_indx:len(
                                        str(ent_dict[relation_e1][-1]).strip())]).strip().lower())
                                    print('--------------------------------------------------------------------')
                                else:
                                    print('------------------------------------------------------------------------')
                                    print('Inter-sentence Relation | e1: %s e2: %s Texts: %s' % (
                                        ent_dict[relation_e1][-1], ent_dict[relation_e2][-1], sent_begin_end_indx_list))
                                    print('filename:', filename)
                                    print('For e1: start_e1_indx=%s begin_indx=%s end_indx=%s' % (start_e1_indx,
                                                                                                  sent_begin_end_indx_list[
                                                                                                      sent_indx][
                                                                                                      1],
                                                                                                  sent_begin_end_indx_list[
                                                                                                      sent_indx][
                                                                                                      2]))

                                    print('for e1:', str(ent_dict[relation_e1][-1]).lower().strip())
                                    print('for e1:', str(str(sent_begin_end_indx_list[sent_indx][0])[
                                                         start_e1_indx - sent_begin_end_indx_list[sent_indx][
                                                             1]:
                                                         (start_e1_indx - sent_begin_end_indx_list[sent_indx][
                                                             1]) + len(
                                                             str(ent_dict[relation_e1][-1]).strip())]).lower().strip())
                                    print('------------------------------------------------------------------------')
                                    mis_match_mention_text_but_within_sent_offsets+=1

                        if e2_found is not True and start_e2_indx >= begin_indx and start_e2_indx < end_indx:

                            if str(ent_dict[relation_e2][-1]).lower().strip() == str(str(sent_begin_end_indx_list[sent_indx][0])[
                                                                                     start_e2_indx - sent_begin_end_indx_list[sent_indx][1]:
                                                                                             (start_e2_indx - sent_begin_end_indx_list[sent_indx][1]) + len(
                                                                                         str(ent_dict[relation_e2][-1]).strip())]).lower().strip():

                                e2_found = True
                                sentence_number_for_e2 = sent_indx
                                # print('--------------------------------------------------------------------')
                                #print(str(ent_dict[relation_e2][-1]).lower().strip())
                                #print(str(sent_begin_end_indx_list[sent_indx][0][start_e2_indx:len(
                                #    str(ent_dict[relation_e2][-1]).strip())]).strip().lower())
                                #print('--------------------------------------------------------------------')
                            else:
                                if str(ent_dict[relation_e2][-1]).lower().strip() in str(
                                        sent_begin_end_indx_list[sent_indx][0]).lower():
                                    e2_found = True
                                    sentence_number_for_e2 = sent_indx
                                    print('--------------------------------------------------------------------')
                                    print(str(ent_dict[relation_e2][-1]).lower().strip())
                                    print(str(sent_begin_end_indx_list[sent_indx][0][start_e2_indx:len(
                                        str(ent_dict[relation_e2][-1]).strip())]).strip().lower())
                                    print('--------------------------------------------------------------------')
                                else:
                                    print('------------------------------------------------------------------------')
                                    print('Inter-sentence Relation | e1: %s e2: %s Texts: %s' % (
                                        ent_dict[relation_e1][-1], ent_dict[relation_e2][-1], sent_begin_end_indx_list))
                                    print('filename:', filename)
                                    print('For e2: start_e2_indx=%s begin_indx=%s end_indx=%s' % (start_e2_indx,
                                                                                                  sent_begin_end_indx_list[
                                                                                                      sent_indx][
                                                                                                      1],
                                                                                                  sent_begin_end_indx_list[
                                                                                                      sent_indx][
                                                                                                      2]))

                                    print('for e2:', str(ent_dict[relation_e2][-1]).lower().strip())
                                    print('for e2:', str(str(sent_begin_end_indx_list[sent_indx][0])[
                                                         start_e2_indx - sent_begin_end_indx_list[sent_indx][
                                                             1]:(start_e2_indx -
                                                                 sent_begin_end_indx_list[sent_indx][1]) + len(
                                                             str(ent_dict[relation_e2][-1]).strip())]).lower().strip())
                                    print('------------------------------------------------------------------------')
                                    mis_match_mention_text_but_within_sent_offsets+=1

                        if e1_found and e2_found:
                            break

                if e1_found and e2_found:
                    if sentence_number_for_e1 == sentence_number_for_e2 and sentence_number_for_e2!=-1:
                        # intra-sentence relationship
                        intra_sent_rel_count += 1
                        intra_rel_count_dict[relation]+=1
                    elif sentence_number_for_e1 !=-1 and sentence_number_for_e2 !=-1:
                        # inter-sentence relationship
                        inter_sent_rel_count += 1
                        inter_rel_count_dict[relation] += 1
                print("sentence_number_for_e1 %s" %sentence_number_for_e1)
                print("sentence_number_for_e2 %s" %sentence_number_for_e2)
                print(sent_begin_end_indx_list[sentence_number_for_e1:sentence_number_for_e2])
                target_sent = ""
                e1_word = ent_dict[relation_e1][-1]
                e2_word = ent_dict[relation_e2][-1]
                for sent_num in range(min(sentence_number_for_e1,sentence_number_for_e2),max(sentence_number_for_e1,sentence_number_for_e2)+1):
                    sent = sent_begin_end_indx_list[sent_num][0]
                    sent_begin_idx = sent_begin_end_indx_list[sent_num][1]
                    if sent_num == sentence_number_for_e1:
                        if sent.count(e1_word) == 1:
                            sent = sent.replace(e1_word, "<e1>" + e1_word + "</e1>", 1)
                        else:
                            start_e1_tag_idx = start_e1_indx - sent_begin_idx
                            end_e1_tag_idx = end_e1_indx - sent_begin_idx
                            sent = sent[:start_e1_tag_idx] + "<e1>" + sent[start_e1_tag_idx:end_e1_tag_idx] +  "</e1>" + sent[end_e1_tag_idx:]
                    if sent_num == sentence_number_for_e2:
                        if sent.count(e2_word) == 1:
                            sent = sent.replace(e2_word, "<e2>" + e2_word + "</e2>", 1)
                        else:
                            start_e2_tag_idx = start_e2_indx - sent_begin_idx
                            end_e2_tag_idx = end_e2_indx - sent_begin_idx
                            sent = sent[:start_e2_tag_idx] + "<e2>" + sent[start_e2_tag_idx:end_e2_tag_idx] +  "</e2>" + sent[end_e2_tag_idx:]
                    target_sent += " " + sent

                with open(os.path.join(output_path,"train_data.txt"),'a') as data_file:
                    # data_file.write(str(filename.split("/")[-1])  + "::" + str(int(sentence_number_for_e2) - int(sentence_number_for_e1)) + "::" + str(relation)  + "::" + str(ent_dict[relation_e1][-1])  + "::" + str(ent_dict[relation_e2][-1]) + "::" + str(relation_e1)  + "::" + str(relation_e2)  + "::" + str(relation_e1_type) + "::" + str(relation_e2_type) + "::" + str(start_e1_indx) + "::" + str(start_e2_indx) + "::" + target_sent)
                    data_file.write(str(filename.split("/")[-1])  + "::" + str(abs(int(sentence_number_for_e2) - int(sentence_number_for_e1))) + "::" + str(relation)  + "::" + str(ent_dict[relation_e1][-1])  + "::" + str(ent_dict[relation_e2][-1]) + "::" + target_sent)
                    data_file.write('\n')

                # exit(0)

        print('intra_sent_rel_count=%s inter_sent_rel_count=%s' %(intra_sent_rel_count, inter_sent_rel_count))
        print('intra_rel_count_dict:', intra_rel_count_dict)
        print('inter_rel_count_dict:', inter_rel_count_dict)
        print('mis_match_mention_text_but_within_sent_offsets:', mis_match_mention_text_but_within_sent_offsets)


    elif data_type == 'REL_ONLY':
        basepath = "D:/thesis_stuff/datasets/BioNLPdata/BB3-data/BB3-event-rel/"
        data_dir = [annotations_path]
        # "BioNLP-ST-2016_BB-event_test"]

        # basepath = "D:/thesis_stuff/datasets/BioNLPdata/BB2013/"
        # data_dir = ["BioNLP-ST-2013_Bacteria_Biotopes_train/task_2/",
        #             "BioNLP-ST-2013_Bacteria_Biotopes_dev/task_2/"]
        #
        annotation_dict = {}
        doc_list_dict = {}
        star_count = 0
        for dir in data_dir:
            list_filepaths = get_filepaths(os.path.join(basepath, dir))
            # print
            print('#doc files:%s in %s' % (len(list_filepaths), dir))
            # read data file
            for filename in list_filepaths:
                ent_dict = dict()
                rel_dict = dict()

                # read corresponding annotation file i.e. entities only
                if filename.endswith(".a1"):
                    key_file = str(str(filename).split('.a1')[0])
                    print('filename:', filename)
                    print('key_file:', key_file)
                    for line in open(filename).readlines():
                        line = str(line).strip('\n')
                        if line.strip().startswith('T'):
                            ner_tag = line.split()[1]
                            if str(ner_tag).lower() == "title" or str(ner_tag).lower() == "paragraph":
                                continue
                            if line.find(';'):
                                line = re.sub('\d+;\d+\s', '', line)
                                ent_id = line.split()[0]
                                begin_ent = line.split()[2]
                                end_ent = line.split()[3]
                                entity = ' '.join(line.split()[4:])
                                # entity = open(key_file + ".txt",'rb').read()[int(begin_ent):int(end_ent)]
                                ent_dict[str(ent_id).strip()] = [line.split()[1], begin_ent, end_ent, entity]
                            else:
                                ent_id = line.split()[0]
                                ent_dict[str(ent_id).strip()] = [line.split()[1], line.split()[2], line.split()[3],
                                                                 str(' '.join(line.split()[4:])).strip()]
                            # NER tag, start indx, end indx, mention
                        else:
                            print('line:', line)
                            print('Error in the file annotation (ent):', filename)
                            exit()

                    for line in open(str(str(filename).split('.a1')[0])+".a3").readlines():
                        line = str(line).strip('\n')
                        if line.strip().startswith('O'):
                            rel_id = line.split()[0]
                            rel_dict[str(rel_id).strip()] = [line.split()[1], line.split()[2], line.split()[3]]
                        elif line.strip().startswith('*'):
                            star_count += 1
                        else:
                            print('line:', line)
                            print('Error in the file annotation (rel):', filename)
                            exit()

                    if annotation_dict.has_key(key_file):
                        print('Error: Annotation file already exists...')
                        exit()

                    annotation_dict[key_file] = [ent_dict, rel_dict]
                    print('annotation_dict[key_file]:', annotation_dict[key_file])
                    # exit(0)

                # read corresponding data/text file
                if filename.endswith(".txt"):
                    doc_key_file = str(str(filename).split('.txt')[0])
                    # read text file
                    print('Text filename:', filename)
                    text = " "
                    # text = open(filename).read().replace('\n', ' ')
                    text = open(filename).read().replace('\n', ' ')
                    # for line in open(filename).readlines():
                    #    line = str(line).strip('\n')
                    #    text = " "+ text + " " + str(line).strip()

                    if doc_list_dict.has_key(doc_key_file):
                        print('Error: doc file already exists...')
                        exit()
                    else:
                        doc_list_dict[doc_key_file] = []

                    # split text into sentences
                    tokenised_text = tokenizer.tokenize(text)
                    start = 0
                    end = start
                    for tok_text in tokenised_text:
                        end = start + len(tok_text)
                        if 0 < start < len(text):
                            space_counter = 0
                            while text[start + space_counter] == ' ':
                                end+=1 # some sentences have initial space which is removed when the sentence is tokenized
                                space_counter+=1
                        # doc_list_dict[doc_key_file].append([tok_text, start, end])  # text, begin, end
                        doc_list_dict[doc_key_file].append([text[start:end], start, end])  # text, begin, end
                        start = end

                    print('filename:%s #sentences=%s' % (filename, len(doc_list_dict[doc_key_file])))
                    print('filename:%s #doc_list_dict=%s' % (filename, doc_list_dict[doc_key_file]))
                    print('-------------------------------------------------------------------------------------')
                    # print('text:', text)
                    # print('tokenised text:', tokenizer.tokenize(text))
        print(annotation_dict)
        # exit(0)
        load_save_pkl.save_as_pkl(annotation_dict, os.path.join(output_path, "BB2016_annotation_dict.pkl"))
        load_save_pkl.save_as_pkl(doc_list_dict, os.path.join(output_path, "BB2016_doc_list_dict.pkl"))
        # exit(0)
        print('star_count:', star_count)
        # exit(0)
        inter_sent_rel_count = 0
        intra_sent_rel_count = 0
        rel_id=0
        target_sent_list_dict = {}

        # compute inter-sentence and intra-sentence relationships
        for filename in annotation_dict.keys():
            ent_dict, rel_dict = annotation_dict[filename]
            # print('rel_dict:', rel_dict)
            for rel_key in rel_dict.keys():
                rel_id+=1
                relation = rel_dict[rel_key][0]
                relation_e1 = str(rel_dict[rel_key][1]).split(':')[-1]
                relation_e2 = str(rel_dict[rel_key][2]).split(':')[-1]
                relation_e1_type = str(rel_dict[rel_key][1]).split(':')[0]
                relation_e2_type = str(rel_dict[rel_key][2]).split(':')[0]
                start_e1_indx = int(ent_dict[relation_e1][1])
                end_e1_indx = int(ent_dict[relation_e1][2])
                start_e2_indx = int(ent_dict[relation_e2][1])
                end_e2_indx = int(ent_dict[relation_e2][2])

                # check in document the range of the entities, if they lie within or across the sentence boundaries
                sent_begin_end_indx_list = doc_list_dict[filename]

                sentence_number_for_e1 = -1
                sentence_number_for_e2 = -1
                e1_found = False
                e2_found = False

                for sent_begin_end_indx, sent_indx in zip(sent_begin_end_indx_list,
                                                          range(len(sent_begin_end_indx_list))):
                    sent = sent_begin_end_indx[0]
                    begin_indx = int(sent_begin_end_indx[1])
                    end_indx = int(sent_begin_end_indx[2])

                    # check if the two entities lie in the same sentences or across the sentence boundaries
                    if start_e1_indx >= begin_indx and start_e1_indx < end_indx and \
                                    start_e2_indx >= begin_indx and start_e2_indx < end_indx:
                        # intra-sentence relationship
                        sentence_number_for_e1 = sent_indx
                        sentence_number_for_e2 = sent_indx
                        e1_found = True
                        e2_found = True
                        print('------------------------------------------------------------------------')
                        print('Intra-sentence Relation | e1: %s e2: %s Texts: %s' % (
                            ent_dict[relation_e1][-1], ent_dict[relation_e2][-1], sent_begin_end_indx_list))
                        print('------------------------------------------------------------------------')
                        break
                    else:
                        # check for intern-sentence relationship
                        if e1_found is not True and start_e1_indx >= begin_indx and start_e1_indx < end_indx:
                            e1_found = True
                            sentence_number_for_e1 = sent_indx

                        if e2_found is not True and start_e2_indx >= begin_indx and start_e2_indx < end_indx:
                            e2_found = True
                            sentence_number_for_e2 = sent_indx

                        if e1_found and e2_found:
                            print('------------------------------------------------------------------------')
                            print('Inter-sentence Relation | e1: %s e2: %s Texts: %s' % (
                                ent_dict[relation_e1][-1], ent_dict[relation_e2][-1], sent_begin_end_indx_list))
                            print('------------------------------------------------------------------------')
                            break

                if e1_found and e2_found:
                    if sentence_number_for_e1 == sentence_number_for_e2:
                        # intra-sentence relationship
                        intra_sent_rel_count += 1
                    else:
                        # inter-sentence relationship
                        inter_sent_rel_count += 1

                print("sentence_number_for_e1 %s" %sentence_number_for_e1)
                print("sentence_number_for_e2 %s" %sentence_number_for_e2)
                print(sent_begin_end_indx_list[sentence_number_for_e1:sentence_number_for_e2])
                target_sent = ""
                containment = False
                e1_word = ent_dict[relation_e1][-1]
                e2_word = ent_dict[relation_e2][-1]
                e1_type = ent_dict[relation_e1][0]
                e2_type = ent_dict[relation_e2][0]
                k_val = abs(int(sentence_number_for_e2) - int(sentence_number_for_e1))
                if k_val <=3:
                    if k_val == 0:
                        sent_num = sentence_number_for_e2
                        sent = sent_begin_end_indx_list[sent_num][0]
                        target_sent = sent
                        # orig_sent = sent_begin_end_indx[0]
                        if start_e1_indx <= start_e2_indx <= end_e1_indx or start_e2_indx <= start_e1_indx <= end_e2_indx:
                            containment = True
                            # if start_e1_indx <= start_e2_indx <= end_e2_indx <= end_e1_indx:
                            #     target_sent = target_sent.replace(e1_word, "<e1>" + e1_word + "</e1>", 1)
                            #     target_sent = target_sent.replace(e2_word, "<e2>" + e2_word + "</e2>", 1)
                            # elif start_e2_indx <= start_e1_indx <= end_e1_indx <= end_e2_indx:
                            #     target_sent = target_sent.replace(e2_word, "<e2>" + e2_word + "</e2>", 1)
                            #     target_sent = target_sent.replace(e1_word, "<e1>" + e1_word + "</e1>", 1)
                            # with open(os.path.join(output_path,"containment_data_BB2016.txt"),'a') as containment_data_file:
                            #     containment_data_file.write(str(rel_id) + "::" + str(filename.split("/")[-1])  + "::" + str(k_val) + "::" + str(relation)  + "::" + str(e1_word)  + "::" + str(e2_word) + "::" + target_sent + "::" + e1_type + "::" + e2_type)
                            #     containment_data_file.write('\n')
                        else:
                            target_sent = target_sent.replace(e1_word, "<e1>" + e1_word + "</e1>", 1)
                            target_sent = target_sent.replace(e2_word, "<e2>" + e2_word + "</e2>", 1)
                            target_sent_list_dict[rel_id]= [target_sent]
                    else:
                        target_sent_list = []
                        for sent_num in range(min(sentence_number_for_e1,sentence_number_for_e2),max(sentence_number_for_e1,sentence_number_for_e2)+1):
                            sent = sent_begin_end_indx_list[sent_num][0]
                            sent_begin_idx = sent_begin_end_indx_list[sent_num][1]
                            if sent_num == sentence_number_for_e1:
                                # if sent.count(e1_word) == 1:
                                sent = sent.replace(e1_word, "<e1>" + e1_word + "</e1>", 1)
                                # else:
                                # start_e1_tag_idx = start_e1_indx - sent_begin_idx
                                # end_e1_tag_idx = end_e1_indx - sent_begin_idx
                                # sent = sent[:start_e1_tag_idx] + "<e1>" + sent[start_e1_tag_idx:end_e1_tag_idx] +  "</e1>" + sent[end_e1_tag_idx:]
                                # e1_word = sent[start_e1_tag_idx:end_e1_tag_idx]
                            if sent_num == sentence_number_for_e2:
                                # if sent.count(e2_word) == 1:
                                sent = sent.replace(e2_word, "<e2>" + e2_word + "</e2>", 1)
                                # else:
                                # start_e2_tag_idx = start_e2_indx - sent_begin_idx
                                # end_e2_tag_idx = end_e2_indx - sent_begin_idx
                                # sent = sent[:start_e2_tag_idx] + "<e2>" + sent[start_e2_tag_idx:end_e2_tag_idx] +  "</e2>" + sent[end_e2_tag_idx:]
                                # e2_word = sent[start_e2_tag_idx:end_e2_tag_idx]
                            target_sent += " " + sent
                            target_sent_list.append(sent)
                        target_sent_list_dict[rel_id]= target_sent_list
                    tags = ['<e1>','</e1>','<e2>','</e2>']
                    if not all(x in target_sent for x in tags) or containment == True:
                        with open(os.path.join(output_path,error_file),'a') as error_data_file:
                            error_data_file.write(str(rel_id) + "::" + str(filename.split("/")[-1])  + "::" + str(k_val) + "::" + str(relation)  + "::" + str(e1_word)  + "::" + str(e2_word) + "::" + target_sent + "::" + e1_type + "::" + e2_type +  "::" + relation_e1 +  "::" + relation_e2 +  "::" + rel_key)
                            error_data_file.write('\n')
                    with open(os.path.join(output_path,train_file),'a') as data_file:
                        data_file.write(str(filename.split("/")[-1])  + "::" + str(k_val) + "::" + str(relation)  + "::" + str(e1_word)  + "::" + str(e2_word) + "::" + target_sent + "::" + e1_type + "::" + e2_type +  "::" + relation_e1 +  "::" + relation_e2 +  "::" + rel_key)
                        data_file.write('\n')
        load_save_pkl.save_as_pkl(target_sent_list_dict, os.path.join(output_path,target_sent_list_dict_file))
