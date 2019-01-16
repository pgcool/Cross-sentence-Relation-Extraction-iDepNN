from __future__ import print_function
import os
import re
import zipfile
import load_save_pkl
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import glob
from collections import defaultdict, Counter
import shutil

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

def run_evaluation(feature_name, train_k_type, test_dataset_type):
    full_results_path = feature_name + '/' + train_k_type + '/test_' + test_dataset_type + '/'
    svm_type = feature_name.split('_')[1]
    # labelled data
    expt_input_path = 'D:/thesis_stuff/code/master_thesis_v2/emnlp_data_preparation/data/BB2016/BB2016_test/' + test_dataset_type + '/test/'
    test_final_data = load_save_pkl.load_pickle_file(os.path.join(expt_input_path, "BB2016_test_data_w_others_" + test_dataset_type + "_sdp_dep.pkl"))
    print('test_final_data %s' %test_final_data)

    # biRNN results
    results_dir = '../../data/svm_bionlp_2016/results/' +  feature_name + '/' + train_k_type + '/test_' + test_dataset_type + '/results_' + svm_type + '/'

    # predictions data
    predictions_file = results_dir + 'results_stats_test' + svm_type + '.txt'
    print(predictions_file)
    results_file_list = open(predictions_file, 'rb').readlines()
    testfile_predictions_data = [int(elem.split(',')[1]) for i, elem in enumerate(results_file_list) if i >0]
    print(len(testfile_predictions_data))

    # annotations
    annotation_results_path = '../../data/svm_bionlp_2016/annotations/' + full_results_path
    print('annotation_results_path %s' %annotation_results_path)

    def ensure_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    ensure_directory(annotation_results_path)

    print('annotation_results_path %s' %annotation_results_path)
    print('len(test_final_data) %s' %len(test_final_data))
    print("# len(testfile_predictions_data) %s" %len(testfile_predictions_data) )
    # testfile_predictions_data = [int(elem.strip())for elem in testfile_predictions_data]
    print('len(testfile_predictions_data) %s' %len(testfile_predictions_data))

    print(Counter(testfile_predictions_data))
    key_file_dir = 'D:/thesis_stuff/datasets/BioNLPdata/BB3-data/BB3-event-rel/BioNLP-ST-2016_BB-event_test/'

    count = 0
    predicted_rel_dict = defaultdict(list)
    for i, elem in enumerate(test_final_data):
        elem=elem[0][0]
        if testfile_predictions_data[i]== 1:
            count+=1
            relation = 'Lives_In'
            doc_id = elem.split('::')[0]
            e1 = elem.split('::')[3]
            e2 = elem.split('::')[4]
            key_file = key_file_dir + doc_id
            # exit(0)
            e1_token_type = elem.split('::')[6]
            e2_token_type = elem.split('::')[7]
            e1_token_num = elem.split('::')[8]
            e2_token_num = elem.split('::')[9]
            relation_id = 'R' + str((len(predicted_rel_dict[doc_id]) + 1))
            predicted_rel_dict[doc_id].append([relation_id, relation, e1_token_type, e1_token_num, e2_token_type, e2_token_num])
    print(count)

    # exit()
    counter = 0
    # write the result annotations
    for key, value in predicted_rel_dict.iteritems():
        # print("Writing for file %s" %key)
        with open(os.path.join(annotation_results_path, key + ".a2"),'a') as ann_result_file:
            for rel_content in value:
                ann_result_file.write(rel_content[0] + "\t" + rel_content[1] + " " + "Bacteria" + ":" + rel_content[3] + " " + "Location" + ":" + rel_content[5])
                ann_result_file.write("\n")
                counter += 1

    print(counter)

if __name__ == '__main__':
    # configurations
    feature_name = 'expt_3_rbf_c10'
    train_k_type = 'train_k_le_1' # [train_k_0, train_k_le_1, train_k_le_2, train_k_le_3]
    test_dataset_types = ['k_0', 'k_le_1', 'k_le_2', 'k_le_3', 'k_le_8', 'k_all']

    for test_dataset_type in test_dataset_types:
        run_evaluation(feature_name, train_k_type, test_dataset_type)

        annotation_results_path = '../../data/svm_bionlp_2016/annotations/' + '/'+ feature_name + '/' \
                                  + train_k_type + '/test_' + test_dataset_type
        print('Creating zip ')
        shutil.make_archive(annotation_results_path, 'zip', annotation_results_path)


