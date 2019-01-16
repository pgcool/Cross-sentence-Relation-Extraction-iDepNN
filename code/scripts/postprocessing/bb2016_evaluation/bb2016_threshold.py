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

def run_evaluation(test_dataset_type, annotation_results_path):
    # labelled data
    full_results_path = "STRNN_RESULTS/EB_RNN_RC__word2vec_update_2018-08-06_3.decision"    
    expt_input_path = ''../../../data/processed_input/BB2016/test/k_le_1/BB2016_test_data_w_others_k_le_1_sdp_dep.pkl'

    test_final_data = load_save_pkl.load_pickle_file(os.path.join(expt_input_path))	
    print('test_final_data %s' %test_final_data)

    
    testfile_predictions_data = open(test_predictions_file).readlines()
    testfile_predictions_data = [int(elem.strip())for elem in testfile_predictions_data]


/*  test_predictions_prob_file = results_dir + '*.prob'
    test_predictions_prob_file = glob.glob(test_predictions_prob_file)[0]
    testfile_predictions_prob_data = open(test_predictions_prob_file).readlines()
    testfile_predictions_prob_data = [float(elem.strip())for elem in testfile_predictions_prob_data]

    print('len(testfile_predictions_data) %s' %len(testfile_predictions_data))
    print('len(testfile_predictions_prob_data) %s' %len(testfile_predictions_prob_data))

    print(Counter(testfile_predictions_data))*/

    count = 0
    predicted_rel_dict = defaultdict(list)
    for i, elem in enumerate(test_final_data):
        elem=elem[0][0]
        if threshold < 0.5:
            if testfile_predictions_data[i]==1 and 0.5 < testfile_predictions_prob_data[i] <= 1-threshold:
                count+=1
                relation = 'Lives_In'
                doc_id = elem.split('::')[0]
                e1 = elem.split('::')[3]
                e2 = elem.split('::')[4]
                e1_token_type = elem.split('::')[6]
                e2_token_type = elem.split('::')[7]
                e1_token_num = elem.split('::')[8]
                e2_token_num = elem.split('::')[9]
                relation_id = 'R' + str((len(predicted_rel_dict[doc_id]) + 1))
                predicted_rel_dict[doc_id].append([relation_id, relation, e1_token_type, e1_token_num, e2_token_type, e2_token_num])

        if testfile_predictions_data[i]==0 and testfile_predictions_prob_data[i] >= threshold:
                count+=1
                relation = 'Lives_In'
                doc_id = elem.split('::')[0]
                e1 = elem.split('::')[3]
                e2 = elem.split('::')[4]
                e1_token_type = elem.split('::')[6]
                e2_token_type = elem.split('::')[7]
                e1_token_num = elem.split('::')[8]
                e2_token_num = elem.split('::')[9]
                relation_id = 'R' + str((len(predicted_rel_dict[doc_id]) + 1))
                predicted_rel_dict[doc_id].append([relation_id, relation, e1_token_type, e1_token_num, e2_token_type, e2_token_num])

    print(count)
    counter = 0
    # write the result annotations
    for key, value in predicted_rel_dict.iteritems():
        with open(os.path.join(annotation_results_path, key + ".a2"),'wb') as ann_result_file:
            for rel_content in value:
                ann_result_file.write(rel_content[0] + "\t" + rel_content[1] + " " + "Bacteria" + ":" + rel_content[3] + " " + "Location" + ":" + rel_content[5])
                ann_result_file.write("\n")
                counter += 1
    print(counter)

if __name__ == '__main__':
    test_dataset_type = 'k_le_1'
    annotation_results_path = "./annotations/"
    def ensure_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    ensure_directory(annotation_results_path)
    run_evaluation(test_dataset_type, annotation_results_path)
    
    print('Creating zip ')
    shutil.make_archive(annotation_results_path, 'zip', annotation_results_path)