from collections import defaultdict
import ast
import xml.etree.ElementTree as ET
import itertools
from nltk.tokenize import sent_tokenize

def generate_key_dict(key_file):
    key_data = open(key_file).readlines()
    key_data_dict = defaultdict(dict)

    for line in key_data:
        if '<TEMPLATE-' in line:
            template_key = line.split(':=')[0].strip()
            curr_key = key_data_dict[template_key]
        elif 'COMMENT: ' in line or 'ON_THE_JOB' in line:
            continue
        else:
            if ': ' in line:
                k, v = [item.lstrip().strip('\n') for item in line.split(': ')]
                if k == 'CONTENT':
                    curr_key[k] = [v]
                else:
                    curr_key[k] = v
            elif ':=' in line:
                key = line.split(':=')[0].strip()
                if key not in key_data_dict[template_key]:
                    key_data_dict[template_key][key] = {}
                curr_key = key_data_dict[template_key][key]
            elif '<SUCCESSION_EVENT-' in line:
                key_data_dict[template_key]['CONTENT'].append(line.strip())
    return key_data_dict


def generate_rel_tuples(key_data_dict):
    doc_id_relation_tuples = defaultdict(list)
    for template_key, value in key_data_dict.items():
        # ignore files without succesion events
        if 'CONTENT' in value:
            doc_no = ast.literal_eval(value['DOC_NR'])
            doc_id = doc_no[:6] + '-' + doc_no[6:]
            succession_events = set(value['CONTENT'])
            for event in succession_events:
                try:
                    in_and_out = value[event]['IN_AND_OUT']
                    io_person_id = value[in_and_out]['IO_PERSON']
                    succ_org = value[event]['SUCCESSION_ORG']
                    per = ast.literal_eval(value[io_person_id]['PER_NAME'])
                    org = ast.literal_eval(value[succ_org]['ORG_NAME'])
                    post = ast.literal_eval(value[event]['POST'])
                    doc_id_relation_tuples[doc_id].append((per, org, post))
                    if 'PER_ALIAS' in value[io_person_id]:
                        per_alias = ast.literal_eval(value[io_person_id]['PER_ALIAS'])
                        doc_id_relation_tuples[doc_id].append((per_alias, org, post))
                except Exception as e:
                    print('Exception for event %s:' %e)
    return doc_id_relation_tuples


def extract_sentences(data_file):
    doc_id_sentences = defaultdict(list)
    with open(data_file) as f:
        it = itertools.chain('<ROOT>', f, '</ROOT>')
        root = ET.fromstringlist(it)
        for DOC in root:
            doc_id = DOC[1].text.strip().replace(".","")
            curr_sentence_list = []
            for neighbor in DOC:
                if neighbor.tag == "TXT":
                    for p in neighbor:
                        formatted_paragraph = p.text.strip().replace('\n', ' ')
                        curr_sentence_list.extend(sent_tokenize(formatted_paragraph))
            doc_id_sentences[doc_id] = curr_sentence_list
    return doc_id_sentences

def annotate_sentence(doc_id, curr_sents, rel_string, e1, e2):
    e1_sent_idxs = []
    e2_sent_idxs = []
    annotated_data = []
    for sent_idx, sent in enumerate(curr_sents):
        if e1 in sent:
            e1_sent_idxs.append(sent_idx)
        if e2 in sent:
            e2_sent_idxs.append(sent_idx)
    ent_pair_list = []
    shortest_k_val = max_k
    curr_shortest_pair = None
    for x in e1_sent_idxs:
        for y in e2_sent_idxs:
            if abs(x - y) <= shortest_k_val:
                shortest_k_val = abs(x - y)
                curr_shortest_pair = (x,y)
    if curr_shortest_pair is not None:
        ent_pair_list.append(curr_shortest_pair)
    for val in ent_pair_list:
        sample_begin = min(val)
        sample_end = max(val)
        k_val = sample_end - sample_begin
        e1_sent_idx, e2_sent_idx = val
        e1_sent_idx, e2_sent_idx = e1_sent_idx - sample_begin, e2_sent_idx - sample_begin
        data_item_sent = ''
        sent_list = curr_sents[sample_begin:sample_end+1]

        for sent_idx, sent in enumerate(sent_list):
            if sent_idx == e1_sent_idx:
                sent = sent.replace(e1, '<e1>' + e1 + '</e1>')
            if sent_idx == e2_sent_idx:
                sent = sent.replace(e2, '<e2>' + e2 + '</e2>')
            data_item_sent += sent
        annotated_data_item = doc_id + '::' + str(k_val) + '::' + rel_string + '::' + e1  + '::' + e2 + '::' + data_item_sent
        annotated_data.append(annotated_data_item)
    return annotated_data

def annotate_sentences(doc_id_sentences, doc_id_relation_tuples):
    muc6_data = []
    for doc_id, relation_tuples in doc_id_relation_tuples.items():
        curr_sents = doc_id_sentences[doc_id]
        for rel_tuple in relation_tuples:
            per, org, post = rel_tuple
            annotated_data = annotate_sentence(doc_id, curr_sents, rel_string='R1-PerPost', e1=per, e2=post)
            muc6_data.extend(annotated_data)
            annotated_data = annotate_sentence(doc_id, curr_sents, rel_string='R2-PerOrg', e1=per, e2=org)
            muc6_data.extend(annotated_data)
            annotated_data = annotate_sentence(doc_id, curr_sents, rel_string='R3-PostOrg', e1=post, e2=org)
            muc6_data.extend(annotated_data)
    return list(set(muc6_data))

if __name__ == "__main__":
    muc6_source_dir = '../../../../data/source/muc6/'
    data_file = muc6_source_dir + './data.txt'
    key_file = muc6_source_dir + './key.txt'
    output_file = muc6_source_dir + 'muc6_annotated_data.txt'
    max_k = 8
    key_data_dict = generate_key_dict(key_file)
    doc_id_relation_tuples = generate_rel_tuples(key_data_dict)
    doc_id_sentences = extract_sentences(data_file)
    muc6_data = annotate_sentences(doc_id_sentences, doc_id_relation_tuples)
    with open(output_file, 'w') as out_fp:
        out_fp.write('\n'.join(muc6_data))