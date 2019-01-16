import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
from theano import config

def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def extract_target_entities_from_sentence(sentence, merge_multi_words_for_target_entities=False):
    '''
    Extracts target entities from the given sentence
    :param sentence:
    :return:
    '''
    s = str(sentence)
    # extract entity e1
    start_idx = s.find("<e1>")
    end_idx = s.find("</e1>")+4
    substr = s[start_idx:end_idx+1]
    substr = substr.replace("<e1>", "")
    substr = substr.replace("</e1>", "")
    #print('substr:', substr)
    substr = substr.lower()

    if merge_multi_words_for_target_entities == True:
        e1_entity = ""
        if len(substr.split()) > 1:
            for sb in substr.split():
                if e1_entity == "":
                    e1_entity = sb
                else:
                    e1_entity = e1_entity + '_' + sb
        else:
            e1_entity = substr
    else:
        e1_entity = substr

    start_idx = s.find("<e2>")
    end_idx = s.find("</e2>")+4
    substr = s[start_idx:end_idx+1]
    substr = substr.replace("<e2>", "")
    substr = substr.replace("</e2>", "")
    substr = substr.lower()

    if merge_multi_words_for_target_entities == True:
        # extract entity e2
        e2_entity = ""
        if len(substr.split()) > 1:
            for sb in substr.split():
                if e2_entity == "":
                    e2_entity = sb
                else:
                    e2_entity = e2_entity + '_' + sb
        else:
            e2_entity = substr
    else:
        e2_entity = substr

    return e1_entity, e2_entity

def replace_numbers_fn(line):
    tmp_line = ''
    # replace numbers
    # check for interger and decimal numbers and replace them by 0
    for word in line.split():
        if not ("<e1>" in word or "<e2>" in word or "<e2>" in word or "</e1>" in word or "</e2>" in word):
            if re.findall(r'\b\d+\b', word):  #re.findall(r"[-+]?\d*\.\d+|\d+", word)
                #print('word as number/decimal:', word)
                word = '0'

        tmp_line = tmp_line + " " + word

    tmp_line.rstrip(" ")
    tmp_line.lstrip(" ")
    return tmp_line


def replace_regex_fn(line):
    tmp_line = ''
    # replace urls and hyphenated words
    for word in line.split():
        if not ("<e1>" in word or "<e2>" in word or "<e2>" in word or "</e1>" in word or "</e2>" in word):
            if re.findall(r'//(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', word):  #re.findall(r"[-+]?\d*\.\d+|\d+", word)
                #print('word as number/decimal:', word)
                word = 'url'
            elif re.findall(r'^[a-z]+(?:-[a-z]+)+$', word):
                word = str(word).replace("-"," ")

        tmp_line = tmp_line + " " + word

    tmp_line.rstrip(" ")
    tmp_line.lstrip(" ")
    return tmp_line


def replace_time_units_fn(line):
    tmp_line = ''
    for word in line.lower().split():
        # check if the word is time unit
        if word in ['hour', 'hours', 'year','years', 'seconds', 'second', 'minute', 'minutes', 'decade', 'decades',
                    'day', 'days', 'week', 'weeks', 'centuries','century', 'quater', 'quaters', 'annum', 'sec',
                    'hr', 'millisecond', 'milliseconds', 'microsecond', 'microseconds', 'moment', 'month',
                    'months', 'millennium', 'january', 'february', 'march', 'april', 'may',
                    'june', 'july', 'august', 'september', 'october', 'november', 'december', 'today', 'tomorrow',
                    'yesterday']:
            word = 'time'

        tmp_line = tmp_line + " " + word

    tmp_line.rstrip(" ")
    tmp_line.lstrip(" ")

    return tmp_line

def replace_currency_symbols_fn(line):
    tmp_line = ''
    for word in line.split():
        # check for currency symbols
        if word in ["$"]:
            word = 'currency'

        tmp_line = tmp_line + " " + word

    tmp_line.rstrip(" ")
    tmp_line.lstrip(" ")

    return tmp_line

# return the extreme location(indices) of entities in the sentence
def get_target_entities_locs(s, slot_names):
    e1_lower_indx = s.index(slot_names[0])
    e1_highest_indx = s.rindex(slot_names[0])
    e2_lower_indx = s.index(slot_names[1])
    e2_highest_indx = s.rindex(slot_names[1])

    start = np.minimum(e1_lower_indx, e2_lower_indx)

    if e1_highest_indx < e2_highest_indx:
        end = e2_highest_indx + len(slot_names[1])
    else:
        end = e1_highest_indx + len(slot_names[0])

    return start, end

def get_text_neighbourhood_to_entity(s, slot_names, target_entities, left_neighbourhood_size=5,
                                     right_neighbourhood_size=5, get_entity_types=False,
                                     label = None, get_unnorm_position_features=False,
                                     position_indicators=False):

    start, end = get_target_entities_locs(s, slot_names)
    words_left_to_e1 = s[:start].lower().split()[-left_neighbourhood_size:]
    words_right_to_e2 = s[end:].lower().split()[:right_neighbourhood_size]

    left= ''
    right = ''
    for w in words_left_to_e1:
        left = left + ' '+ w
    for w in words_right_to_e2:
        right = right + ' ' + w

    # get text left to e1, text between the target entities including target entities and text rights of e2
    s = left + ' '+ s[start:end]+ ' '+ right

    if get_entity_types == True:
        entity_types_for_words_in_s = []
        if str(getRelation(label)).lower() == 'other':
            relation_terms_for_e1_e2 = ['other_e1', 'other_e2']
        else:
            relation_terms_for_e1_e2 = get_NER_for_entity_from_relation_type(label)

        e1_string = slot_names[0]
        e2_string = slot_names[1]

        for word in s.split():
            e1_split = str(e1_string).split()
            e2_split = str(e2_string).split()

            found_in_e1_split = 0
            found_in_e2_split = 0
            # search in e1_split
            for e1 in e1_split:
                if word.strip() == e1.strip():
                    word = word.strip()
                    if position_indicators == True and ('<e1>' in word or '</e1>' in word):
                        # for <e1>
                        entity_types_for_words_in_s.append('other')
                        entity_types_for_words_in_s.append(relation_terms_for_e1_e2[0])
                        # for </e1>
                        entity_types_for_words_in_s.append('other')
                    else:
                        entity_types_for_words_in_s.append(relation_terms_for_e1_e2[0])
                    found_in_e1_split = 1
                    break

            if found_in_e1_split == 0:
                # search in e2_split
                for e2 in e2_split:
                    if word.strip() == e2.strip():
                        word = word.strip()
                        if position_indicators == True and ('<e2>' in word or '</e2>' in word):
                            # for <e2>
                            entity_types_for_words_in_s.append('other')
                            entity_types_for_words_in_s.append(relation_terms_for_e1_e2[1])
                            # for </e2>
                            entity_types_for_words_in_s.append('other')
                        else:
                            entity_types_for_words_in_s.append(relation_terms_for_e1_e2[1])
                        found_in_e2_split = 1
                        break

            if found_in_e1_split == 0 and found_in_e2_split == 0:
                entity_types_for_words_in_s.append('other')

    #print('before replace, s:', s)
    s = s.replace(slot_names[0], target_entities[0])
    s = s.replace(slot_names[1], target_entities[1])
    #print('after replace, s:', s)

    if position_indicators == True:
        s = s.replace("<e1>", " <e1> ")
        s = s.replace("</e1>", " </e1> ")
        s = s.replace("<e2>", " <e2> ")
        s = s.replace("</e2>", " </e2> ")



    if get_entity_types == True:
        return s,entity_types_for_words_in_s
    else:
        return s


def label_words_by_NE(tokenised_sentence, target_entities):
    word_NEs = []
    for word in tokenised_sentence:
        if str(word).strip() == str(target_entities[0]).strip():
            word_NEs.append(str(target_entities[0]).strip())
        elif str(word).strip() == str(target_entities[1]).strip():
            word_NEs.append(str(target_entities[1]).strip())
        else:
            word_NEs.append('other')
    return word_NEs

# load data and remove puntuations
def read_slot_fill(path, slot_names, target_entities, data_type,
                   get_text_btwn_entities=False, exclude_entity_terms=True,
                   text_neighbourhood_to_entity = False,
                   left_neighbourhood_size = 5,
                   right_neighbourhood_size = 5,
                   NER_for_target_entities=False,
                   get_entity_types=False,
                   get_unnorm_position_feat=False,
                   get_position_indicators=False):
    # To replace the numbers by 0
    replace_numbers = True
    # To replace the currency symbols by 'currency' token
    replace_currency_symbols = True
    # To replace the time units by 'time' token
    replace_time_units = True

    labels = []
    sentences = []
    pos_featues_s = []
    #named_entities = []
    lines = [line.rstrip('\n\r') for line in open(path)]

    if get_entity_types == True:
        entity_types_for_words = [[] for x in range(len(lines))]

    no_filler_name_count = 0
    #target_entities = [[]]
    #target_entities.append([])
    tmp_line = ''
    empty_count = 0
    line_count = 0
    for line in lines:
        if data_type == 'train':
            splits = line.split('::')
            label = splits[1].strip()
            s = str(splits[5]).strip()
        elif data_type == 'test' or data_type == 'dev':
            splits = line.split(':')
            label = str(splits[0]).strip()
            s = str(splits[5]).strip()

        if label == '+':
            label = 1
        elif label == '-':
            label = 0
        else:
            print('invalid label:', label)
            exit()

        if slot_names[0] not in s or slot_names[1] not in s:
            #print('slot_name(s) is not present')
            #print('skip the sentence:', s)
            no_filler_name_count = no_filler_name_count + 1
            if data_type == 'train':
                continue
            else:
                s = '<name> <filler>'
                #label_each_word_by_NE = [target_entities[0], target_entities[1]]

        #print('sentence:', s)
        # replace currency symbols by 'currency'
        if replace_currency_symbols == True:
            s = repalce_currency_symbols_fn(s)

        # replace time units by 'time'
        if replace_time_units == True:
            s = replace_time_units_fn(s)

        # replace numbers(int/float) by '0'
        if replace_numbers == True:
            s = replace_numbers_fn(s)

        if get_position_indicators == True:
            s = s.replace(slot_names[0]," <e1> " +slot_names[0]+" </e1> ")
            s = s.replace(slot_names[1]," <e2> " +slot_names[1]+" </e2> ")

        if get_text_btwn_entities == True:
            e1_lower_indx = s.index(slot_names[0])
            e1_highest_indx = s.rindex(slot_names[0])
            e2_lower_indx = s.index(slot_names[1])
            e2_highest_indx = s.rindex(slot_names[1])

            if e1_lower_indx < e2_lower_indx:
                start = e1_lower_indx + len(slot_names[0])
            else:
                start = e2_lower_indx + len(slot_names[1])

            #end = np.max(e1_highest_indx, e2_highest_indx)
            end = e2_highest_indx if e2_highest_indx > e1_highest_indx else e1_highest_indx
            s = s[start:end]

            if s == ' ' or len(s.split()) == 0:
                    empty_count +=1
                    s = '<unk>'

            if get_unnorm_position_feat == True:
                tmp_s = slot_names[0] + ' '+ s +' '+ slot_names[1]
                # compute position features for each word in the sentence,
                # returna a list of [d_e1, d_e2] for each word in the sentence
                pos_feat = get_position_features_for_each_token_slot_fill(tmp_s.split(),
                                                                            [slot_names[0], slot_names[1]])
                pos_feat = pos_feat[1:]
                pos_feat = pos_feat[:-1]
                pos_featues_s.append(pos_feat)

                if len(pos_feat) != len(s.split()):
                    raise ValueError('Number of words in unnormalised_pos_feat_seq_train and seq_train are not equal !!')

                if get_position_indicators == True:
                    s =  '</e1>' + ' '+ s + ' '+'<e2>'

            # return NE list
            #label_each_word_by_NE = label_words_by_NE(s.split(), slot_names)
            if len(s) == 0:
                empty_count +=1


        if text_neighbourhood_to_entity == True:
            if get_entity_types == True:
                s, entity_types_for_words_in_s = get_text_neighbourhood_to_entity(s, slot_names=slot_names,
                                                                                    target_entities=[target_entities],
                                                                                    left_neighbourhood_size=left_neighbourhood_size,
                                                                                    right_neighbourhood_size=right_neighbourhood_size,
                                                                                    get_entity_types=get_entity_types,
                                                                                    label=label,
                                                                                    position_indicators=get_position_indicators)
                if len(s.split()) != len(entity_types_for_words_in_s):
                    raise ValueError("Dimension mismatch for s and entity_types_for_words_in_s")

                for ne_type in entity_types_for_words_in_s:
                    entity_types_for_words[line_count].append(ne_type)
            else:
                s = get_text_neighbourhood_to_entity(s, slot_names=slot_names,target_entities=target_entities,
                                                     left_neighbourhood_size=left_neighbourhood_size,
                                                     right_neighbourhood_size=right_neighbourhood_size)

            if get_unnorm_position_feat == True:
                # compute position features for each word in the sentence,
                # returns a list of [d_e1, d_e2] for each word in the sentence
                pos_feat = get_position_features_for_each_token_slot_fill(s.lower().split(), [slot_names[0], slot_names[1]])
                pos_featues_s.append(pos_feat)
                #pos_featues_s.append(pos_feat)

            # return NE list
            #label_each_word_by_NE = label_words_by_NE(s.split(), target_entities)

        if exclude_entity_terms == True:
            if get_unnorm_position_feat == True:
                i = 0
                pos_feat = pos_featues_s[-1]
                print('pos_feat at last:', pos_feat)
                indicres_to_del = []
                for word in s.split():
                    if word.strip() == str(slot_names[0]).strip() or word.strip() == str(slot_names[1]).strip():
                        indicres_to_del.append(i)
                    i+=1
                pos_feat = np.delete(pos_feat, indicres_to_del, axis=0)
                s = s.replace(slot_names[0], ' ')
                s = s.replace(slot_names[1], ' ')
                pos_featues_s[-1] = pos_feat

            if get_position_indicators == True:
                import re
                # regular  expression to replace all between <e1> and </e1>; and <e2 and </e2>
                print('s before regular expres:', s)
                s = re.sub("^<e1>\w*<\e1>$", '<e1>  <\e1>', s)
                print('s after regular expres:', s)

                print('s before regular expres:', s)
                s = re.sub(r"<e2>\w*<\e2>", '<e2>  <\e2>', s)
                print('s after regular expres:', s)

            # return NE list
            #label_each_word_by_NE = label_words_by_NE(s.split(), slot_names)

        if exclude_entity_terms != True and text_neighbourhood_to_entity != True and get_text_btwn_entities !=True:
            # substitute the slot names by target entities
            s = s.replace(slot_names[0], target_entities[0])
            s = s.replace(slot_names[1], target_entities[1])
            # return NE list
            #label_each_word_by_NE = label_words_by_NE(s.split(), target_entities)

        if len([word for word in s.lower().split() if word not in stoplist]) == 0:
            #print(label, ': ', s)
            #s = '<unk>'
            continue

        if get_position_indicators is not True:
            s = s.replace(slot_names[0], target_entities[0])
            s = s.replace(slot_names[1], target_entities[1])
        else:
            s = s.replace(slot_names[0], " <e1> "+str(target_entities[0])+ " </e1> ")
            s = s.replace(slot_names[1], " <e2> "+str(target_entities[1])+ " </e2> ")

        labels.append(label)
        sentences.append(s)
        line_count+=1
        #named_entities.append(label_each_word_by_NE)

    print('empty count:', empty_count)
    print('no_filler_name_count:', no_filler_name_count)

    if get_entity_types == True:
        return sentences, labels, target_entities, entity_types_for_words, pos_featues_s
    else:
        return sentences, labels, target_entities, None, pos_featues_s

    #return sentences, labels, named_entities

def read_filler_candidate_sent_list(cand_sentence_list, slot_names, target_entities, data_type,
                   get_text_btwn_entities=False, exclude_entity_terms=True,
                   text_neighbourhood_to_entity = False,
                   left_neighbourhood_size = 5,
                   right_neighbourhood_size = 5,
                   NER_for_target_entities=False,
                   get_entity_types=False,
                   get_unnorm_position_feat=False,
                   get_position_indicators=False):

    # To replace the numbers by 0
    replace_numbers = True
    # To replace the currency symbols by 'currency' token
    replace_currency_symbols = True
    # To replace the time units by 'time' token
    replace_time_units = True

    lines = cand_sentence_list

    labels = []
    sentences = []
    pos_featues_s = []

    if get_entity_types == True:
        entity_types_for_words = [[] for x in range(len(lines))]

    no_filler_name_count = 0
    #target_entities = [[]]
    #target_entities.append([])
    tmp_line = ''
    empty_count = 0
    line_count = 0
    for line in lines:
        s = line.strip()

        if slot_names[0] not in s or slot_names[1] not in s:
            #print('slot_name(s) is not present')
            #print('skip the sentence:', s)
            no_filler_name_count = no_filler_name_count + 1
            if data_type == 'train':
                continue
            else:
                s = '<name> <filler>'
                #label_each_word_by_NE = [target_entities[0], target_entities[1]]

        #print('sentence:', s)
        # replace currency symbols by 'currency'
        if replace_currency_symbols == True:
            s = repalce_currency_symbols_fn(s)

        # replace time units by 'time'
        if replace_time_units == True:
            s = replace_time_units_fn(s)

        # replace numbers(int/float) by '0'
        if replace_numbers == True:
            s = replace_numbers_fn(s)

        if get_position_indicators == True:
            s = s.replace(slot_names[0]," <e1> " +slot_names[0]+" </e1> ")
            s = s.replace(slot_names[1]," <e2> " +slot_names[1]+" </e2> ")

        if get_text_btwn_entities == True:
            e1_lower_indx = s.index(slot_names[0])
            e1_highest_indx = s.rindex(slot_names[0])
            e2_lower_indx = s.index(slot_names[1])
            e2_highest_indx = s.rindex(slot_names[1])

            if e1_lower_indx < e2_lower_indx:
                start = e1_lower_indx + len(slot_names[0])
            else:
                start = e2_lower_indx + len(slot_names[1])

            #end = np.max(e1_highest_indx, e2_highest_indx)
            end = e2_highest_indx if e2_highest_indx > e1_highest_indx else e1_highest_indx
            s = s[start:end]

            if s == ' ' or len(s.split()) == 0:
                    empty_count +=1
                    s = '<unk>'

            if get_unnorm_position_feat == True:
                tmp_s = slot_names[0] + ' '+ s +' '+ slot_names[1]
                # compute position features for each word in the sentence,
                # returna a list of [d_e1, d_e2] for each word in the sentence
                pos_feat = get_position_features_for_each_token_slot_fill(tmp_s.split(),
                                                                            [slot_names[0], slot_names[1]])
                pos_feat = pos_feat[1:]
                pos_feat = pos_feat[:-1]
                pos_featues_s.append(pos_feat)

                if len(pos_feat) != len(s.split()):
                    raise ValueError('Number of words in unnormalised_pos_feat_seq_train and seq_train are not equal !!')

                if get_position_indicators == True:
                    s =  '</e1>' + ' '+ s + ' '+'<e2>'

            # return NE list
            #label_each_word_by_NE = label_words_by_NE(s.split(), slot_names)
            if len(s) == 0:
                empty_count +=1


        if text_neighbourhood_to_entity == True:
            if get_entity_types == True:
                s, entity_types_for_words_in_s = get_text_neighbourhood_to_entity(s, slot_names=slot_names,
                                                                                    target_entities=[target_entities],
                                                                                    left_neighbourhood_size=left_neighbourhood_size,
                                                                                    right_neighbourhood_size=right_neighbourhood_size,
                                                                                    get_entity_types=get_entity_types,
                                                                                    label=label,
                                                                                    position_indicators=get_position_indicators)
                if len(s.split()) != len(entity_types_for_words_in_s):
                    raise ValueError("Dimension mismatch for s and entity_types_for_words_in_s")

                for ne_type in entity_types_for_words_in_s:
                    entity_types_for_words[line_count].append(ne_type)
            else:
                s = get_text_neighbourhood_to_entity(s, slot_names=slot_names,target_entities=target_entities,
                                                     left_neighbourhood_size=left_neighbourhood_size,
                                                     right_neighbourhood_size=right_neighbourhood_size)

            if get_unnorm_position_feat == True:
                # compute position features for each word in the sentence,
                # returns a list of [d_e1, d_e2] for each word in the sentence
                pos_feat = get_position_features_for_each_token_slot_fill(s.lower().split(), [slot_names[0], slot_names[1]])
                pos_featues_s.append(pos_feat)
                #pos_featues_s.append(pos_feat)

            # return NE list
            #label_each_word_by_NE = label_words_by_NE(s.split(), target_entities)

        if exclude_entity_terms == True:
            if get_unnorm_position_feat == True:
                i = 0
                pos_feat = pos_featues_s[-1]
                print('pos_feat at last:', pos_feat)
                indicres_to_del = []
                for word in s.split():
                    if word.strip() == str(slot_names[0]).strip() or word.strip() == str(slot_names[1]).strip():
                        indicres_to_del.append(i)
                    i+=1
                pos_feat = np.delete(pos_feat, indicres_to_del, axis=0)
                s = s.replace(slot_names[0], ' ')
                s = s.replace(slot_names[1], ' ')
                pos_featues_s[-1] = pos_feat

            if get_position_indicators == True:
                import re
                # regular  expression to replace all between <e1> and </e1>; and <e2 and </e2>
                print('s before regular expres:', s)
                s = re.sub("^<e1>\w*<\e1>$", '<e1>  <\e1>', s)
                print('s after regular expres:', s)

                print('s before regular expres:', s)
                s = re.sub(r"<e2>\w*<\e2>", '<e2>  <\e2>', s)
                print('s after regular expres:', s)

            # return NE list
            #label_each_word_by_NE = label_words_by_NE(s.split(), slot_names)

        if exclude_entity_terms != True and text_neighbourhood_to_entity != True and get_text_btwn_entities !=True:
            # substitute the slot names by target entities
            s = s.replace(slot_names[0], target_entities[0])
            s = s.replace(slot_names[1], target_entities[1])
            # return NE list
            #label_each_word_by_NE = label_words_by_NE(s.split(), target_entities)

        if len([word for word in s.lower().split() if word not in stoplist]) == 0:
            #print(label, ': ', s)
            #s = '<unk>'
            continue

        if get_position_indicators is not True:
            s = s.replace(slot_names[0], target_entities[0])
            s = s.replace(slot_names[1], target_entities[1])
        else:
            s = s.replace(slot_names[0], " <e1> "+str(target_entities[0])+ " </e1> ")
            s = s.replace(slot_names[1], " <e2> "+str(target_entities[1])+ " </e2> ")

        sentences.append(s)
        line_count+=1
        #named_entities.append(label_each_word_by_NE)

    print('empty count:', empty_count)
    print('no_filler_name_count:', no_filler_name_count)

    if get_entity_types == True:
        return sentences, labels, target_entities, entity_types_for_words, pos_featues_s
    else:
        return sentences, labels, target_entities, None, pos_featues_s


# load candidate sentence list and remove puntuations
def read_filler_candidate_sent_list_bk(cand_sentence_list, slot_names,
                                    target_entities, get_text_btwn_entities=False, exclude_entity_terms=False,
                                    text_neighbourhood_to_entity = False,
                                    left_neighbourhood_size = 5,
                                    right_neighbourhood_size = 5):
    # To replace the numbers by 0
    replace_numbers = True
    # To replace the currency symbols by 'currency' token
    replace_currency_symbols = True
    # To replace the time units by 'time' token
    replace_time_units = True

    sentences = []
    lines = cand_sentence_list

    no_filler_name_count = 0
    empty_count = 0
    for line in lines:
        s = line.strip()

        if slot_names[0] not in s or slot_names[1] not in s:
            no_filler_name_count = no_filler_name_count + 1
            continue

        # replace currency symbols by 'currency'
        if replace_currency_symbols == True:
            s = repalce_currency_symbols_fn(s)

        # replace time units by 'time'
        if replace_time_units == True:
            s = replace_time_units_fn(s)

        # replace numbers(int/float) by '0'
        if replace_numbers == True:
            s = replace_numbers_fn(s)

        if get_text_btwn_entities == True:
            e1_lower_indx = s.index(slot_names[0])
            e1_highest_indx = s.rindex(slot_names[0])
            e2_lower_indx = s.index(slot_names[1])
            e2_highest_indx = s.rindex(slot_names[1])

            if e1_lower_indx < e2_lower_indx:
                start = e1_lower_indx + len(slot_names[0])
            else:
                start = e2_lower_indx + len(slot_names[1])

            end = np.max(e1_highest_indx, e2_highest_indx)
            s = s[start:end]
            if len(s) == 0:
                empty_count +=1

        if text_neighbourhood_to_entity == True:
            s = get_text_neighbourhood_to_entity(s, slot_names=slot_names,target_entities=target_entities,
                                                 left_neighbourhood_size=left_neighbourhood_size,
                                                 right_neighbourhood_size=right_neighbourhood_size)

        if exclude_entity_terms == True:
            s = s.replace(slot_names[0], ' ')
            s = s.replace(slot_names[1], ' ')

        if exclude_entity_terms != True and text_neighbourhood_to_entity != True and get_text_btwn_entities !=True:
            # substitute the slot names by target entities
            s = s.replace(slot_names[0], target_entities[0])
            s = s.replace(slot_names[1], target_entities[1])

        if len([word for word in s.lower().split() if word not in stoplist]) == 0:
            #print(label, ': ', s)
            #s = '<unk>'
            continue
        sentences.append(s)

    return sentences


#from data.getMacroFScore import getRelation, getIndex
def get_NER_for_entity_from_relation_type(label_idx):
    relation_type = getRelation(label_idx)
    relation_terms  = str(relation_type).lower().split('-', 1)

    if relation_terms[1].find("e1") > relation_terms[1].find("e2"):
        tmp = relation_terms[0].lower()
        relation_terms[0] = relation_terms[1].split('(')[0].lower()
        relation_terms[1] = tmp
    else:
        relation_terms[0] = relation_terms[0].lower()
        relation_terms[1] = relation_terms[1].split('(')[0].lower()

    return relation_terms


def read_semVal_train(path, get_text_btwn_entities=False,
                      exclude_entity_terms=False,
                      text_neighbourhood_to_entity = False,
                      left_neighbourhood_size = 5,
                      right_neighbourhood_size = 5,
                      remove_other_class = False,
                      NER_for_target_entities=False,
                      merge_multi_words_for_target_entities=False,
                      get_entity_types=False, get_unnorm_position_feat=False,
                      get_position_indicators=False):
    '''
    Reads data and extract target entities from the sentences
    Also, removes <e1>, </e1>,<e2>,</e2> and replaces by respective entities.
    Words in multi-word entity are concatenated by '_' and form single word composite entity
    :param path:
    :return:
        sentences:
        labels:
        target_entities:
    '''

    # To replace the numbers by 0
    replace_numbers = True
    # To replace the currency symbols by 'currency' token
    replace_currency_symbols = True
    # To replace the time units by 'time' token
    replace_time_units = True

    labels = []
    sentences = []
    pos_featues_s = []
    lines = [line.rstrip('\n\r') for line in open(path)]

    line_count = 0
    if remove_other_class == True:
        lines_without_other_class_count = 0
        for line in lines:
            #drop OTHERS
            if int(line.split(':')[0]) != 18:
                lines_without_other_class_count += 1
        target_entities = [[]  for x in range(lines_without_other_class_count)]
    else:
        target_entities = [[]  for x in range(np.array(lines).shape[0])]
    #print('in train: target_entities.shape', np.array(target_entities).shape)

    for line in lines:
        if remove_other_class == True:
            #drop OTHERS
            if int(line.split(':')[0]) != 18:
                # extract target entities from each sentence
                s = line.split(':', 1)[1]

                # replace currency symbols by 'currency'
                if replace_currency_symbols == True:
                    s = repalce_currency_symbols_fn(s)

                # replace time units by 'time'
                if replace_time_units == True:
                    s = replace_time_units_fn(s)

                # replace numbers(int/float) by '0'
                if replace_numbers == True:
                    s = replace_numbers_fn(s)

                #label = labels[line_count]
                label = int(line.split(':', 1)[0])
                if NER_for_target_entities == True:
                    relation_terms = get_NER_for_entity_from_relation_type(label)
                    e1 = relation_terms[0]
                    e2 = relation_terms[1]
                else:
                    # extract target entities and merge the tokens in entity mentions if it is multi-word
                    # 'merge_multi_words_for_target_entities' is False then e1 and e1 are the string with multiple words
                    # seperated ny ' '
                    e1, e2 = extract_target_entities_from_sentence(s,
                                                                   merge_multi_words_for_target_entities
                                                                   =merge_multi_words_for_target_entities)
                #print('e1:', e1, ' e2:', e2)
                target_entities[line_count].append(e1)
                target_entities[line_count].append(e2)
                line_count+=1
        else:
            # extract target entities from each sentence
            s = line.split(':', 1)[1]

            # replace currency symbols by 'currency'
            if replace_currency_symbols == True:
                s = repalce_currency_symbols_fn(s)

            # replace time units by 'time'
            if replace_time_units == True:
                s = replace_time_units_fn(s)

            # replace numbers(int/float) by '0'
            if replace_numbers == True:
                s = replace_numbers_fn(s)

            #label = labels[line_count]
            label = int(line.split(':', 1)[0])
            if NER_for_target_entities == True:
                relation_terms = get_NER_for_entity_from_relation_type(label)
                e1 = relation_terms[0]
                e2 = relation_terms[1]
            else:
                # extract target entities and merge the toked in entity  in it is multi-word and also merge the multi-word
                # okens of target entity in the sentence nd return the sentence
                e1, e2 = extract_target_entities_from_sentence(s, merge_multi_words_for_target_entities=
                                                               merge_multi_words_for_target_entities)

            if len(str(e1).split())> 1:
                e1 = str(e1).split()[-1].strip()

            if len(str(e2).split())> 1:
                e2 = str(e2).split()[-1].strip()

            target_entities[line_count].append(e1)
            target_entities[line_count].append(e2)
            line_count+=1

    if get_entity_types == True:
        entity_types_for_words = [[] for x in range(line_count)]

    #target_entities = [[]]
    #target_entities.append([])
    tmp_line = ''
    line_count = 0
    empty_count = 0
    for line in lines:
        #drop OTHERS
        if remove_other_class == True and int(line.split(':')[0]) == 18:
            continue
        else:

            # extract target entities from each sentence
            s = line.split(':', 1)[1]

            # replace currency symbols by 'currency'
            if replace_currency_symbols == True:
                s = repalce_currency_symbols_fn(s)

            # replace time units by 'time'
            if replace_time_units == True:
                s = replace_time_units_fn(s)

            # replace numbers(int/float) by '0'
            if replace_numbers == True:
                s = replace_numbers_fn(s)

            label = int(line.split(':', 1)[0])


            if NER_for_target_entities == True:
                relation_terms = get_NER_for_entity_from_relation_type(label)
                e1 = relation_terms[0]
                e2 = relation_terms[1]
                #print('e1:', e1)
                #print('e2:', e2)
            else:
                # extract target entities and merge the tokens in entity in it is multi-word
                e1, e2= extract_target_entities_from_sentence(s, merge_multi_words_for_target_entities=
                                                              merge_multi_words_for_target_entities)


            # take last word of entity mentions for multi-word entity mention
            if len(str(e1).split())> 1:
                e1_string = "<e1>"+str(e1).split()[-1].strip()+"</e1>"
                e1_remain = ''
                for r in str(e1).split()[:-1]:
                    e1_remain = e1_remain + ' ' + r
            else:
                e1_remain = ''
                e1_string = "<e1>"+e1.strip()+"</e1>"

            if len(str(e2).split())> 1:
                e2_string = "<e2>"+str(e2).split()[-1].strip()+"</e2>"
                e2_remain = ''
                for r in str(e2).split()[:-1]:
                    e2_remain = e2_remain + ' ' + r
            else:
                e2_remain = ''
                e2_string = "<e2>"+e2.strip()+"</e2>"


            '''
            # substitue with last word from multi-word target entities
            e1_string = "<e1>"+e1.strip()+"</e1>"
            e2_string = "<e2>"+e2.strip()+"</e2>"
            '''

            #print('line before extracting entities:\n', s)
            #replace '...<e1>...<\e1>....' by ...e1...
            start_idx = s.find("<e1>")
            end_idx = s.find("</e1>")+4
            s = s[:start_idx]+ ' ' + e1_remain + ' ' + e1_string+ ' ' +s[end_idx+1:]
            #replace '...<e2>...<\e2>....' by ...e2...
            start_idx = s.find("<e2>")
            end_idx = s.find("</e2>")+4
            s = s[:start_idx]+ ' ' + e2_remain + ' ' + e2_string + ' ' +s[end_idx+1:]
            #print('line after extracting entities:\n', s)

            # get_text_btwn_entities including target entity terms
            if get_text_btwn_entities == True:
                e1_lower_indx = s.index(e1_string)
                e1_highest_indx = s.rindex(e1_string)
                e2_lower_indx = s.index(e2_string)
                e2_highest_indx = s.rindex(e2_string)

                if e1_lower_indx < e2_lower_indx:
                    start = e1_lower_indx + len(e1_string)
                else:
                    start = e2_lower_indx + len(e2_string)

                end = e2_highest_indx if e2_highest_indx > e1_highest_indx else e1_highest_indx

                s = s[start:end]
                if s == ' ' or len(s.split()) == 0:
                    empty_count +=1
                    s = '<unk>'
                    #continue

                if get_unnorm_position_feat == True:
                    tmp_s = e1_string + ' '+ s +' '+ e2_string
                    # compute position features for each word in the sentence,
                    # returna a list of [d_e1, d_e2] for each word in the sentence
                    pos_feat = get_position_features_for_each_token_slot_fill(tmp_s.split(),
                                                                              [e1_string, e2_string])
                    pos_feat = pos_feat[1:]
                    pos_feat = pos_feat[:-1]
                    pos_featues_s.append(pos_feat)

                    if len(pos_feat) != len(s.split()):
                        raise ValueError('Number of words in unnormalised_pos_feat_seq_train and seq_train are not equal !!')

                if get_position_indicators == True:
                    s =  '</e1>' + ' '+ s + ' '+'<e2>'

            # get text neighbourhood to target entities
            if text_neighbourhood_to_entity == True:
                if get_entity_types == True:

                    s, entity_types_for_words_in_s = get_text_neighbourhood_to_entity(s, slot_names=[e1_string,e2_string],
                                                                                        target_entities=[e1_string, e2_string],
                                                                                        left_neighbourhood_size=left_neighbourhood_size,
                                                                                        right_neighbourhood_size=right_neighbourhood_size,
                                                                                        get_entity_types=get_entity_types,
                                                                                        label=label,
                                                                                        position_indicators=get_position_indicators)
                    if len(s.split()) != len(entity_types_for_words_in_s):
                        raise ValueError("Dimension mismatch for s and entity_types_for_words_in_s")

                    #print('s:', s)
                    #print('entity_types_for_words_in_s:', entity_types_for_words_in_s)

                    for ne_type in entity_types_for_words_in_s:
                        entity_types_for_words[line_count].append(ne_type)
                else:
                    s = get_text_neighbourhood_to_entity(s, slot_names=[e1_string,e2_string],
                                                            target_entities=[e1_string, e2_string],
                                                            left_neighbourhood_size=left_neighbourhood_size,
                                                            right_neighbourhood_size=right_neighbourhood_size
                                                            )


                if get_unnorm_position_feat == True:
                    # compute position features for each word in the sentence,
                    # returns a list of [d_e1, d_e2] for each word in the sentence
                    pos_feat = get_position_features_for_each_token_slot_fill(s.lower().split(), [e1_string, e2_string])
                    pos_featues_s.append(pos_feat)
                    #pos_featues_s.append(pos_feat)

            # exclude target entity terms
            if exclude_entity_terms == True:
                if get_unnorm_position_feat == True:
                    i = 0
                    pos_feat = pos_featues_s[-1]
                    print('pos_feat at last:', pos_feat)
                    indicres_to_del = []
                    for word in s.split():
                        if word.strip() == e1_string.strip() or word.strip() == e2_string.strip():
                            indicres_to_del.append(i)
                        i+=1
                    pos_feat = np.delete(pos_feat, indicres_to_del, axis=0)
                    s = s.replace(e1_string, ' ')
                    s = s.replace(e2_string, ' ')
                    pos_featues_s[-1] = pos_feat

                if get_position_indicators == True:
                    import re
                    # regular  expression to replace all between <e1> and </e1>; and <e2 and </e2>
                    print('s before regular expres:', s)
                    s = re.sub("^<e1>\w*<\e1>$", '<e1>  <\e1>', s)
                    print('s after regular expres:', s)

                    print('s before regular expres:', s)
                    s = re.sub(r"<e2>\w*<\e2>", '<e2>  <\e2>', s)
                    print('s after regular expres:', s)

            if len([word for word in s.lower().split()]) == 0:
                continue

            if get_position_indicators is not True:
                s = s.replace("<e1>", "")
                s = s.replace("</e1>", "")
                s = s.replace("<e2>", "")
                s = s.replace("</e2>", "")
            else:
                s = s.replace("<e1>", " <e1> ")
                s = s.replace("</e1>", " </e1> ")
                s = s.replace("<e2>", " <e2> ")
                s = s.replace("</e2>", " </e2> ")


            #labels.append(int(line.split(':')[0]))
            #sentences.append(line.split(':')[1])

            labels.append(label)
            sentences.append(s)
            line_count+=1

            #pos_indicators(pos_ind)

    #print('labels:\n', labels)
    #print('sentences:\n', sentences)
    #sentences = remove_punctuation_from_sentences(sentences)

    if get_entity_types == True:
        return sentences, labels, target_entities, entity_types_for_words, pos_featues_s
    else:
        return sentences, labels, target_entities, None, pos_featues_s

def read_semVal_test(test_file_path, test_label_file_path, remove_other_class=True,
                     get_text_btwn_entities=False, exclude_entity_terms=True,
                     text_neighbourhood_to_entity = False,
                     left_neighbourhood_size = 5,
                     right_neighbourhood_size = 5,
                     NER_for_target_entities=False,
                     merge_multi_words_for_target_entities=
                     True):
    '''
    Reads data and extract target entities from the sentences
    Also, removes <e1>, </e1>,<e2>,</e2> and replaces by respective entities.
    Words in multi-word entity are concatenated by '_' and form single word composite entity
    :param path:
    :return:
        sentences:
        target_entities:
    '''

    # To replace the numbers by 0
    replace_numbers = True
    # To replace the currency symbols by 'currency' token
    replace_currency_symbols = True
    # To replace the time units by 'time' token
    replace_time_units = True

    sentences = []
    lines = [line.rstrip('\n\r') for line in open(test_file_path)]

    labels = []
    label_lines = [label_line.rstrip('\n\r') for label_line in open(test_label_file_path)]

    # get the index of other class from label file and remove those elements at indices from test data
    index_of_other_class = []

    if remove_other_class == True:
        for label_line in label_lines:
            #print('label_line:', label_line)
            if label_line.split('\t',1)[1] != "Other":
                labels.append(label_line.split('\t',1)[1].strip())
            else:
                index_of_other_class.append(int(label_line.split('\t',1)[0]))
    else:
        for label_line in label_lines:
            labels.append(label_line.split('\t',1)[1].strip())

    target_entities = [[]  for x in range(np.array(labels).shape[0])]

    index_count = 1
    line_count = 0
    empty_count = 0
    for line in lines:
        if remove_other_class == True:
            if index_count not in index_of_other_class:
                s = line.split(':', 1)[1]

                # replace currency symbols by 'currency'
                if replace_currency_symbols == True:
                    s = repalce_currency_symbols_fn(s)

                # replace time units by 'time'
                if replace_time_units == True:
                    s = replace_time_units_fn(s)

                # replace numbers(int/float) by '0'
                if replace_numbers == True:
                    s = replace_numbers_fn(s)

                if NER_for_target_entities == True:
                    label = label_lines[index_count-1]
                    label = label.split('\t',1)[1].strip()
                    relation_terms = get_NER_for_entity_from_relation_type(getIndex(label))
                    e1 = relation_terms[0]
                    e2 = relation_terms[1]
                else:
                    # extract target entitiees annd merge the toked in entity  in it is multi-word
                    e1, e2 = extract_target_entities_from_sentence(s, merge_multi_words_for_target_entities=
                                                                   merge_multi_words_for_target_entities)

                target_entities[line_count].append(e1)
                target_entities[line_count].append(e2)
                line_count+=1

            index_count = index_count + 1
        else:
            s = line.split(':', 1)[1]
            # replace currency symbols by 'currency'
            if replace_currency_symbols == True:
                s = repalce_currency_symbols_fn(s)

            # replace time units by 'time'
            if replace_time_units == True:
                s = replace_time_units_fn(s)

            # replace numbers(int/float) by '0'
            if replace_numbers == True:
                s = replace_numbers_fn(s)

            e1, e2 = extract_target_entities_from_sentence(s, merge_multi_words_for_target_entities=
                                                           merge_multi_words_for_target_entities)
            target_entities[line_count].append(e1)
            target_entities[line_count].append(e2)
            line_count+=1

    #count start from 1 in label file
    index_count = 1
    count = 0
    empty_count = 0
    for line in lines:
        if remove_other_class == True:
            if index_count not in index_of_other_class:
                #print('index_count:',index_count)
                #print('label:', labels[count])
                #print('sentence:', line.split(':',1)[1])
                s = line.split(':', 1)[1]

                # replace currency symbols by 'currency'
                if replace_currency_symbols == True:
                    s = repalce_currency_symbols_fn(s)

                # replace time units by 'time'
                if replace_time_units == True:
                    s = replace_time_units_fn(s)

                # replace numbers(int/float) by '0'
                if replace_numbers == True:
                    s = replace_numbers_fn(s)

                if NER_for_target_entities == True:
                    label = label_lines[index_count-1]
                    label = label.split('\t',1)[1].strip()
                    relation_terms = get_NER_for_entity_from_relation_type(getIndex(label))
                    e1 = relation_terms[0]
                    e2 = relation_terms[1]
                else:
                    # extract target entitiees annd merge the toked in entity  in it is multi-word
                    e1, e2 = extract_target_entities_from_sentence(s, merge_multi_words_for_target_entities=
                                                                   merge_multi_words_for_target_entities)

                # substitue with concatented multi-word entities
                e1_string = "<e1>"+e1.strip()+"</e1>"
                e2_string =  "<e2>"+e2.strip()+"</e2>"

                #print('e1:', e1, ' e2:', e2)
                #print('line before extracting entities:\n', s)
                #replace '...<e1>...<\e1>....' by ...e1...
                start_idx = s.find("<e1>")
                end_idx = s.find("</e1>")+4
                s = s[:start_idx]+e1_string+s[end_idx+1:]

                #replace '...<e2>...<\e2>....' by ...e2...
                start_idx = s.find("<e2>")
                end_idx = s.find("</e2>")+4
                s = s[:start_idx]+e2_string+s[end_idx+1:]
                #print('line after extracting entities:\n', s)

                # get_text_btwn_entities including target entity terms
                if get_text_btwn_entities == True:
                    e1_lower_indx = s.index(e1_string)
                    e1_highest_indx = s.rindex(e1_string)
                    e2_lower_indx = s.index(e2_string)
                    e2_highest_indx = s.rindex(e2_string)

                    if e1_lower_indx < e2_lower_indx:
                        start = e1_lower_indx + len(e1_string)
                    else:
                        start = e2_lower_indx + len(e2_string)

                    end = np.max(e1_highest_indx, e2_highest_indx)
                    s = s[start:end]
                    if len(s) == 0:
                        empty_count +=1

                # get text neighbourhood to target entities
                if text_neighbourhood_to_entity == True:
                    s = get_text_neighbourhood_to_entity(s, slot_names=[e1_string,e2_string],
                                                         target_entities=[e1_string, e2_string],
                                                         left_neighbourhood_size=left_neighbourhood_size,
                                                         right_neighbourhood_size=right_neighbourhood_size)
                # exclude target entity terms
                if exclude_entity_terms == True:
                    s = s.replace(e1_string, ' ')
                    s = s.replace(e2_string, ' ')

                if len([word for word in s.lower().split() if word not in stoplist]) == 0:
                    continue

                s = s.replace("<e1>", " ")
                s = s.replace("</e1>", " ")
                s = s.replace("<e2>", " ")
                s = s.replace("</e2>", " ")

                sentences.append(s)
                count = count +1

            index_count = index_count + 1
        else:
            line = line.replace("<e1>", " ")
            line = line.replace("</e1>", " ")
            line = line.replace("<e2>", " ")
            line =line.replace("</e2>", " ")
            sentences.append(line.split(':',1)[1])

    #print('sentences:\n', sentences)
    return sentences,labels, target_entities

def read_postag(postag_file, corpus_list):
    postag_seq = []
    for line in open(str(postag_file).split('.txt')[0] + '.pos', 'rb'):
        postag_list = line.split()
        postag_seq.append([corpus_list.index(x) for x in postag_list if x in corpus_list])
    return postag_seq

def read_iob(iob_file, corpus_list):
    iob_seq = []
    for line in open(str(iob_file).split('.txt')[0] + '.iob', 'rb'):
        iob_list = line.split()
        iob_seq.append([corpus_list.index(x) for x in iob_list if x in corpus_list])
    return iob_seq

def read_train(path, get_text_btwn_entities=False,
                   exclude_entity_terms=False,
                   text_neighbourhood_to_entity = False,
                   left_neighbourhood_size = 5,
                   right_neighbourhood_size = 5,
                   remove_other_class = False,
                   NER_for_target_entities=False,
                   merge_multi_words_for_target_entities=False,
                   get_entity_types=False, get_unnorm_position_feat=False,
                   get_position_indicators=False):
    '''
    Reads data and extract target entities from the sentences
    Also, removes <e1>, </e1>,<e2>,</e2> and replaces by respective entities.
    Words in multi-word entity are concatenated by '_' and form single word composite entity
    :param path:
    :return:
        sentences:
        labels:
        target_entities:
    '''
    # To replace the numbers by 0
    replace_numbers = True
    # To replace the currency symbols by 'currency' token
    replace_currency_symbols = True
    # To replace the time units by 'time' token
    replace_time_units = True
    replace_regex_vals = True

    labels = []
    sentences = []
    pos_featues_s = []
    lines = [line.rstrip('\n\r') for line in open(path)]

    line_count = 0
    if remove_other_class == True:
        lines_without_other_class_count = 0
        for line in lines:
            #drop OTHERS
            if int(line.split('::')[0]) != 18:
                lines_without_other_class_count += 1
        target_entities = [[]  for x in range(lines_without_other_class_count)]
    else:
        target_entities = [[]  for x in range(np.array(lines).shape[0])]
    # print('in train: target_entities.shape', np.array(target_entities).shape)

    for line in lines:
        if remove_other_class == True:
            #drop OTHERS
            if int(line.split(':')[0]) != 18:
                # extract target entities from each sentence
                s = line.split(':', 1)[1]

                # replace currency symbols by 'currency'
                if replace_currency_symbols == True:
                    s = repalce_currency_symbols_fn(s)

                # replace time units by 'time'
                if replace_time_units == True:
                    s = replace_time_units_fn(s)

                # replace numbers(int/float) by '0'
                if replace_numbers == True:
                    s = replace_numbers_fn(s)

                #label = labels[line_count]
                label = int(line.split(':', 1)[0])
                if NER_for_target_entities == True:
                    relation_terms = get_NER_for_entity_from_relation_type(label)
                    e1 = relation_terms[0]
                    e2 = relation_terms[1]
                else:
                    # extract target entities and merge the tokens in entity mentions if it is multi-word
                    # 'merge_multi_words_for_target_entities' is False then e1 and e1 are the string with multiple words
                    # seperated ny ' '
                    e1, e2 = extract_target_entities_from_sentence(s,
                                                                   merge_multi_words_for_target_entities
                                                                   =merge_multi_words_for_target_entities)
                #print('e1:', e1, ' e2:', e2)
                target_entities[line_count].append(e1)
                target_entities[line_count].append(e2)
                line_count+=1
        else:
            # extract target entities from each sentence
            s = line.split('::')[5]

            # replace currency symbols by 'currency'
            if replace_currency_symbols == True:
                s = repalce_currency_symbols_fn(s)

            # replace time units by 'time'
            if replace_time_units == True:
                s = replace_time_units_fn(s)

            # replace numbers(int/float) by '0'
            if replace_numbers == True:
                s = replace_numbers_fn(s)

            if replace_regex_vals == True:
                s = replace_regex_fn(s)

            #label = labels[line_count]
            label = str(line.split('::')[2]).strip()

            if label == 'Lives_In':
                label = 0
            elif label == 'Others':
                label = 1
            else:
                print('Error: Invalid Relation Label.'+str(label))
                exit()

            if NER_for_target_entities == True:
                relation_terms = get_NER_for_entity_from_relation_type(label)
                e1 = relation_terms[0]
                e2 = relation_terms[1]
            else:
                # extract target entities and merge the toked in entity  in it is multi-word and also merge the multi-word
                # okens of target entity in the sentence nd return the sentence
                e1, e2 = extract_target_entities_from_sentence(s, merge_multi_words_for_target_entities=
                merge_multi_words_for_target_entities)

            if len(str(e1).split())> 1:
                e1 = str(e1).split()[-1].strip()

            if len(str(e2).split())> 1:
                e2 = str(e2).split()[-1].strip()

            target_entities[line_count].append(e1)
            target_entities[line_count].append(e2)
            line_count+=1

    if get_entity_types == True:
        entity_types_for_words = [[] for x in range(line_count)]

    #target_entities = [[]]
    #target_entities.append([])
    tmp_line = ''
    line_count = 0
    empty_count = 0
    for line in lines:
        #drop OTHERS
        if remove_other_class == True and int(line.split(':')[0]) == 18:
            continue
        else:
            # extract target entities from each sentence
            s = line.split('::')[5]

            # replace currency symbols by 'currency'
            if replace_currency_symbols == True:
                s = repalce_currency_symbols_fn(s)

            # replace time units by 'time'
            if replace_time_units == True:
                s = replace_time_units_fn(s)

            # replace numbers(int/float) by '0'
            if replace_numbers == True:
                s = replace_numbers_fn(s)

            label = str(line.split('::')[2]).strip()

            if label == 'Lives_In':
                label = 0
            elif label == 'Others':
                label = 1
            else:
                print('Error: Invalid Relation Label.'+str(label))
                exit()

            if NER_for_target_entities == True:
                relation_terms = get_NER_for_entity_from_relation_type(label)
                e1 = relation_terms[0]
                e2 = relation_terms[1]
                #print('e1:', e1)
                #print('e2:', e2)
            else:
                # extract target entities and merge the tokens in entity in it is multi-word
                e1, e2= extract_target_entities_from_sentence(s, merge_multi_words_for_target_entities=
                merge_multi_words_for_target_entities)


            # take last word of entity mentions for multi-word entity mention
            # if len(str(e1).split())> 1:
            #    e1_string = "<e1>"+str(e1).split()[-1].strip()+"</e1>"
            #    e1_remain = ''
            #    for r in str(e1).split()[:-1]:
            #       e1_remain = e1_remain + ' ' + r
            # else:
            #    e1_remain = ''
            #    e1_string = "<e1>"+e1.strip()+"</e1>"
#
            # if len(str(e2).split())> 1:
            #    e2_string = "<e2>"+str(e2).split()[-1].strip()+"</e2>"
            #    e2_remain = ''
            #    for r in str(e2).split()[:-1]:
            #        e2_remain = e2_remain + ' ' + r
            # else:
            #    e2_remain = ''
            #    e2_string = "<e2>"+e2.strip()+"</e2>"

            e1_remain = ''
            e1_string = "<e1>"+e1.strip()+"</e1>"
            e2_remain = ''
            e2_string = "<e2>"+e2.strip()+"</e2>"
            
            #print('line before extracting entities:\n', s)
            #replace '...<e1>...<\e1>....' by ...e1...
            start_idx = s.find("<e1>")
            end_idx = s.find("</e1>")+4
            s = s[:start_idx]+ ' ' + e1_remain + ' ' + e1_string+ ' ' +s[end_idx+1:]
            #replace '...<e2>...<\e2>....' by ...e2...
            start_idx = s.find("<e2>")
            end_idx = s.find("</e2>")+4
            s = s[:start_idx]+ ' ' + e2_remain + ' ' + e2_string + ' ' +s[end_idx+1:]
            #print('line after extracting entities:\n', s)

            # get_text_btwn_entities including target entity terms
            if get_text_btwn_entities == True:
                e1_lower_indx = s.index(e1_string)
                e1_highest_indx = s.rindex(e1_string)
                e2_lower_indx = s.index(e2_string)
                e2_highest_indx = s.rindex(e2_string)

                if e1_lower_indx < e2_lower_indx:
                    start = e1_lower_indx + len(e1_string)
                else:
                    start = e2_lower_indx + len(e2_string)

                end = e2_highest_indx if e2_highest_indx > e1_highest_indx else e1_highest_indx

                s = s[start:end]
                if s == ' ' or len(s.split()) == 0:
                    empty_count +=1
                    s = '<unk>'
                    #continue

                if get_unnorm_position_feat == True:
                    tmp_s = e1_string + ' '+ s +' '+ e2_string
                    # compute position features for each word in the sentence,
                    # returns a list of [d_e1, d_e2] for each word in the sentence
                    pos_feat = get_position_features_for_each_token_slot_fill(tmp_s.split(),
                                                                              [e1_string, e2_string])
                    pos_feat = pos_feat[1:]
                    pos_feat = pos_feat[:-1]
                    pos_featues_s.append(pos_feat)

                    if len(pos_feat) != len(s.split()):
                        raise ValueError('Number of words in unnormalised_pos_feat_seq_train and seq_train are not equal !!')

                if get_position_indicators == True:
                    s =  '</e1>' + ' '+ s + ' '+'<e2>'

            # get text neighbourhood to target entities
            if text_neighbourhood_to_entity == True:
                if get_entity_types == True:

                    s, entity_types_for_words_in_s = get_text_neighbourhood_to_entity(s, slot_names=[e1_string,e2_string],
                                                                                      target_entities=[e1_string, e2_string],
                                                                                      left_neighbourhood_size=left_neighbourhood_size,
                                                                                      right_neighbourhood_size=right_neighbourhood_size,
                                                                                      get_entity_types=get_entity_types,
                                                                                      label=label,
                                                                                      position_indicators=get_position_indicators)
                    if len(s.split()) != len(entity_types_for_words_in_s):
                        raise ValueError("Dimension mismatch for s and entity_types_for_words_in_s")

                    #print('s:', s)
                    #print('entity_types_for_words_in_s:', entity_types_for_words_in_s)

                    for ne_type in entity_types_for_words_in_s:
                        entity_types_for_words[line_count].append(ne_type)
                else:
                    s = get_text_neighbourhood_to_entity(s, slot_names=[e1_string,e2_string],
                                                         target_entities=[e1_string, e2_string],
                                                         left_neighbourhood_size=left_neighbourhood_size,
                                                         right_neighbourhood_size=right_neighbourhood_size
                                                         )


                if get_unnorm_position_feat == True:
                    # compute position features for each word in the sentence,
                    # returns a list of [d_e1, d_e2] for each word in the sentence
                    pos_feat = get_position_features_for_each_token_slot_fill(s.lower().split(), [e1_string, e2_string])
                    pos_featues_s.append(pos_feat)
                    #pos_featues_s.append(pos_feat)

            # exclude target entity terms
            if exclude_entity_terms == True:
                if get_unnorm_position_feat == True:
                    i = 0
                    pos_feat = pos_featues_s[-1]
                    print('pos_feat at last:', pos_feat)
                    indicres_to_del = []
                    for word in s.split():
                        if word.strip() == e1_string.strip() or word.strip() == e2_string.strip():
                            indicres_to_del.append(i)
                        i+=1
                    pos_feat = np.delete(pos_feat, indicres_to_del, axis=0)
                    s = s.replace(e1_string, ' ')
                    s = s.replace(e2_string, ' ')
                    pos_featues_s[-1] = pos_feat

                if get_position_indicators == True:
                    import re
                    # regular  expression to replace all between <e1> and </e1>; and <e2 and </e2>
                    print('s before regular expres:', s)
                    s = re.sub("^<e1>\w*<\e1>$", '<e1>  <\e1>', s)
                    print('s after regular expres:', s)

                    print('s before regular expres:', s)
                    s = re.sub(r"<e2>\w*<\e2>", '<e2>  <\e2>', s)
                    print('s after regular expres:', s)

            if len([word for word in s.lower().split()]) == 0:
                continue

            #if get_position_indicators is not True:
            #    s = s.replace("<e1>", "")
            #    s = s.replace("</e1>", "")
            #    s = s.replace("<e2>", "")
            #    s = s.replace("</e2>", "")
            #else:
            #    s = s.replace("<e1>", " <e1> ")
            #    s = s.replace("</e1>", " </e1> ")
            #    s = s.replace("<e2>", " <e2> ")
            #    s = s.replace("</e2>", " </e2> ")

            s = s.replace("<e1>", " <e1> ")
            s = s.replace("</e1>", " </e1> ")
            s = s.replace("<e2>", " <e2> ")
            s = s.replace("</e2>", " </e2> ")

            #labels.append(int(line.split(':')[0]))
            #sentences.append(line.split(':')[1])

            labels.append(label)
            sentences.append(s)
            line_count+=1

            #pos_indicators(pos_ind)

    #print('labels:\n', labels)
    #print('sentences:\n', sentences)
    #sentences = remove_punctuation_from_sentences(sentences)

    if get_entity_types == True:
        return sentences, labels, target_entities, entity_types_for_words, pos_featues_s
    else:
        return sentences, labels, target_entities, None, pos_featues_s

# remove comma and the apostophe
stoplist = set('a an and .  - = * + : ; ! " # % & ( ) * + - . / : ; < = > ? @ [ \\ ] ^ _ ` { | } ~  '.split())
punctuations = '!()-[]{};:"\<>./?@#%^&*_~'

#remove punctuation and lowercasing the sentences
def remove_punctuation_from_sentences(sentences):
    s_count = 0
    for sentence in sentences:
        no_punct = ""
        for char in sentence:
            if char not in punctuations:
                no_punct = no_punct + str(char).lower()
        sentences[s_count] = no_punct
        #print('sentences:','[',s_count,']:',sentences[s_count])
        s_count += 1

    return sentences

def remove_stop_words(line):

    x = ''
    for word in line.split():
        if word not in stoplist:
            x = x + ' ' + word

    x = x.lstrip()
    return x

def get_stop_word_list():
    return stoplist

def get_word_vectoriser_vocab_size_tokenized_sents(tokenized_sentences):
    '''
    :param sentences:
    :return:
        vectorizer : vectoriser to transform word to 1-hot-vector representation
        vocab_size : vacab size (int0
    '''
    #collect all words to generate dictionary
    texts = []
    for tokenized_sentence in tokenized_sentences:
        for word in tokenized_sentence:
            word = str(word).strip().lower()
            if word not in stoplist:
                texts.append(word)
                #print('sentence:', sentence)
                #print('texts:', texts)


    print('Vectorizing:....')
    print('np.unique(texts).shape:', np.array(np.unique(texts)).shape)

    print('np.unique(texts):', np.array(np.unique(texts)))
    #If max_features not None, build a vocabulary that only consider the top
    #max_features ordered by term frequency across the corpus.
    vectorizer = CountVectorizer(analyzer = "word", \
                                 tokenizer = None, \
                                 stop_words = None,
                                 vocabulary=np.unique(texts), strip_accents='unicode')

    train_data_features = vectorizer.fit_transform(texts)
    # Numpy arrays are easy to work with, so convert the result to an
    # array
    # it is an array with 1 at the index where the word is present
    train_data_features = train_data_features.toarray()
    print('data_features.shape:', train_data_features.shape)
    #print(train_data_features)
    vocab_size = train_data_features.shape[1]
    #print('sentences.shape:', np.array(sentences).shape)

    return vectorizer, vocab_size

def get_word_vectoriser_vocab_size(sentences):
    '''
    :param sentences:
    :return:
        vectorizer : vectoriser to transform word to 1-hot-vector representation
        vocab_size : vacab size (int0
    '''
    #collect all words to generate dictionary
    texts = []
    for sentence in sentences:
        for word in sentence.lower().split():
            word = str(word).strip()
            if word not in stoplist:
                texts.append(word)

    print('Vectorizing:....')
    print('np.unique(texts).shape:', np.array(np.unique(texts)).shape)

    print('np.unique(texts):', np.array(np.unique(texts)))
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             stop_words = None,
                             vocabulary=np.unique(texts), strip_accents='unicode')

    train_data_features = vectorizer.fit_transform(texts)
    # Numpy arrays are easy to work with, so convert the result to an
    # array
    # it is an array with 1 at the index where the word is present
    train_data_features = train_data_features.toarray()
    print('data_features.shape:', train_data_features.shape)
    vocab_size = train_data_features.shape[1]

    return vectorizer, vocab_size

def get_train_test_word_indices_slot_fill(count_vec, seq_train, target_entities_train, seq_test, target_entities_test):
    n_train = np.array(seq_train).shape[0]
    n_test = np.array(seq_test).shape[0]
    seq_train_indx = [[] for x in range(n_train)]
    seq_test_indx = [[] for x in range(n_test)]

    s_count = 0
    for s in seq_train:
        for word in str(s).lower().split():
            if word not in stoplist:
                seq_train_indx[s_count].append(count_vec.vocabulary_.get(word))
        s_count+=1

    s_count = 0
    for s in seq_test:
        for word in str(s).lower().split():
            if word not in stoplist:
                seq_test_indx[s_count].append(count_vec.vocabulary_.get(word))
        s_count+=1

    target_entities_train_indx = []
    target_entities_test_indx = []

    for entity in target_entities_train:
        target_entities_train_indx.append(count_vec.vocabulary_.get(entity))

    for entity in target_entities_test:
        target_entities_test_indx.append(count_vec.vocabulary_.get(entity))

    return seq_train_indx, target_entities_train_indx, seq_test_indx , target_entities_test_indx

def get_train_test_word_indices(count_vec, seq_train, target_entities_train, seq_test, target_entities_test):
    n_train = np.array(seq_train).shape[0]
    n_test = np.array(seq_test).shape[0]
    seq_train_indx = [[] for x in range(n_train)]
    seq_test_indx = [[] for x in range(n_test)]

    s_count = 0
    for s in seq_train:
        for word in str(s).lower().split():
            if word not in stoplist:
                seq_train_indx[s_count].append(count_vec.vocabulary_.get(word))
        s_count+=1

    s_count = 0
    for s in seq_test:
        for word in str(s).lower().split():
            if word not in stoplist:
                seq_test_indx[s_count].append(count_vec.vocabulary_.get(word))
        s_count+=1

    target_entities_train_indx = [[] for x in range(n_train)]
    target_entities_test_indx = [[] for x in range(n_test)]

    s_count= 0
    for entities in target_entities_train:
        target_entities_train_indx[s_count].append(count_vec.vocabulary_.get(entities[0]))
        target_entities_train_indx[s_count].append(count_vec.vocabulary_.get(entities[1]))
        s_count+=1

    s_count= 0
    for entities in target_entities_test:
        target_entities_test_indx[s_count].append(count_vec.vocabulary_.get(entities[0]))
        target_entities_test_indx[s_count].append(count_vec.vocabulary_.get(entities[1]))
        s_count+=1

    return seq_train_indx, target_entities_train_indx, seq_test_indx , target_entities_test_indx


def get_position_features_for_each_token(tokenised_sentence, sentence_target_entities):
    '''
    Get position features for each token in the sentence given
    A vector [D_e2, D_e1] is computed for each token.

    :param tokenised_sentence: tokenised sentence with stop words removed

    :param sentence_target_entities: ['target_entity_e1', 'target_entity_e2'] for sentence

    :return:
    position_features: array-like, shape=(n_words, 2)
    '''

    # find the index of e1 in the given sentence
    entity_indices = []  # e1_index, e2_index word positions

    # find the word index of e1 and e2 in the sentence
    # number of words in the sentence
    n_words = len(tokenised_sentence)
    for entity in sentence_target_entities:
        for word_count in range(n_words):
            if entity == tokenised_sentence[word_count]:
                entity_indices.append(word_count)
                break

    if np.array(entity_indices).shape[0] !=2:
        print('tokenised_sentence:', tokenised_sentence)
        print('entity_indices:', entity_indices)
        print('entity_indices.shape:', np.array(entity_indices).shape)
        print('sentence_target_entities:', sentence_target_entities)
        raise ValueError('Error: Entity Indices should have size 2.\n '
                         '1. Possibly if <name> and/or <filler> is not present in the sentence. 2. exact string match '
                         'for <name> and or <filer> fails.')

    # number of words in sentence * n_target_entities
    #print('entity_indices:', entity_indices)
    position_features = np.zeros(shape=(n_words, 2))
    for count in range(n_words):
        # distance of current word from e1
        position_features[count, 0] = np.abs(int(entity_indices[0]) - count)
        # distance of current word from e2
        position_features[count, 1] = np.abs(int(entity_indices[1]) - count)
        count +=1

    #print('position_features:\n',position_features)
    return position_features


def get_position_features_for_each_token_slot_fill(tokenised_sentence, sentence_target_entities):
    '''
    Get position features for each token in the sentence given
    A vector [D_e2, D_e1] is computed for each token.

    :param tokenised_sentence: tokenised sentence with stop words removed

    :param sentence_target_entities: ['target_entity_e1', 'target_entity_e2'] for sentence

    :return:
    position_features: array-like, shape=(n_words, 2)
    '''

    # find the index of e1 in the given sentence
    e1_entity_indices = []  # e1_index word positions
    e2_entity_indices = []  # e2_index word positions

    # find the word index of e1 and e2 in the sentence
    # number of words in the sentence
    n_words = len(tokenised_sentence)

    #print('tokenised_sentence:', tokenised_sentence)
    #print('sentence_target_entities:', sentence_target_entities)

    e1_split = str(sentence_target_entities[0]).split()
    e2_split = str(sentence_target_entities[1]).split()
    #print('e1_split', e1_split)
    #print('e2_split', e2_split)
    for word_count in range(n_words):
        found_in_e1_split = 0
        found_in_e2_split = 0
        search_word_in_e2_split_also = 0

        ## search in e1_split
        for e1 in e1_split:
            if str(tokenised_sentence[word_count]).strip() == str(e1).strip():
                e1_entity_indices.append(word_count)
                found_in_e1_split = 1
                break

        # search in e2_split
        for e2 in e2_split:
            if str(tokenised_sentence[word_count]).strip() == str(e2).strip():
                found_in_e2_split = 1
                break

        if found_in_e1_split == 1 and found_in_e2_split == 1:
            #print('word is present in both e1 and e2 entity mentions')
            search_word_in_e2_split_also = 1

        if found_in_e1_split == 0 or search_word_in_e2_split_also == 1:
            # search in e2_split
            for e2 in e2_split:
                if str(tokenised_sentence[word_count]).strip() == str(e2).strip():
                    e2_entity_indices.append(word_count)
                    found_in_e2_split = 1
                    break

    if len(e1_entity_indices) == 0 or len(e2_entity_indices) == 0:
        print('tokenised_sentence:', tokenised_sentence)
        print('e1_entity_indices:', e1_entity_indices)
        print('e1_entity_indices.shape:', np.array(e1_entity_indices).shape)
        print('e2_entity_indices:', e2_entity_indices)
        print('e2_entity_indices.shape:', np.array(e2_entity_indices).shape)
        print('sentence_target_entities:', sentence_target_entities)
        raise ValueError('Error: Entity Indices should have size 2.\n '
                         '1. Possibly if <name> and/or <filler> is not present in the sentence. 2. exact string match '
                         'for <name> and or <filler> fails.')

    # number of words in sentence * n_target_entities
    #print('entity_indices:', entity_indices)
    position_features = np.zeros(shape=(n_words, 2))
    for count in range(n_words):
        distances_from_e1s = []
        distances_from_e2s = []
        # distance of current word from e1 type entities
        for e1_idx in e1_entity_indices:
            distances_from_e1s.append(count - int(e1_idx))
        # distance of current word from e2 type entities
        for e2_idx in e2_entity_indices:
            distances_from_e2s.append(count - int(e2_idx))

        position_features[count, 0] = int(min(np.min(distances_from_e1s),
                                          np.max(distances_from_e1s), key=abs))
        # distance of current word from e2
        position_features[count, 1] = int(min(np.min(distances_from_e2s),
                                          np.max(distances_from_e2s), key=abs))
        count +=1

    return position_features

# one-hot representation of words in a sentence, s
# s is list of words in the sentence
def get_one_hot_word_representation(vectorizer, s, target_entities=None, position_features=False,
                                    augment_entity_presence=False, first_entity_word=None,
                                    second_entity_word = None, not_entity_word=None, normalise_pos_feat=True,
                                    get_NEs = False):
    x = []
    pos_feat = []
    if position_features == True:
        #print('S:', s)
        pos_feat = get_position_features_for_each_token(s, target_entities)
        # normalise the position features by the number of words in the sentence
        if normalise_pos_feat == True:
            n_words = len(s)
            for i in range(pos_feat.shape[0]):
                # more importance to words closure to target entities
                pos_feat[i, 0] = 1.0 - (pos_feat[i, 0]/n_words)
                pos_feat[i, 1] = 1.0 - (pos_feat[i, 1]/n_words)

    word_count = 0
    for word in s:
        #print('word:', word)
        sample = vectorizer.transform(str(word).split(' ')).toarray()[0]

        if position_features == True:
            sample = np.concatenate((sample,pos_feat[word_count, :]),axis=1)

        if augment_entity_presence == True:
            # get target entities for the sentence
            if word == target_entities[0]:
                # add [0,1] for the first entity
                sample = np.concatenate((sample,first_entity_word),axis=1)
            elif word == target_entities[1]:
                # add [1,0] for the second entity
                sample = np.concatenate((sample,second_entity_word),axis=1)
            else:
                # add [0,0] for the word since it is not a target entity
                sample = np.concatenate((sample,not_entity_word),axis=1)

        x.append(sample)
        word_count += 1

    if get_NEs == True:
        x_NEs = label_words_by_NE(s, target_entities)
        return x, x_NEs
    else:
        return x


def get_entity_presence_in_context_window(vectorizer, cwords, target_entities, first_entity_word, second_entity_word ,
                        not_entity_word, both_entities_in_window):
    cwords_count = 0

    for cword in cwords:
        first_entity_pres = False
        second_entity_pres = False
        for word in cword:
            # get target entities for the sentence
            if word == target_entities[0]:
                first_entity_pres = True
            elif word == target_entities[1]:
                second_entity_pres = True

        if first_entity_pres == True and second_entity_pres == True:
            cwords[cwords_count] = np.concatenate((cwords[cwords_count], both_entities_in_window), axis=1)
        elif first_entity_pres == True:
            cwords[cwords_count] = np.concatenate((cwords[cwords_count], first_entity_word), axis=1)
        elif second_entity_pres == True:
            cwords[cwords_count] = np.concatenate((cwords[cwords_count], second_entity_word), axis=1)
        else:
            cwords[cwords_count] = np.concatenate((cwords[cwords_count], not_entity_word), axis=1)

        #print('cwords[',cwords_count,']:',cwords[cwords_count])
        cwords_count += 1

    return cwords

def get_entity_presence_position_features_NEs(tokenised_sentence_indices, target_entities,
                                                          position_features=False,
                                                          augment_entity_presence=False, first_entity_word=None,
                                                          second_entity_word = None, not_entity_word=None,
                                                          normalise_pos_feat=True,
                                                          get_NEs = False, relation_type=None,
                                                          ne_label_encoder = None):

    if position_features is not True and augment_entity_presence is not True and get_NEs == False:
        return tokenised_sentence_indices, None , None

    x = []
    pos_feat = []
    ent_pres = [[] for e in range(len(tokenised_sentence_indices))]

    if position_features == True:
        #pos_feat = get_position_features_for_each_token(tokenised_sentence, target_entities)
        #print('target_entities:', target_entities)
        pos_feat = get_position_features_for_each_token_slot_fill(tokenised_sentence_indices, target_entities)

        if pos_feat.shape[0] != len(tokenised_sentence_indices):
            #print('pos_feat.shape[0]:',pos_feat.shape[0])
            #print('word2vecs.shape[0]:',word2vecs.shape[0])
            raise ValueError('Dimension mismatch in append_entity_presence_position_features_to_word2vecs()')
        # normalise the position features by the number of words in the sentence
        if normalise_pos_feat == True:
            n_words = len(tokenised_sentence_indices)
            for i in range(pos_feat.shape[0]):
                # more importance to words closure to target entities
                pos_feat[i, 0] = (1.0 - (np.abs(pos_feat[i, 0])/n_words)) * np.sign(pos_feat[i, 0])
                pos_feat[i, 1] = (1.0 - (np.abs(pos_feat[i, 1])/n_words)) * np.sign(pos_feat[i, 1])

    word_count = 0
    e1_split = str(target_entities[0]).split()
    e2_split = str(target_entities[1]).split()
    e1_done = False
    e2_done = False

    for word in tokenised_sentence_indices:
        if augment_entity_presence == True:
            # get target entities for the sentence
            found_in_e1_split = 0
            found_in_e2_split = 0

            # search in e1_split
            for e1 in e1_split:
                if str(word).strip() == str(e1).strip():
                    found_in_e1_split = 1
                    break

            # search in e2_split
            for e2 in e2_split:
                if str(word).strip() == str(e2).strip():
                    found_in_e2_split = 1
                    break

            if found_in_e1_split == 1 and found_in_e2_split == 1:
                if e1_done == False:
                    ent_pres[word_count] = first_entity_word
                    e1_done = True
                else:
                    ent_pres[word_count] = second_entity_word
            else:
                found_in_e1_split = 0
                found_in_e2_split = 0
                # search in e1_split
                for e1 in e1_split:
                    if str(word).strip() == str(e1).strip():
                        # add [0,1] for the first entity
                        ent_pres[word_count] = first_entity_word
                        found_in_e1_split = 1
                        break

                if found_in_e1_split == 0:
                    # search in e2_split
                    for e2 in e2_split:
                        if str(word).strip() == str(e2).strip():
                            # add [1,0] for the second entity
                            ent_pres[word_count] = second_entity_word
                            found_in_e2_split = 1
                            break

                if found_in_e1_split==0 and found_in_e2_split==0:
                    # add [0,0] for the word since it is not a target entity
                    ent_pres[word_count] = not_entity_word

        word_count += 1

    entity_types_for_words_in_s = []
    if get_NEs == True:
        if str(getRelation(relation_type)).lower() == 'other':
            relation_terms_for_e1_e2 = ['other_e1', 'other_e2']
        else:
            relation_terms_for_e1_e2 = get_NER_for_entity_from_relation_type(relation_type)

        #e1_string = target_entities[0]
        #e2_string = target_entities[1]

        e1_done = False
        e2_done = False
        e1_split = str(target_entities[0]).split()
        e2_split = str(target_entities[1]).split()
        for word in tokenised_sentence_indices:
            found_in_e1_split = 0
            found_in_e2_split = 0

            # search in e1_split
            for e1 in e1_split:
                if str(word).strip() == e1.strip():
                    found_in_e1_split = 1
                    break

            # search in e2_split
            for e2 in e2_split:
                if str(word).strip() == e2.strip():
                    found_in_e2_split = 1
                    break

            if found_in_e1_split == 1 and found_in_e2_split == 1:
                #word is present in both e1 and e2 entity mentions
                if e1_done == False:
                    entity_types_for_words_in_s.append(ne_label_encoder.transform(relation_terms_for_e1_e2[0]))
                    e1_done = True
                else:
                    entity_types_for_words_in_s.append(ne_label_encoder.transform(relation_terms_for_e1_e2[1]))
            else:
                found_in_e1_split = 0
                found_in_e2_split = 0
                # search in e1_split
                for e1 in e1_split:
                    if str(word).strip() == e1.strip():
                        entity_types_for_words_in_s.append(ne_label_encoder.transform(relation_terms_for_e1_e2[0]))
                        found_in_e1_split = 1
                        break

                if found_in_e1_split == 0:
                    # search in e2_split
                    for e2 in e2_split:
                        if str(word).strip() == e2.strip():
                            entity_types_for_words_in_s.append(ne_label_encoder.transform(relation_terms_for_e1_e2[1]))
                            found_in_e2_split = 1
                            break

                if found_in_e1_split == 0 and found_in_e2_split == 0:
                    entity_types_for_words_in_s.append(ne_label_encoder.transform('other'))

        #x_NEs = get_NER_for_entity_from_relation_type(tokenised_sentence, relation_type)
        #print('tokenised sen:', tokenised_sentence)

    if augment_entity_presence == True and position_features == True and get_NEs == True:
        if not(len(pos_feat) == len(ent_pres) and len(ent_pres) == len(entity_types_for_words_in_s)):
            raise ValueError('Dimension mismatch for position features, entitty_pres and ent types')
    elif position_features == True and get_NEs == True:
        if len(pos_feat) != len(entity_types_for_words_in_s):
            raise ValueError('Dimension mismatch for position features, and ent types')
    elif position_features == True and augment_entity_presence == True:
        if len(pos_feat) != len(ent_pres):
            raise ValueError('Dimension mismatch for position features, and ent pres')
    elif get_NEs == True and augment_entity_presence == True:
        if len(ent_pres) != len(entity_types_for_words_in_s):
            raise ValueError('Dimension mismatch for ent pres, and ent types')
    return pos_feat, ent_pres, entity_types_for_words_in_s

def get_word2vec_emb_entity_presence_position_features(word2vec_dict, tokenised_sentence, target_entities,
                                                          position_features=False,
                                                          augment_entity_presence=False, first_entity_word=None,
                                                          second_entity_word = None, not_entity_word=None,
                                                          normalise_pos_feat=True,
                                                          get_NEs = False, relation_type=None,
                                                          ne_label_encoder = None):

    word2vecs, tokenised_sentence = get_word2vec_emb(word2vec_dict=word2vec_dict, tokenised_sentence=tokenised_sentence)
    word2vecs = np.array(word2vecs)

    if word2vecs.shape[0] == 0:
        raise ValueError('empty word2vec disctionary')

    if position_features is not True and augment_entity_presence is not True and get_NEs == False:
        return word2vecs

    x = []
    pos_feat = []
    if position_features == True:
        #pos_feat = get_position_features_for_each_token(tokenised_sentence, target_entities)
        #print('target_entities:', target_entities)
        pos_feat = get_position_features_for_each_token_slot_fill(tokenised_sentence, target_entities)

        if pos_feat.shape[0] != word2vecs.shape[0]:
            #print('pos_feat.shape[0]:',pos_feat.shape[0])
            #print('word2vecs.shape[0]:',word2vecs.shape[0])
            raise ValueError('Dimension mismatch in append_entity_presence_position_features_to_word2vecs()')
        # normalise the position features by the number of words in the sentence
        if normalise_pos_feat == True:
            n_words = len(tokenised_sentence)
            for i in range(pos_feat.shape[0]):
                # more importance to words closure to target entities
                pos_feat[i, 0] = (1.0 - (np.abs(pos_feat[i, 0])/n_words)) * np.sign(pos_feat[i, 0])
                pos_feat[i, 1] = (1.0 - (np.abs(pos_feat[i, 1])/n_words)) * np.sign(pos_feat[i, 1])

    word_count = 0
    e1_split = str(target_entities[0]).split()
    e2_split = str(target_entities[1]).split()
    e1_done = False
    e2_done = False
    for word, word2vec in zip(tokenised_sentence, word2vecs):
        if augment_entity_presence == True:
            # get target entities for the sentence
            found_in_e1_split = 0
            found_in_e2_split = 0

            # search in e1_split
            for e1 in e1_split:
                if str(word).strip() == e1.strip():
                    found_in_e1_split = 1
                    break

            # search in e2_split
            for e2 in e2_split:
                if str(word).strip() == e2.strip():
                    found_in_e2_split = 1
                    break

            if found_in_e1_split == 1 and found_in_e2_split == 1:
                if e1_done == False:
                    word2vec = np.concatenate((word2vec,first_entity_word),axis=1)
                    e1_done = True
                else:
                    word2vec = np.concatenate((word2vec,second_entity_word),axis=1)

            else:
                found_in_e1_split = 0
                found_in_e2_split = 0
                # search in e1_split
                for e1 in e1_split:
                    if str(word).strip() == e1.strip():
                        # add [0,1] for the first entity
                        word2vec = np.concatenate((word2vec,first_entity_word),axis=1)
                        found_in_e1_split = 1
                        break

                if found_in_e1_split == 0:
                    # search in e2_split
                    for e2 in e2_split:
                        if str(word).strip() == e2.strip():
                            # add [1,0] for the second entity
                            word2vec = np.concatenate((word2vec,second_entity_word),axis=1)
                            found_in_e2_split = 1
                            break

                if found_in_e1_split==0 and found_in_e2_split==0:
                    # add [0,0] for the word since it is not a target entity
                    word2vec = np.concatenate((word2vec,not_entity_word),axis=1)

            '''
            if word == target_entities[0]:
                # add [0,1] for the first entity
                word2vec = np.concatenate((word2vec,first_entity_word),axis=1)
            elif word == target_entities[1]:
                # add [1,0] for the second entity
                word2vec = np.concatenate((word2vec,second_entity_word),axis=1)
            else:
                # add [0,0] for the word since it is not a target entity
                word2vec = np.concatenate((word2vec,not_entity_word),axis=1)
            '''

        if position_features == True:
            word2vec = np.concatenate((word2vec,pos_feat[word_count, :]),axis=1)

        x.append(word2vec)
        word_count += 1

    if get_NEs == True:
        entity_types_for_words_in_s = []
        if str(getRelation(relation_type)).lower() == 'other':
            relation_terms_for_e1_e2 = ['other_e1', 'other_e2']
        else:
            relation_terms_for_e1_e2 = get_NER_for_entity_from_relation_type(relation_type)

        #e1_string = target_entities[0]
        #e2_string = target_entities[1]

        e1_done = False
        e2_done = False
        e1_split = str(target_entities[0]).split()
        e2_split = str(target_entities[1]).split()
        for word in tokenised_sentence:
            found_in_e1_split = 0
            found_in_e2_split = 0

            # search in e1_split
            for e1 in e1_split:
                if str(word).strip() == e1.strip():
                    found_in_e1_split = 1
                    break

            # search in e2_split
            for e2 in e2_split:
                if str(word).strip() == e2.strip():
                    found_in_e2_split = 1
                    break

            if found_in_e1_split == 1 and found_in_e2_split == 1:
                #word is present in both e1 and e2 entity mentions
                if e1_done == False:
                    entity_types_for_words_in_s.append(ne_label_encoder.transform(relation_terms_for_e1_e2[0]))
                    e1_done = True
                else:
                    entity_types_for_words_in_s.append(ne_label_encoder.transform(relation_terms_for_e1_e2[1]))

            else:
                found_in_e1_split = 0
                found_in_e2_split = 0
                # search in e1_split
                for e1 in e1_split:
                    if str(word).strip() == e1.strip():
                        entity_types_for_words_in_s.append(ne_label_encoder.transform(relation_terms_for_e1_e2[0]))
                        found_in_e1_split = 1
                        break

                if found_in_e1_split == 0:
                    # search in e2_split
                    for e2 in e2_split:
                        if str(word).strip() == e2.strip():
                            entity_types_for_words_in_s.append(ne_label_encoder.transform(relation_terms_for_e1_e2[1]))
                            found_in_e2_split = 1
                            break

                if found_in_e1_split == 0 and found_in_e2_split == 0:
                    entity_types_for_words_in_s.append(ne_label_encoder.transform('other'))

            '''
            if word.strip() == e1_string:
                entity_types_for_words_in_s.append(ne_label_encoder.transform(relation_terms_for_e1_e2[0]))
            elif word.strip() == e2_string:
                entity_types_for_words_in_s.append(ne_label_encoder.transform(relation_terms_for_e1_e2[1]))
            else:
                entity_types_for_words_in_s.append(ne_label_encoder.transform('other'))
            '''

        #x_NEs = get_NER_for_entity_from_relation_type(tokenised_sentence, relation_type)
        #print('tokenised sen:', tokenised_sentence)
        return x, entity_types_for_words_in_s
    else:
        return x


import random
def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)

def minibatch(l, bs):
    '''
    l :: list of word idxs
    return a list of minibatches of indexes
    which size is equal to bs
    border cases are treated as follow:
    eg: [0,1,2,3] and bs = 3
    will output:
    [[0],[0,1],[0,1,2],[1,2,3]]
    '''
    out  = [l[:i] for i in range(1, min(bs,len(l)+1) )]
    out += [l[i-bs:i] for i in range(bs,len(l)+1) ]
    assert len(l) == len(out)
    return out

def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = win/2 * [-1] + l + win/2 * [-1]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out


def generate_dict_for_word2vec_emb(word2vec_file_path):
    lines = [line.rstrip('\n\r') for line in open(word2vec_file_path)]

    words = []
    word2vecs = []

    for line in lines:
        words.append(str(line).split(' ', 1)[0])
        word2vecs.append(str(line).split(' ', 1)[1])

    word2vec_dict = {}
    words_vecs = zip(words, word2vecs)
    #print(words_vecs)
    for word, vec in words_vecs:
        if str(word).strip() != 'PADDING':
            word = str(word).lower().strip()

        word2vec_dict[word] = np.fromstring(vec, dtype=np.float, sep=' ')
        #print(word2vec_dict[word])
    return word2vec_dict

# tokenised sentence in lower()
def get_word2vec_emb(word2vec_dict, tokenised_sentence, data_type = 'sf'):
    word_vec_s = []
    s= []
    for word in tokenised_sentence:
        '''
        if data_type == 'sf':
            word = str(word).strip(punctuations).strip()
        else:
            word = str(word).strip()
        '''
        word = str(word).strip()
        if word != 'PADDING':
            word = str(word).lower()

        if word2vec_dict.has_key(word):
            word_vec_s.append(word2vec_dict[word])
        else:
            word_vec_s.append(word2vec_dict['<unk>'])
        s.append(word)

    return word_vec_s, s

# return the w2v embedding matrix, vocab dictionary of the words that exists in the w2v dictionary
def get_w2v_emb_dict_full_vocab(word2vec_file_path):
    word2vec_dict = generate_dict_for_word2vec_emb(word2vec_file_path)
    emb_dict = {}
    # sotre word and indiices for faster lookup
    dict_word_indices_for_emb = {}
    embedding = []
    index = 0
    for word in word2vec_dict:
        emb_dict[word] = word2vec_dict[word]
        dict_word_indices_for_emb[word] = index
        embedding.append(emb_dict[word])
        index += 1

    return embedding, emb_dict, dict_word_indices_for_emb, len(emb_dict)

# return the w2v embedding matrix, vocab dictionary of the words that exists in the w2v dictionary
def get_w2v_emb_dict_vocab(word2vec_file_path, data, data_type = 'sf', get_position_indicators=False):
    word2vec_dict = generate_dict_for_word2vec_emb(word2vec_file_path)
    emb_dict = {}
    # sotre word and indiices for faster lookup
    dict_word_indices_for_emb = {}
    embedding = []
    index = 0
    for sentence in data:
        for word in sentence.split():
            if data_type == 'sf':
                word = str(word).strip(punctuations).strip()
            else:
                word = str(word).strip()

            word = word.lower()
            # to handle duplicates
            if word2vec_dict.has_key(word) and emb_dict.has_key(word) == False:
                emb_dict[word] = word2vec_dict[word]
                dict_word_indices_for_emb[word] = index
                embedding.append(emb_dict[word])
                index += 1

    emb_dict['<unk>'] = word2vec_dict['<unk>']
    embedding.append(emb_dict['<unk>'])
    dict_word_indices_for_emb['<unk>'] = index
    index += 1

    if get_position_indicators == True:
        # random embedding init to represent the position indicator <e1>
        emb_dict['<e1>'] = 0.2 * word2vec_dict['<unk>']
        embedding.append(emb_dict['<e1>'])
        dict_word_indices_for_emb['<e1>'] = index
        index += 1

        # random embedding init to to represent the position indicator <e1>
        #emb_dict['</e1>'] = 0.1 * np.random.uniform(-1.0, 1.0, (1, len(word2vec_dict['<unk>'])))
        emb_dict['</e1>'] = 0.1 * word2vec_dict['<unk>']
        embedding.append(emb_dict['</e1>'])
        dict_word_indices_for_emb['</e1>'] = index
        index += 1

        # random embedding init to to represent the position indicator <e1>
        emb_dict['<e2>'] = 0.3 * word2vec_dict['<unk>']
        embedding.append(emb_dict['<e2>'])
        dict_word_indices_for_emb['<e2>'] = index
        index += 1

        # random embedding init to to represent the position indicator <e1>
        emb_dict['</e2>'] = 0.4 * word2vec_dict['<unk>']
        embedding.append(emb_dict['</e2>'])
        dict_word_indices_for_emb['</e2>'] = index
        index += 1

    # to represent the beginning of the sentence
    emb_dict['PADDING'] = word2vec_dict['PADDING']
    embedding.append(emb_dict['PADDING'])
    dict_word_indices_for_emb['PADDING'] = index
    index += 1

    return embedding, emb_dict, dict_word_indices_for_emb, len(emb_dict)


def get_train_test_word_indices_from_emb_dict(emb_dict, dict_word_indices_for_emb, seq_train,
                                              target_entities_train, seq_test, target_entities_test, slot_fill=False,
                                              get_NEs = False, target_entity_types_for_words_train=None,
                                              target_entity_types_for_words_test=None):
    n_train = np.array(seq_train).shape[0]
    n_test = np.array(seq_test).shape[0]
    seq_train_indx = [[] for x in range(n_train)]
    seq_test_indx = [[] for x in range(n_test)]

    targets_train_NEs = [[] for x in range(n_train)]
    targets_test_NEs = [[] for x in range(n_test)]

    s_count = 0
    for s in seq_train:
        if get_NEs == True:
            for word, et in zip(str(s).lower().split(),target_entity_types_for_words_train):
                #if word not in stoplist and emb_dict.has_key(word) == True:
                if emb_dict.has_key(word) == True:
                    seq_train_indx[s_count].append(dict_word_indices_for_emb[word])
                    targets_train_NEs[s_count].append(et)

                    '''
                    if get_NEs == True:
                        if slot_fill == True:
                            if word == target_entities_train[0] or word == target_entities_train[1]:
                                targets_train_NEs[s_count].append(label_encoder.transform(word))
                            else:
                                targets_train_NEs[s_count].append(label_encoder.transform('other'))
                        else:
                            if word == target_entities_train[s_count][0] or word == target_entities_train[s_count][1]:
                                targets_train_NEs[s_count].append(label_encoder.transform(word))
                            else:
                                targets_train_NEs[s_count].append(label_encoder.transform('other'))
                    '''
                else:
                    seq_train_indx[s_count].append(dict_word_indices_for_emb['<unk>'])
                    targets_train_NEs[s_count].append(et)
        else:
            for word in str(s).lower().split():
                #if word not in stoplist and emb_dict.has_key(word) == True:
                if emb_dict.has_key(word) == True:
                    seq_train_indx[s_count].append(dict_word_indices_for_emb[word])
                else:
                    seq_train_indx[s_count].append(dict_word_indices_for_emb['<unk>'])
        s_count+=1

    s_count = 0
    for s in seq_test:
        if get_NEs == True:
            for word, et in zip(str(s).lower().split(),target_entity_types_for_words_test) :
                #if word not in stoplist and emb_dict.has_key(word) == True:
                if emb_dict.has_key(word) == True:
                    seq_test_indx[s_count].append(dict_word_indices_for_emb[word])
                    targets_test_NEs[s_count].append(et)
                    '''
                    if get_NEs == True:
                        if slot_fill == True:
                            if word == target_entities_test[0] or word == target_entities_test[1]:
                                targets_test_NEs[s_count].append(label_encoder.transform(word))
                            else:
                                targets_test_NEs[s_count].append(label_encoder.transform('other'))
                        else:
                            if word == target_entities_test[s_count][0] or word == target_entities_test[s_count][1]:
                                targets_test_NEs[s_count].append(label_encoder.transform(word))
                            else:
                                targets_test_NEs[s_count].append(label_encoder.transform('other'))
                    '''
                else:
                    seq_test_indx[s_count].append(dict_word_indices_for_emb['<unk>'])
                    targets_test_NEs[s_count].append(et)
        else:
            for word in str(s).lower().split():
                if emb_dict.has_key(word) == True:
                    seq_test_indx[s_count].append(dict_word_indices_for_emb[word])
                else:
                    seq_test_indx[s_count].append(dict_word_indices_for_emb['<unk>'])

        s_count+=1

    if slot_fill == True:
        target_entities_train_indx = []
        target_entities_test_indx = []

        if emb_dict.has_key(target_entities_train[0]):
            target_entities_train_indx.append(dict_word_indices_for_emb[target_entities_train[0]])
        else:
            target_entities_train_indx.append(dict_word_indices_for_emb['<unk>'])

        if emb_dict.has_key(target_entities_train[1]):
            target_entities_train_indx.append(dict_word_indices_for_emb[target_entities_train[1]])
        else:
            target_entities_train_indx.append(dict_word_indices_for_emb['<unk>'])

        if emb_dict.has_key(target_entities_test[0]):
            target_entities_test_indx.append(dict_word_indices_for_emb[target_entities_test[0]])
        else:
            target_entities_test_indx.append(dict_word_indices_for_emb['<unk>'])

        if emb_dict.has_key(target_entities_test[1]):
            target_entities_test_indx.append(dict_word_indices_for_emb[target_entities_test[1]])
        else:
            target_entities_test_indx.append(dict_word_indices_for_emb['<unk>'])

    elif slot_fill == 'conll':
        target_entities_train_indx = [[] for x in range(n_train)]
        target_entities_test_indx = [[] for x in range(n_test)]
    else:
        target_entities_train_indx = [[] for x in range(n_train)]
        target_entities_test_indx = [[] for x in range(n_test)]
        s_count= 0
        for entities in target_entities_train:
            if emb_dict.has_key(entities[0]):
                target_entities_train_indx[s_count].append(dict_word_indices_for_emb[entities[0]])
            else:
                target_entities_train_indx[s_count].append(dict_word_indices_for_emb['<unk>'])

            if emb_dict.has_key(entities[1]):
                target_entities_train_indx[s_count].append(dict_word_indices_for_emb[entities[1]])
            else:
                target_entities_train_indx[s_count].append(dict_word_indices_for_emb['<unk>'])

            s_count+=1

        s_count= 0
        for entities in target_entities_test:
            if emb_dict.has_key(entities[0]):
                target_entities_test_indx[s_count].append(dict_word_indices_for_emb[entities[0]])
            else:
                target_entities_test_indx[s_count].append(dict_word_indices_for_emb['<unk>'])

            if emb_dict.has_key(entities[1]):
                target_entities_test_indx[s_count].append(dict_word_indices_for_emb[entities[1]])
            else:
                target_entities_test_indx[s_count].append(dict_word_indices_for_emb['<unk>'])

            s_count+=1

    return seq_train_indx, target_entities_train_indx, seq_test_indx , target_entities_test_indx,\
           targets_train_NEs, targets_test_NEs






def get_train_dev_test_word_indices_from_emb_dict(emb_dict, dict_word_indices_for_emb, seq_train,
                                              target_entities_train, seq_dev, target_entities_dev, seq_test, target_entities_test, slot_fill=False,
                                              get_NEs = False, target_entity_types_for_words_train=None, target_entity_types_for_words_dev=None,
                                              target_entity_types_for_words_test=None):
    n_train = np.array(seq_train).shape[0]
    n_dev = np.array(seq_dev).shape[0]
    n_test = np.array(seq_test).shape[0]
    seq_train_indx = [[] for x in range(n_train)]
    seq_dev_indx = [[] for x in range(n_dev)]
    seq_test_indx = [[] for x in range(n_test)]

    targets_train_NEs = [[] for x in range(n_train)]
    targets_dev_NEs = [[] for x in range(n_dev)]
    targets_test_NEs = [[] for x in range(n_test)]

    s_count = 0
    for s in seq_train:
        if get_NEs == True:
            for word, et in zip(str(s).lower().split(),target_entity_types_for_words_train):
                #if word not in stoplist and emb_dict.has_key(word) == True:
                if emb_dict.has_key(word) == True:
                    seq_train_indx[s_count].append(dict_word_indices_for_emb[word])
                    targets_train_NEs[s_count].append(et)

                    '''
                    if get_NEs == True:
                        if slot_fill == True:
                            if word == target_entities_train[0] or word == target_entities_train[1]:
                                targets_train_NEs[s_count].append(label_encoder.transform(word))
                            else:
                                targets_train_NEs[s_count].append(label_encoder.transform('other'))
                        else:
                            if word == target_entities_train[s_count][0] or word == target_entities_train[s_count][1]:
                                targets_train_NEs[s_count].append(label_encoder.transform(word))
                            else:
                                targets_train_NEs[s_count].append(label_encoder.transform('other'))
                    '''
                else:
                    seq_train_indx[s_count].append(dict_word_indices_for_emb['<unk>'])
                    targets_train_NEs[s_count].append(et)
        else:
            for word in str(s).lower().split():
                #if word not in stoplist and emb_dict.has_key(word) == True:
                if emb_dict.has_key(word) == True:
                    seq_train_indx[s_count].append(dict_word_indices_for_emb[word])
                else:
                    seq_train_indx[s_count].append(dict_word_indices_for_emb['<unk>'])
        s_count+=1

    s_count = 0
    for s in seq_dev:
        if get_NEs == True:
            for word, et in zip(str(s).lower().split(),target_entity_types_for_words_dev) :
                #if word not in stoplist and emb_dict.has_key(word) == True:
                if emb_dict.has_key(word) == True:
                    seq_dev_indx[s_count].append(dict_word_indices_for_emb[word])
                    targets_dev_NEs[s_count].append(et)
                    '''
                    if get_NEs == True:
                        if slot_fill == True:
                            if word == target_entities_dev[0] or word == target_entities_dev[1]:
                                targets_dev_NEs[s_count].append(label_encoder.transform(word))
                            else:
                                targets_dev_NEs[s_count].append(label_encoder.transform('other'))
                        else:
                            if word == target_entities_dev[s_count][0] or word == target_entities_dev[s_count][1]:
                                targets_dev_NEs[s_count].append(label_encoder.transform(word))
                            else:
                                targets_dev_NEs[s_count].append(label_encoder.transform('other'))
                    '''
                else:
                    seq_dev_indx[s_count].append(dict_word_indices_for_emb['<unk>'])
                    targets_dev_NEs[s_count].append(et)
        else:
            for word in str(s).lower().split():
                if emb_dict.has_key(word) == True:
                    seq_dev_indx[s_count].append(dict_word_indices_for_emb[word])
                else:
                    seq_dev_indx[s_count].append(dict_word_indices_for_emb['<unk>'])

        s_count+=1

    s_count = 0

    for s in seq_test:
        if get_NEs == True:
            for word, et in zip(str(s).lower().split(),target_entity_types_for_words_test) :
                #if word not in stoplist and emb_dict.has_key(word) == True:
                if emb_dict.has_key(word) == True:
                    seq_test_indx[s_count].append(dict_word_indices_for_emb[word])
                    targets_test_NEs[s_count].append(et)
                    '''
                    if get_NEs == True:
                        if slot_fill == True:
                            if word == target_entities_test[0] or word == target_entities_test[1]:
                                targets_test_NEs[s_count].append(label_encoder.transform(word))
                            else:
                                targets_test_NEs[s_count].append(label_encoder.transform('other'))
                        else:
                            if word == target_entities_test[s_count][0] or word == target_entities_test[s_count][1]:
                                targets_test_NEs[s_count].append(label_encoder.transform(word))
                            else:
                                targets_test_NEs[s_count].append(label_encoder.transform('other'))
                    '''
                else:
                    seq_test_indx[s_count].append(dict_word_indices_for_emb['<unk>'])
                    targets_test_NEs[s_count].append(et)
        else:
            for word in str(s).lower().split():
                if emb_dict.has_key(word) == True:
                    seq_test_indx[s_count].append(dict_word_indices_for_emb[word])
                else:
                    seq_test_indx[s_count].append(dict_word_indices_for_emb['<unk>'])

        s_count+=1

    if slot_fill == True:
        target_entities_train_indx = []
        target_entities_dev_indx = []
        target_entities_test_indx = []

        if emb_dict.has_key(target_entities_train[0]):
            target_entities_train_indx.append(dict_word_indices_for_emb[target_entities_train[0]])
        else:
            target_entities_train_indx.append(dict_word_indices_for_emb['<unk>'])

        if emb_dict.has_key(target_entities_train[1]):
            target_entities_train_indx.append(dict_word_indices_for_emb[target_entities_train[1]])
        else:
            target_entities_train_indx.append(dict_word_indices_for_emb['<unk>'])

        if emb_dict.has_key(target_entities_dev[0]):
            target_entities_dev_indx.append(dict_word_indices_for_emb[target_entities_dev[0]])
        else:
            target_entities_dev_indx.append(dict_word_indices_for_emb['<unk>'])

        if emb_dict.has_key(target_entities_dev[1]):
            target_entities_dev_indx.append(dict_word_indices_for_emb[target_entities_dev[1]])
        else:
            target_entities_dev_indx.append(dict_word_indices_for_emb['<unk>'])

        if emb_dict.has_key(target_entities_test[0]):
            target_entities_test_indx.append(dict_word_indices_for_emb[target_entities_test[0]])
        else:
            target_entities_test_indx.append(dict_word_indices_for_emb['<unk>'])

        if emb_dict.has_key(target_entities_test[1]):
            target_entities_test_indx.append(dict_word_indices_for_emb[target_entities_test[1]])
        else:
            target_entities_test_indx.append(dict_word_indices_for_emb['<unk>'])

    elif slot_fill == 'conll':
        target_entities_train_indx = [[] for x in range(n_train)]
        target_entities_dev_indx = [[] for x in range(n_dev)]
        target_entities_test_indx = [[] for x in range(n_test)]
    else:
        target_entities_train_indx = [[] for x in range(n_train)]
        target_entities_dev_indx = [[] for x in range(n_dev)]
        target_entities_test_indx = [[] for x in range(n_test)]
        s_count= 0
        for entities in target_entities_train:
            if emb_dict.has_key(entities[0]):
                target_entities_train_indx[s_count].append(dict_word_indices_for_emb[entities[0]])
            else:
                target_entities_train_indx[s_count].append(dict_word_indices_for_emb['<unk>'])

            if emb_dict.has_key(entities[1]):
                target_entities_train_indx[s_count].append(dict_word_indices_for_emb[entities[1]])
            else:
                target_entities_train_indx[s_count].append(dict_word_indices_for_emb['<unk>'])

            s_count+=1

        s_count= 0
        for entities in target_entities_dev:
            if emb_dict.has_key(entities[0]):
                target_entities_dev_indx[s_count].append(dict_word_indices_for_emb[entities[0]])
            else:
                target_entities_dev_indx[s_count].append(dict_word_indices_for_emb['<unk>'])

            if emb_dict.has_key(entities[1]):
                target_entities_dev_indx[s_count].append(dict_word_indices_for_emb[entities[1]])
            else:
                target_entities_dev_indx[s_count].append(dict_word_indices_for_emb['<unk>'])

            s_count+=1

        s_count= 0
        for entities in target_entities_test:
            if emb_dict.has_key(entities[0]):
                target_entities_test_indx[s_count].append(dict_word_indices_for_emb[entities[0]])
            else:
                target_entities_test_indx[s_count].append(dict_word_indices_for_emb['<unk>'])

            if emb_dict.has_key(entities[1]):
                target_entities_test_indx[s_count].append(dict_word_indices_for_emb[entities[1]])
            else:
                target_entities_test_indx[s_count].append(dict_word_indices_for_emb['<unk>'])

            s_count+=1

    return seq_train_indx, target_entities_train_indx, seq_dev_indx , target_entities_dev_indx, seq_test_indx , target_entities_test_indx, \
           targets_train_NEs, targets_dev_NEs, targets_test_NEs



