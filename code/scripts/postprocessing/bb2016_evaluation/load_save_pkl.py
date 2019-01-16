from six.moves import cPickle
import json

def load_pickle_file(pickle_file_path):
    f = open(pickle_file_path, 'rb')
    loaded_pkl = cPickle.load(f)
    f.close()
    return loaded_pkl

def save_as_pkl_json(data, path):
    f1 = open(path, 'wb')
    cPickle.dump(data, f1, protocol=cPickle.HIGHEST_PROTOCOL)
    f1.close()
    with open(path + '.json', 'w') as f2:
        json.dump(data, f2, sort_keys = True, indent = 4)
        f2.close()

def save_as_pkl(data, path):
    f1 = open(path, 'wb')
    cPickle.dump(data, f1, protocol=cPickle.HIGHEST_PROTOCOL)
    f1.close()