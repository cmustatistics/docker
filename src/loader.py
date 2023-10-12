import pickle
from pathlib import Path

def save_to(obj, object_str, path):
    Path(path).mkdir(parents=True, exist_ok=True)
    file = '{0}/{1}.pkl'.format(path, object_str)
    with open(file, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def load(object_str, path):
    name = '{0}{1}.pkl'.format(path, object_str)
    with open(name, 'rb') as f:
        obj = pickle.load(f)
    return obj
