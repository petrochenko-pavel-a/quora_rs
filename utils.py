import pickle
import numpy as np
import yaml
import os
from sklearn.metrics import f1_score

def load(s):
    with open(s,"rb") as f:
        return pickle.load(f)

def load_yaml(s):
    with open(s,"rb") as f:
        return yaml.load(f)
def writeText(s,val):
    with open(s,"w",encoding="utf8") as f:
        for x in val:
            f.write(str(x)+"\r")

def save(s,o):
    with open(s,"wb") as f:
        pickle.dump(o,f,pickle.HIGHEST_PROTOCOL)

def eval_treshold_and_score(pred_val_y_1, y_val):
    thresholds = []

    for thresh in np.arange(0.1, 0.501, 0.01):
        thresh = np.round(thresh, 2)
        res = f1_score(y_val, (pred_val_y_1 > thresh).astype(int))
        thresholds.append([thresh, res])
        #print("F1 score at threshold {0} is {1}".format(thresh, res))

    thresholds.sort(key=lambda x: x[1], reverse=True)
    return thresholds[0]
def ensure_exists(directory):
    try:
        os.makedirs(directory);
    except:
        pass
def encode(bin_str, sbase):
    n = int(bin_str,2)
    code = []
    base = len(sbase)
    while n > 0:
        n, c = divmod(n, base)
        code.append(sbase[c])
    code = reversed(code)
    return ''.join(code)

def decode(code, sbase, size):
    code = reversed(code)
    base = len(sbase)
    bin_str =  bin(sum([sbase.index(c)*base**i for i, c in enumerate(code)]))[2:]
    bin_str = (size-len(bin_str))*'0' + bin_str
    return bin_str