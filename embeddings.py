import os
import gensim
import numpy as np
import gc
from tqdm import tqdm
from utils import load,save



def load_binary(words, max_features, EMBEDDING_FILE):
    wm=gensim.models.keyedvectors.Word2VecKeyedVectors(300);
    mmm=wm.load_word2vec_format(EMBEDDING_FILE,binary=True)
    nb_words = min(max_features, len(words))
    embedding_matrix = np.random.normal(0, 0.01, (nb_words, 300))
    numErrors=0
    missedWords=[]
    for word, i in words.items():
        if i >= max_features: continue
        if word in mmm.vocab:
            embedding_vector = mmm.word_vec(word)
            embedding_matrix[i] = embedding_vector
        else:
            if word.lower() in mmm.vocab:
                embedding_vector = mmm.word_vec(word.lower())
                embedding_matrix[i] = embedding_vector
            else:
                if "'s" in word:
                    if word.lower()[:-2] in mmm.vocab:
                        embedding_vector = mmm.word_vec(word.lower()[:-2])
                        embedding_matrix[i] = embedding_vector

                    else:
                        numErrors=numErrors+1
                        missedWords.append(word)
    print("Embeddings are missed for:" + str(numErrors / max_features) + " of words in index")
    return embedding_matrix,missedWords


def load_text(words, max_features, EMBEDDING_FILE, e=False):

    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    if not e:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)
    else: embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE,encoding="utf8",errors="ignore") if len(o)>100)
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    word_index = words
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    numErrors=0
    missedWords=[]
    for word, i in tqdm(word_index.items()):
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        else:
            embedding_vector = embeddings_index.get(word.lower())
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

            else:
                if "'s" in word:
                    embedding_vector = embeddings_index.get(word.lower()[:-2])
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector

                    else:
                        numErrors=numErrors+1
                        missedWords.append(word)
    del embeddings_index; gc.collect()
    print("Embeddings are missed for:"+str(numErrors/max_features)+" of words in index")

    return embedding_matrix,missedWords

class WordEmbeddings:

    def __init__(self,EMB_DIR,name:str,wn,binary=False):
        if ".bin" in name:
            binary=True
        name=name
        if os.path.exists(name):
            self.emb=load(name)
            return
        EMBEDDING_FILE = EMB_DIR + name[0:name.rindex(".")] + "/" + name
        print("Preparing:"+EMBEDDING_FILE)
        if not os.path.exists(EMBEDDING_FILE):
            EMBEDDING_FILE=EMB_DIR + name
        if not binary:
            self.emb,missed = load_text(wn.wn, len(wn.wn), EMBEDDING_FILE, True)
        else:
            self.emb,missed = load_binary(wn.wn, len(wn.wn), EMBEDDING_FILE)
        save(name,self.emb)
        with open(name+".missed_embeddings.words","w",encoding="utf8") as f:
            f.writelines([x+"\r" for x in missed])