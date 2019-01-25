import os
import operator
from nltk.tokenize import casual_tokenize
from utils import save,load
import numpy as np
from keras.preprocessing import text

class Tokenize:

    def __init__(self,ds,name="train"):
        fn="tokens_" + name + ".dat"
        if os.path.exists(fn):
            self.tokens,self.cases=load(fn)
            return
        data=[]
        cases=[]
        print("Tokenizing:"+name)
        for f in ds.texts():
            ts=casual_tokenize(f, preserve_case=True);
            case=[]
            tw=[]
            for x in ts:
                tw.append(x)
                if x.upper()==x:
                    case.append(2)
                elif x.lower()!=x:
                    case.append(1)
                else:
                    case.append(0)
            cases.append(case)
            data.append([x for x in tw])
        self.tokens=np.array(data)
        self.cases=np.array(cases)
        save(fn,[self.tokens,self.cases])

    def __getitem__(self, item):
        return self.tokens[item]

    def __len__(self):
        return len(self.tokens)


class TokenizeChars:

    def __init__(self,ds,name="train",num_chars=200):
        fn="tokens_chars_" + name + ".dat"
        if os.path.exists(fn):
            self.tokens=load(fn)
            return
        data=[]
        cases=[]
        print("Tokenizing chars:"+name)
        ts=text.Tokenizer(num_words=num_chars, char_level=True)
        ts.fit_on_texts(ds.texts())
        mm=ts.texts_to_sequences(ds.texts())
        self.tokens=mm
        save(fn,mm)

    def __getitem__(self, item):
        return self.tokens[item]

    def __len__(self):
        return len(self.tokens)


class Vocabulary:
    def __init__(self,d,name=""):
        fn="vocabulary_" + name + ".dat"
        self._items = None
        if os.path.exists(fn):
            self.vocab=load(fn)
            return

        print("Vocabulary:"+name)
        vocab={}
        for ts in d:
            for f in ts.tokens:
                for w in f:
                    if w in vocab:
                        vocab[w]=vocab[w]+1
                    else:
                        vocab[w]=1
        self.vocab=vocab
        save(fn,self.vocab)

    def items(self):
        if self._items is None:
            self._items=sorted(list(self.vocab.items()),key=operator.itemgetter(1),reverse=True)
        return self._items


    def __getitem__(self, item):
        return self.items()[item]

    def __len__(self):
        return len(self.vocab)

class WordNumber:
    def __init__(self,ts,vocab,name="train",maxWords=-1):
        self.ts=ts;
        self.vocab=vocab
        fn = "word_indese" + name + ".dat"

        self.vn = None
        if os.path.exists(fn):
            self.wi,self.wn,self.nw = load(fn)
            return
        voc=vocab.vocab

        wn={}
        nw = {}
        if maxWords==-1:
            it=vocab.items()
        else: it=vocab.items()[:maxWords]
        for m in it:
            c=len(wn)
            wn[m[0]]=c
            nw[c]=m[0]

        self.vn=[]
        for f in ts.tokens:
            numbers=[]
            for w in f:
                if w in wn:
                    numbers.append(wn[w])
            self.vn.append(numbers)
        self.wi=np.array(self.vn)
        self.wn=wn
        self.nw=nw
        save(fn,[self.wi,wn,nw])

    def tokens(self):
        return self.vn