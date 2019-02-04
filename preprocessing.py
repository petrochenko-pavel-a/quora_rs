import os
import operator
from nltk.tokenize import casual_tokenize
from utils import save,load
import numpy as np
from keras.preprocessing import text
import re
import string
import unicodedata

def spacing_misspell(text):
    """
    'deadbody' -> 'dead body'
    """
    misspell_list = [
        '(S|s)hit', '(F|f)uck'  # ,'Trump'
    ]
    misspell_re = re.compile('(%s)' % '|'.join(misspell_list))
    return misspell_re.sub(r" \1 ", text)


def clean_latex(text):
    """
    replace latex math with 'mathematical formula' tag
    """
    corr_t = []
    for t in text.split(" "):
        t = t.strip()
        if t != '':
            corr_t.append(t)
    text = ' '.join(corr_t)
    text = re.sub(r'\[math].+?\[/math]', 'mathematical formula', text)
    return text


def normalize_unicode(text):
    """
    unicode string normalization
    """
    return unicodedata.normalize('NFKD', text)


def remove_newline(text):
    """
    remove \n and  \t
    """
    text = re.sub('\n', ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub('\b', ' ', text)
    text = re.sub('\r', ' ', text)
    return text


def decontracted(text):
    """
    add space before and after punctuation and symbols
    """
    apo = "'"
    re_tok = re.compile(f'([{apo}])')
    return re_tok.sub(r' \1', text)

    # quora
    text = re.sub(r"(Q|q)uoran", "quora contributor", text)
    text = re.sub(r"(Q|q)uorans", "quora contributors", text)

    return text


def spacing_punctuation(text):
    """
    add space before and after punctuation and symbols
    """
    regular_punct = list(string.punctuation)
    extra_punct = [
        ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', '$', '&',
        '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£',
        '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›',
        '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”',
        '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',
        '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼',
        '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
        'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',
        '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
        '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡', '§', '£', '₤']
    all_punct = ''.join(sorted(list(set(regular_punct + extra_punct))))
    re_tok = re.compile(f'([{all_punct}])')
    return re_tok.sub(r' \1 ', text)


def spacing_digit(text):
    """
    add space before and after digits
    """
    re_tok = re.compile('([0-9])')
    return re_tok.sub(r' \1 ', text)


def spacing_number(text):
    """
    add space before and after numbers
    """
    re_tok = re.compile('([0-9]{1,})')
    return re_tok.sub(r' \1 ', text)


def remove_number(text):
    """
    numbers are not toxic
    """
    return re.sub('\d+', ' ', text)


def remove_space(text):
    """
    remove extra spaces and ending space if any
    """
    text = re.sub('\s+', ' ', text)
    text = re.sub('\s+$', '', text)
    return text


def substitute(text):
    """
    substitute some words after de-contraction
    """
    # text = re.sub(r" e g ", " eg ", text)
    # text = re.sub(r" b g ", " bg ", text)
    # text = re.sub(r" u s ", " US ", text)
    # text = re.sub(r" u s a ", " USA ", text)
    text = re.sub(r"e - mail", "email", text)

    return text


def preprocess(text, sub=False, remove_num=True):
    ## remove new line
    text = remove_newline(text)
    ## de-contract
    text = decontracted(text)
    ## space misspell
    text = spacing_misspell(text)
    ## clean_latex
    text = clean_latex(text)
    ## space
    text = spacing_punctuation(text)
    ## substitute after decontract
    if sub: text = substitute(text)
    ## handle numbers
    if remove_num:
        text = remove_number(text)
    else:
        text = spacing_number(text)
        text = spacing_digit(text)
        # remove space
    text = remove_space(text)

    return text

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
        if os.path.exists(fn) :
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
        #if self._items is None:
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
        outOfVocb=set()
        for f in ts.tokens:
            numbers=[]
            for w in f:
                if w in wn:
                    numbers.append(wn[w])
                else:
                    if w.lower()==w:
                        w="something"
                        numbers.append(wn[w])
                    else:
                        w = "Something"
                        numbers.append(wn[w])
            self.vn.append(numbers)
        self.wi=np.array(self.vn)
        self.wn=wn
        self.nw=nw
        for i in outOfVocb:
            print(i)
        save(fn,[self.wi,wn,nw])

    def tokens(self):
        return self.vn