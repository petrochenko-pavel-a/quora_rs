import numpy as np

from  preprocessing import Vocabulary,WordNumber
import os
import string
import singularize
#os.chdir("data1")


#voc=Vocabulary(None)
se=singularize.English()

from tqdm import tqdm

def is_number(s):
    s=s.replace("-","")
    s = s.replace("/", "")
    s = s.replace(",", "")
    s = s.replace(":", "")
    try:
        float(s)
        return True
    except ValueError:
        return False

def test(voc,w1,original=None):
    if len(w1)==0:
        return True
    if w1 in voc.vocab :
        if voc.vocab[w1] > 5:
            return True
        if original is not None:
            if original not in voc.vocab:
                return True
            if voc.vocab[original]*2<voc.vocab[w1]:
                return True
    if w1[0] in string.ascii_lowercase:
        if test(voc,w1[0].upper()+w1[1:]):
            return True
    return False

def find(voc,w:str):
    for x in range(1,len(w)):
        w1=w[0:x]
        w2 = w[x:]
        if len(w1)>2 or w1=="a" or w1=="of" or w1=="at" or w1=="do" or w1=="un" or w1=="in" or w1=="is" or w1=='i' or w1=="to" or w1=="my":
            if len(w2)>2 or w2=='of' or w2=="an" or w2=="the":
                if test(voc,w1) and test(voc,w2):
                    return [w1,w2]
    for i in range(1,len(w)):
        w1=w[:i]+w[i+1:]
        if test(voc,w1,w):
           return w1
    for i in range(1,len(w)):
        for e in string.ascii_lowercase:
            w1 = w[:i] +e+ w[i + 1:]
            if test(voc,w1,w):
               return w1
    for i in range(1,len(w)):
        for e in string.ascii_lowercase:
            w1 = w[:i] +e+ w[i:]
            if test(w1,w):
                return w1
    for i in range(1,len(w)-1):
          w1 = w[:i] +w[i+1]+w[i]+ w[i+2:]
          if test(voc,w1,w):
              return w1
    return None

class VocabularyReplacer:

    def __init__(self):
        self.replacements={}
        self.words=set()
        self.cases={}
        self.voc=None

    def init(self,voc:Vocabulary):
        self.voc=voc
        for w, c in voc.items():
            try:
                self.append(w)
            except:
                pass
        for word in self.words:
            try:
                if len(word) > 1:
                    uword = word[0].upper() + word[1:]
                    try:

                        if word in voc.vocab and uword in voc.vocab:
                            if voc.vocab[word]*2 < voc.vocab[uword]:
                                self.cases[word] = uword

                    except:
                        print(word, uword)
            except:
                pass
        pass

    def fix(self,words):
        result=[]
        for w in words:
            if w in self.replacements:
                x=self.replacements[w]
                if isinstance(x,list):
                    for v in x:
                        result.append(str(v))

                else:
                    # if x in self.cases:
                    #     x=self.cases[x]
                    result.append(x)
            else:
                result.append(w)
        return result

    def init_numbers(self,n:WordNumber):
        self.ns=n
        self.nreplacements={}
        for q in self.replacements:
            if q in n.wn:
                z=self.replacements[q]
                if isinstance(z,str):
                    if z in n.wn:
                        self.nreplacements[n.wn[q]]=n.wn[z]
                        self.nreplacements[n.wn[z]] = n.wn[q]

    def augment(self,items):
        m=[]
        for v in tqdm(range(len(items))):
            v=items[v]
            rs=[]
            for i in range(len(v)):
                q=np.random.randint(0, 15)
                if q<3:
                    if v[i] in self.nreplacements:
                        rs.append(self.nreplacements[v[i]])
                    continue
                if q==4:
                    continue
                if q==5 and len(rs)>2:
                    z=rs.pop()
                    rs.append(v[i])
                    rs.append(z)
                    continue
                rs.append(v[i])
            m.append(rs)
        r= np.array(m)
        print(r.shape)
        return r


    def append(self,word:str):
        voc=self.voc
        original=word
        if word.upper()==word and len(word)<5 and voc.vocab[word]>3:
            #this is abbr lets keep them
            if not word.lower() in voc.vocab or voc.vocab[word.lower()]<10:
                self.replacements[original] = word
                self.words.add(word)
                return
        #Collapse word case
        if word.lower()!=word:
            word=word.lower()
        #collapse urls
        if "://" in word or ".com" in word or ".org" in word or ".edu" in word or ".ru" in word or ".net" in word or "www." in word:
            word="http//google.com"

        #Collapse `s
        if word[-2:] == "\'s" :
            word = word[:-2]

        #Collapse plural forms
        if word[-1] == 's':
            ms = se.singularize(word)
            if (ms in voc.vocab and voc.vocab[ms]>1) or ms[-2:]=="ns" or ms[-2:]=="ts" or ms[-2:]=="rs" or ms[-2:]=="ps" or ms[-2:]=="ds" or ms[-2:]=="hs":
                word=ms

        if is_number(word):
            word="8"
        if "-" in word and word!='-':
            words=word.split("-")
            #print("A")
            word=words
        if len(word)>0:
            if word[0].isdigit():
                for i in range(len(word)):
                    if not word[i].isdigit():
                        word=[word[:i],word[i:]]
                        break
        if isinstance(word, str):

            if not test(voc,word):
                we=find(voc,word)
                if we is not None:
                    word=we
        self.replacements[original]=word
        if isinstance(word,list):
            for x in word:
                if isinstance(x,str):
                    if not test(voc,x):
                        we = find(voc,x)
                        if isinstance(we,str):
                            x = we
                    self.words.add(x)
        else: self.words.add(word)
        pass


#r=VocabularyReplacer()
#r.init(voc)

#print(r.fix(["I","like","potatoes","but","i","don't","like","trumpists","and","their","downtrace"]))
#
# print(len(r.words))
#
#
#
# print(len(r.cases))
# for w in r.cases:
#     print(w,r.cases[w])
#
#
# with open("replacements.txt","w",encoding="utf8") as f:
#     for c in r.replacements.items():
#         if c[0]!=c[1]:
#             f.write(c[0]+" "+str(c[1])+"\r")
# with open("bad_replacements.txt", "w", encoding="utf8") as f:
#         for word in r.words:
#             if not word in voc.vocab or voc.vocab[word] < 2:
#                 if len(word)>1:
#                     ow=word
#                     word=word[0].upper()+word[1:]
#                     if not word in voc.vocab or voc.vocab[word] < 2:
#                         f.write(ow+"\r")
#
#
#
# print(len(r.words))
#
#
#
