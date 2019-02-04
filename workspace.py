from preprocessing import *
from embeddings import *
from keras.preprocessing import sequence
from custom_vect import *
from datasets import *
from  utils import *
from keras.layers import Embedding,concatenate,Lambda
import tokenfix
import keras
from sklearn.decomposition import PCA

class ClassificationWorkspace:

    def __init__(self,config,EMB_DIR,path):
        self.EMB_DIR=EMB_DIR
        self.path=path
        self.train_dataset=None
        self.test_dataset=None
        self.num_words=config["num_words"]
        self.num_chars=config["num_chars"]
        self.max_words_seq_length=config["max_words_seq"]
        self.max_chars_seq_length=config["max_chars_seq"]
        self.embeddings=config["embeddings"]
        self.computedEmbeddings=None
        self.train_words = None
        self.train_chars = None
        self.test_words = None
        self.test_chars = None
        self.fold_count=config["folds"]
        self.folds=None
        self.config=config
        self.ew=None
        self.meta=None
        self.holdout_split=None

    def augment(self,td):
        t0=sequence.pad_sequences(self.aug.augment(td[0]),self.max_words_seq_length)
        print(t0.shape)
        t1 = sequence.pad_sequences(self.aug.augment(td[1]), self.max_words_seq_length)
        t2=td[2]
        return [t0,t1,t2]

    def prepare(self,train_dataset,test_dataset):
        self.train_dataset=train_dataset
        self.test_dataset=test_dataset
        train_words = Tokenize(train_dataset)
        train_chars = TokenizeChars(train_dataset,"train",num_chars=self.num_chars)
        test_words = Tokenize(test_dataset, "test")
        test_chars = TokenizeChars(test_dataset, "test",num_chars=self.num_chars)

        vocab = Vocabulary([train_words, test_words])
        repl=tokenfix.VocabularyReplacer()
        repl.init(vocab)
        # newT=[]
        # ls=0;
        # for x in train_words:
        #     ra=repl.fix(x)
        #     if str(x).lower()!=str(ra).lower():
        #         if self.train_dataset.predictions()[ls]==1:
        #             print(x,ra)
        #     ls=ls+1
        #     newT.append(ra)
        # self.train_words=newT
        # train_words.tokens=self.train_words
        # newTest = []
        # for x in test_words:
        #     newTest.append(repl.fix(x))
        # self.test_words = newTest
        # test_words.tokens=self.test_words
        # vocab = Vocabulary([train_words, test_words])
        # print("Voc:"+str(len(vocab.items())))
        wn = WordNumber(train_words, vocab, "train_words", self.num_words)
        self.wn=wn
        wnt = WordNumber(test_words, vocab, "test_words", self.num_words)

        repl.init_numbers(wnt)
        self.aug=repl
        self.wnt=wnt
        self.computedEmbeddings=[]

        for x in self.embeddings:
            if isinstance(x,str):
                e=WordEmbeddings(self.EMB_DIR,x, wn)
                self.computedEmbeddings.append(e.emb)
            else:
                train_embeddings_if_needed(list(train_dataset.texts())+list(test_dataset.texts()),self.EMB_DIR+x["name"],x["size"],x["window"],x["min_count"])
                e = WordEmbeddings(self.EMB_DIR,x["name"], wn)
                self.computedEmbeddings.append(e.emb)

        self.train_words = sequence.pad_sequences(wn.wi, self.max_words_seq_length)
        self.train_words1=sequence.pad_sequences([list(reversed(x)) for x in wn.wi], self.max_words_seq_length)
        self.train_chars = sequence.pad_sequences(train_chars.tokens, self.max_chars_seq_length)
        self.test_words = sequence.pad_sequences(wnt.wi, self.max_words_seq_length)
        self.test_words1 = sequence.pad_sequences([list(reversed(x)) for x in wnt.wi], self.max_words_seq_length)
        self.test_chars = sequence.pad_sequences(test_chars.tokens, self.max_chars_seq_length)
        self.meta=np.array([[len(x) / 100] for x in wn.wi])
        self.meta_test = np.array([[len(x) / 100] for x in wnt.wi])


        self.holdout_split = holdout(train_dataset, self.config["holdout"],self.config["holdout_seed"])

        ts=[]
        ts1=[]

        mn0 = np.zeros((100))
        mn1 = np.zeros((100))


        writeText("holdout_indexes.txt",self.holdout_split[1])
        self.folds = KFoldDataSet(train_dataset, self.holdout_split[0], self.config["folds"], self.config["folds_seed"], self.config["stratify_folds"])
        self.folds.indexes=[ (np.array(self.holdout_split[0])[x[0]],np.array(self.holdout_split[0])[x[1]]) for x in self.folds.indexes]
        for i in range(len(self.folds.indexes)):
            writeText("fold_indexes_train" + str(i) +".txt", self.folds.indexes[i][0])
            writeText("fold_indexes_val" + str(i) + ".txt", self.folds.indexes[i][1])

    def create_keras_word_embedings_layer(self,num,iter):
        if self.ew is not None:
             weights=self.ew
        else:
            weights =  [np.concatenate(self.computedEmbeddings, axis=1)]
            self.ew=weights
        weights[0]=weights[0]
        #pca = PCA(n_components=600)
        #principalComponents = pca.fit_transform(weights[0])
        #print(principalComponents)
        # #conc = np.concatenate(workspace.computedEmbeddings)
        # i = keras.layers.Input((weights[0].shape[1],))
        # d = keras.layers.Dense(300, activation="relu")(i)
        # #d = keras.layers.Dense(600, activation="relu")(d)
        # o = keras.layers.Dense(weights[0].shape[1], activation=None)(d)
        # mz = keras.Model(i, o)
        # mz.compile("adam", keras.losses.mean_squared_error, metrics=[keras.metrics.mean_squared_error])
        # mz.fit(weights, weights, epochs=13, validation_split=0.1, verbose=1, batch_size=1024)
        #
        #
        # ma=keras.Model(i, mz.layers[-2].output)
        # eee=ma.predict(weights)
        # #print(pred.shape)
        # eee=np.concatenate(eee)
        emb = Embedding(weights[0].shape[0],weights[0].shape[1], weights=weights, trainable=False)

        # if "dim_num_embedding" in self.config:
        #     emb=concatenate([emb,Embedding(self.num_words,self.config["dim_num_embedding"],trainable=False)
        return emb

    def get_data(self,foldNum):
        train = self.folds.indexes[foldNum][0]
        val = self.folds.indexes[foldNum][1]
        train_ = self.train_words[train]
        train_chars_ = self.train_words1[train]
        train_len=self.meta[train]
        train_ = [train_, train_chars_,train_len]
        pred_train = self.train_dataset.predictions()[train]
        pred_val = self.train_dataset.predictions()[val]
        val_ = [self.train_words[val], self.train_words1[val], self.meta[val]]
        return pred_train, pred_val, train_, val_



    def get_test(self):
        return [self.test_words, self.test_words1,self.meta_test]

    def get_holdout(self):
        indexes = self.holdout_split[1]

        words = self.train_words[indexes]
        chars = self.train_words1[indexes]
        combined = [words, chars, self.meta[indexes]]
        predictions = self.train_dataset.predictions()[indexes]

        return predictions, combined