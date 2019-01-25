from preprocessing import *
from embeddings import *
from keras.preprocessing import sequence
from custom_vect import *
from datasets import *
from  utils import *
from keras.layers import Embedding

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
        self.holdout_split=None


    def prepare(self,train_dataset,test_dataset):
        self.train_dataset=train_dataset
        self.test_dataset=test_dataset
        train_words = Tokenize(train_dataset)
        train_chars = TokenizeChars(train_dataset,"train",num_chars=self.num_chars)
        test_words = Tokenize(test_dataset, "test")
        test_chars = TokenizeChars(test_dataset, "test",num_chars=self.num_chars)

        vocab = Vocabulary([train_words, test_words])

        wn = WordNumber(train_words, vocab, "train_words", self.num_words)
        self.wn=wn
        wnt = WordNumber(test_words, vocab, "test_words", self.num_words)
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
        self.train_chars = sequence.pad_sequences(train_chars.tokens, self.max_chars_seq_length)
        self.test_words = sequence.pad_sequences(wnt.wi, self.max_chars_seq_length)
        self.test_chars = sequence.pad_sequences(test_chars.tokens, self.max_chars_seq_length)

        self.holdout_split = holdout(train_dataset, self.config["holdout"],self.config["holdout_seed"])
        writeText("holdout_indexes.txt",self.holdout_split[1])
        self.folds = KFoldDataSet(train_dataset, self.holdout_split[0], self.config["folds"], self.config["folds_seed"], self.config["stratify_folds"])
        self.folds.indexes=[ (np.array(self.holdout_split[0])[x[0]],np.array(self.holdout_split[0])[x[1]]) for x in self.folds.indexes]
        for i in range(len(self.folds.indexes)):
            writeText("fold_indexes_train" + str(i) +".txt", self.folds.indexes[i][0])
            writeText("fold_indexes_val" + str(i) + ".txt", self.folds.indexes[i][1])

    def create_keras_word_embedings_layer(self, words_input):
        return Embedding(self.num_words, 1350,
                                    weights=[np.concatenate(self.computedEmbeddings, axis=1)],
                                    trainable=False)(words_input)

    def get_data(self,foldNum):
        train = self.folds.indexes[foldNum][0]
        val = self.folds.indexes[foldNum][1]
        train_ = self.train_words[train]
        train_chars_ = self.train_chars[train]
        train_ = [train_, train_chars_]
        pred_train = self.train_dataset.predictions()[train]
        pred_val = self.train_dataset.predictions()[val]
        val_ = [self.train_words[val], self.train_chars[val]]
        return pred_train, pred_val, train_, val_

    def get_test(self):
        return [self.test_words, self.test_chars]

    def get_holdout(self):
        indexes = self.holdout_split[1]

        words = self.train_words[indexes]
        chars = self.train_chars[indexes]
        combined = [words, chars]
        predictions = self.train_dataset.predictions()[indexes]

        return predictions, combined