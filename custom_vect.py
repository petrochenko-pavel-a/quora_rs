import gensim
import logging
import os

def train_embeddings_if_needed(texts, name, size, window, min_count):

    if os.path.exists(name):
        return
    print("Training custom embeddings:" + name + " size:" + str(size) + ", window:" + str(window) + ", mincount:" + str(min_count))
    with open(name+".input.txt","w",encoding="utf8") as f:
       f.write("\r\n".join(texts))




    def read_input(input_file):
        """This method reads the input file which is in gzip format"""

        logging.info("reading file {0}...this may take a while".format(input_file))
        with open(input_file, 'r',encoding="utf8") as f:
            for i, line in enumerate(f):

                if (i % 10000 == 0):
                    logging.info("read {0} reviews".format(i))

                yield gensim.utils.simple_preprocess(line)

    documents=[]
    for v in read_input(name+".input.txt"):
        documents.append(v)

    model = gensim.models.Word2Vec(
            documents,
            size=size,
            window=window,
            min_count=min_count,
            workers=10)
    model.train(documents, total_examples=len(documents), epochs=10)
    model.wv.save_word2vec_format(name)