# quora_rs

How to train model: 

`pipeline.py --data ./data --input ./experiments/exp1/config.yaml --quora D:/quora`


Example config:

```yaml
num_words: 200000
num_chars: 200

max_words_seq: 40
max_chars_seq: 150

embeddings:
  - glove.840B.300d.txt
  - wiki-news-300d-1M.vec
  - paragram_300_sl999.txt
  - GoogleNews-vectors-negative300.bin
  - name: custom.vect
    size: 150
    window: 4
    min_count: 2

holdout: 0.2
holdout_seed: 211
folds: 5
folds_seed: 33
stratify_folds: true

model:
  words_dropout: 0.3
  words_dropout_kind: spatial
  branches:

#     gf:
#        type: global_features_network
#        input: words
#        dropout: 0.3
#        output_channels: 16
#        mode: both

#     cnn:
#        type: cnn
#        input: chars
#        channels: [100,100,100,100,100]
#        pooling: [2,2,2,2,2]
#        windows: [2,2,2,2,2]
#        dropout: 0.3
#        output_channels: 16
     lstm:
       type: cnn
       input: chars
       channels: [100,100]
       dropout: 0.0
       output_channels: 16

#     gru:
#        type: residual_gru_with_attention
#        input: words
#        channels: 120
#        last_layer_channels: 60
#        output_channels: 40
#        dropout: 0.2

optimizer: adam
metrics: ['accuracy']
loss: binary_crossentropy
epochs: 3
batch: 1024

```

model branches are composed using functions in `blocks.py` 

metrics and weights will be stored in the parent of folder of experiment configuration file

Note: if you are changing preprocessing related parameters you should clean data folder