from keras.layers import *
import layers
import keras
import workspace

def embeddings_with_dropout(all_embeddings, cfg ):
    if cfg.spatial:
        all_emb = SpatialDropout1D(cfg.dropout)(all_embeddings)
    else:
        all_emb = Dropout(cfg.dropout)(all_embeddings)
    return all_emb


def global_features_network(l):
    q = Masking()(l)
    f20 = GlobalAveragePooling1D()(q)
    f21 = GlobalMaxPooling1D()(l)
    f = Dropout(0.3)(concatenate([f20, f21]))
    global_features = Dense(16, activation="relu")(f)
    return global_features


def char_cnn(characters):
    m2 = Conv1D(100, 2, activation="relu")(characters)
    m2 = AveragePooling1D(2)(m2)
    m2 = Conv1D(100, 2, activation="relu")(m2)
    m2 = AveragePooling1D(2)(m2)
    m2 = Conv1D(100, 2, activation="relu")(m2)
    m2 = AveragePooling1D(2)(m2)
    m2 = Conv1D(100, 2, activation="relu")(m2)
    m20 = GlobalAveragePooling1D()(m2)
    m21 = GlobalMaxPooling1D()(m2)
    m2 = Dropout(0.3)(concatenate([m20, m21]))
    m2 = Dense(16, activation="relu")(m2)
    return m2


def residual_gru_with_attention(units, dropout, feature_count, input):
    m = Bidirectional(CuDNNGRU(units, return_sequences=True))(input)
    m = layers.attention_3d_block(m)
    m1 = Bidirectional(CuDNNGRU(units, return_sequences=True))(m)
    m =  add([m, m1])
    m = Bidirectional(CuDNNGRU(units // 2, return_sequences=True))(m)
    m = layers.AttLayer()(m)
    m = Dropout(dropout)(m)
    m = Dense(feature_count, activation="relu")(m)
    return m


def build_network(word_embeddings,  characters, cfg):
    all_emb = embeddings_with_dropout(  word_embeddings,cfg)
    main_branch = residual_gru_with_attention(cfg.count * 25, cfg.dropout1, 128, all_emb)
    results=[main_branch]
    if cfg.addGlobal:
        results.append(global_features_network(all_emb))
    if cfg.addChars:
        results.append(char_cnn(characters))
    if len(results)>1:
        return concatenate(results)
    else: return main_branch


def create_model(cfg,workspace:workspace.ClassificationWorkspace):
    words = Input(shape=(workspace.max_words_seq_length,))
    chars = Input(shape=(workspace.max_chars_seq_length,))
    word_embeddings  = workspace.create_keras_word_embedings_layer(words)
    chars_embedding = Embedding(workspace.num_chars, 50, trainable=False)(chars)
    out = Dense(1, activation="sigmoid")(build_network(word_embeddings,  chars_embedding, cfg))
    m = keras.models.Model([words,chars], out)
    m.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return m