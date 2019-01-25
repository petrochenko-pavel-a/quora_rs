from keras.layers import *
import layers

def global_features_network(input, dropout, output_channels,mode="ave"):
    masked = Masking()(input)
    ave = GlobalAveragePooling1D()(masked)
    max = GlobalMaxPooling1D()(input)
    i=ave
    if mode == "both":
        i=concatenate([ave, max])
    elif mode == "max":
        i=max
    f = Dropout(dropout)(i)
    global_features = Dense(output_channels, activation="relu")(f)
    return global_features

def cnn(input,dropout,output_channels,mode="ave",channels=(100,100,100,100,100),pooling=(2,2,2,2,2),windows=(2,2,2,2,2),activation="relu"):
    x=input
    for i in range(len(channels)):
        x=Conv1D(channels[i], windows[i], activation=activation)(x)
        if pooling[i]>1:
            x=AveragePooling1D(pooling[i])(x)

    ave = GlobalAveragePooling1D()(x)
    max = GlobalMaxPooling1D()(x)
    i = ave
    if mode == "both":
        i = concatenate([ave, max])
    elif mode == "max":
        i = max
    elif mode == "dense":
        i = Flatten()(x)

    f = Dropout(dropout)(i)

    result = Dense(output_channels, activation="relu")(f)
    return result

def lstm(input,dropout,output_channels,channels=(100,100)):
    x=input
    for i in range(len(channels)):
        if i==len(channels)-1:
            x = Bidirectional(CuDNNLSTM(channels[i], return_sequences=False))(x)
        else: x=Bidirectional(CuDNNLSTM(channels[i],return_sequences=True))(x)

    f = Dropout(dropout)(x)
    result = Dense(output_channels, activation="relu")(f)
    return result

def residual_gru_with_attention(input,channels,last_layer_channels, dropout, output_channels ):
    m = Bidirectional(CuDNNGRU(channels, return_sequences=True))(input)
    m = layers.attention_3d_block(m)
    m1 = Bidirectional(CuDNNGRU(channels, return_sequences=True))(m)
    m =  add([m, m1])
    m = Bidirectional(CuDNNGRU(last_layer_channels, return_sequences=True))(m)
    m = layers.AttLayer()(m)
    m = Dropout(dropout)(m)
    m = Dense(output_channels, activation="relu")(m)
    return m