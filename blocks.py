from keras.layers import *
import keras as keras
import layers
from tensorflow.python.framework.ops import Tensor

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

def lstmA(input,channels=128):
    x=input
    x=Bidirectional(CuDNNLSTM(channels,return_sequences=True))(x)
    x=layers.SeqWeightedAttention()(x)
    #f = Dropout(0.2)(x)

    result = Dense(16, activation="relu")(x)
    return x

def lstmB(input,channels=128):
    x=input
    x=Bidirectional(CuDNNLSTM(channels,return_sequences=True))(x)
    x=layers.SeqSelfAttention(attention_activation='sigmoid')(x)
    #f = Dropout(0.2)(x)
    x=GlobalAveragePooling1D()(x)
    result = Dense(16,name="B", activation="relu")(x)
    return x

def convLstm(input,dropout,output_channels,channels=(100,100)):
    x =input
    x=Conv1D(100,1,activation="tanh")(x)
    tc=[]
    for i in range(10):
        m = Bidirectional(CuDNNGRU(30, return_sequences=True))(x)
        m = layers.attention_3d_block(m)
        m1 = Bidirectional(CuDNNGRU(30, return_sequences=True))(m)
        m = add([m, m1])
        m = Bidirectional(CuDNNGRU(15, return_sequences=True))(m)
        m = layers.AttLayer(15)(m)
        tc.append(m)
    m=add(tc)
    m = Dense(15, activation="relu")(m)
    #result = Dense(output_channels, activation="relu")(x)
    return m

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

def fullSeqDense(input, inputType, num_words, embedding_vector_length = 200, dropout = 0.0,
                 dropout2 = 0.0, activation="relu", third_layer = False,
                 embedding_regularizer=0.0,
                 dense_regularizer=0.0,
                 output_channels = 64):
    m = None
    if inputType == "wordNumbers":
        embRegularizer = None
        if embedding_regularizer > 0: embRegularizer=keras.regularizers.l2(embedding_regularizer)

        m = Embedding(input_dim=num_words, output_dim=embedding_vector_length,
                      trainable=True, embeddings_regularizer=embRegularizer,
                      name="EmbeddingDense")(input)

        if dropout > 0: m = Dropout(dropout)(m)
    elif inputType == "words":
        m = input

    m = Flatten()(m)

    denseRegularizer = None
    if dense_regularizer > 0: denseRegularizer = keras.regularizers.l2(dense_regularizer)

    if third_layer:
        m = Dense(int((embedding_vector_length + output_channels)/2),
                  activation=activation, kernel_regularizer=denseRegularizer, name="SecondDense")(m)

        if dropout2 > 0: m = Dropout(dropout2)(m)

    m = Dense(output_channels, kernel_regularizer=denseRegularizer, activation=activation,
              name="ThirdDense")(m)

    return m

def inception(input: Tensor,dropout=0.0, depth = 1,
              channels=64, cardinality = 1, mode = "all",
              inner_channels=64, inner_cardinality = 1, inner_mode = "3and1pool",
              output_channels=64, bottleneck=True, pool:str = None, activation="relu"):
    """
    Creates inception-like convlotion-pool network for sequence analysis
    :param input: input Tensor, expected word vector, channels cant be higher than word vector length
    :param dropout: final dropout
    :param channels: first layer features number
    :param inner_channels: inner convolution layers features number
    :param mode: conv blocks in the first layer "all", "3only", "5only", "1pool", "3and1pool", "5and1pool"
    :param inner_mode: conv blocks in the inner layers "all", "3only", "5only", "1pool", "3and1pool", "5and1pool"
    :param output_channels: - number of output channels, default to 64
    :param activation: default to relu
    :param pool: None, "1x" and "2x". "1x" reduces output size by 2 once per 2 depth layers, "2x" reduces output size by 2 once per depth layer.
    :return:
    """

    featuresNum = input.shape.dims[input.shape.ndims-1].value
    if channels > featuresNum: channels = featuresNum
    if inner_channels > featuresNum: second_channels = featuresNum

    m = input

    if depth == 1:
        m = Resnext1DBlock(m, channels, mode, activation, cardinality, bottleneck, pool is not None)
    elif depth == 2:
        m = Resnext1DPairBlock(m, channels, mode, activation, cardinality, bottleneck, pool,
                               inner_channels = inner_channels, inner_mode = inner_mode,
                               inner_cardinality = inner_cardinality)
    elif depth > 2:
        numOfPairs = depth // 2
        #initial pair:
        m = Resnext1DPairBlock(m, channels, mode, activation, cardinality, bottleneck, pool,
                               inner_channels=inner_channels, inner_mode=inner_mode,
                               inner_cardinality = inner_cardinality)
        #additional pairs:
        for pairIndex in range(numOfPairs - 1):
            m = Resnext1DPairBlock(m, inner_channels, inner_mode, activation,
                                   inner_cardinality, bottleneck, pool)

        #even layer if applicable
        if depth % 2 == 1:
            m = Resnext1DBlock(m, inner_channels, inner_mode, activation,
                               inner_cardinality, bottleneck, pool is not None)

    globalAve = GlobalAveragePooling1D()(m)
    globalMax = GlobalMaxPooling1D()(m)
    m = concatenate([globalAve, globalMax])

    if dropout > 0: m = Dropout(dropout)(m)

    m = Dense(output_channels, activation="relu")(m)

    return m

def Resnext1DPairBlock(input: Tensor, channels: int, mode: str, activation: str, cardinality,
                       bottleneck: bool, pool: str,
                       inner_channels = None, inner_mode = None, inner_cardinality = None):
    """

    :param input:
    :param channels:
    :param mode:
    :param activation:
    :param cardinality:
    :param bottleneck:
    :param pool: supported values: None, "1x" and "2x". "1x" reduces output size twice per block, "2x" reduces size 4 times per block.
    :param inner_channels:
    :param inner_mode:
    :param inner_cardinality:
    :return:
    """
    if inner_channels is None: inner_channels = channels
    if inner_mode is None: inner_mode = mode
    if inner_cardinality is None: inner_cardinality = cardinality

    first = Resnext1DBlock(input, channels, mode, activation, cardinality,
                           bottleneck, pool == "2x")

    second = Resnext1DBlock(first, inner_channels, inner_mode, activation,
                            inner_cardinality, bottleneck, pool is not None)

    pooledInput = input
    if pool is not None: pooledInput = MaxPool1D(3, 2, padding="same")(pooledInput)
    if pool == "2x": pooledInput = MaxPool1D(3, 2, padding="same")(pooledInput)

    return concatenate([pooledInput, second])

def Resnext1DBlock(input: Tensor, channels: int, mode: str, activation: str, cardinality: int,
                   bottleneck: bool, pool: bool):
    if cardinality == 1:
        return Inception1DBlock(input, channels, mode, activation, bottleneck, pool)

    inceptionBlocks = []
    for i in range(cardinality):
        inceptionBlocks.append(Inception1DBlock(input, channels, mode, activation, bottleneck, pool))

    return concatenate(inceptionBlocks)

def Inception1DBlock(input: Tensor, channels: int, mode: str, activation: str, bottleneck: bool,
                     pool: bool):
    strides = 1
    if pool: strides = 2

    conv1 = Conv1D(channels, 1, padding="same", strides=strides, activation=activation)(input)

    if bottleneck:
        conv3 = Conv1D(channels//3, 1, padding="same", activation=activation)(input)
        conv3 = Conv1D(channels, 3, padding="same", strides=strides, activation=activation)(conv3)
    else:
        conv3 = Conv1D(channels, 3, padding="same", strides=strides, activation=activation)(input)

    if bottleneck:
        conv5 = Conv1D(channels//4, 1, padding="same", activation=activation)(input)
        conv5 = Conv1D(channels, 5, padding="same", strides=strides, activation=activation)(conv5)
    else:
        conv5 = Conv1D(channels, 5, padding="same", strides=strides, activation=activation)(input)

    pool = MaxPool1D(pool_size=3, padding="same", strides=strides)(input)
    pool = Conv1D(channels, 1, padding="same", activation=activation)(pool)

    if mode == "all":
        m = concatenate([conv1, conv3, conv5, pool])
    elif mode == "3only":
        m = conv3
    elif mode == "5only":
        m = conv5
    if mode == "1pool":
        m = concatenate([conv1, pool])
    if mode == "3and1pool":
        m = concatenate([conv1, conv3, pool])
    if mode == "5and1pool":
        m = concatenate([conv1, conv5, pool])

    return m