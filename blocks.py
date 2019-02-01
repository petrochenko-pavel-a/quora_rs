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

class CropWindow(Wrapper):

    def __init__(self,count,layer,**kwargs):
        self.count=count
        super(CropWindow, self).__init__(layer, **kwargs)
        self.return_sequences=True
        self.return_state=False

    def compute_output_shape(self, input_shape):
        print(input_shape)
        output_shape = self.layer.compute_output_shape((None,20,input_shape[2]))
        print("Out:")
        print(output_shape)
        return output_shape

    def build(self, input_shape=None):

        self.layer.build((input_shape[0],20,input_shape[2]))
        self.built = True

    def call(self, inputs, **kwargs):
        print(inputs)
        fi=[inputs[0][0:self.count,:]]
        return self.layer.call([fi,inputs[1]],**kwargs)

    def get_config(self):
        return self.layer.get_config()

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from keras.layers import deserialize as deserialize_layer
        rnn_layer = CuDNNGRU(**config)
        num_constants = config.pop('num_constants', None)
        layer = cls(20,rnn_layer)
        layer._num_constants = num_constants
        return layer


def residual_gru_with_attention(input,channels,last_layer_channels, dropout, output_channels,mode="atention"):
    m = CuDNNGRU(channels, return_sequences=True)(input)
    m = layers.attention_3d_block(m)
    # m1 = CuDNNGRU(channels, return_sequences=True)(m)
    # m =  add([m, m1])
    m = CuDNNGRU(last_layer_channels, return_sequences=False)(m)

    # if mode=="atention":
    #     m = layers.AttLayer()(m)
    # else:
    #     if mode =="max":
    #         m=layers.MaxPool1D(m)
    #     else: m=layers.AveragePooling1D(m)

    m = Dropout(dropout)(m)
    m = Dense(output_channels, activation="relu")(m)
    return m