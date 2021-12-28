import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import optimizers, metrics, layers, Model
import numpy as np
import time

vocabulary_map = {'v9_3digit':5571, 'v9':5679, 'v10': 7930}

class EmbeddingDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, demograph, icd, Y,
                 batch_size,
                 shuffle=True):
        self.demograph = demograph
        self.icd = icd
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.Y=Y
        self.n = len(self.Y)
    
    def __getitem__(self, index):
        start = index*self.batch_size
        end = min(start+self.batch_size, self.n)
        max_len = max([len(l) for l in self.icd[start:end]])
        icds = np.full((end-start, max_len), -1)
        for i in range(start, end): 
            icds[i-start, :len(self.icd[i])] = self.icd[i]
        return [self.demograph[start:end,:], icds+1], self.Y[start:end]

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

def icd_list_onehot(icd_list, unique_codes = 12683):
    onehot = np.zeros((len(icd_list), unique_codes), dtype=np.float32)
    for i, icd_row in enumerate(icd_list):
        onehot[i][icd_row]=1
    return onehot

def create_embedding_model(vocabulary, demographic_size, embedding_dim=1024):
    # Receive the user as an input.
    demograph_input = layers.Input(name="demograph_input", shape=(demographic_size))
    icd_input = layers.Input(name="icd_input", shape=(vocabulary))
    # Get user embedding.
    icd_embedding = layers.Embedding(vocabulary+1, embedding_dim, mask_zero=True)(icd_input)
    icd_embedding_sum = keras.backend.sum(icd_embedding, axis=1)
    icd_fc = layers.Dense(256, activation='relu')(icd_embedding_sum)
    concat = layers.Concatenate(axis=1)([icd_fc, demograph_input])
    fc1 = layers.Dense(128, activation='relu')(concat)
    fc2 = layers.Dense(50, activation='relu')(fc1)
    predict = layers.Dense(1, activation='sigmoid')(fc2)
    model = Model(
        inputs=[demograph_input, icd_input], outputs=predict, name="embedding_model"
    )
    return model

def create_base_model(vocabulary, demographic_size, embedding_dim=1024):
    # Receive the user as an input.
    demograph_input = layers.Input(name="demograph_input", shape=(demographic_size))
    icd_input = layers.Input(name="icd_input", shape=(vocabulary))
    fc1 = layers.Dense(100, activation='relu')(demograph_input)
    predict = layers.Dense(1, activation='sigmoid')(fc1)
    model = Model(
        inputs=[demograph_input, icd_input], outputs=predict, name="embedding_model"
    )
    return model

def setup_embedding_data(df_train_embed, df_test_embed, X_train, y_train, X_test, y_test, batch_size):
    icd_train_list = [eval(x) for x in df_train_embed['icd_encoded_list'].copy()]
    icd_test_list = [eval(x) for x in df_test_embed['icd_encoded_list'].copy()]
    X_train = X_train.to_numpy(dtype=np.float32)
    y_train = y_train.to_numpy(dtype=np.float32)
    X_test = X_test.to_numpy(dtype=np.float32)
    y_test = y_test.to_numpy(dtype=np.float32)

    train_gen = EmbeddingDataGen(X_train, icd_train_list, y_train, batch_size=batch_size)
    test_gen = EmbeddingDataGen(X_test, icd_test_list, y_test, batch_size=batch_size)
    return train_gen, test_gen
