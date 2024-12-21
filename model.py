import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, LSTM, Reshape, Dense, RepeatVector, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = open("shakespeare.txt", "r").read().lower()

# unique characters
chars = sorted(set(data)) + [' ']

char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}

num_hidden_units = 70
reshaper = Reshape((1, len(chars)))
reshaper_hidden = Reshape((1, num_hidden_units))
lstm_cell_1 = LSTM(num_hidden_units, return_state=True,name="LSTM1")
lstm_cell_2 = LSTM(num_hidden_units, return_state=True, name="LSTM2")
densor = Dense(len(chars), activation='softmax')
# length of input sequence
Tx = 15

def get_model(Tx, num_hidden_units):
    """
    Return the model specified by sequence length of 'Tx' and number of hidden units for both layers of the LSTM
    """
    X = Input(shape=(Tx, len(chars)))
    layer_1_a0 = Input(shape=(num_hidden_units,), name="layer_1_a0")
    layer_2_a0 = Input(shape=(num_hidden_units,), name="layer_2_a0")
    layer_1_c0 = Input(shape=(num_hidden_units,), name="layer_1_c0")
    layer_2_c0 = Input(shape=(num_hidden_units,), name="layer_2_c0")

    layer_1_a = layer_1_a0
    layer_2_a = layer_2_a0
    layer_1_c = layer_1_c0
    layer_2_c = layer_2_c0

    for t in range(Tx):
        x = X[:, t, :]
        x = reshaper(x)
        output1, layer_1_a, layer_1_c = lstm_cell_1(x, initial_state=[layer_1_a, layer_1_c])

        layer_1_a_reshaped = reshaper_hidden(layer_1_a)

        _, layer_2_a, layer_2_c = lstm_cell_2(layer_1_a_reshaped, initial_state=[layer_2_a, layer_2_c])

    out = densor(layer_2_a)

    return Model(inputs= [X, layer_1_a0, layer_1_c0 , layer_2_a0, layer_2_c0], outputs=out)

model = get_model(Tx, num_hidden_units)

opt = Adam(learning_rate=0.01 ,beta_1=0.9, beta_2=0.999)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'] * Tx)

def pre_process(Tx, the_data):
    """
    preprocess data of text into chunks of "Tx" characters
    return:
        X: Shape(m, Tx, len(chars)) , m is the number of samples of length 'Tx'
        Y: Shape(m, len(chars)) , contains the ending character of each sample m

    len(chars) is the number of unique characters in the document
    X and Y are one hot encoded
    """
    x = []
    y = []
    for i in range(0, len(the_data) - Tx, 3):
        x.append(the_data[i: i + Tx])
        y.append(the_data[i + Tx])

    m = len(x)
    X = np.zeros((m, Tx, len(chars)))
    Y = np.zeros((m, len(chars)))
    # one hot encoding
    for i, sentence in enumerate(x):
      for t, char in enumerate(sentence):
        X[i, t, char_to_index[char]] = 1

      Y[i, char_to_index[y[i]]] = 1

    return X, Y

X, Y = pre_process(Tx, data)
m = X.shape[0]
layer_1_a0 = np.zeros((m, num_hidden_units))
layer_2_a0 = np.zeros((m, num_hidden_units))
layer_1_c0 = np.zeros((m, num_hidden_units))
layer_2_c0 = np.zeros((m, num_hidden_units))

# with tf.device('/GPU:0'):
  # history = model.fit([X, layer_1_a0, layer_1_c0, layer_2_a0, layer_2_c0], Y, epochs=50, verbose =1)

loaded_model = tf.keras.models.load_model("model.keras")

lstm_cell_1 = loaded_model.get_layer(name="LSTM1")
lstm_cell_2 = loaded_model.get_layer(name="LSTM2")
densor = loaded_model.get_layer(name="dense")

# print(f"loss at epoch 1: {history.history['loss'][0]}")
# print(f"loss at epoch 50: {history.history['loss'][49]}")

# model.save("model2.keras")

def generate_seq_model(input_Tx, Ty):
    """
    return a model that generates a sequence of length 'Ty' character by character with a fixed context window of size 'input_Tx'
    starting with a sentence of length 'my_Tx'

    Todo: Implementing a context size of 'Tx' (which the model was trained/optimized for) instead of 'input_Tx' would ensure optimal performance
      - if input_Tx << 15 , pad the input and apply masking (so that the padding doesn't influence the predictions) until the input with the generated letters >= 'Tx',
        then use the most recent 'Tx' characters
      - if input_Tx >> 15, use the recent 'Tx' characters and ignore the rest

    """
    input_X = Input(shape=(input_Tx, len(chars)))
    layer_1_a0 = Input(shape=(num_hidden_units,), name="layer_1_a0")
    layer_2_a0 = Input(shape=(num_hidden_units,), name="layer_2_a0")
    layer_1_c0 = Input(shape=(num_hidden_units,), name="layer_1_c0")
    layer_2_c0 = Input(shape=(num_hidden_units,), name="layer_2_c0")

    X = input_X
    layer_1_a = layer_1_a0
    layer_2_a = layer_2_a0
    layer_1_c = layer_1_c0
    layer_2_c = layer_2_c0

    # return one hot encoding of the new letter specified by the argmax of the output
    onehot_argmax_layer = Lambda(lambda x: tf.one_hot(tf.math.argmax(x, axis=1), depth=len(chars)))
    # slide the context window by 1
    newX_layer = Lambda(lambda inputs: tf.concat([inputs[0][:, 1:, :], tf.expand_dims(inputs[1], axis=1)], axis=1), output_shape = (input_Tx, len(chars)), mask=None)

    outputs = []

    for t in range(Ty):
        for t2 in range(input_Tx):
            x = X[:, t2, :]
            x = reshaper(x)

            _, layer_1_a, layer_1_c = lstm_cell_1(x, initial_state=[layer_1_a, layer_1_c])

            layer_1_a_reshaped = reshaper_hidden(layer_1_a)

            _, layer_2_a, layer_2_c = lstm_cell_2(layer_1_a_reshaped, initial_state=[layer_2_a, layer_2_c])

        out = densor(layer_2_a)

        outputs.append(out)

        new_letter = onehot_argmax_layer(out)

        X = newX_layer([X, new_letter])

    return Model(inputs=[input_X, layer_1_a0, layer_1_c0, layer_2_a0, layer_2_c0], outputs=outputs)


def predict_and_sample(Ty, input_sentence):
    """
    return indices of the generated sequence of length 'Ty'
    starting with 'input_sentence'
    """

    inference_model = generate_seq_model(len(input_sentence), Ty)

    x_initializer = np.zeros((1, len(input_sentence), len(chars)))
    # one hot encode the input
    for i, char in enumerate(input_sentence):
        x_initializer[0, i, char_to_index[char.lower()]] = 1

    layer_1_a_initializer = np.zeros((1, num_hidden_units))
    layer_2_a_initializer = np.zeros((1, num_hidden_units))
    layer_1_c_initializer = np.zeros((1, num_hidden_units))
    layer_2_c_initializer = np.zeros((1, num_hidden_units))

    preds = inference_model.predict([x_initializer, layer_1_a_initializer, layer_1_c_initializer,layer_2_a_initializer, layer_2_c_initializer], verbose=1)

    indices = np.argmax(np.array(preds), axis=2)

    return indices


sentence = "From fairest creatures "
indices = predict_and_sample(100, sentence)

print(sentence, end="")
for index in indices:
  print(index_to_char[index[0]], end="")

output = "From fairest creatures the sty be blood and love of thee thy self are painting all many, will thee thy silge the world thee"

