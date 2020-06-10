# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
from keras.utils import np_utils, plot_model
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Embedding, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(filepath, input_shape=20):
    df = pd.read_csv(filepath)

    labels, vocabulary = list(df['label'].unique()), list(df['text'].unique())

    string = ''
    for word in vocabulary:
        string += word
        string += ' '

    vocabulary = set(string.split())
    print(vocabulary)

    word_dictionary = {word: i+1 for i, word in enumerate(vocabulary)}
    with open('word_dict.pk', 'wb') as f:
        pickle.dump(word_dictionary, f)
    inverse_word_dictionary = {i+1: word for i, word in enumerate(vocabulary)}
    label_dictionary = {label: i for i, label in enumerate(labels)}
    with open('label_dict.pk', 'wb') as f:
        pickle.dump(label_dictionary, f)
    output_dictionary = {i: labels for i, labels in enumerate(labels)}

    vocab_size = len(word_dictionary.keys())
    label_size = len(label_dictionary.keys())


    content = ''

    for kv in word_dictionary.items():
    # d.iteritems: an iterator over the (key, value) items
        content = content + str(kv)
    with open('./vocabulary', 'w') as f:
        f.write(content)
        f.close()
    for kv in label_dictionary.items():
    # d.iteritems: an iterator over the (key, value) items
        print(kv)

    x = []
    for sent in df['text']:
        print(sent)
        cur = []
        lineList = sent.split()
        print(lineList)
        for word in lineList:
            print(word)
            print(word_dictionary[word])
            cur.append(word_dictionary[word])
        x.append(cur)

    #x = [[word_dictionary[word] for word in sent.split()] for sent in df['text']]
    x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
    y = [[label_dictionary[sent]] for sent in df['label']]
    y = [np_utils.to_categorical(label, num_classes=label_size) for label in y]
    y = np.array([list(_[0]) for _ in y])

    return x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary

def load_test_data(filepath, input_shape=20):
    df = pd.read_csv(filepath)

    labels, vocabulary = list(df['id'].unique()), list(df['text'].unique())

    string = ''
    for word in vocabulary:
        string += word
        string += ' '

    vocabulary = set(string.split())

    word_dictionary = {word: i+1 for i, word in enumerate(vocabulary)}
    with open('word_test_dict.pk', 'wb') as f:
        pickle.dump(word_dictionary, f)
    inverse_word_dictionary = {i+1: word for i, word in enumerate(vocabulary)}
    label_dictionary = {label: i for i, label in enumerate(labels)}
    with open('label_test_dict.pk', 'wb') as f:
        pickle.dump(label_dictionary, f)
    output_dictionary = {i: labels for i, labels in enumerate(labels)}

    vocab_size = len(word_dictionary.keys())
    label_size = len(label_dictionary.keys())

    x = []
    for sent in df['text']:
        cur = []
        lineList = sent.split()
        for word in lineList:
            cur.append(word_dictionary[word])
        x.append(cur)
    x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
    y = [[label_dictionary[sent]] for sent in df['id']]
    y = [np_utils.to_categorical(label, num_classes=label_size) for label in y]
    y = np.array([list(_[0]) for _ in y])

    return x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary

# Embedding + LSTM + Softmax.
def create_LSTM(n_units, input_shape, output_dim, filepath):
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = load_data(filepath)
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=output_dim,
                        input_length=input_shape, mask_zero=True))
    model.add(LSTM(n_units, input_shape=(x.shape[0], x.shape[1])))
    model.add(Dropout(0.2))
    model.add(Dense(label_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #plot_model(model, to_file='./model_lstm.png', show_shapes=True)
    model.summary()

    return model

def model_train(input_shape, filepath, model_save_path):

    # input_shape = 100
    x, y, output_dictionary, vocab_size, label_size, inverse_word_dictionary = load_data(filepath, input_shape)
    #train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.1, random_state = 42)

    test_x, test_y, test_output_dictionary, test_vocab_size, test_label_size, test_inverse_word_dictionary = load_test_data('./test.csv', input_shape)
    train_x = x
    train_y = y

    n_units = 50
    batch_size = 16
    epochs = 10
    output_dim = 20

    lstm_model = create_LSTM(n_units, input_shape, output_dim, filepath)
    lstm_model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    lstm_model.save(model_save_path)

    N = test_x.shape[0]
    predict = []
    label = []
    for start, end in zip(range(0, N, 1), range(1, N+1, 1)):
        sentence = [inverse_word_dictionary[i] for i in test_x[start] if i != 0]
        y_predict = lstm_model.predict(test_x[start:end])
        label_predict = output_dictionary[np.argmax(y_predict[0])]
        #label_true = output_dictionary[np.argmax(test_y[start:end])]
        print(''.join(sentence), label_predict)
        predict.append(label_predict)
        #label.append(label_true)
    
    with open('result.csv', 'w') as f:
        f.write('id,label\n')
        i = 0
        for lp in predict:
            f.write(str(i)+','+lp+'\n')
            i=i+1
        f.close()

if __name__ == '__main__':
    filepath = './train.csv'
    input_shape = 80
    model_save_path = './result.h5'
    model_train(input_shape, filepath, model_save_path)