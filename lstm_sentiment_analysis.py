import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

# Input data files are available in the "../input/" directory.
# cleaning input files
data = pd.read_csv('../input/Sentiment Analysis Dataset 2.csv', skiprows=[8835,535881])
data = data[['Sentiment', 'SentimentText']]

# extracting important data and preprocessing the text
data['SentimentText'] = data['SentimentText'].apply(lambda x: x.lower())
data['SentimentText'] = data['SentimentText'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['SentimentText'].values)
X = tokenizer.texts_to_sequences(data['SentimentText'].values)
X = pad_sequences(X)


embed_dim = 50
lstm_out = 80
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())


Y = pd.get_dummies(data['Sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


batch_size = 32

# training the model and giving the predict the accuracy over training data
model.fit(X_train, Y_train, nb_epoch = 1, batch_size=batch_size, verbose = 2)


validation_size = 1500
X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)

# testing data accuracy
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))



