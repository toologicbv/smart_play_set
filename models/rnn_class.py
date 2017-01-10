from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from preprocessing.process_data import get_data
import numpy as np

train_data_f, train_labels_f, train_game_f, dta_dict_f = get_data('20161206', force=False,
                                                                  apply_window_func=False, calc_mag=True,
                                                                  extra_label="20hz_1axis_low8hz_330_12_1",
                                                                  optimal_w_size=False,
                                                                  f_type='low', lowcut=8, b_order=5)

X_train = train_data_f
# for cross validation train labels has to have 1 dim tensor
Y_train = np.squeeze(train_labels_f)

model = Sequential()
# model.add(Embedding(max_features, 256, input_length=maxlen))
model.add(Dense(12, input_shape=(12, 1)))
model.add(LSTM(output_dim=12, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=16, nb_epoch=10)
score = model.evaluate(X_train, Y_train, batch_size=16)
