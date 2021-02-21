import pickle
import os
import numpy as np
from keras.models import Model
from keras.layers import add, Dropout, Embedding, LSTM, Dense
from keras import Input

train_data = pickle.load(open('train_data.pickle', 'rb'))
train_label = pickle.load(open('train_label.pickle', 'rb'))
val_data = pickle.load(open('val_data.pickle', 'rb'))
val_label = pickle.load(open('val_label.pickle', 'rb'))

train_img, train_cap = [], []
for x, y in train_data:
	train_img.append(x)
	train_cap.append(y)
train_img = np.asarray(train_img)
train_cap = np.asarray(train_cap)

val_img, val_cap = [], []
for x, y in val_data:
	val_img.append(x)
	val_cap.append(y)
val_img = np.asarray(val_img)
val_cap = np.asarray(val_cap)

print(val_img.shape)
print(val_label.shape)

VOCAB = pickle.load(open('VOCAB.pickle', 'rb'))
vocab_size = len(VOCAB) + 1
max_length = 34

# encode và decode
word2i, i2word = dict(), dict()
cnt = 1
for word in VOCAB:
	word2i[word] = cnt
	i2word[cnt] = word
	cnt += 1

# sử dụng pretrained model Glovel để embedding chữ
glovel_path = 'E:\Machine Learning\Deep Learning Basic\Glovel model\glove.6B.200d.txt'

embedding_word = dict()
embedding_dim = 200 # mỗi từ là một vector (200, )

with open(glovel_path, encoding='utf-8') as f:
	for line in f:
		line = line.split()
		word = line[0]
		vect = np.asarray(line[1:], dtype='float32')
		embedding_word[word] = vect

# tạo embedding vector cho từng từ trong vocab
embedding_vocab = np.zeros((vocab_size, embedding_dim))

for word in VOCAB:
	if word in embedding_word:
		embedding_vocab[word2i[word]] = embedding_word[word]
		# nếu từ này không có trong embedding_word của ta thì xem như nó là 1 vector (0,0,...,0)

# Tạo model
input_1 = Input(shape=(2048,))
Img1 = Dropout(0.25)(input_1)
Img2 = Dense(256, activation='relu')(Img1)
input_2 = Input(shape=(max_length,))
Text1 = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(input_2) # mask_zero=True nghĩa là padding bằng cách thêm số 0
Text2 = Dropout(0.25)(Text1)
Text3 = LSTM(units=256)(Text2)
Com1 = add([Img2, Text3])
Com2 = Dense(256, activation='relu')(Com1)
output = Dense(vocab_size, activation='softmax')(Com2)

model = Model(inputs=[input_1, input_2], outputs=output)

# model.summary()

# layer 2 của model sẽ sử dụng Glovel ở trên
model.layers[2].set_weights([embedding_vocab])
model.layers[2].trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit([train_img, train_cap], train_label, epochs=10, batch_size=128, validation_data=([val_img, val_cap], val_label))

model.save('Model_1.h5')