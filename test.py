import pickle
from keras.models import load_model, Model
import matplotlib.pyplot as plt
import keras
import os
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import cv2
import numpy as np

test_ids = pickle.load(open('test_ids.pickle', 'rb'))
test_vects = pickle.load(open('test_vects.pickle', 'rb'))

model = load_model('Model_1.h5')

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

base_model = InceptionV3()
base_model = Model(base_model.input, base_model.layers[-2].output)

# load ảnh theo id (là filename luôn)
def load_img(path):
    img = cv2.imread(path)
    return img

def embedding_img(img):
    img = cv2.resize(img, (299, 299))
    img = np.expand_dims(img, axis=0)  # thêm 1 chiều cho ảnh để dùng khi embedding ảnh thàn hvector
    img = preprocess_input(img)
    vector = base_model.predict(img)
    # reshape từ (1,2048) thành (2048, )
    vector = np.reshape(vector, vector.shape[1])
    return vector

def Predict_cap(img):
    vect = img.reshape((1, 2048))
    text = 'startseq'
    for i in range(max_length):
        seq = [word2i[w] for w in text.split() if w in word2i]
        seq = keras.preprocessing.sequence.pad_sequences([seq], maxlen=max_length, padding='pre')[0]
        seq = seq.reshape((1, max_length))
        next_word = model.predict([vect, seq])
        next_word = np.argmax(next_word)
        text = text + ' ' + i2word[next_word]
        if i2word[next_word] == 'endseq':
                break
    ans = text.split()
    ans = ans[1:-1]
    ans = ' '.join(ans)
    return ans

FOLDER = 'Test img'
NAME = 'img.jpg'
PATH = os.path.join(FOLDER, NAME)

def Predict_and_show(PATH):
    ID = test_ids[0]
    pre_run = Predict_cap(test_vects[ID])
    img = load_img(PATH)
    img_ar = embedding_img(img)
    cap = Predict_cap(img_ar)
    plt.imshow(img[:,:,::-1])
    plt.xlabel(cap)
    plt.yticks([])
    plt.xticks([])
    plt.show()
    
Predict_and_show(PATH)