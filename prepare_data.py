import os
import cv2
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np
import pickle
import tensorflow as tf

text_path = 'E:\Machine Learning\Deep Learning Basic\Image Captioning\Data\Flickr8k_text'
img_path = 'E:\Machine Learning\Deep Learning Basic\Image Captioning\Data\Flicker8k_Dataset'

# 1. Xử lý phần "chữ"
# đọc file txt
def load_doc(filename):
	path = os.path.join(text_path, filename)
	f = open(path, 'r')
	text = f.read()
	f.close()
	return text

# caption của các ảnh
CAPS_RAW = load_doc('Flickr8k.token.txt')

# tạo ra một dict có dạng (id ảnh : ['cap 1', 'cap 2',...])
def load_caps(docs):
	data = dict()
	for line in docs.split('\n'):
		cap = line.split()
		ID = cap[0].split('#')[0] # id ảnh chỉ tính ngang đuôi .jpg
		cap = ' '.join([word for word in cap[1:] if len(word)>=2]) # bỏ những kí tự có độ dài < 2 (ví dụ a, dấu câu)
		if ID not in data:
			data[ID] = []
		data[ID].append('startseq ' + cap.lower() + ' endseq')
		# ta thêm startseq và endseq vào đầu và cuối mỗi caption
	return data

CAPS_DATA = load_caps(CAPS_RAW)
print(CAPS_DATA['1000268201_693b08cb0e.jpg'])

# chia ra những ảnh train, val, test theo id
train_id_raw = load_doc('Flickr_8k.trainImages.txt').split('\n')
val_id_raw = load_doc('Flickr_8k.devImages.txt').split('\n')
test_id_raw = load_doc('Flickr_8k.testImages.txt').split('\n')

# để cho chắc chắn, ta check xem id này có nằm trong file caps không, rồi mới thêm vào train_id thật
# dùng thêm set luôn để đề phòng một id xuất hiện nhiều lần
train_ids = list(set([ID for ID in train_id_raw if ID in CAPS_DATA]))
val_ids = list(set([ID for ID in val_id_raw if ID in CAPS_DATA]))
test_ids = list(set([ID for ID in test_id_raw if ID in CAPS_DATA]))

# Tách những caps của các ảnh train và valid ra
def caps_kind(kind):
	data = dict()
	for ID in CAPS_DATA:
		if ID in kind:
			if ID not in data:
				data[ID] = []
			data[ID].extend(CAPS_DATA[ID])
	return data

train_caps = caps_kind(train_ids)
val_caps = caps_kind(val_ids)

# Tạo VOCAB, chỉ lấy những từ xuất hiện >= 5 lần
words = dict()
for ID in train_caps:
	for cap in train_caps[ID]:
		for w in cap.split():
			words[w] = words.get(w,0) + 1
VOCAB = [w for w in words if words[w] >= 5]
with open('VOCAB.pickle', 'wb') as f:
	pickle.dump(VOCAB, f)

vocab_size = len(VOCAB) + 1 # 1 kí tự dùng để padding
print(vocab_size)

# decode và encode chuỗi, dựa vào VOCAB
word2i, i2word = dict(), dict()
cnt = 1 
for word in VOCAB:
	word2i[word] = cnt
	i2word[cnt] = word
	cnt += 1

print(train_caps['2513260012_03d33305cf.jpg'])
#2. Xử lý phần ảnh
#sử dụng pretrained model để embedding ảnh thành vector, ta sẽ sử dụng vector này để đưa vào model chính
#bỏ đi layers cuối của inception V3 (layer cuối này là layer phân loại, ta không dùng chúng)
base_model = InceptionV3()
base_model = Model(base_model.input, base_model.layers[-2].output)

# load ảnh theo id (là filename luôn)
def load_img(filename):
	path = os.path.join(img_path, filename)
	# model inception v3 ta dự định sử dụng nhận đầu vào là ảnh 299 x 299 x 3
	img_ar = cv2.imread(path)
	img_ar = cv2.resize(img_ar, (299, 299))
	img_ar = np.expand_dims(img_ar, axis=0) # thêm 1 chiều cho ảnh để dùng khi embedding ảnh thàn hvector
	return preprocess_input(img_ar) # preprocess_input sẽ giúp ta chuẩn hóa ảnh về dạng chuẩn của inception_v3

def embedding_img(filename):
	img = load_img(filename)
	vector = base_model.predict(img)
	# reshape từ (1,2048) thành (2048, )
	vector = np.reshape(vector, vector.shape[1])
	return vector

def dict_vector(id_file):
	data = dict()
	for ID in id_file:
		data[ID] = embedding_img(ID)
	return data

# tạo dict chứa id ảnh cùng với embedding vector của nó
train_vects = dict_vector(train_ids)
val_vects = dict_vector(val_ids)
test_vects = dict_vector(test_ids)

with open('test_ids.pickle', 'wb') as f:
	pickle.dump(test_ids, f)
with open('test_vects.pickle', 'wb') as f:
	pickle.dump(test_vects, f)


# 3. Tạo input, output cho tập train và validation
# Lưu ý:
# Input: vector embedding của ảnh và 1 số từ
# Ouput: từ tiếp theo
# Ví dụ: ảnh + A -> girl
# 		 ảnh + A girl -> walk
# ...

def data_generator(caps, vect, max_length=34):
	INPUT, OUTPUT = [], []
	for ID in caps:
		for cap in caps[ID]:
			encoded_cap = [word2i[w] for w in cap.split() if w in word2i]
			for i in range(1, len(encoded_cap)): # đi từ 1 ví từ đầu tiên của mỗi cap là startseq
				in_seq = encoded_cap[:i]
				out_seq = encoded_cap[i]
				# padding in_seq cho dài bằng maxlen
				in_seq = tf.keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=max_length, padding='pre')[0]
				# one-hot encoding từ out_seq
				out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
				INPUT.append([vect[ID], in_seq])
				OUTPUT.append(out_seq)
	return np.asarray(INPUT), np.asarray(OUTPUT)

train_data, train_label = data_generator(train_caps, train_vects)
val_data, val_label = data_generator(val_caps, val_vects)

with open('train_data.pickle', 'wb') as f:
	pickle.dump(train_data, f)
with open('train_label.pickle', 'wb') as f:
	pickle.dump(train_label, f)
with open('val_data.pickle', 'wb') as f:
	pickle.dump(val_data, f)
with open('val_label.pickle', 'wb') as f:
	pickle.dump(val_label, f)
