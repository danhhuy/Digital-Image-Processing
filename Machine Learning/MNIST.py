import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from keras.utils import np_utils
import cv2

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
    
# load du lieu MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_val, y_val = x_train[50000:60000,:], y_train[50000:60000] 
x_train, y_train = x_train[:50000,:], y_train[:50000] 

# reshape lai du lieu cho dung kich thuoc ma keras yeu cau
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# one hot encoding label (y)
y_train = np_utils.to_categorical(y_train, 10)
y_val = np_utils.to_categorical(y_val, 10)
y_test = np_utils.to_categorical(y_test, 10)
print('Dœ li»u y sau one-hot encoding ',y_train[0])

# dinh nghia Model
model = Sequential()
# Them convolutional layer voi 32 kernel, kich thuoc kernel 3*3
# Dung ham sigmoid lam activation va chi ro input_shape cho layer dau tien
model.add(Conv2D(32, (3,3), activation='sigmoid', input_shape=(28,28,1)))
# Them convolutional layer
model.add(Conv2D(32, (3,3), activation='sigmoid'))
# Them Max pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))
# Flatten layer chuyen tu tensor sang vector
model.add(Flatten())
# Them fully connected layer voi 128 nodes va dung ham sigmoid
model.add(Dense(128, activation='sigmoid'))
# Output layer voi 10 nodes va dung softmax function de chuyen sang xac suat
model.add(Dense(10, activation='softmax'))

# Compile model, chi ro ham loss_function nao duoc su dung, phuong thuc dung de toi uu ham loss function
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Thuc hien train model voi data
H = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=3, verbose=1)

# Ve do thi loss, accuracy cua training set va validation set
import matplotlib.pyplot as plt
fig = plt.figure()

numEpoch = 3
plt.plot(np.arange(0, numEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numEpoch), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, numEpoch), H.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()
plt.show()
cv2.waitKey(0)

# Danh gia model voi du lieu test set
score = model.evaluate(x_test, y_test, verbose=0)
print(score)

# Du doan anh
plt.imshow(x_test[0].reshape(28,28), cmap='gray')
plt.show()
cv2.waitKey(0)
y_predict = model.predict(x_test[0].reshape(1,28,28,1))
print(y_predict)
print('Gia tri du doan: ', np.argmax(y_predict))