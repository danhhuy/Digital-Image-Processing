# Them thu vien
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from imutils import paths
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
import numpy as np
import random
import os

# Lay cac duong dan den anh
image_path = list(paths.list_images('dataset/'))

# Doi ngau nhien vi tri cac duong dan anh
random.shuffle(image_path)

# Duong dan anh se la dataset/ten loai hoa/ten anh dataset/Bluebell/image_0241.jpg n
labels = [p.split(os.path.sep)[-2] for p in image_path]

# Chuyen ten cac loai hoc thanh so
le = LabelEncoder()
labels = le.fit_transform(labels)

# One-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Load anh va resize ve dung kich thuoc ma VGG 16 can là (224,224)
list_image = []
for (j, imagePath) in enumerate(image_path):
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, 0)
    image = imagenet_utils.preprocess_input(image)
    list_image.append(image)
list_image = np.vstack(list_image)

# Load model VGG 16 cua ImageNet dataset, include_top=False đe bỏ phan Fully connected lay
baseModel = VGG16(weights='imagenet', include_top=False,input_tensor=Input(shape=(224, 224, 3)))

# Xay them cac layer
# Lay output cıa ConvNet trong VGG16
fcHead = baseModel.output

# Flatten trước khi dùng FCs
fcHead = Flatten(name='flatten')(fcHead)

# Them FC
fcHead = Dense(256, activation='relu')(fcHead)
fcHead = Dropout(0.5)(fcHead)

# Output layer với softmax activation
fcHead = Dense(17, activation='softmax')(fcHead)

# Xay dựng model bang viec noi ConvNet cıa VGG16 và fcHead
model = Model(inputs=baseModel.input, outputs=fcHead)

# Chia traing set, test set ti le 80-20
X_train, X_test, y_train, y_test = train_test_split(list_image, labels, \
test_size=0.2, random_state=42)

# augmentation cho training data
aug_train = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# augementation cho test
aug_test= ImageDataGenerator(rescale=1./255)

# freeze VGG model
for layer in baseModel.layers:
    layer.trainable = False
opt = RMSprop(0.001)
model.compile(opt, 'categorical_crossentropy', ['accuracy'])
numOfEpoch = 25
H = model.fit_generator(aug_train.flow(X_train, y_train, batch_size=32), steps_per_epoch=len(X_train)//32, validation_data=(aug_test.flow(X_test, y_test, batch_size=32)), validation_steps=len(X_test)//32, epochs=numOfEpoch)

# unfreeze some last CNN layer:
for layer in baseModel.layers[15:]:
    layer.trainable = True

numOfEpoch = 35
opt = SGD(0.001)
model.compile(opt, 'categorical_crossentropy', ['accuracy'])
H = model.fit_generator(aug_train.flow(X_train, y_train, batch_size=32),steps_per_epoch=len(X_train)//32,validation_data=(aug_test.flow(X_test, y_test, batch_size=32)),validation_steps=len(X_test)//32,epochs=numOfEpoch)