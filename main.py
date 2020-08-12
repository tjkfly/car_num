import tensorflow as tf
import os
import matplotlib.pyplot as pl
import pathlib
import cv2 as cv
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

data_path = 'E:\\code\\tf\\proj\\car_num\\data'
# # data_root = pathlib.Path(data_path)
# # print(data_root)
# # for item in data_root.iterdir():
# #     print(item)
# character_folders = os.listdir(data_path)
# for character_folder in character_folders:
#     character_imgs = os.path.join(data_path, character_folder)
#     print(character_imgs)
character_folders = os.listdir(data_path)
label = 0
LABEL_temp = {}
if(os.path.exists('./train_data.list')):
    os.remove('./train_data.list')
if(os.path.exists('./test_data.list')):
    os.remove('./test_data.list')
for character_folder in character_folders:
    with open('./train_data.list', 'a') as f_train:
        with open('./test_data.list', 'a') as f_test:
            if character_folder == '.DS_Store' or character_folder == '.ipynb_checkpoints' or character_folder == 'data23617':
                continue
            print(character_folder + " " + str(label))
            LABEL_temp[str(label)] = character_folder     #存储一下标签的对应关系
            character_imgs = os.listdir(os.path.join(data_path, character_folder))
            for i in range(len(character_imgs)):
                if i%10 == 0:
                    f_test.write(os.path.join(os.path.join(data_path, character_folder), character_imgs[i]) + "\t" + str(label) + '\n')
                else:
                    f_train.write(os.path.join(os.path.join(data_path, character_folder), character_imgs[i]) + "\t" + str(label) + '\n')
    label = label + 1
print('图像列表已生成')

all_image_paths = []
all_image_labels = []
test_image_paths = []
test_image_labels = []
with open('./train_data.list', 'r') as f:
    lines = f.readlines()
    for line in lines:
        img, label = line.split('\t')
        all_image_paths.append(img)
        all_image_labels.append(int(label))
with open('./test_data.list', 'r') as f:
    lines = f.readlines()
    for line in lines:
        img, label = line.split('\t')
        test_image_paths.append(img)
        test_image_labels.append(int(label))
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image,dtype=tf.float32)
    image = tf.image.resize(image, [20, 20])
    image /= 255.0  # normalize to [0,1] range
    # image = tf.reshape(image,[100*100*3])
    return image

def load_and_preprocess_image(path,label):
    image = tf.io.read_file(path)
    return preprocess_image(image), label


ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
train_data = ds.map(load_and_preprocess_image).batch(64)
db = tf.data.Dataset.from_tensor_slices((test_image_paths, test_image_labels))
test_data = db.map(load_and_preprocess_image).batch(64)

# imgs, lables = next(iter(train_data))
# print(imgs.shape)
# img, lable = imgs[0], lables[0]
# cv.imshow("1",img.numpy())
# cv.waitKey(0)
# pl.imshow(img)
# pl.show()

def train_model(train_data,test_data):
    #构建模型
    network = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        keras.layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
        keras.layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        keras.layers.BatchNormalization(),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(65)])
    network.build(input_shape=(None, 20, 20, 3))
    network.summary()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    network.compile(optimizer=optimizers.SGD(lr=0.001),
                    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    # network.fit(train_data, epochs=100, validation_data=test_data, callbacks=[reduce_lr])

    network.evaluate(test_data)
    # tf.saved_model.save(network, 'E:\\code\\tf\proj\\car_num\\model\\')
train_model(train_data,test_data)






















