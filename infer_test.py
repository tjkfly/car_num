import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import os
import cv2
import pathlib
import random
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

label_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
              'C', '川',
              'D', 'E', '鄂', 'F',
              'G', '赣', '甘', '贵', '桂',
              'H', '黑', '沪',
              'J', '冀', '津', '京', '吉',
              'K', 'L', '辽', '鲁', 'M', '蒙', '闽',
              'N', '宁',
              'P', 'Q', '青', '琼',
              'R', 'S', '陕', '苏', '晋',
              'T', 'U', 'V', 'W', '皖',
              'X', '湘', '新',
              'Y', '豫', '渝', '粤', '云',
              'Z', '藏', '浙']

print(len(label_list))

DEFAULT_FUNCTION_KEY = "serving_default"
loaded = tf.saved_model.load('.\\model\\')

network = loaded.signatures[DEFAULT_FUNCTION_KEY]
print(list(loaded.signatures.keys()))

print('加载 weights 成功')

# license_plate = cv2.imread('E:\\code\\tf\\proj\\car_num\\yuA.png')
license_plate = cv2.imread('E:\\code\\tf\\proj\\car_num\\num_plate.png')
gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_RGBA2GRAY)
ret, binary_plate = cv2.threshold(gray_plate, 175, 255, cv2.THRESH_BINARY)
cv2.imshow('1', binary_plate)
print(binary_plate.shape)
cv2.waitKey(0)
result = []
for col in range(binary_plate.shape[1]):
    result.append(0)
    for row in range(binary_plate.shape[0]):
        result[col] = result[col] + binary_plate[row][col] / 255  # 每一列的像素累加
character_dict = {}
num = 0
i = 0
while i < len(result):
    if result[i] == 0:
        i += 1
    else:
        index = i + 1
        while result[index] != 0:
            index += 1
        character_dict[num] = [i, index - 1]
        num += 1
        i = index
for i in range(8):
    if i == 2:
        continue
    padding = (170 - (character_dict[i][1] - character_dict[i][0])) / 2
    # 让图片W:H = 1 : 1
    ndarray = np.pad(binary_plate[:, character_dict[i][0]:character_dict[i][1]], ((0, 0), (int(padding), int(padding))),
                     'constant', constant_values=(0, 0))
    # ndarray = binary_plate[:, character_dict[i][0]:character_dict[i][1]]
    print(ndarray.shape)
    ndarray = cv2.resize(ndarray, (20, 20))
    cv2.imwrite('./' + str(i) + '.png', ndarray)

path = 'E:\\code\\tf\\proj\\car_num\\0.png'

test_image = plt.imread(path)
image = tf.io.read_file(path)
plt.imshow(test_image)

plt.show()
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [20, 20])
image1 = image / 255.0  # normalize to [0,1] range
print(image1.shape)
image1 = tf.expand_dims(image1, axis=0)

print(image1.shape)

print(image1.shape)

pred = network(image1)
pred = tf.nn.softmax(pred['output_1'], axis=1)
print("预测softmax后", pred)

pred = tf.argmax(pred, axis=1)
print("最终测试结果", pred.numpy())

print("预测结果原始结果", label_list[int(pred)])
