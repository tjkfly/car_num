import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import matplotlib.pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
print('tf版本：',tf.__version__)



DEFAULT_FUNCTION_KEY = "serving_default"
loaded = tf.saved_model.load('.\\model\\')

network = loaded.signatures[DEFAULT_FUNCTION_KEY]
print(list(loaded.signatures.keys()))

print('加载 weights 成功')


# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: network(x))
full_model = full_model.get_concrete_function(
tf.TensorSpec(network.inputs[0].shape, network.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
		logdir="./frozen_models",
		name="frozen_graph.pb",
		as_text=False)