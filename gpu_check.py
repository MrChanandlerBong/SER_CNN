import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPUs visible to TF:", tf.config.list_physical_devices("GPU"))
