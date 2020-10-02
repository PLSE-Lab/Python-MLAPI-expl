# 1. Turn on to print which unit (CPU/GPU) is used for an operation?
import tensorflow as tf
tf.debugging.set_log_device_placement(True)
 

# 2. Check how many GPUs are usable?
import tensorflow as tf
tf.debugging.set_log_device_placement(True)

print('Available GPUs: ', tf.test.is_gpu_available())
print(tf.config.experimental.list_logical_devices('GPU'))

mat1, mat2 = tf.random.uniform((1, 3)), tf.random.uniform((3, 3))
print(tf.matmul(mat1, mat2))
 

# 3-1. Choose which unit is used to use for the operation.
import tensorflow as tf
tf.debugging.set_log_device_placement(True)

with tf.device('/CPU:0'):
    mat1, mat2 = tf.random.uniform((1, 3)), tf.random.uniform((3, 3))
    print(tf.matmul(mat1, mat2))

with tf.device('/GPU:0'):
    mat1, mat2 = tf.random.uniform((1, 3)), tf.random.uniform((3, 3))
    print(tf.matmul(mat1, mat2))

with tf.device('/GPU:1'):
    mat1, mat2 = tf.random.uniform((1, 3)), tf.random.uniform((3, 3))
    print(tf.matmul(mat1, mat2))

with tf.device('/GPU:2'):
    mat1, mat2 = tf.random.uniform((1, 3)), tf.random.uniform((3, 3))
    print(tf.matmul(mat1, mat2))
 

# 3-2. Set which unit is used to use for the operation forcibly.
import tensorflow as tf
idx_gpu_used = 2
list_gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(list_gpu[idx_gpu_used], 'GPU')

tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.list_logical_devices(device_type='GPU')
 

# 4. Set memory growth is working, which means that GPU memory won't be occupied wholly at initialization.
import tensorflow as tf
tf.debugging.set_log_device_placement(True)
list_gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in list_gpu:
    tf.config.experimental.set_memory_growth(gpu, True)
 

# 5. Allow the system to choose the available GPU to use
import tensorflow as tf
tf.config.set_soft_device_placement(True)
mat1, mat2 = tf.random.uniform((1, 3)), tf.random.uniform((3, 3))
print(tf.matmul(mat1, mat2))
 

# 6. Parallel computing via multi-GPUs
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
  inputs = tf.keras.layers.Input(shape=(64,))
  predictions = tf.keras.layers.Dense(64)(inputs)
  model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
  model.compile(optimizer='adam', loss='mse')