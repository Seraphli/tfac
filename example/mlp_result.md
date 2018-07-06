on TensorFlow 1.4.1

```
2018-04-23 15:59:41.114081: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
running original
Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz
2018-04-23 15:59:41.290164: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties:
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.8225
pciBusID: 0000:03:00.0
totalMemory: 7.92GiB freeMemory: 7.04GiB
2018-04-23 15:59:41.290186: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:03:00.0, compute capability: 6.1)
Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz
2018-04-23 15:59:41.720123: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:03:00.0, compute capability: 6.1)
runtime: 51.436022996902466 sec


running tfac
Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz
2018-04-23 16:00:43.531815: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:03:00.0, compute capability: 6.1)
runtime: 43.995368242263794 sec
```

on TensorFlow 1.8
```
2018-07-06 15:43:41.569585: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-07-06 15:43:41.696757: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1356] Found device 0 with properties:
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.8225
pciBusID: 0000:03:00.0
totalMemory: 7.92GiB freeMemory: 6.87GiB
2018-07-06 15:43:41.696779: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-07-06 15:43:41.912794: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-07-06 15:43:41.912820: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0
2018-07-06 15:43:41.912825: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N
2018-07-06 15:43:41.913009: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6627 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:03:00.0, compute capability: 6.1)
WARNING:tensorflow:From /home/seraphli/Workspace/Github/Seraphli/tfac/example/mlp.py:162: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
running original
WARNING:tensorflow:From /home/seraphli/Workspace/envs/py365/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
Instructions for updating:
Please write your own downloading logic.
WARNING:tensorflow:From /home/seraphli/Workspace/envs/py365/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.
Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz
WARNING:tensorflow:From /home/seraphli/Workspace/envs/py365/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use tf.data to implement this functionality.
Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz
WARNING:tensorflow:From /home/seraphli/Workspace/envs/py365/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.
Instructions for updating:
Please use alternatives such as official/mnist/dataset.py from tensorflow/models.
Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz
2018-07-06 15:43:42.510521: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-07-06 15:43:42.510552: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-07-06 15:43:42.510558: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0
2018-07-06 15:43:42.510561: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N
2018-07-06 15:43:42.510700: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6627 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:03:00.0, compute capability: 6.1)
runtime: 86.23988580703735 sec


running tfac
Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz
2018-07-06 15:45:19.190923: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1435] Adding visible gpu devices: 0
2018-07-06 15:45:19.190963: I tensorflow/core/common_runtime/gpu/gpu_device.cc:923] Device interconnect StreamExecutor with strength 1 edge matrix:
2018-07-06 15:45:19.190968: I tensorflow/core/common_runtime/gpu/gpu_device.cc:929]      0
2018-07-06 15:45:19.190972: I tensorflow/core/common_runtime/gpu/gpu_device.cc:942] 0:   N
2018-07-06 15:45:19.191136: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1053] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6627 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:03:00.0, compute capability: 6.1)
runtime: 66.41560959815979 sec
```