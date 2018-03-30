```
running original
Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz
2018-03-30 19:00:49.774903: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2018-03-30 19:00:50.008648: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties:
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.8225
pciBusID: 0000:03:00.0
totalMemory: 7.92GiB freeMemory: 6.73GiB
2018-03-30 19:00:50.008673: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:03:00.0, compute capability: 6.1)
runtime: 78.95094037055969 sec


running tfac
Extracting /tmp/tensorflow/mnist/input_data/train-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/train-labels-idx1-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-images-idx3-ubyte.gz
Extracting /tmp/tensorflow/mnist/input_data/t10k-labels-idx1-ubyte.gz
2018-03-30 19:02:18.850375: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080, pci bus id: 0000:03:00.0, compute capability: 6.1)
2018-03-30 19:03:19.194830: W tensorflow/core/kernels/queue_base.cc:295] _0_fifo_queue: Skipping cancelled enqueue attempt with queue not closed
2018-03-30 19:03:19.194909: W tensorflow/core/kernels/queue_base.cc:295] _1_fifo_queue_1: Skipping cancelled enqueue attempt with queue not closed
runtime: 60.876787185668945 sec
```