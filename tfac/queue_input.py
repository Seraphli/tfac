import threading
import tensorflow as tf


class EnqueueThread(threading.Thread):
    def __init__(self, inputs, size):
        super(EnqueueThread, self).__init__()
        self.daemon = True
        self.inputs = inputs
        self.size = size
        self.queue = tf.FIFOQueue(
            self.size, [i.dtype for i in self.inputs], self._get_shape())
        self.should_stop = False

    def _get_shape(self):
        shapes = []
        for i in self.inputs:
            shape = i.get_shape()
            shape = tf.TensorShape(shape.dims[1:])
            shapes.append(shape)
        return shapes

    def prepare(self, sess, sample_fn):
        self.sess = sess
        self.sample_fn = sample_fn
        self._op = self.queue.enqueue_many(self.inputs)

    def dequeue_many_op(self, n):
        sample = self.queue.dequeue_many(n)
        for s, i in zip(sample, self.inputs):
            s.set_shape(i.get_shape())
        return sample

    def run(self):
        while not self.should_stop:
            data = self.sample_fn()
            feed_dict = dict(zip(self.inputs, data))
            self.sess.run(self._op, feed_dict=feed_dict)


class QueueInput(object):
    def __init__(self, features, labels, queue_size=50):
        self.features = features
        self.labels = labels
        self.queue_size = queue_size
        self.inputs = list(self.features.values())
        self.split_idx = len(self.inputs)
        self.inputs.extend(list(self.labels.values()))
        self._threads = []
        self._ops = []

    def build_op(self, batch_size):
        thread = EnqueueThread(self.inputs, self.queue_size)
        self._threads.append(thread)
        dequeue = thread.dequeue_many_op(batch_size)
        _features = dict(zip(self.features.keys(), dequeue[:self.split_idx]))
        _labels = dict(zip(self.labels.keys(), dequeue[self.split_idx:]))
        idx = len(self._ops)
        self._ops.append((_features, _labels))
        return idx, _features, _labels

    def get_op(self, idx):
        return self._ops[idx]

    def run(self, sess, sample_fn):
        assert len(sample_fn) == len(self._threads)
        for idx, t in enumerate(self._threads):
            t.prepare(sess, sample_fn[idx])
            t.start()

    def stop(self):
        for t in self._threads:
            t.should_stop = True
