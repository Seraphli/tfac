import threading
import tensorflow as tf


class EnqueueThread(threading.Thread):
    def __init__(self, inputs, capacity, queue_class):
        """Thread for enqueue and dequeue 
        
        Args:
            inputs (list): List of placeholder
            capacity (int): Size of queue 
            queue_class : Function for initialize queue
        """
        super(EnqueueThread, self).__init__()
        self.daemon = True
        self.inputs = inputs
        self.capacity = capacity
        self.queue = queue_class(
            capacity=self.capacity,
            dtypes=[i.dtype for i in self.inputs],
            shapes=self._get_shape()
        )
        self.should_stop = False
        self.close_op = self.queue.close(True)
        self.is_close_op = self.queue.is_closed()

    def _get_shape(self):
        shapes = []
        for i in self.inputs:
            shape = i.get_shape()
            shape = tf.TensorShape(shape.dims[1:])
            shapes.append(shape)
        return shapes

    def prepare(self, sess, sample_fn):
        """Prepare for run
        
        Args:
            sess: Tensorflow session 
            sample_fn: Function to draw one sample from data
        """
        self.sess = sess
        self.sample_fn = sample_fn
        self._op = self.queue.enqueue_many(self.inputs)

    def dequeue_many_op(self, n):
        """Get tensorflow dequeue op
        
        Args:
            n (int): Size to dequeue 

        Returns:
            Dequeue op
            
        """
        sample = self.queue.dequeue_many(n)
        for s, i in zip(sample, self.inputs):
            s.set_shape(i.get_shape())
        return sample

    def run(self):
        while not self.should_stop:
            try:
                data = self.sample_fn()
                feed_dict = dict(zip(self.inputs, data))
                self.sess.run(self._op, feed_dict=feed_dict)
            except tf.errors.CancelledError:
                # Avoid exception output when run close_op
                pass
            except Exception as e:
                print(e)

    def close_queue(self):
        self.sess.run(self.close_op)


class QueueInput(object):
    def __init__(self, features, labels, queue_size,
                 queue_class=tf.FIFOQueue):
        """Initialize based on features and labels
        
        Args:
            features (dict): Dictionary represent features 
            labels (dict): Dictionary represent labels
            queue_size (int): Size of queue 
            queue_class: Class or function to initialize queue
        """
        self.features = features
        self.labels = labels
        self.queue_size = queue_size
        self.queue_class = queue_class
        self.inputs = list(self.features.values())
        self.split_idx = len(self.inputs)
        self.inputs.extend(list(self.labels.values()))
        self._threads = []
        self._ops = []
        self.idx = 0

    def build_op(self, batch_size):
        """Build tensorflow op based on batch_size
        
        Args:
            batch_size (int): batch size

        Returns:
            index of op, features op, labels op
            
        """
        thread = EnqueueThread(self.inputs, self.queue_size[self.idx],
                               self.queue_class)
        self._threads.append(thread)
        dequeue = thread.dequeue_many_op(batch_size)
        _features = dict(zip(self.features.keys(), dequeue[:self.split_idx]))
        _labels = dict(zip(self.labels.keys(), dequeue[self.split_idx:]))
        self.idx = len(self._ops)
        self._ops.append((_features, _labels))
        return self.idx, _features, _labels

    def get_op(self, idx):
        """Get tensorflow op according to index
        
        Args:
            idx (int): index of op

        Returns:
            tensorflow op

        """
        return self._ops[idx]

    def run(self, sess, sample_fn):
        """Run queue with provided session
        
        Args:
            sess: Tensorflow session
            sample_fn (list): List of sample function for every op 

        Returns:

        """
        assert len(sample_fn) == len(self._threads)
        for idx, t in enumerate(self._threads):
            t.prepare(sess, sample_fn[idx])
            t.start()

    def stop(self):
        for t in self._threads:
            t.should_stop = True
            t.close_queue()
