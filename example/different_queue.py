import tensorflow as tf
from tfac.queue_input import QueueInput
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial
import time


class ModeKeys(object):
    """Standard names for model modes.

    The following standard keys are defined:

    * `TRAIN`: training mode.
    * `EVAL`: evaluation mode.
    * `PREDICT`: inference mode.
    """

    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'infer'


def build_net(features, labels, mode, reuse, prefix):
    # Remember to set reuse to tf.AUTO_REUSE
    dense_1 = tf.layers.dense(features["x"], 128, name=prefix + "dense_1",
                              reuse=reuse, activation=tf.nn.relu)
    dense_2 = tf.layers.dense(dense_1, 256, name=prefix + "dense_2",
                              reuse=reuse, activation=tf.nn.relu)
    dense_3 = tf.layers.dense(dense_2, 256, name=prefix + "dense_3",
                              reuse=reuse, activation=tf.nn.relu)
    logits = tf.layers.dense(dense_3, 10, name=prefix + "logit",
                             reuse=reuse)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == ModeKeys.PREDICT:
        net = {"predictions": predictions}
        return net

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels["y"],
                                                  logits=logits)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())

    net = {
        "predictions": predictions,
        "loss": loss,
        "train_op": train_op
    }
    return net


def measure_time(func):
    def wrapper():
        start = time.time()
        func()
        stop = time.time()
        print("runtime: " + str(stop - start) + " sec")

    return wrapper


@measure_time
def with_fifo_queue():
    # define features and labels
    features = {
        "x": tf.placeholder(tf.float32, [None, 784])
    }

    labels = {
        "y": tf.placeholder(tf.int64, [None])
    }

    # use QueueInput to build input op
    qi = QueueInput(features, labels, [400, 100], tf.FIFOQueue)
    idx, batch_features, batch_labels = qi.build_op(32)
    idx, pred_features, pred_labels = qi.build_op(50)

    # build net on two different input with same weights
    train_net = build_net(batch_features, batch_labels,
                          ModeKeys.TRAIN, False, "fifo_queue_")
    pred_net = build_net(pred_features, pred_labels,
                         ModeKeys.PREDICT, tf.AUTO_REUSE, "fifo_queue_")

    # build sample function
    mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data')
    sample_fn = []
    sample_fn.append(partial(mnist.train.next_batch, 100))
    sample_fn.append(partial(mnist.test.next_batch, 50))

    # initialize tensorflow session
    sess = tf.InteractiveSession()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # run QueueInput
    qi.run(sess, sample_fn)

    # training
    for _ in range(100):
        sess.run(train_net["train_op"])

    # predict
    pred = sess.run(pred_net["predictions"])

    # stop QueueInput
    qi.close()

    sess.close()


@measure_time
def with_random_shuffle_queue():
    # define features and labels
    features = {
        "x": tf.placeholder(tf.float32, [None, 784])
    }

    labels = {
        "y": tf.placeholder(tf.int64, [None])
    }

    # use QueueInput to build input op
    qc = lambda **kwargs: tf.RandomShuffleQueue(**kwargs, min_after_dequeue=50)
    qi = QueueInput(features, labels, [400, 100], queue_class=qc)
    idx, batch_features, batch_labels = qi.build_op(32)
    idx, pred_features, pred_labels = qi.build_op(50)

    # build net on two different input with same weights
    train_net = build_net(batch_features, batch_labels,
                          ModeKeys.TRAIN, False, "rsq_")
    pred_net = build_net(pred_features, pred_labels,
                         ModeKeys.PREDICT, tf.AUTO_REUSE, "rsq_")

    # build sample function
    mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data')
    sample_fn = []
    sample_fn.append(partial(mnist.train.next_batch, 100))
    sample_fn.append(partial(mnist.test.next_batch, 50))

    # initialize tensorflow session
    sess = tf.InteractiveSession()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # run QueueInput
    qi.run(sess, sample_fn)

    # training
    for _ in range(10000):
        sess.run(train_net["train_op"])

    # predict
    pred = sess.run(pred_net["predictions"])

    # stop QueueInput
    qi.close()

    sess.close()


def main():
    print("running with_fifo_queue")
    with_fifo_queue()
    print("\n")
    print("running with_random_shuffle_queue")
    with_random_shuffle_queue()


if __name__ == '__main__':
    main()
