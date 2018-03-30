import tensorflow as tf
from tfac.queue_input import QueueInput
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial


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


def build_net(features, labels, mode):
    # One Layer, just for test
    # Remember to set reuse to tf.AUTO_REUSE
    logits = tf.layers.dense(features["x"], 10, name="dense",
                             reuse=tf.AUTO_REUSE)

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


def main():
    # define features and labels
    features = {
        "x": tf.placeholder(tf.float32, [None, 784])
    }

    labels = {
        "y": tf.placeholder(tf.int64, [None])
    }

    # use QueueInput to build input op
    qi = QueueInput(features, labels, 100)
    idx, batch_features, batch_labels = qi.build_op(32)
    idx, pred_features, pred_labels = qi.build_op(1)

    # build net on two different input with same weights
    train_net = build_net(batch_features, batch_labels, ModeKeys.TRAIN)
    pred_net = build_net(pred_features, pred_labels, ModeKeys.PREDICT)

    # build sample function
    mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data')
    sample_fn = []
    sample_fn.append(partial(mnist.train.next_batch, 50))
    sample_fn.append(partial(mnist.test.next_batch, 50))

    # initialize tensorflow session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # run QueueInput
    qi.run(sess, sample_fn)

    # training
    for _ in range(1000):
        sess.run(train_net["train_op"])

    # predict
    pred = sess.run(pred_net["predictions"])
    print(pred)

    # stop QueueInput
    qi.stop()


if __name__ == '__main__':
    main()
