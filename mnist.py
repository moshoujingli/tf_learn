import tensorflow as tf
import numpy as np
import input_data
from tensorflow.python.framework import ops

def init_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)


def model(X, W_conv1, b_conv1, pool1, W_conv2, b_conv2, pool2, W_fc, b_fc, w_o, b_o, p_keep_hidden):
    # reshape
    # 28*28*1 -> 28*28*32 -> 14*14*32
    # 14*14*32 -> 14*14*64 -> 7*7*64
    # 7*7*64 -> 1*1024
    # 1*1024 -> 1*10
    X_2d = tf.reshape(X, [-1, 28, 28, 1])
    conv_1_X_2d = tf.nn.conv2d(X_2d, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
    conv_1_X_2d = tf.nn.relu(conv_1_X_2d + b_conv1)
    poll_1_X_2d = tf.nn.max_pool(conv_1_X_2d, ksize=pool1, strides=pool1, padding='SAME')

    conv_2_X_2d = tf.nn.conv2d(poll_1_X_2d, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    conv_2_X_2d = tf.nn.relu(conv_2_X_2d + b_conv2)
    poll_2_X_2d = tf.nn.max_pool(conv_2_X_2d, ksize=pool2, strides=pool2, padding='SAME')

    fc_vec_len = W_fc.get_shape().as_list()[0]
    fc_1d = tf.reshape(poll_2_X_2d, [-1, fc_vec_len])
    h2 = tf.nn.relu(tf.matmul(fc_1d, W_fc) + b_fc)
    h2 = tf.nn.dropout(h2, p_keep_hidden)
    return tf.nn.softmax(tf.matmul(h2, w_o) + b_o)


def defineGraph():
    X = tf.placeholder("float", [None, 784])
    Y = tf.placeholder("float", [None, 10])

    W_conv1 = init_weights([5, 5, 1, 8], "w_conv1")
    b_conv1 = init_weights([8], "b_conv1")
    pool1 = [1, 2, 2, 1]
    W_conv2 = init_weights([3, 3, 8, 16], "w_conv2")
    b_conv2 = init_weights([16], "b_conv2")
    pool2 = [1, 2, 2, 1]
    W_fc = init_weights([7 * 7 * 16, 1024], "w_fc")
    b_fc = init_weights([1024], "b_fc")
    w_o = init_weights([1024, 10], "w_o")
    b_o = init_weights([10], "b_o")

    p_keep_input = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

    py_x = model(X, W_conv1, b_conv1, pool1, W_conv2, b_conv2, pool2, W_fc, b_fc, w_o, b_o, p_keep_hidden)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    return predict_op, train_op, X, Y, p_keep_input, p_keep_hidden


class MnistModel(object):
    """docstring for MinstModel"""

    def __init__(self):
        super(MnistModel, self).__init__()
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    def train(self, modelSavePath):
        predict_op, train_op, X, Y, p_keep_input, p_keep_hidden = defineGraph()
        mnist = self.mnist
        sess = tf.Session()
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess.run(init)

        for i in range(200):
            batch = mnist.train.next_batch(100)
            sess.run(train_op, feed_dict={X: batch[0], Y: batch[1],
                                          p_keep_input: 1.0, p_keep_hidden: 0.7})
        saver.save(sess, modelSavePath)

    def test(self, modelSavePath):
        ops.reset_default_graph()
        predict_op, train_op, X, Y, p_keep_input, p_keep_hidden = defineGraph()
        teX, teY = self.mnist.test.images, self.mnist.test.labels
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, modelSavePath)

        print np.mean(np.argmax(teY, axis=1) ==
                      sess.run(predict_op, feed_dict={X: teX, Y: teY,
                                                      p_keep_input: 1.0,
                                                      p_keep_hidden: 1.0}))

    def predicate(self, modelSavePath):
        pass
