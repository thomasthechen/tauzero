import tensorflow as tf
import numpy as np

lr = 0.1  # learning rate

n_input = 69
n_hidden_1 = 120
n_hidden_2 = 60
n_output = 1

def nn(X):

    with tf.name_scope('Hidden 1'):
        W_h1 = weight_variable([n_input, n_hidden_1])
        b_h1 = bias_variable([n_hidden_1])
        hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_h1), b_h1))

    with tf.name_scope('Hidden 2'):
        W_h2 = weight_variable([n_hidden_1, n_hidden_2])
        b_h2 = bias_variable([n_hidden_2])
        hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_h2), b_h2))

    with tf.name_scope('Output'):
        W_o = weight_variable([n_hidden_2, n_output])
        b_o = bias_variable([n_output])
        output = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_o), b_o))

    return output

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    W = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(W)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    b = tf.constant(0.1, shape=shape)
    return tf.Variable(b)

def main():

    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_output])

    print('1')

if __name__ == '__main__':
    main()