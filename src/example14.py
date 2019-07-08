import csv
import numpy as np
import tensorflow as tf


def csv_reader(filename, is_train=True):
    with open(filename, 'r', encoding='UTF8') as csvfile:
        read_csv = csv.reader(csvfile, delimiter=',')

        train_data, test_data = list(), list()
        for i, row in enumerate(read_csv):
            if i == 0:  # exclude text information
                continue

            if is_train:
                train_data.append(row[1:-1])  # exclude id
                test_data.append(row[-1])  # last column
            else:
                train_data.append(row[1:])  # exclude id

        if is_train:
            train_data = np.asarray(train_data).astype(np.float32)
            test_data = np.asarray(test_data).astype(np.float32)

            return train_data, np.expand_dims(test_data, axis=1)
        else:
            train_data = np.asarray(train_data).astype(np.float32)

            return train_data


# Read training and test data
x_train, y_train = csv_reader('./data/lecture3_train_data.csv', is_train=True)
x_test = csv_reader('./data/lecture3_test_data.csv', is_train=False)

# Define placeholders
X = tf.compat.v1.placeholder(tf.float32, shape=[None, x_train.shape[1]], name='x')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='y')

# Define Variables
W = tf.Variable(tf.random.normal([x_train.shape[1], 1], dtype=tf.float32, name='weight'))
b = tf.Variable(tf.random.normal([1]), dtype=tf.float32, name='bias')

# Our multi-variable model hypothesis = X * W + b
hypothesis = tf.linalg.matmul(X, W) + b
# Cost/loss function
cost = tf.math.reduce_mean(tf.math.square(hypothesis - y))

# Define optimizer
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Initilize session and variables
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# For loop
for step in range(2001):
    _, cost_val, W_val, b_val = sess.run(
        [train, cost, W, b], feed_dict={X: x_train, y: y_train})

    if step % 20 == 0:
        msg = "step: {}, cost: {:.3f}, W: {}, b:{}"
        print(msg.format(step, cost_val, W_val, b_val))

# Test stage
preds = sess.run(hypothesis, feed_dict={X: x_test})
for i in range(len(preds)):
    msg = "Id: {}, Predicted score: {}"
    print(msg.format(i, preds[i]))

