import csv
import numpy as np
import tensorflow as tf


def read_csv(filename, ratio=(5, 5)):
    with open(filename, 'r', encoding='UTF8') as csvfile:
        csv_handler = csv.reader(csvfile, delimiter=',')

        data = list()
        for row in csv_handler:
            data.append(row)

        # Convert to ndarray with np.float32
        data = np.asarray(data).astype(np.float32)

        # Split the data into training and test dataset
        num_trains = int(np.ceil((ratio[0] / np.sum(ratio)) * data.shape[0]))

        x_train = data[:num_trains, :-1]
        y_train = data[:num_trains, [-1]]
        x_test = data[num_trains:, :-1]
        y_test = data[num_trains:, [-1]]

        return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = read_csv('./data/diabetes.csv', ratio=(5, 5))
print('x_train shape: {}'.format(x_train.shape))
print('y_train shape: {}'.format(y_train.shape))
print('x_test shape: {}'.format(x_test.shape))
print('y_tet shape: {}'.format(y_test.shape))

# Define input and outptu placeholders
X = tf.compat.v1.placeholder(tf.float32, shape=[None, x_train.shape[1]], name='X')
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='Y')
# Variables for W and b
W = tf.Variable(tf.random.normal([x_train.shape[1], 1]), dtype=tf.float32, name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# Hypothesis using tf.math.sigmoid
hypothesis = tf.math.sigmoid(tf.linalg.matmul(X, W) + b)
# Cost/loss of the sigmoid cross entropy
cost = -tf.math.reduce_mean(Y * tf.math.log(hypothesis) + (1 - Y) * tf.math.log(1 - hypothesis))
# Define optimizer and minimizing the cost
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Predicted value 0.0 or 1.0 and accuracy
predicted = tf.dtypes.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.math.reduce_mean(tf.dtypes.cast(tf.math.equal(predicted, Y), dtype=tf.float32)) * 100.

# Define session and initialize all variables
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# For loop function with 10001
for step in range(20001):
    _, cost_val, acc_val = sess.run([train, cost, accuracy], feed_dict={X: x_train, Y: y_train})

    if step % 100 == 0:
        msg = "step: {}, cost:{:.3f}, trainining acc: {:.2f}%"
        print(msg.format(step, cost_val, acc_val))

# Test stage
pred_test, acc_test = sess.run([predicted, accuracy], feed_dict={X: x_test, Y: y_test})
print('Predict: {}'.format(pred_test))
print('Test acc: {:.2f}%'.format(acc_test))

