import numpy as np
import tensorflow as tf


xy = np.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973],
               [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
               [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
               [816, 820.958984, 1008100, 815.48999, 819.23999],
               [819.359985, 823, 1188100, 818.469971, 818.97998],
               [819, 823, 1198100, 816, 820.450012],
               [811.700012, 815.25, 1098100, 809.780029, 813.669983],
               [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])


def MinMaxScaler(data):
    return (data - data.min()) / (data.max() - data.min())


xy = MinMaxScaler(xy)
print(xy)

x_data = xy[:, :-1]
y_data = xy[:, [-1]]

# placeholders for a tensor that will be always fed.
X = tf.compat.v1.placeholder(tf.float32, shape=[None, x_data.shape[1]])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random.normal([x_data.shape[1], 1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = tf.linalg.matmul(X, W) + b
cost = tf.math.reduce_mean(tf.math.square(hypothesis - Y))
# Minimize
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-3)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

