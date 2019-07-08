import matplotlib.pyplot as plt
import tensorflow as tf

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.compat.v1.placeholder(tf.float32)
# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.math.reduce_mean(tf.square(hypothesis - Y))

# Launch the graph in a session.
sess = tf.compat.v1.Session()
# Initializes global variables in the graph.
sess.run(tf.compat.v1.global_variables_initializer())

# Variables for plotting cost function
W_val = list()
cost_val = list()

for i in range(-30, 50):
    feed_W = i * 0.1
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

# Show the cost function
plt.plot(W_val, cost_val)
plt.xlabel('W')
plt.ylabel('Cost')
plt.show()
