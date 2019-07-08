import tensorflow as tf

# X and Y data
# x_train = [1, 2, 3]
# y_train = [1, 2, 3]

# Now we can use X and Y in place of x_train and y_train
# placeholders for a tensor that will be always fed using feed_dict
X = tf.compat.v1.placeholder(tf.float32, name='X')
Y = tf.compat.v1.placeholder(tf.float32, name='Y')

W = tf.Variable(tf.random.normal([1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# Our hypothesis XW + b
hypothesis = X * W + b

# cost/loss function
cost = tf.math.reduce_mean(tf.math.square(hypothesis - Y))

# Minimize
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch a graph in a session.
sess = tf.compat.v1.Session()
# Initialize global variables in the graph.
sess.run(tf.compat.v1.global_variables_initializer())

# Fit the line
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3], Y: [1, 2, 3]})

    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# Testing our model
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

