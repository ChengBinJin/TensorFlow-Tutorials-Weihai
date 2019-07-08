import tensorflow as tf

X = tf.compat.v1.placeholder(tf.float32, shape=[None], name='X')
Y = tf.compat.v1.placeholder(tf.float32, shape=[None], name='Y')
W = tf.compat.v1.get_variable(name='weight', shape=[1], initializer=tf.random_normal_initializer)
b = tf.compat.v1.get_variable(name='bias', shape=[1], initializer=tf.random_normal_initializer)

# Our hypothesis XW+b
hypothesis = X * W + b
# cost/loss function
cost = tf.math.reduce_mean(tf.math.square(hypothesis - Y))
# Minimize
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.compat.v1.Session()
# Initializes global variables in the graph.
sess.run(tf.compat.v1.global_variables_initializer())

# Fit the line with new training data
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
        feed_dict={X: [1, 2, 3, 4, 5],
                   Y: [2.1, 3.1, 4.1, 5.1, 6.1]})

    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# Testing our model
print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))

