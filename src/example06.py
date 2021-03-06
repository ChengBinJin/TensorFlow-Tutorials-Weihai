import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random.normal([1], name='weight'))
X = tf.compat.v1.placeholder(tf.float32)
Y = tf.compat.v1.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.math.reduce_mean(tf.math.square(hypothesis - Y))

# Minimize: Gradient Descent Magic
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.compat.v1.Session()
# Initializes global variables in the graph.
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(21):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

