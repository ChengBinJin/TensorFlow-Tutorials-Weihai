import tensorflow as tf

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
y_data = [152., 185., 180., 196., 142.]

# Placeholders for a tensor that will be always fed.
x1 = tf.compat.v1.placeholder(tf.float32)
x2 = tf.compat.v1.placeholder(tf.float32)
x3 = tf.compat.v1.placeholder(tf.float32)
Y = tf.compat.v1.placeholder(tf.float32)

w1 = tf.Variable(tf.random.normal([1], name='weight1'))
w2 = tf.Variable(tf.random.normal([1], name='weight2'))
w3 = tf.Variable(tf.random.normal([1], name='weight3'))
b = tf.Variable(tf.random.normal([1]), name='bias')

hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b

# cost/loss function
cost = tf.math.reduce_mean(tf.math.square(hypothesis - Y))
# Minimize. Need a very small learning rate for this data set
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.compat.v1.Session()
# Initializes global variabels in the graph.
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={x1: x1_data,
                                              x2: x2_data,
                                              x3: x3_data,
                                              Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

