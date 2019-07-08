import tensorflow as tf

# Initialize the placeholders of the x and y
x = tf.compat.v1.placeholder(tf.float32, name='x')
y = tf.compat.v1.placeholder(tf.float32, name='y')
# Initialize W and b using tf.Variable()
W = tf.Variable(tf.random.normal([1]), dtype=tf.float32, name='weight')
b = tf.Variable(tf.random.normal([1]), dtype=tf.float32, name='bias')

# Our linear model, hypothesis = W * x + b
hypothesis = x * W + b
# Define the cost using L2 distance
cost = tf.math.reduce_mean(tf.math.square(hypothesis - y))

# Define the Gradient Descent optimizer with learning rate 0.01
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# Minimize the cost using the optimizer
train = optimizer.minimize(cost)

# Define session
sess = tf.compat.v1.Session()
# Initialize the global variables
sess.run(tf.compat.v1.global_variables_initializer())

# For Loop function with 500 iterations
for step in range(500):
    _, cost_val, W_val, b_val = sess.run(
        [train, cost, W, b], feed_dict={x: [10, 9, 3, 2], y: [90, 80, 50, 30]})

    # print step, cost, W, and b for 20 iterations
    if step % 20 == 0:
        msg = "step: {}, cost: {:.3f}, W: {:.2f}, b: {:.2f}"
        print(msg.format(step, cost_val, W_val[0], b_val[0]))

msg = "Studied {} hours, the expected score: {}"
print(msg.format(3.5, sess.run(hypothesis, feed_dict={x: 3.5})))
print(msg.format(4.0, sess.run(hypothesis, feed_dict={x: 4.0})))
print(msg.format(5.0, sess.run(hypothesis, feed_dict={x: 5.0})))

