import matplotlib.pyplot as plt
import tensorflow as tf

x_data = [3, 4.5, 5.5, 6.5, 7.5, 8.5, 8, 9, 9.5, 10]
y_data = [8.49, 11.93, 16.18, 18.08, 21.45, 24.35, 21.24, 24.84, 25.94, 26.02]

# Initialize the placholders of the x and y
x = tf.compat.v1.placeholder(tf.float32, name='x')
y = tf.compat.v1.placeholder(tf.float32, name='y')
# Initialize W and b using tf.Variable()
W = tf.Variable(tf.random.normal([1]), dtype=tf.float32, name='weight')
b = tf.Variable(tf.random.normal([1]), dtype=tf.float32, name='bias')

# Our linear model, hypothesis = x * W + b
hypothesis = x * W + b
# Define the cost
cost = tf.math.reduce_mean(tf.math.square(hypothesis - y))

# Stochastic Gradient Descent optimizer
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# Minimize the cost
train = optimizer.minimize(cost)

# Initialize session
sess = tf.compat.v1.Session()
# Initialize the global variables
sess.run(tf.compat.v1.global_variables_initializer())

# For loop with 500 iterations
for step in range(501):
    _, cost_val, W_val, b_val = sess.run(
        [train, cost, W, b], feed_dict={x: x_data, y: y_data})

    if step % 20 == 0:
        msg = "step: {}, cost: {:.3f}, W: {:.2f}, b: {:.2f}"
        print(msg.format(step, cost_val, W_val[0], b_val[0]))

# Draw data samples and prediced line
x_min, x_max = min(x_data), max(x_data)
x_cords, y_cords = list(), list()
for x_cord in range(x_min - 1, x_max + 1, 1):
    x_cords.append(x_cord)
    y_cords.append(sess.run(hypothesis, feed_dict={x: x_cord})[0])

# matplotlib.pyplot scatter, plot, and show
plt.scatter(x_data, y_data)
plt.plot(x_cords, y_cords, color='red')
plt.show()
