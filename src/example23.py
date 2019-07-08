import tensorflow as tf

x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = list()

try_operation = input("Please enter the operation ['and', 'or', 'xor']:")
if try_operation.lower() == 'and':
    y_data = [[0], [0], [0], [1]]
elif try_operation.lower() == 'or':
    y_data = [[0], [1], [1], [1]]
elif try_operation.lower() == 'xor':
    y_data = [[0], [1], [1], [0]]
else:
    exit('Please enter the correct operation')

# Define placeholders
X = tf.compat.v1.placeholder(tf.float32, shape=[None, 2], name='X')
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='Y')

# Define two variables W and b
W = tf.Variable(tf.random.normal([2, 1]), dtype=tf.float32, name='weight')
b = tf.Variable(tf.random.normal([1]), dtype=tf.float32, name='bias')

# Hypothesis: sigmoid(X * W + b)
hypothesis = tf.linalg.matmul(X, W) + b
logits = tf.nn.sigmoid(hypothesis)
prediction = tf.dtypes.cast(logits > 0.5, dtype=tf.float32)

# Loss function of the binary cross entropy
loss = tf.math.reduce_mean(-Y * tf.math.log(logits) - (1 - Y) * tf.math.log(1 - logits))

# Define optimizer and minimize the loss
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

# Define session and initialize global variables
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# For loop
for step in range(501):
    _, loss_val = sess.run([optimizer, loss], feed_dict={X: x_data, Y: y_data})

    if step % 50 == 0:
        msg = "Step: {}, Loss: {:.3f}"
        print(msg.format(step, loss_val))

pred_val, label = sess.run([prediction, Y], feed_dict={X: x_data, Y: y_data})
print("Prediction: {}\nLabel: {}".format(pred_val, label))

