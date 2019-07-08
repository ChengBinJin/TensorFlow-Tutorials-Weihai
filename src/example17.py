import tensorflow as tf

x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4],
          [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6],
          [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1],
          [0, 1, 0], [0, 1, 0], [0, 1, 0],
          [1, 0, 0], [1, 0, 0]]

nb_classes = 3
X = tf.compat.v1.placeholder(tf.float32, shape=[None, 4], name='X')
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, nb_classes], name='Y')
W = tf.Variable(tf.random.normal([4, nb_classes]), dtype=tf.float32, name='weight')
b = tf.Variable(tf.random.normal([nb_classes]), dtype=tf.float32, name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.linalg.matmul(X, W) + b)
pred_cls = tf.math.argmax(hypothesis, axis=1)

# Corss entropy cost/loss
cost = tf.math.reduce_mean(tf.math.reduce_sum(- Y * tf.math.log(hypothesis), axis=1))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2001):
    _, cost_val, pred_val = sess.run([train, cost, pred_cls], feed_dict={X: x_data, Y: y_data})
    if step % 200 == 0:
        msg = "step: {}\ncost: {:.3f}\npred: {}"
        print(msg.format(step, cost_val, pred_val))

# Testing & One-hot encoding
hy_val, pred_val = sess.run([hypothesis, pred_cls], feed_dict={X: [[1, 11, 7, 9]]})
print('hy_val: {}'.format(hy_val))
print('pred_val: {}'.format(pred_val))
