import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cifar10
from cifar10 import num_classes, img_shape

# Load CIFAR-10 data
cifar10.data_path = './data/CIFAR-10'
cifar10.maybe_download_and_extract()

class_names = cifar10.load_class_names()
print("Number of classes: {}".format(num_classes))
print("CIFAR-10 class names: {}".format(class_names))
print("Image shape: {}".format(img_shape))

# cls_train: 0~9, labels_train is one-hot representation
imgs_train, cls_train, labels_train = cifar10.load_training_data()
imgs_test, cls_test, labels_test = cifar10.load_test_data()

print("Size of:")
print("- Training-set: \t{}".format(len(imgs_train)))
print("- Test-set: \t\t{}".format(len(imgs_test)))

# Calculate mean RGB and subtract the mean to zero-centering
RGB_mean = np.mean(imgs_train, axis=0)
print("RGB_mean shape: {}".format(RGB_mean.shape))


def conv_layer(x, output_dim, name, k_h=5, k_w=5, d_h=1, d_w=1):
    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable('weight', shape=[k_h, k_w, x.get_shape()[-1], output_dim],
                            initializer=tf.compat.v1.initializers.he_normal())
        b = tf.compat.v1.get_variable('bias', shape=[output_dim], initializer=tf.constant_initializer(0.))

        conv = tf.nn.conv2d(x, w, strides=[1, d_h, d_w, 1], padding='VALID')
        conv = tf.nn.bias_add(conv, b)
        output = tf.nn.relu(conv)

    return output

def fc_layer(x, output_dim, name, is_active=True):
    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable('weight', shape=[x.get_shape()[-1], output_dim],
                            initializer=tf.compat.v1.initializers.he_normal())
        b = tf.compat.v1.get_variable('bias', shape=[output_dim],
                                      initializer=tf.constant_initializer(0.))
        output = tf.linalg.matmul(x, w) + b

        if is_active:
            return tf.nn.relu(output)
        else:
            return output

def print_activation(t):
    print("{}: \t{}".format(t.op.name, t.get_shape().as_list()))

##########################################################################################
# Initialize number of epoches and batch size
num_epoches = 20
batch_size = 128

# Initialize learning rate
learning_rate = 0.01  # 0.1, 0.01, 0.001, 0.0001, et al.

# Initialize the probability of the dropout
drop_prob = 0.1  # rate = 1 - drop_prob
##########################################################################################

# Define placeholders
X = tf.compat.v1.placeholder(tf.float32, shape=[None, *img_shape], name="X")
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes], name="Y")
drop_prob_ph = tf.compat.v1.placeholder(tf.float32, name='drop_prob_ph')

# Conv1 layer: 5x5 with 6 filters, stride 1, and padding=VALID
output = conv_layer(X, output_dim=6, name='conv1')
print_activation(output)

# Max-pooling: 2d max-pooling with stride 2x2
output = tf.nn.max_pool2d(output, ksize=[2, 2], strides=[2, 2], padding='SAME')
print_activation(output)

# Conv2 layer: 5x5 with 16 filters, stride 1, and padding=VALID
output = conv_layer(output, output_dim=16, name='conv2')
print_activation(output)

# Max-pooling: 2d max-pooling with stride 2x2
output = tf.nn.max_pool2d(output, ksize=[2, 2], strides=[2, 2], padding='SAME')
print_activation(output)

# Reshape to become a vector
output = tf.reshape(output, shape=[-1, tf.math.reduce_prod(output.get_shape()[1:])])
print_activation(output)

# Fully-connection layer 1: neurons=120
output = fc_layer(output, output_dim=120, name='fc1')
output = tf.nn.dropout(output, rate=drop_prob)
print_activation(output)

# Fully-connection layer 2: neurons=84
output = fc_layer(output, output_dim=82, name='fc2')
output = tf.nn.dropout(output, rate=drop_prob)
print_activation(output)

# Last fully-connection layer: neurons=10
output = fc_layer(output, output_dim=num_classes, name='last_fc', is_active=False)
print_activation(output)

# Accuracy
prediction = tf.math.argmax(output, axis=1)
is_correct = tf.math.equal(prediction, tf.math.argmax(Y, axis=1))
accuracy = tf.math.reduce_mean(tf.dtypes.cast(is_correct, dtype=tf.float32))

# Softmax cross entropy loss
loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=Y))

# Optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Session
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

total_iterations = int(np.ceil(len(imgs_train) / batch_size))
for epoch in range(num_epoches):
    for iteration in range(total_iterations):
        idxes = np.random.randint(low=0, high=len(imgs_train), size=batch_size)
        x_batch, y_batch = imgs_train[idxes], labels_train[idxes]
        # Zero-centering
        x_batch -= RGB_mean

        _, loss_val, batch_acc = sess.run([optimizer, loss, accuracy],
                                          feed_dict={X: x_batch, Y: y_batch, drop_prob_ph: drop_prob})
        if iteration % 50 == 0 or iteration == total_iterations - 1:
            msg = "Epoch: {:2d}, Iter: {:3d}, Loss: {:.3f}, Batch-Acc.: {:.2%}"
            print(msg.format(epoch, iteration, loss_val, batch_acc))

# Test stage
total_acc = 0.
num_feeds = 0
for start_idx in range(0, len(imgs_test), batch_size):
    if start_idx + batch_size >= len(imgs_test):
        end_idx = len(imgs_test)
    else:
        end_idx = start_idx + batch_size

    x_batch, y_batch = imgs_test[start_idx:end_idx].copy(), labels_test[start_idx:end_idx]
    # Zero-centering
    x_batch -= RGB_mean

    test_acc = sess.run(accuracy, feed_dict={X: x_batch, Y: y_batch, drop_prob_ph: 0.})
    total_acc += test_acc
    num_feeds +=1

print("Test acc: {:.2%}".format(total_acc / num_feeds))

# Showing test images and prediction results
num_tests = 20
idxes = np.random.randint(low=0, high=len(imgs_test), size=num_tests)

# Zero-centring and sess.run()
imgs, labels = imgs_test[idxes], cls_test[idxes]
x_batch = imgs - RGB_mean
pred_vals = sess.run(prediction, feed_dict={X: x_batch, drop_prob_ph: 0.})

# Create figure with sub-plots.
fig, axes = plt.subplots(nrows=2, ncols=num_tests // 2, figsize=(15, 4))
fig.subplots_adjust(hspace=0.3, wspace=0.6)

for i, ax in enumerate(axes.flat):
    # Plot image
    ax.imshow(imgs[i], interpolation='spline16')

    # Name of the true class
    cls_true_name = class_names[labels[i]]

    # Predicted classes
    cls_pred_name = class_names[pred_vals[i]]

    # Show true and predicted classes
    xlabel = "True: {}\nPred: {}".format(cls_true_name, cls_pred_name)

    # Show the classes as the label on the x-axis.
    ax.set_xlabel(xlabel)

    # Remove ticks from the plot.
    ax.set_xticks([])
    ax.set_yticks([])

# Ensure the plot is shown correctly with multiple plots.
# in a single Notebook cell.
plt.show()
