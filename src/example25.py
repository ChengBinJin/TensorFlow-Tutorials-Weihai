import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cifar10
from cifar10 import img_shape, num_classes, img_size_flat

# Load CIFAR-10 data
cifar10.maybe_download_and_extract('./data/CIFAR-10/')

class_names = cifar10.load_class_names()
print("Number of classes: {}".format(num_classes))
print("CIFAR-10 class names: {}".format(class_names))
print("Image shape: {}".format(img_shape))
print("Image flatten size: {}".format(img_size_flat))

# cls_train: 0~9, labels_train is one-hot representation
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

# Calculate mean RGB and subtract the mean to zero-centering
mean_RGB = np.mean(np.reshape(images_train, [-1, img_size_flat]), axis=0)
print("mean_RGB shape: {}".format(mean_RGB.shape))

# Define placeholders
X = tf.compat.v1.placeholder(tf.float32, shape=[None, img_size_flat], name='X')
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, num_classes], name='Y')

###########################################################################################################
# Control number of layers and numer of neurons in hidden layer
num_hiddens = [256, 128, num_classes]

# Initialize number of epoches and batch size
num_epochs = 20
batch_size = 128

# Initialize learning rate
learning_rate = 0.01  # 0.1, 0.01, 0.001, 0.0001, etc.

# Activation function
# [tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu, tf.nn.leaky_relu, tf.nn.elu]
act_func = tf.nn.relu
###########################################################################################################

output = X
for i, num_hidden in enumerate(num_hiddens):
    # Define variables W and b
    W = tf.Variable(tf.random.normal([output.get_shape().as_list()[1], num_hidden]), name='weight'+str(i))
    b = tf.Variable(tf.random.normal([num_hidden]), name='bias'+str(i))
    # Hypothesis: X * W + b
    output = tf.linalg.matmul(output, W) + b

    # The last layer doesn't use the activation function
    if i == len(num_hiddens) - 1:
        break
    else:
        # Activation function
        output = act_func(output)

# Prediction
predictions = tf.math.argmax(output, axis=1)

# Define loss tensor
loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=Y))

# Define accuracy tensor
is_correct = tf.math.equal(tf.math.argmax(output, axis=1), tf.math.argmax(Y, axis=1))
accuracy = tf.math.reduce_mean(tf.dtypes.cast(is_correct, dtype=tf.float32))

# Define optimizer
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Define session and initialize global variables
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# Calculate the iterations for one epoch
total_iterations = int(np.ceil(len(images_train)/ batch_size))

# For loop
for epoch in range(num_epochs):
    for iteration in range(total_iterations):
        # Random-sampling images and labels from training data
        sample_idxes = np.random.randint(low=0, high=len(images_train), size=batch_size)
        x_batch, y_batch = images_train[sample_idxes], labels_train[sample_idxes]

        # Reshape the original (N, 32, 32, 3) to (N, 3072)
        x_batch = np.reshape(x_batch, [-1, img_size_flat])

        # Zero-centering for image data
        x_batch -= mean_RGB

        # sess.run()
        _, loss_val, batch_acc = sess.run([optimizer, loss, accuracy],
                                          feed_dict={X: x_batch, Y: y_batch})

        if iteration % 100 == 0 or iteration == total_iterations - 1:
            msg = "Epoch: {:3d}, Iteration: {:3d}, Loss: {:5.3f}, Batch-Acc: {:.2%}"
            print(msg.format(epoch, iteration, loss_val, batch_acc))


# Test stage
test_accuracy = 0.
num_feeds = 0
for start_idx in range(0, len(images_test), batch_size):
    if start_idx + batch_size >= len(images_test):
        end_idx = len(images_test) - 1
    else:
        end_idx = start_idx + batch_size

    # Extract images and labels
    x_batch = images_test[start_idx:end_idx]
    y_batch = labels_test[start_idx:end_idx]

    # Flatten images
    x_batch = np.reshape(x_batch, [-1, img_size_flat])
    # Subtract mean_RGB
    x_batch = x_batch - mean_RGB

    acc_test = sess.run(accuracy,feed_dict={X: x_batch, Y: y_batch})
    test_accuracy += acc_test
    num_feeds += 1

print("Test accuracy: {:.2%}".format(test_accuracy / num_feeds))

# Showimg test images and prediction results
num_tests = 20
idxes = np.random.randint(low=0, high=len(images_test), size=num_tests)

# Reshape, zero-centering, and sess.run()
imgs = images_test[idxes]
img_flattens = np.reshape(imgs, [-1, img_size_flat]) - mean_RGB
pred_vals = sess.run(predictions, feed_dict={X: img_flattens})

# Labels
labels = cls_test[idxes]

# Creat figure with sub-plots.
fig, axes = plt.subplots(nrows=2, ncols=num_tests // 2, figsize=(15, 4))
fig.subplots_adjust(hspace=0.3, wspace=0.6)

for i, ax in enumerate(axes.flat):
    # Plot image
    ax.imshow(imgs[i], interpolation='spline16')

    # Name of the truce class
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

