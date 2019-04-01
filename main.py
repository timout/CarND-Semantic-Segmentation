#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import numpy as np
import random
import cv2

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    graph = tf.get_default_graph()
    input = graph.get_tensor_by_name('image_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    l3 = graph.get_tensor_by_name('layer3_out:0')
    l4 = graph.get_tensor_by_name('layer4_out:0')
    l7 = graph.get_tensor_by_name('layer7_out:0')
    
    return input, keep_prob, l3, l4, l7

def conv1x1(inputs, filters):
    """
    1x1 tf.layers.conv2d
    """
    k_init = 0.01 # std for kernel_initializer
    k_reg = 0.001 # scale for kernel_regualarizer
    padding = "same"
    return tf.layers.conv2d(inputs, 
                            filters, 
                            1, 
                            padding=padding,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=k_init),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=k_reg))

def conv_transpose(inputs, filters, kernel_size, strides):
    """
    tf.layers.conv2d_transpose 
    """
    padding = "same" # "valid" or "same"
    k_init = 0.01    # std for kernel_initializer
    k_reg = 0.001    # scale for kernel_regualarizer
    return tf.layers.conv2d_transpose(inputs,
                                      filters, 
                                      kernel_size, 
                                      strides, 
                                      padding,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=k_init),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=k_reg))


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    l7 = conv1x1(inputs=vgg_layer7_out, filters=num_classes)
    l4 = conv1x1(inputs=vgg_layer4_out, filters=num_classes)
    l3 = conv1x1(inputs=vgg_layer3_out, filters=num_classes)

    # up-sample
    res = conv_transpose(inputs=l7, filters=num_classes, kernel_size=4, strides=2) 
    # skip connection
    res = tf.add(res, l4)    
    # up-sample                                                        
    res = conv_transpose(inputs=res, filters=num_classes, kernel_size=4, strides=2) 
    # skip connection
    res = tf.add(res, l3)                                                            
    # up-sample
    return conv_transpose(inputs=res, filters=num_classes, kernel_size=16, strides=8)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return logits, optimizer, loss


def transform_image(img):
    return img[:, ::-1, :]

def augment_colors(img):
    img = np.float32(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    # Hue, Saturation and Brightness change
    img[:,:,0] = img[:,:,0] * random.uniform(0.5,2.0)
    img[:,:,1] = img[:,:,1] * random.uniform(0.5,2.0)
    img[:,:,2] = img[:,:,2] * random.uniform(0.5,2.0)
    img = np.uint8(np.clip(img, 0, 255))
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def augment(image, label):
    image_aug = np.copy(image)
    label_aug = np.copy(label)
    for i in range(image.shape[0]):
        #flip
        image_aug[i] = transform_image(image[i])
        label_aug[i] = transform_image(label[i])
        # change colors
        image_aug[i] = augment_colors(image_aug[i])

    return np.concatenate( (image, image_aug), axis=0), np.concatenate( (label, label_aug), axis=0)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, do_augment=False):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param do_augment: Turn on/off image augmentation
    """
    with open('loss.csv', 'w') as file: 
        for epoch in range(epochs):
            for batch, (image, label) in enumerate(get_batches_fn(batch_size)):
                # augment images
                if do_augment: 
                    image, label = augment(image, label)
                feed_dict = {input_image: image, correct_label: label, keep_prob: 0.75, learning_rate: 0.0001}
                _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
                loss_str = 'Epoch: {}, batch: {}, loss: {}'.format(epoch, batch, loss)
                print(loss_str)
                file.write(loss_str+'\n')


def train(correct_label, data_dir, image_shape, input_image, keep_prob, learning_rate, loss, optimizer, sess):
    epochs = 25
    batch_size = 5
    training_path = os.path.join(data_dir, 'data_road/training')
    get_batches_fn = helper.gen_batch_function(training_path, image_shape)
    sess.run(tf.global_variables_initializer())
    train_nn(sess, epochs, batch_size, get_batches_fn, optimizer, loss, input_image, correct_label, keep_prob, learning_rate, True)


def run_tests(data_dir):
    tests.test_for_kitti_dataset(data_dir)
    tests.test_load_vgg(load_vgg, tf)
    tests.test_layers(layers)
    tests.test_optimize(optimize)
    tests.test_train_nn(train_nn)

def run(data_dir, runs_dir):
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')
    learning_rate = tf.placeholder(tf.float32)
    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])

    with tf.Session() as sess:
        input, keep_prob, l3, l4, l7 = load_vgg(sess, vgg_path)
        output = layers(l3, l4, l7, num_classes)
        logits, optimizer, loss = optimize(output, correct_label, learning_rate, num_classes)
        print("Training...")
        train(correct_label, data_dir, image_shape, input, keep_prob, learning_rate, loss, optimizer, sess)
        print("Save samples...")
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input)

if __name__ == '__main__':
    data_dir = './data'
    runs_dir = './runs'
    run_tests(data_dir)
    run(data_dir, runs_dir)
