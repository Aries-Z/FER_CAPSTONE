from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.regularizers import l2
import pickle
import numpy
import csv
import scipy.misc
import scipy
from scipy import ndimage
import imutils
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import itertools
import tensorflow as tf
import json


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    #  images are 48x48 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 48, 48, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 48, 48, 1]
    # Output Tensor Shape: [batch_size, 48, 48, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 48, 48, 32]
    # Output Tensor Shape: [batch_size, 24, 24, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 32 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 24, 24, 32]
    # Output Tensor Shape: [batch_size, 24, 24, 32]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=[4, 4],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 24, 24, 32]
    # Output Tensor Shape: [batch_size, 12, 12, 32]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Convolutional Layer #3
    # Input Tensor Shape: [batch_size, 12, 12, 32]
    # Output Tensor Shape: [batch_size, 12 * 12 * 64]

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    # Input Tensor Shape: [batch_size, 12, 12, 64]
    # Output Tensor Shape: [batch_size, 6 * 6 * 64]


    pool3_flat = tf.reshape(pool3, [-1, 6 * 6 * 64])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool3_flat, units=3072, activation=tf.nn.relu)

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 7]
    logits = tf.layers.dense(inputs=dropout, units=7)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": logits,
        "dense_layer": dropout
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def denormalize(x, y, frameWidth, frameHeight):
    x1 = frameWidth / 2 + x * frameWidth / 2
    y1 = frameHeight / 2 + y * frameHeight / 2
    return int(x1), int(y1)


class EmotionSubsystem():
    def __init__(self):
        pass

    def __call__(self, video_dir, people_boxes, skeletons, model_dir):

        # characters will be a dictionary of character to a list of bounding_box
        # namedtuple for that character. The list will have same length as number
        # of frames in the video.
        # prediction field should be a dictionary of character
        # to prediction value for each possible prediction.
        # lastlayer should be a dictionary of numpy arrays with keys corresponding
        # to those in prediction
        face_size = 27

        fer_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir=model_dir)
        cap = cv2.VideoCapture(video_dir)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        buf = np.empty((frameCount, frameHeight, frameWidth,), np.dtype('uint8'))
        print(frameCount, frameWidth, frameHeight)
        fc = 0
        ret = True

        while (fc < frameCount and ret):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            buf[fc] = gray
            fc += 1

        cap.release()

        video_arrays = buf
        # croping and classifying part


        with open(people_boxes) as csvfile:
            result = {"prediction": {}, "dense_layer": {}}

            readCSV = csv.reader(csvfile, delimiter=',')
            boxes = []
            flag = 0
            for i in readCSV:
                if flag == 0:
                    flag = 1
                else:
                    boxes.append(i)

            boxes_dict = {}
            for i in boxes:
                if i[1] in boxes_dict:
                    boxes_dict[i[1]].append(i)
                else:
                    boxes_dict[i[1]] = []
            for unique in boxes_dict:
                x_input = []

                result["prediction"][unique] = 0
                result["dense_layer"][unique] = {}
                frame_ids = []
                for i in boxes_dict[unique]:
                    #             print(i)
                    frame_ids.append(int(i[0]))
                    frame = video_arrays[int(i[0])]

                    img = frame[int(i[3]):int(i[3]) + int(i[5]), int(i[2]):int(i[2]) + int(i[4])]

                    with open(skeletons, 'r') as load_f:
                        load_dict = json.load(load_f)

                        for item in load_dict["data"]:

                            if item["frame_index"] == int(i[0]):
                                for item2 in item["skeleton"]:
                                    x, y = denormalize(item2["pose"][0], item2["pose"][1], frameWidth, frameHeight)
                                    #                             print(x,y)
                                    #                             print(int(i[2]))
                                    #                             print(int(i[2])+int(i[4]))
                                    #                             print(int(i[3]))
                                    #                             print(int(i[3])+int(i[5]))

                                    if int(i[2]) <= x and x <= int(i[2]) + int(i[4]) and int(i[3]) <= y and y <= int(
                                            i[3]) + int(i[5]):
                                        #                                 print(x,y)
                                        img2 = frame[y - face_size:y + face_size, x - face_size:x + face_size]

                                        #                                 cv2.namedWindow('frame 10')
                                        #                                 cv2.imshow('frame 10', img)

                                        #                                 cv2.namedWindow('frame 11')
                                        #                                 cv2.imshow('frame 11', img2)
                                        #                                 cv2.waitKey(0)
                                        crop_img = cv2.resize(img2, (48, 48), interpolation=cv2.INTER_CUBIC)
                                        crop_img = crop_img.flatten()

                                        for i1 in crop_img:
                                            x_input.append(i1)
                x_input = np.array(x_input)
                print(np.shape(x_input))
                x_input = np.reshape(x_input, (-1, 2304))
                print(np.shape(x_input))
                x_input = x_input.astype(numpy.float64)
                print(x_input.dtype)

                eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": x_input},
                    num_epochs=1,
                    shuffle=False)
                predict_results = fer_classifier.predict(input_fn=eval_input_fn)

                z = 0
                list_class = []
                results = {}
                for re in predict_results:
                    results[frame_ids[z]] = re
                    print(re)
                    z += 1
                    list_class.append(re['classes'])
                print(results)
                occerance = []
                for i in range(7):
                    occerance.append(list_class.count(i))
                classs = np.argmax(occerance)
                print(occerance)
                result["prediction"][unique] = classs
                max = 0
                max_id = 0
                for i in results:
                    print(i)
                    if results[i]["classes"] == classs:
                        if max < results[i]["probabilities"][classs]:
                            max = results[i]["probabilities"][classs]
                            max_id = i

                for i in results:
                    if i == max_id:
                        result["dense_layer"][unique] = results[i]["dense_layer"]
            print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            return result



a=EmotionSubsystem()
s=a('handShake_0029.avi','handShake_0029.avi.csv','handShake_0029.json',"/tmp/mnist_convnet_model")
print(np.shape(s["dense_layer"]["3"]))