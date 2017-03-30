import tensorflow as tf
from tfRecords import TFRecords_Reader
import numpy as np
from PIL import Image,ImageDraw
import os

sess = tf.InteractiveSession()

NUM_EXAMPLE = 5548
BATCH_SIZE = 50
NUM_EPOCH = 10
SAVER_STEP = (NUM_EXAMPLE * NUM_EPOCH) / BATCH_SIZE / 2

isTrain = True
checkpoint_dir = './model/'

img_dir = 'VP_224/'
points_dir =  'vanishingpoint_record.txt'
records_name =  'train.tfrecords'

a = TFRecords_Reader(NUM_EXAMPLE)
#a.write_records(img_dir,points_dir,records_name)
image,_x,_y,_index=a.readbatch_by_queue(records_name,batch_size=BATCH_SIZE,num_epoch=NUM_EPOCH)

def computLoss(x_y,x,y):
    imx = tf.transpose(x_y)[0]
    imx = tf.expand_dims(imx, 1)
    imy = tf.transpose(x_y)[1]
    imy = tf.expand_dims(imy, 1)
    print(imx.shape)
    print(imy.shape)
    derta_x = imx - x
    derta_y = imy - y
    derta_x_abs = tf.pow(derta_x,2)
    derta_y_abs = tf.pow(derta_y,2)
    derta_x2_sum = tf.reduce_mean(derta_x_abs,reduction_indices=0)
    derta_y2_sum = tf.reduce_mean(derta_y_abs,reduction_indices=0)
    loss = tf.sqrt(derta_x2_sum + derta_y2_sum)
    return loss,derta_x,derta_y


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

im = tf.placeholder(tf.float32, shape=[None, 224,224,3])/255 #[10,224,224,3]
x = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.placeholder(tf.float32, shape=[None, 1])
#i_s = tf.placeholder(tf.float32, shape=[1])

keep_prob = tf.placeholder(tf.float32)


## conv1 layer ##
W_conv1 = tf.Variable(tf.truncated_normal([11,11, 3,96], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[96]))
conv1 = tf.nn.conv2d(im, W_conv1, strides=[1, 4, 4, 1], padding='SAME')
h_conv1 = tf.nn.relu(conv1 + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

## conv2 layer ##
W_conv2 = tf.Variable(tf.truncated_normal([5,5,96,256], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[256]))
conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
h_conv2 = tf.nn.relu(conv2 + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

## conv3 layer ##
W_conv3 = tf.Variable(tf.truncated_normal([3,3,256,384], stddev=0.1))
b_conv3 = tf.Variable(tf.constant(0.1, shape=[384]))
conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
h_conv3 = tf.nn.relu(conv3 + b_conv3)

## conv4 layer ##
W_conv4 = tf.Variable(tf.truncated_normal([3,3,384,384], stddev=0.1))
b_conv4 = tf.Variable(tf.constant(0.1, shape=[384]))
conv4 = tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME')
h_conv4 = tf.nn.relu(conv4 + b_conv4)

## conv5 layer ##
W_conv5 = tf.Variable(tf.truncated_normal([3,3,384,256], stddev=0.1))
b_conv5 = tf.Variable(tf.constant(0.1, shape=[256]))
conv5 = tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='SAME')
h_conv5 = tf.nn.relu(conv5 + b_conv5)
h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

# [n_samples, 6, 6, 256] ->> [n_samples, 9216]
h_pool5_flat = tf.reshape(h_pool5, [-1, 9216])

## fc6 layer ##
W_fc6 = tf.Variable(tf.truncated_normal([9216,4096], stddev=0.1))
b_fc6 = tf.Variable(tf.constant(0.1, shape=[4096]))
h_fc6 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc6) + b_fc6)
h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob)

## fc7 layer ##
W_fc7 = tf.Variable(tf.truncated_normal([4096,4096], stddev=0.1))
b_fc7 = tf.Variable(tf.constant(0.1, shape=[4096]))
h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop, W_fc7) + b_fc7)
h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

## fc8 layer ##
W_fc8 = tf.Variable(tf.truncated_normal([4096,2], stddev=0.1))
b_fc8 = tf.Variable(tf.constant(0.1, shape=[2]))
x_y = tf.matmul(h_fc7_drop, W_fc8) + b_fc8

loss_L2 = computLoss(x_y,x,y)[0]

#learning_rate = tf.train.exponential_decay(1,10 * i_s,1000,0.95,staircase=True)
train_step = tf.train.AdamOptimizer(0.1).minimize(loss_L2)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

    #show W and B
    print('W_conv1',sess.run(W_conv1,
                feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
    print('b_conv1', sess.run(W_conv1,
                feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
    print('W_conv2', sess.run(W_conv1,
                feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
    print('b_conv2', sess.run(W_conv1,
                feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
    print('W_conv3', sess.run(W_conv1,
                feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
    print('b_conv3', sess.run(W_conv1,
                feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
    print('W_conv4', sess.run(W_conv1,
                 feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
    print('b_conv4', sess.run(W_conv1,
                 feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
    print('W_conv5', sess.run(W_conv1,
                 feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
    print('b_conv5', sess.run(W_conv1,
                 feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
    print('W_fc6', sess.run(W_conv1,
                 feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
    print('b_fc6', sess.run(W_conv1,
                 feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
    print('W_fc7', sess.run(W_conv1,
                 feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
    print('b_fc7', sess.run(W_conv1,
                 feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
    print('W_fc8', sess.run(W_conv1,
                 feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
    print('b_fc8', sess.run(W_conv1,
                 feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
    #show image
    # j = 0
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # try:
    #     while not coord.should_stop():
    #         img_batch, x_batch, y_batch, i = sess.run([image, _x, _y, _index])
    #
    #         xy = sess.run(x_y, feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1})
    #         xy_pre = np.array(np.multiply(xy, 224), np.uint16)
    #         x_batch = np.array(np.multiply(x_batch, 224), np.uint16)
    #         y_batch = np.array(np.multiply(y_batch, 224), np.uint16)
    #         print(xy_pre.size)
    #         for k in range(BATCH_SIZE):
    #             x_pre = xy_pre[k][0]
    #             y_pre = xy_pre[k][1]
    #             x_real = x_batch[k][0]
    #             y_real = y_batch[k][0]
    #             one_img = np.array(img_batch[k], np.uint8)
    #             one_img = Image.fromarray(one_img)
    #             draw = ImageDraw.Draw(one_img)
    #             draw.ellipse((x_real, y_real, x_real + 5, y_real + 5), fill='red')
    #             draw.ellipse((x_pre, y_pre, x_pre + 5, y_pre + 5), fill='blue')
    #             one_img.show()
    #             print('i=',i[k],'\tx_real=',x_real,'\ty_real=',y_real,'\tx_pre=',x_pre,'\ty_pre=',y_pre)
    #             input()
    #
    #         j = j + 1
    #         print('j=', j, '\tloss_L2=',sess.run(loss_L2, feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1}))
    #
    # except tf.errors.OutOfRangeError:
    #     print('Done training')
    # finally:
    #     coord.request_stop()
    # coord.join(threads)

else:
    pass


sess.close()