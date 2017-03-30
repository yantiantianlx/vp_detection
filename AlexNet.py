import tensorflow as tf
from tfRecords import TFRecords_Reader

sess = tf.InteractiveSession()

NUM_EXAMPLE = 5548
BATCH_SIZE = 10
NUM_EPOCH = 10
SAVER_STEP = (NUM_EXAMPLE * NUM_EPOCH) / BATCH_SIZE / 2

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
    derta_x_2 = tf.pow(derta_x,2)
    derta_y_2 = tf.pow(derta_y,2)
    derta_x2_sum = tf.reduce_mean(derta_x_2,reduction_indices=0)
    derta_y2_sum = tf.reduce_mean(derta_y_2,reduction_indices=0)
    loss = derta_x2_sum + derta_y2_sum
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
W_fc7 = tf.Variable(tf.truncated_normal([4096,512], stddev=0.1))
b_fc7 = tf.Variable(tf.constant(0.1, shape=[512]))
h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop, W_fc7) + b_fc7)
h_fc7_drop = tf.nn.dropout(h_fc7, keep_prob)

## fc8 layer ##
W_fc8 = tf.Variable(tf.truncated_normal([512,32], stddev=0.1))
b_fc8 = tf.Variable(tf.constant(0.1, shape=[32]))
h_fc8 = tf.nn.relu(tf.matmul(h_fc7_drop, W_fc8) + b_fc8)
h_fc8_drop = tf.nn.dropout(h_fc8, keep_prob)

## fc9 layer ##
W_fc9 = tf.Variable(tf.truncated_normal([32,2], stddev=0.1))
b_fc9 = tf.Variable(tf.constant(0.1, shape=[2]))
x_y = tf.matmul(h_fc8_drop, W_fc9) + b_fc9

loss_L2 = computLoss(x_y,x,y)[0]

#learning_rate = tf.train.exponential_decay(1,10 * i_s,1000,0.95,staircase=True)
train_step = tf.train.AdadeltaOptimizer(0.01).minimize(loss_L2)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


j = 0
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
try:
    while not coord.should_stop():
        img_batch, x_batch, y_batch, i = sess.run([image, _x, _y, _index])
        train_step.run(feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 0.5})
        print sess.run(x_y,feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 0.5})
        j = j + 1
        #print 'j=', j, '\tloss_L2=', sess.run(loss_L2, feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1})
        print sess.run(computLoss(x_y, x, y)[1], feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1})
        print sess.run(computLoss(x_y, x, y)[2], feed_dict={im: img_batch, x: x_batch, y: y_batch, keep_prob: 1})
        if (j + 1) % SAVER_STEP == 0:
            saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=j + 1)

except tf.errors.OutOfRangeError:
    print('Done training')
finally:
    coord.request_stop()

coord.join(threads)

sess.close()