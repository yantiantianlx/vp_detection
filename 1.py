from tfRecords import TFRecords_Reader
import tensorflow as tf
import numpy as np
import os

sess = tf.InteractiveSession()


current_dir = os.getcwd()
img_dir = 'VP_224/'
points_dir =  'vanishingpoint_record.txt'
records_name =  'train_one_color.tfrecords'

a = TFRecords_Reader(40)
a.write_records(img_dir,points_dir,records_name)
_index=a.readbatch_by_queue(records_name,batch_size=7,num_epoch=2)

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
try:
     while not coord.should_stop():
       i= sess.run(_index)
       print( i)
except tf.errors.OutOfRangeError:
    print('Done reading')
finally:
    coord.request_stop()

coord.join(threads)
sess.close()