from tfRecords import TFRecords_Reader
import tensorflow as tf
import numpy as np
import os

sess = tf.InteractiveSession()


#current_dir = os.getcwd()
img_dir = 'VP_224/'
points_dir =  'vanishingpoint_record.txt'
records_name =  'train_one_color.tfrecords'

a = TFRecords_Reader(5548)
a.write_records(img_dir,points_dir,records_name)
image,_x,_y,_index=a.readbatch_by_queue(records_name,batch_size=10,num_epoch=3)

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
try:
     while not coord.should_stop():
        img,x,y,i= sess.run([image,_x,_y,_index])
        my_img = np.array(img[0], np.uint8)
        my_img = Image.fromarray(my_img)
        my_img.show()
        print('i=', i[0],'x=',x[0][0],'y=',y[0][0])
        input()
except tf.errors.OutOfRangeError:
    print('Done reading')
finally:
    coord.request_stop()

coord.join(threads)
sess.close


testbyRyantestbyRyantestbyRyantestbyRyantestbyRyantestbyRyantestbyRyantestbyRyan
