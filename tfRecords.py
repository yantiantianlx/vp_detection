import os
import tensorflow as tf
# from PIL import Image
import numpy as np

class TFRecords_Reader(object):
    def __init__(self,num_examples):
        self.__num_examples = num_examples

    def write_records(self,img_dir=None,points_dir=None,records_name=None):
        points_txt = open(points_dir, 'r')
        writer = tf.python_io.TFRecordWriter(records_name)
        for i in range(self.__num_examples):
            j=int(points_txt.readline())
            x=float(points_txt.readline())
            y=float(points_txt.readline())

            example = tf.train.Example(features=tf.train.Features(feature={
                "point_x" : tf.train.Feature(float_list=tf.train.FloatList(value=[x])),
                "point_y": tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
                "index": tf.train.Feature(float_list=tf.train.FloatList(value=[j])),
                #"img_row" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
        writer.close()

    def readbatch_by_queue(self,records_name=None,batch_size=None,num_epoch=None):
        filename_queue = tf.train.string_input_producer([records_name],num_epoch)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               "point_x" : tf.FixedLenFeature([],tf.float32),
                                               "point_y": tf.FixedLenFeature([], tf.float32),
                                               "index": tf.FixedLenFeature([], tf.float32),
                                               # "img_row" : tf.FixedLenFeature([],tf.string)
                                           })
        # img = tf.decode_raw(features["img_row"],tf.uint8)
        # img = tf.reshape(img,[224,224,3])
        # img = tf.cast(img, tf.float32)
        points_x = features["point_x"]
        points_y = features["point_y"]
        index = features["index"]
        min_after_dequeue = np.mod(self.__num_examples, batch_size)
        x_batch,y_batch,i_batch = tf.train.shuffle_batch(
            [points_x, points_y,index], batch_size=batch_size,
            capacity=self.__num_examples, min_after_dequeue=min_after_dequeue)
        x_batch = tf.expand_dims(x_batch,1)
        y_batch = tf.expand_dims(y_batch,1)
        return i_batch


