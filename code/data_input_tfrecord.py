import tensorflow as tf
import numpy as np
import random
import gdal,re

def input_batches(batch_size, output_size, TFrecord_path, out_height, out_width, 
                  data='l8', regression=False, augmentation=True, normalization=True):
    
    dataset = tf.contrib.data.TFRecordDataset([TFrecord_path])
    
    if regression:
        if data == 'l8':
            dataset = dataset.map(parse_l8_regression)
        if data == 's1':
            dataset = dataset.map(parse_s1_regression)
    else:
        dataset = dataset.map(parse_classification)

    dataset = dataset.map(lambda img, label: resize_augmentation_ops(
        img, label, output_size, out_height, out_width, regression, augmentation, normalization))
    

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch = iterator.get_next()
    
    return images_batch, labels_batch

def parse_l8_regression(example_proto):
    feature_dict = {
        'image_l8': tf.FixedLenFeature([], tf.string),
        'density_log2': tf.FixedLenFeature([], tf.float32)
    }
    parsed_features = tf.parse_single_example(example_proto, features=feature_dict)
    label = parsed_features['density_log2']
    rgb_image = tf.decode_raw(parsed_features['image_l8'], tf.uint8)
    rgb_image = tf.reshape(rgb_image, [150, 150, 3]) # we have to define the shape
    return rgb_image, label 

def parse_s1_regression(example_proto):
    feature_dict = {
        'image_s1': tf.FixedLenFeature([], tf.string),
        'density_log2': tf.FixedLenFeature([], tf.float32)
    }
    parsed_features = tf.parse_single_example(example_proto, features=feature_dict)
    label = parsed_features['density_log2']
    rgb_image = tf.decode_raw(parsed_features['image_s1'], tf.uint8)
    rgb_image = tf.reshape(rgb_image, [450, 450, 3]) # we have to define the shape
    return rgb_image, label 

def parse_classification(example_proto):
    feature_dict = {
        'image_l8': tf.FixedLenFeature([], tf.string),
        'density_class': tf.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.parse_single_example(example_proto, features=feature_dict)
    label = parsed_features['density_class']
    rgb_image = tf.decode_raw(parsed_features['image_l8'], tf.uint8)
    rgb_image = tf.reshape(rgb_image, [500, 500, 3]) # we have to define the shape
    return rgb_image, label 
    

def resize_augmentation_ops(img, label, output_size, out_height, out_width, regression, augmentation, normalization):
    
    img.set_shape([None, None, None])
    if not regression:
        label = tf.cast(tf.one_hot(label, output_size), tf.int64)
    
    single_image = tf.cast(tf.image.resize_images(img, [out_height, out_width]), tf.uint8)

    # Data Augmentation
    if augmentation:
        single_image = tf.image.resize_image_with_crop_or_pad(single_image, int(1.15*out_height), int(1.15*out_width))
        single_image = tf.random_crop(single_image, [out_height, out_width, 3])
        single_image = tf.image.random_flip_left_right(single_image)
    
    if normalization:
        single_image = tf.image.per_image_standardization(single_image)
    
    return single_image, label

def create_sample_data(num_files):
    """
    This function creates input_files and input_classes for testing.
    The same tif file is used for all samples.
    """
    filename = "/home/timhu/test_tif/l8_median_india_vis_500x500_402382.0.tif"
    possible_classes = list(range(16))
    
    input_files = np.empty((num_files,), dtype=object)
    input_labels = np.zeros((num_files,), dtype=np.int64)
    
    for f in range(num_files):
        input_files[f] = filename
        input_labels[f] = random.choice(possible_classes)
        
    return input_files, input_labels

if __name__ == '__main__':
    # testing code
    input_files, input_labels = create_sample_data(50000)
    
    tf.reset_default_graph()
    sess = tf.Session()
    batch_size = 128

    with tf.device('/cpu:0'):
        images_batch, labels_batch = input_batches(batch_size, input_files, input_labels, \
                                                                            500, 500, \
                                                                            augmentation=False, shuffle=True)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print("BATCH 1:")
    curr_image_batch, curr_label_batch = sess.run([images_batch, labels_batch])
    #print(curr_image_batch)
    print(curr_label_batch)
    
    print("BATCH 2:")
    curr_image_batch, curr_label_batch = sess.run([images_batch, labels_batch])
    #print(curr_image_batch)
    print(curr_label_batch)
