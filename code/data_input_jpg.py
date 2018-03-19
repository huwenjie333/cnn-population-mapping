import tensorflow as tf
import numpy as np
import random
import gdal,re

def input_batches(batch_size, output_size, input_files, input_labels, input_ids, out_height, out_width, out_channel, nl_jpg=None,
                  regression=False, augmentation=True, normalization=True):
    '''
     Args:
        batch_size (int): number of images per batch.
        output_size (int): number of classes for classification, 1 for regression.
        input files: np.array shaped (:,K), all input image paths, could be K image paths in each example
        input_labels: np.array shaped (:,), all input image labels
        input_ids: np.array shaped (:,), all input village_id.
        out_{height/width/channel} (int): ouptut shape of image
        nl_jpg: np.array shaped (:,), all input nightlight image jpg path, with 5x5x1 size
        regression (bool): whether the training is regression
        augmentation (bool): whether to perform input image augmentation (random cropp and flip)
        normalization (bool): wether to use tf.image.per_image_standardization function
        
    Return:
        images_batch: tf.tensor (B,W,H,C)
        images_batch: tf.tensor (B,25)
        labels_batch: tf.tensor (B,output_size)
    '''
    images_per_label = input_files.shape[1]
#     dataset = tf.contrib.data.TFRecordDataset([TFrecord_path])

    if nl_jpg is None:
        dataset = tf.contrib.data.Dataset.from_tensor_slices((input_files, input_labels, input_ids))

        dataset = dataset.map(lambda files, labels, ids: parse_filename(
            files, labels, ids, images_per_label, out_height, out_width))

        dataset = dataset.map(lambda img, label, ids: resize_augmentation_ops(
            img, label, ids, output_size, out_height, out_width, out_channel, regression, augmentation, normalization))
    else:
        dataset = tf.contrib.data.Dataset.from_tensor_slices((input_files, input_labels, nl_jpg, input_ids))

        dataset = dataset.map(lambda files, labels, nl_jpg, ids: parse_filename(
            files, labels, ids, images_per_label, out_height, out_width, nl_jpg=nl_jpg))

        dataset = dataset.map(lambda img, label, nl_vector, ids: resize_augmentation_ops(
            img, label, ids, output_size, out_height, out_width, out_channel, 
            regression, augmentation, normalization, nl_vector=nl_vector))
    

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    
    iterator = dataset.make_one_shot_iterator()
    
    if nl_jpg is None:
        images_batch, labels_batch, ids_batch = iterator.get_next()
        return images_batch, labels_batch, ids_batch
    else:
        images_batch, labels_batch, nl_batch, ids_batch = iterator.get_next()
        return images_batch, nl_batch, labels_batch, ids_batch


def parse_filename(filename, label, ids, images_per_label, out_height, out_width, nl_jpg=None):
    image_list = []
    for i in range(images_per_label):
        image_string = tf.read_file(filename[i])
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.cast(tf.image.resize_images(image_decoded, [out_height, out_width]), tf.uint8)
        image_list.append(image_resized)
    image = tf.concat(values=image_list, axis=2)
    
    if nl_jpg is None:
        return image, label, ids
    else:
        image_string = tf.read_file(nl_jpg)
        image_decoded = tf.image.decode_jpeg(image_string)
        nl_vector = tf.reshape(image_decoded, [-1])
        return image, label, nl_vector, ids
        
    

def resize_augmentation_ops(single_image, label, ids, output_size, out_height, out_width, out_channel, 
                            regression, augmentation, normalization, nl_vector=None):
    
    single_image.set_shape([None, None, None])
    if not regression:
        label = tf.cast(tf.one_hot(label, output_size), tf.int64)
    
#     single_image = tf.cast(tf.image.resize_images(img, [out_height, out_width]), tf.uint8)

    # Data Augmentation
    if augmentation:
        single_image = tf.image.resize_image_with_crop_or_pad(single_image, int(1.15*out_height), int(1.15*out_width))
        single_image = tf.random_crop(single_image, [out_height, out_width, out_channel])
        single_image = tf.image.random_flip_left_right(single_image)
    
    if normalization:
        single_image = tf.image.per_image_standardization(single_image)
    if nl_vector is None:
        return single_image, label, ids
    else:
        return single_image, label, nl_vector, ids

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
