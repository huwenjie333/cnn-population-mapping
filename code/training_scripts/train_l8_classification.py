from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os,sys,time,re
import numpy as np
import pandas as pd
# np.set_printoptions(threshold=np.inf)
sys.path.insert(0,'..')

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.slim.python.slim.nets import vgg
from tensorflow.contrib.slim.python.slim.nets import resnet_v1

import tensorflow.contrib.slim as slim

import data_input_jpg as dataset


#################################### Traning Parameters for model tuning #################################################
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# log directory to store trained checkpoints and tensorboard summary
# LOG_DIR = '/home/timhu/dfd-pop/logs/regression_l8s1_state24_lr-6_drop08_vgg_Mar7_test'
# LOG_DIR = '/home/timhu/logs/regression_l8s1_inputcombo_state24_lr-5_decay-1_wd5e-3_drop08_vgg_Jul26'
# LOG_DIR = '/home/timhu/logs/regression_l8_state24_lr-5_decay-1_wd5e-3_drop08_vgg_Jul26'
LOG_DIR = '/home/timhu/logs/classification_l8_allstate_lr-5_decay-1_wd5e-3_drop08_vgg_Sep1'


# Basic model parameters as external flags.
FLAGS = argparse.Namespace(learning_rate= 1e-5,
                           lr_decay_rate = 1e-1, # exponential learning rate decay 
                           weight_decay=5e-3, 
                           dropout_keep= 0.8, 
                           max_epoch = 40, # maximum number of epoch
                           batch_size= 48, 
                           output_size = 12) # class number = 1 for regression output

# CNN architecture and weights
MODEL = 'vgg' # VGG 16
PRETRAIN_WEIGHTS = '/home/timhu/weights/vgg_16.ckpt'
# MODEL = 'resnet' # Resnet V1 152
# PRETRAIN_WEIGHTS = '/home/timhu/dfd-pop/weights/resnet_v1_152.ckpt'

# input image dimensions, used for image preprocessing before CNN model
IMAGE_HEIGHT = 224 
IMAGE_WIDTH = 224 

# input traning data
#'/home/timhu/data/all_jpgpaths_clean_538k_May17.csv' # state24_jpgpaths_clean_17k_May17.csv
ANNOS_CSV = '/home/timhu/data/all_jpgpaths_classification_clean_538k_Sep1.csv'  
JPG_DIR = '/home/timhu/data/all_jpg/' 
DATA = 'l8' # l8,s1,l8s1
IMAGE_CHANNEL = 3 # 3 if l8 or s1,6 if l8+s1

################################# ### Traning script (no need to change) #################################################

def main(_):
    if tf.gfile.Exists(LOG_DIR): # whether the path exits
        raise Exception('LOG_DIR already exits:', LOG_DIR)
    else:
        tf.gfile.MakeDirs(LOG_DIR) # mkdir
        run_training()

def run_training():
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.Session(config = config)
#     sess = tf.Session() # config=tf.ConfigProto(log_device_placement=True)) 
    
    # create input path and labels np.array from csv annotations
    df_annos = pd.read_csv(ANNOS_CSV, index_col=0)
    df_annos = df_annos.sample(frac=1).reset_index(drop=True) # shuffle the whole datasets
    if DATA == 'l8':
        path_col = ['l8_vis_jpg']
    elif DATA == 's1':
        path_col = ['s1_vis_jpg']
    elif DATA == 'l8s1':
        path_col = ['l8_vis_jpg', 's1_vis_jpg']

    input_files_train = JPG_DIR + df_annos.loc[df_annos.partition == 'train', path_col].values
    input_labels_train = df_annos.loc[df_annos.partition == 'train', 'pop_density_log2_class'].values
    input_files_val = JPG_DIR + df_annos.loc[df_annos.partition == 'val', path_col].values
    input_labels_val = df_annos.loc[df_annos.partition == 'val', 'pop_density_log2_class'].values
    input_id_train = df_annos.loc[df_annos.partition == 'train', 'village_id'].values
    input_id_val = df_annos.loc[df_annos.partition == 'val', 'village_id'].values
 
    
    print('input_files_train shape:', input_files_train.shape)
    train_set_size = len(input_labels_train)
    
    # data input 
    with tf.device('/cpu:0'):
        train_images_batch, train_labels_batch, _ = \
        dataset.input_batches(FLAGS.batch_size, FLAGS.output_size, input_files_train, input_labels_train, input_id_train,
                              IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL, regression=False, augmentation=True, normalization=True)
        val_images_batch, val_labels_batch, _ = \
        dataset.input_batches(FLAGS.batch_size, FLAGS.output_size, input_files_val, input_labels_val, input_id_val,
                              IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL, regression=False, augmentation=False, normalization=True)

    images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL]) 
    labels_placeholder = tf.placeholder(tf.float32, shape=[None, FLAGS.output_size])
    print('finish data input')

    TRAIN_BATCHES_PER_EPOCH = int(train_set_size / FLAGS.batch_size) # number of training batches/steps in each epoch
    MAX_STEPS = TRAIN_BATCHES_PER_EPOCH * FLAGS.max_epoch # total number of training batches/steps

    # CNN forward reference
    if MODEL == 'vgg':
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=FLAGS.weight_decay)):
            outputs, _ = vgg.vgg_16(images_placeholder, num_classes=FLAGS.output_size, 
                                   dropout_keep_prob=FLAGS.dropout_keep, is_training=True)
#             outputs = tf.squeeze(outputs) # change shape from (B,1) to (B,), same as label input
#     if MODEL == 'resnet':
#         with slim.arg_scope(resnet_v1.resnet_arg_scope()):
#             outputs, _ = resnet_v1.resnet_v1_152(images_placeholder, num_classes=FLAGS.output_size, is_training=True)
#             outputs = tf.squeeze(outputs) # change shape from (B,1) to (B,), same as label input


    # loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=labels_placeholder, logits=outputs, name='softmax_loss_batch')
    loss = tf.reduce_mean(cross_entropy, name='softmax_loss_mean')
    tf.summary.scalar('loss', loss) 


    # accuracy (R2)
    correct = tf.equal(tf.argmax(outputs, 1), tf.argmax(labels_placeholder, 1))
    correct = tf.cast(correct, tf.float32)
    accuracy = tf.reduce_mean(correct)
    tf.summary.scalar('accuracy', accuracy)

    
    # determine the model vairables to restore from pre-trained checkpoint
    if MODEL == 'vgg':
        if DATA == 'l8s1':
            model_variables = slim.get_variables_to_restore(exclude=['vgg_16/fc8', 'vgg_16/conv1']) 
        else:
            model_variables = slim.get_variables_to_restore(exclude=['vgg_16/fc8']) 
    if MODEL == 'resnet':
        model_variables = slim.get_variables_to_restore(exclude=['resnet_v1_152/logits', 'resnet_v1_152/conv1']) 


    # training step and learning rate
    global_step = tf.Variable(0, name='global_step', trainable=False) #, dtype=tf.int64)
    learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate, # initial learning rate
        global_step = global_step, # current step
        decay_steps = MAX_STEPS, # total numbers step to decay 
        decay_rate = FLAGS.lr_decay_rate) # final learning rate = FLAGS.learning_rate * decay_rate
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#     optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)


    # to only update gradient in first and last layer
#     vars_update = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vgg_16/(conv1|fc8)')
#     print('variables to update in traing: ', vars_update)

    train_op = optimizer.minimize(loss, global_step=global_step) #, var_list = vars_update)

    # summary output in tensorboard
    summary = tf.summary.merge_all()
    summary_writer_train = tf.summary.FileWriter(os.path.join(LOG_DIR, 'log_train'), sess.graph)
    summary_writer_val = tf.summary.FileWriter(os.path.join(LOG_DIR, 'log_val'), sess.graph)

    # variable initialize
    init = tf.global_variables_initializer()
    sess.run(init)

    # restore the model from pre-trained checkpoint
    restorer = tf.train.Saver(model_variables)
    restorer.restore(sess, PRETRAIN_WEIGHTS)
    print('loaded pre-trained weights: ', PRETRAIN_WEIGHTS)

    # saver object to save checkpoint during training
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)


    print('start training...')
    epoch = 0  
    for step in xrange(MAX_STEPS):
        if step % TRAIN_BATCHES_PER_EPOCH == 0:
            epoch += 1

        start_time = time.time() # record the time used for each batch 

        images_out, labels_out = sess.run([train_images_batch, train_labels_batch]) # inputs of this batch, numpy array format
        
        duration_batch = time.time() - start_time
 
        if step == 0:
            print("finished reading batch data")
            print("images_out shape:", images_out.shape)
        feed_dict = {images_placeholder: images_out, labels_placeholder: labels_out}
        _, train_loss, train_accuracy, train_outputs, lr = \
            sess.run([train_op, loss, accuracy, outputs, learning_rate], feed_dict=feed_dict)

        duration = time.time() - start_time

        if step % 10 == 0 or (step + 1) == MAX_STEPS: # print traing loss every 10 batches 
            print('Step %d epoch %d lr %.3e: train loss = %.4f train accuracy = %.4f (%.3f sec, %.3f sec(each batch))' \
                  % (step, epoch, lr, train_loss, train_accuracy, duration*10, duration_batch))
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer_train.add_summary(summary_str, step)
            summary_writer_train.flush()

        if step % 50 == 0 or (step + 1) == MAX_STEPS: # calculate and print validation loss every 50 batches 
            images_out, labels_out = sess.run([val_images_batch, val_labels_batch])
            feed_dict = {images_placeholder: images_out, labels_placeholder: labels_out}

            val_loss, val_accuracy = sess.run([loss, accuracy], feed_dict=feed_dict)
            print('Step %d epoch %d: val loss = %.4f val accuracy = %.4f ' % (step, epoch, val_loss, val_accuracy))

            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer_val.add_summary(summary_str, step)
            summary_writer_val.flush()
            
        # Save a checkpoint every epoch
        if step % TRAIN_BATCHES_PER_EPOCH == 0 or (step + 1) == MAX_STEPS:
            checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=step, write_state=True)
    
if __name__ == '__main__':
    tf.app.run(main=main)
