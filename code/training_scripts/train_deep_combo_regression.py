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

# from tensorflow.contrib.slim.python.slim.nets import vgg
# from tensorflow.contrib.slim.python.slim.nets import resnet_v1
import vgg_deep_combo as vgg

import tensorflow.contrib.slim as slim

import data_input_jpg as dataset


#################################### Traning Parameters for model tuning #################################################
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# log directory to store trained checkpoints and tensorboard summary
# LOG_DIR = '/home/timhu/dfd-pop/logs/regression_l8s1_deepcombo_state24_lr-5_decay-1_wd5e-3_drop07_vgg_Mar12'
LOG_DIR = '/home/timhu/logs/regression_l8s1_deepcombo_allstate_lr-5_decay-1_wd5e-3_drop08_vgg_Jul28'

# Basic model parameters as external flags.
FLAGS = argparse.Namespace(learning_rate= 1e-5,
                           lr_decay_rate = 1e-1, # exponential learning rate decay
                           weight_decay=5e-3, 
                           dropout_keep= 0.8, 
                           max_epoch = 40, # maximum number of epoch
                           batch_size= 48, 
                           output_size = 1) # class number = 1 for regression output

# CNN architecture and weights
MODEL = 'vgg' # VGG 16
PRETRAIN_WEIGHTS = '/home/timhu/weights/vgg_16.ckpt'
# MODEL = 'resnet' # Resnet V1 152
# PRETRAIN_WEIGHTS = '/home/timhu/dfd-pop/weights/resnet_v1_152.ckpt'

# input image dimensions, used for image preprocessing before CNN model
IMAGE_HEIGHT = 224 
IMAGE_WIDTH = 224 

# input traning data
# ANNOS_CSV = '/home/timhu/dfd-pop/data/annos_csv/state24_jpgpaths_density_nolaps_12k_Mar6.csv'
ANNOS_CSV = '/home/timhu/data/all_jpgpaths_clean_538k_May17.csv'  # state24_jpgpaths_clean_17k_May17.csv
JPG_DIR = '/home/timhu/data/all_jpg/'
DATA = 'l8s1' #l8, s1, l8s1, l8s1nl
IMAGE_CHANNEL = 6 # 3,3,6,6


################################# ### Traning script (no need to change) #################################################

def main(_):
    if tf.gfile.Exists(LOG_DIR): # whether the path exits
        raise Exception('LOG_DIR already exits:', LOG_DIR)
    else:
        tf.gfile.MakeDirs(LOG_DIR) # mkdir
        run_training()

def run_training():
    sess = tf.Session() # config=tf.ConfigProto(log_device_placement=True)) 
    
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
    input_labels_train = df_annos.loc[df_annos.partition == 'train', 'pop_density_log2'].values
    input_files_val = JPG_DIR + df_annos.loc[df_annos.partition == 'val', path_col].values
    input_labels_val = df_annos.loc[df_annos.partition == 'val', 'pop_density_log2'].values
    input_id_train = df_annos.loc[df_annos.partition == 'train', 'village_id'].values
    input_id_val = df_annos.loc[df_annos.partition == 'val', 'village_id'].values
    
    print('input_files_train shape:', input_files_train.shape)
    train_set_size = len(input_labels_train)
    
    # data input 
    with tf.device('/cpu:0'):
        train_images_batch, train_labels_batch, _ = \
        dataset.input_batches(FLAGS.batch_size, FLAGS.output_size, input_files_train, input_labels_train, input_id_train,
                              IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL, regression=True, augmentation=True, normalization=True)
        val_images_batch, val_labels_batch, _ = \
        dataset.input_batches(FLAGS.batch_size, FLAGS.output_size, input_files_val, input_labels_val, input_id_val,
                              IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL, regression=True, augmentation=False, normalization=True)


    images_l8_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3]) 
    images_s1_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    labels_placeholder = tf.placeholder(tf.float32, shape=[None,])
    print('finish data input')

    TRAIN_BATCHES_PER_EPOCH = int(train_set_size / FLAGS.batch_size) # number of training batches/steps in each epoch
    MAX_STEPS = TRAIN_BATCHES_PER_EPOCH * FLAGS.max_epoch # total number of training batches/steps

    # CNN forward reference
    if MODEL == 'vgg':
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=FLAGS.weight_decay)):
            outputs, _ = vgg.vgg_16(images_l8_placeholder, images_s1_placeholder, num_classes=FLAGS.output_size, 
                                   dropout_keep_prob=FLAGS.dropout_keep, is_training=True)
        outputs = tf.squeeze(outputs) # change shape from (B,1) to (B,), same as label input
    if MODEL == 'resnet':
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            outputs, _ = resnet_v1.resnet_v1_152(images_placeholder, num_classes=FLAGS.output_size, is_training=True)
            outputs = tf.squeeze(outputs) # change shape from (B,1) to (B,), same as label input
  

    # loss
    labels_real = tf.pow(2.0, labels_placeholder) 
    outputs_real = tf.pow(2.0, outputs)
    
    # only loss_log2_mse are used for gradient calculate, model minimize this value
    loss_log2_mse = tf.reduce_mean(tf.squared_difference(labels_placeholder, outputs), name='loss_log2_mse')   
    loss_real_rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(labels_real, outputs_real)), name='loss_real_rmse')
    loss_real_mae = tf.losses.absolute_difference(labels_real, outputs_real)
    
    tf.summary.scalar('loss_log2_mse', loss_log2_mse) 
    tf.summary.scalar('loss_real_rmse', loss_real_rmse) 
    tf.summary.scalar('loss_real_mae', loss_real_mae)     
    
    # accuracy (R2)
    def r_sqaured(labels, outputs):
        sst = tf.reduce_sum(tf.squared_difference(labels, tf.reduce_mean(labels)))
        sse = tf.reduce_sum(tf.squared_difference(labels, outputs))
        return (1.0 - tf.div(sse, sst))
    
    r2_log2 = r_sqaured(labels_placeholder, outputs)
    r2_real = r_sqaured(labels_real, outputs_real)
    
    tf.summary.scalar('r2_log2', r2_log2)
    tf.summary.scalar('r2_real', r2_real)


    # determine the model vairables to restore from pre-trained checkpoint
    if MODEL == 'vgg':
        model_variables = slim.get_variables_to_restore(exclude=[
            'vgg_16/fc5','vgg_16/fc7/dim_reduce','vgg_16/fc8','vgg_16/l8', 'vgg_16/s1']) 
    if MODEL == 'resnet':
        model_variables = slim.get_variables_to_restore(exclude=['resnet_v1_152/logits']) #, 'resnet_v1_152/conv1']) 


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

    train_op = optimizer.minimize(loss_log2_mse, global_step=global_step) #, var_list = vars_update)

    # summary output in tensorboard
    summary = tf.summary.merge_all()
    summary_writer_train = tf.summary.FileWriter(os.path.join(LOG_DIR, 'log_train'), sess.graph)
    summary_writer_val = tf.summary.FileWriter(os.path.join(LOG_DIR, 'log_val'), sess.graph)
    
    # variable initialize
    init = tf.global_variables_initializer()
    sess.run(init)

    ##### restore the model from pre-trained checkpoint for new VGG archtecture #####
    
    # restore the weights for the layers that are nor modified in the new arch (excep conv1, fc8)
    restorer = tf.train.Saver(model_variables)
    restorer.restore(sess, PRETRAIN_WEIGHTS)
    print('loaded pre-trained weights: ', PRETRAIN_WEIGHTS)
    for ww in model_variables:
        print(ww)
    
    # a fake layer to hold the new variables to restore
    with tf.variable_scope("vgg_16"): 
        fake_net = slim.repeat(images_l8_placeholder, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        fake_net = slim.max_pool2d(fake_net, [2, 2], scope='pool1')
        fake_net = slim.repeat(fake_net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        fake_net = slim.max_pool2d(fake_net, [2, 2], scope='pool2')
        fake_net = slim.repeat(fake_net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        fake_net = slim.max_pool2d(fake_net, [2, 2], scope='pool3')
        fake_net = slim.repeat(fake_net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        fake_net = slim.max_pool2d(fake_net, [2, 2], scope='pool4')
        fake_net = slim.repeat(fake_net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        fake_net = slim.max_pool2d(fake_net, [2, 2], scope='pool5')
    
    # print out the vairables in fake layer
    dup_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vgg_16/conv[1-5]/')
    
    # restore the vairables of fake layer with checkpoint weights
    restorer = tf.train.Saver(dup_weights)
    restorer.restore(sess, PRETRAIN_WEIGHTS)
    var_list = []
    print('loaded pre-trained weights: ', PRETRAIN_WEIGHTS)
#     print('restore duplicated weights to update: ')
    for ww in dup_weights:
        var_list.append(ww.name.replace('vgg_16/',''))
#     for v in var_list:
#         print(v)
    
#     # get the vairables for the fakes layer 
#     with tf.variable_scope("vgg_16/conv1", reuse=True):
#         weights1 = tf.get_variable("conv1_1/weights")
#         bias1 = tf.get_variable("conv1_1/biases")
#         weights2 = tf.get_variable("conv1_2/weights")
#         bias2 = tf.get_variable("conv1_2/biases")
    
#     l8_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'vgg_16/l8/')
#     print('new l8 weights to update: ')
#     for ww in l8_weights:
#         print(ww.name)
        
    assign_ops = []
    # assign the weights of fake layer to true model vairables 
    with tf.variable_scope("vgg_16", reuse=True):
        for v in var_list:
            v = v.replace(':0','')
            with tf.variable_scope("l8", reuse=True):
                new_var = tf.get_variable(v)
            old_var = tf.get_variable(v)
            assign_ops.append(tf.assign(new_var, old_var))
            print(new_var.name, '<==', old_var.name)
            with tf.variable_scope("s1", reuse=True):
                new_var = tf.get_variable(v)
            assign_ops.append(tf.assign(new_var, old_var))
            print(new_var.name, '<==', old_var.name)
            

    
    sess.run([assign_ops])
    ###########################################################################
    
    # saver object to save checkpoint during training
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)


    print('start training...')
    epoch = 0
    best_r2 = -float('inf')
    for step in xrange(MAX_STEPS):
        if step % TRAIN_BATCHES_PER_EPOCH == 0:
            epoch += 1

        start_time = time.time() # record the time used for each batch 

        images_out, labels_out = sess.run([train_images_batch, train_labels_batch]) # inputs of this batch, numpy array format
        
        duration_batch = time.time() - start_time
 
        if step == 0:
            print("finished reading batch data")
            print("images_out shape:", images_out.shape)
        feed_dict = {images_l8_placeholder: images_out[:,:,:,:3],
                     images_s1_placeholder: images_out[:,:,:,3:],
                     labels_placeholder: labels_out}
        
        _, train_loss, train_accuracy, train_outputs, lr = \
            sess.run([train_op, loss_log2_mse, r2_log2, outputs, learning_rate], feed_dict=feed_dict)

        duration = time.time() - start_time

        if step % 10 == 0 or (step + 1) == MAX_STEPS: # print traing loss every 10 batches 
            print('Step %d epoch %d lr %.3e: log2 MSE loss = %.4f log2 R2 = %.4f (%.3f sec, %.3f sec(each batch))' \
                  % (step, epoch, lr, train_loss, train_accuracy, duration*10, duration_batch))
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer_train.add_summary(summary_str, step)
            summary_writer_train.flush()

        if step % 50 == 0 or (step + 1) == MAX_STEPS: # calculate and print validation loss every 50 batches 
            images_out, labels_out = sess.run([val_images_batch, val_labels_batch])
            feed_dict = {images_l8_placeholder: images_out[:,:,:,:3],
                         images_s1_placeholder: images_out[:,:,:,3:],
                         labels_placeholder: labels_out}

            val_loss, val_accuracy = sess.run([loss_log2_mse, r2_log2], feed_dict=feed_dict)
            print('Step %d epoch %d: val log2 MSE = %.4f val log2 R2 = %.4f ' % (step, epoch, val_loss, val_accuracy))

            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer_val.add_summary(summary_str, step)
            summary_writer_val.flush()
            
            # in each epoch, if the validation R2 is higher than best R2, save the checkpoint
            if step % (TRAIN_BATCHES_PER_EPOCH - TRAIN_BATCHES_PER_EPOCH % 50) == 0:
                if val_accuracy > best_r2:
                    best_r2 = val_accuracy
                    checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step=step, write_state=True)

#         # Save a checkpoint every 3 epoch
#         if step % (TRAIN_BATCHES_PER_EPOCH * 3) == 0 or (step + 1) == MAX_STEPS:
#             checkpoint_file = os.path.join(LOG_DIR, 'model.ckpt')
#             saver.save(sess, checkpoint_file, global_step=step, write_state=True)
    
if __name__ == '__main__':
    tf.app.run(main=main)
