import tensorflow as tf
import pandas as pd
import numpy as np
import random
import gdal
from PIL import Image

import os,io,time
os.environ['CUDA_VISIBLE_DEVICES'] = ''

###################################### parameters #############################################

# the diretory of images where the paths in ANNOS_CSV relative to
IMG_DIR = '/home/timhu/all_tif'

# central crop width and height
Wl8 = Hl8 = 150 # Landsat 30m resolution per pixel
Ws1 = Hs1 = 450 # Sentinel-1 10m resolution per pixel

# annotation csv for image relative path and labels
ANNOS_CSV = '/home/timhu/dfd-pop/data/annos_csv/state24_paths_density_labels_13k_Feb10.csv'

# read csv files
df_annos = pd.read_csv(ANNOS_CSV, index_col=0)
len_train = len(df_annos[df_annos.partition == 'train'])
len_val = len(df_annos[df_annos.partition == 'val'])
len_test = len(df_annos[df_annos.partition == 'test'])

# set the output path for new TFRecord file, and 
record_train_path = '/home/timhu/dfd-pop/data/TFrecord/state24_l8s1_density_train_'+str(len_train)+'.tfrecord'
record_val_path = '/home/timhu/dfd-pop/data/TFrecord/state24_l8s1_density_val_'+str(len_val)+'.tfrecord'
record_test_path = '/home/timhu/dfd-pop/data/TFrecord/state24_l8s1_density_test_'+str(len_test)+'.tfrecord'

#############################################################################################

# helper functions to convert Python values into instances of Protobuf "Messages"
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# helper functions to load TIF image path as numpy array 
def load_tif_from_file(file, crop_width, crop_height):
    img_file = gdal.Open(file)
    width, height = img_file.RasterXSize, img_file.RasterYSize
    if crop_width > width:
        raise Exception("Requested width exceeds tif width.")
    if crop_height > height:
        raise Exception("Requested height exceeds tif height.")  
    # returns np array of shape (C, H, W)
    img_data = img_file.ReadAsArray((width - crop_width)//2, (height - crop_height)//2, crop_width, crop_height) 
    img_data = np.moveaxis(img_data, 0, -1)
    
    return img_data

# compress the the image numpy array to JPEG bytes, which saves file sizes of TFrecord
def convert_jpeg_bytes(image_path, crop_width, crop_height):
    im_array = load_tif_from_file(image_path, crop_width, crop_height)
    im = Image.fromarray(im_array) 
    im_bytes = io.BytesIO()
    im.save(im_bytes, format='JPEG')
    
    return im_bytes.getvalue()
    

if __name__ == '__main__':
    sess = tf.Session()
    
    # create a TFRecordWriter for each writer
    train_writer = tf.python_io.TFRecordWriter(record_train_path)
    val_writer = tf.python_io.TFRecordWriter(record_val_path)
    test_writer = tf.python_io.TFRecordWriter(record_test_path)


    # shuffle the whole datasets
    df_annos = df_annos.sample(frac=1).reset_index(drop=True)
    len_df = len(df_annos)
    
    start_time = time.time()
    # loop through each row in CSV and write images and labels to TFrecord
    for (i, row) in df_annos.iterrows():
        if i % 100 == 0:
            duration = time.time() - start_time
            start_time = time.time()
            print ('finish writing: %d/%d (%.3f sec)' % (i, len_df, duration)) 

        # read in each row and take the image and labels needed for TFrecord    
        l8_path = os.path.join(IMG_DIR, row.l8_vis_path)
        s1_path = os.path.join(IMG_DIR, row.s1_vis_path)
#         l8_bytes = convert_jpeg_bytes(l8_path, Wl8, Hl8)
#         s1_bytes = convert_jpeg_bytes(s1_path, Ws1, Hs1)
        l8_array = load_tif_from_file(l8_path, Wl8, Hl8)
        s1_array = load_tif_from_file(s1_path, Ws1, Hs1)


        # covert python values to the format of Protobuf "Messages"
        next_features = {'image_l8': _bytes_feature(l8_array.tostring()),
                         'image_s1': _bytes_feature(s1_array.tostring()),
                        'density_class': _int64_feature(row.pop_density_class), 
                        'density_val': _float_feature(row.pop_density), 
                        'density_log2': _float_feature(row.pop_density_log2),
                        'longitude': _float_feature(row.longitude),
                        'latitude': _float_feature(row.latitude)}
        # Create an instance of an Example protocol buffer
        next_example = tf.train.Example(features=tf.train.Features(feature=next_features))

        if row.partition == 'train':
            # Serialize to string and write to the file
            train_writer.write(next_example.SerializeToString())
        if row.partition == 'val':
            val_writer.write(next_example.SerializeToString())
        if row.partition == 'test':
            test_writer.write(next_example.SerializeToString())

    train_writer.close()
    val_writer.close()
    test_writer.close()