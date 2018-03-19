import numpy as np
import pandas as pd
import re,os,time, math, gdal
from PIL import Image

df = pd.read_csv('/home/timhu/dfd-pop/data/annos_csv/state24_paths_density_labels_13k_Feb10.csv')

TIF_DIR = '/home/timhu/all_tif'
JPG_DIR = '/home/timhu/all_jpg'

Wl8 = Hl8 = 150 # Landsat 30m resolution per pixel
Ws1 = Hs1 = 450 # Sentinel-1 10m resolution per pixel



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

def jpg_convert_save(tif_path, crop_width, crop_height):
    jpg_path = re.sub('tif', 'jpg', tif_path)
    jpg_path = re.sub('500x500', str(crop_width)+'x'+str(crop_height), jpg_path)
    
    img_array = load_tif_from_file(tif_path, crop_width, crop_height)
    im = Image.fromarray(img_array)
    
    im.save(os.path.join(JPG_DIR, jpg_path), format='JPEG')
    

for i in range(len(df)):
    if i % 100 == 0:
        print('finish:', i)
    l8_path = os.path.join(TIF_DIR, df.l8_vis_path[i])
    s1_path = os.path.join(TIF_DIR, df.s1_vis_path[i])
    
    jpg_convert_save(l8_path, Wl8, Hl8)
    jpg_convert_save(s1_path, Ws1, Hs1)
    
    