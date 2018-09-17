# cnn-population-mapping

In this project, we aim to predict the population density of rural villages of India from satellite imageries by utilizing Convolutional Neural Network (CNN) models. With the availability of high-frequency satellite images, we can predict population density every few days, saving the costs of on-site census surveys and avoiding the inaccuracies caused by the infrequency of census surveys. We demonstrate state-of-the-art prediction performance in villages of all states in India. By using satellite images with 10-30 meter resolution, our best models can predict aggregated village population in one Subdistrict (akin to a US county) with $R^2$ of 0.93, and individual village $\log_2$ population density with $R^2$ of 0.44.

Since we cannot share the data in this project, the codes in this repository are for reference only. 

## Data Requirement

This project requires datasets and csv annotations provided by the instructors of CS325B in Stanford. 

The required data and files include:  
 - Groud truth India village population survey datasets, available at Google Cloud Buckets: 
     - `gs://es262-poverty/India_pov_pop.csv`
 - Landsat-8 and Sentinel-1 Satellite RGB images for each village, also available at Google Cloud Buckets: 
     - `gs://es262-poverty/l8_median_india_vis_500x500_*.0.tif`
     - `gs://es262-poverty/s1_median_india_vis_500x500_*.0.tif`
 - The metadata info for images and csv are in:
     - `data/Readme_poverty.rtf`
 - VGG-16 pretrain weights
     - https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models
     - http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
 - Optional data
     - DMSP/OLS nightlights (.tif) : https://ngdc.noaa.gov/eog/dmsp/downloadV4composites.html
     - Landscan (.tif) : https://earthworks.stanford.edu/?featured=geospatial_data&q=landscan

## Data Preprocessing

The required data should be preprocessed to have the format needed for CNN training, which can be done by modify and run the codes in: 

 1. code/pop_data_preprocess.ipynb
     - Input:
        - data/India_pov_pop_May17.csv: ground truth village population dataset
        - data/output.txt: contained all image filenames
     - Output:
        - data/India_pov_pop_May17_density.csv: initially cleaned dataset with population density conversion
        - data/India_pov_pop_May17_partiton.csv: dataset with train/val/test partition by sub-distrcts
        - data/all_paths_density_544k_May17.csv: dataset with image tif paths
        - data/all_jpgpaths_density_544k_May17.csv: dataset with image jpg paths
        - data/all_jpgpaths_clean_538k_May17.csv: dataset with outlier removed
        - data/state24_jpgpaths_clean_17k_May17.csv: dataset with only one state data 
    - Sections:
        - initial cleanup: 
           - Select necessary field of csv
           - remove rows with missing are/lat/long
        - calculate density and class
           - area in units of m2, calculate density as nums/km2
           - Calculate log2 density: pop_density_log2
           - => India_pov_pop_May17_density.csv
        - Village area distribution
           - Used to determine the crop size of image
        - train/val/test partition without image overlap
           - 70% train, 20% Val, 10% test parition by subdistrict_id
           - remove training partition rows that have overlap with any val/test rows
           - => India_pov_pop_May17_partiton.csv
        - add tif imge path
           - Relative path for dataset column l8_median_india_vis and s1_median_india_vis
           - Takes 15min to add 500k image paths as seperate column
           - => all_paths_density_544k_May17.csv
        - create jpg path column in csv
           - => all_jpgpaths_density_544k_May17.csv
        - Data Cleaning / Extrme values removal
           - remove 1% extrme data, filter out [0.5%, 99.5%] of pop_density_log2
           - => all_jpgpaths_clean_538k_May17.csv
        - Select single state (24)
           - select a single state data to reduce training load initially
           - => state24_jpgpaths_clean_17k_May17.csv
                        
  2. code/jpg_convertion.py  (convert satellite TIF image to jpg format)
 
	   - Input: 
	       - data/all_jpgpaths_clean_538k_May17.csv
	       - original TIF images in `/data/all_tif`
	   - output:
	       - Cropped jpg images saved in `data/all_jpg`

## Data Loading Test

Once the data are preprocessed, they can be tested and visualized in the notebook:

- tim_input_test.ipynb
    * tif load test
        * Input:
            * data/all_jpgpaths_clean_538k_May17.csv
            * original TIF images in `/data/all_tif`
        * Output:
            * L8 and S1 cropped images
    * CNN input test
        * Input:
            * vgg_deeep_combo.py
            * data_input_jpg.py
            * weights/vgg_16.ckpt
            * data/all_jpgpaths_clean_538k_May17.csv
            * Cropped jpg images saved in `data/all_jpg`
        * Output:
            * Batch input of S1, L8, pop, lat, long, village id, etc.
            
## CNN training

This projects uses seperate training script for different combination of datasets and architectures. The Training scripts are in the directory `code/training_scripts`, where the uses of each script are listed below. To run the training, firstly modify the parameters at the beginning section of the script, and then run it with `python <train_script.py>`.

(L8: Landsat-8, S1: Sentinel-1, NL: Nightlight)

 - scripts:
    * train_l8_regression.py, train_s1_regression.py, 
        * L8/S1 only: DATA = ‘l8’/’s1’, IMAGE_CHANNEL = 3
        * L8+S1 concat initially: DATA = ‘l8s1’, IMAGE_CHANNEL = 6
    * train_shallow_combo_regression.py
        * L8+S1 combined after conv1
    * train_deep_combo_regression.py
        * L8+S1 combined after all conv1-conv5
    * train_deep_combo_nl_regression.py
        * L8+S1+NL, deep combo
    * train_l8_classification.py
        * the training that use 2017 paper's approach with only L8 data and classification output
 - parameters to change before training:
    * JPG_DIR: jpg images directory 
    * LOG_DIR: directory that save checkpoints
    * FLAGS: training hyperparameters
    * PRETRAIN_WEIGHTS
    * IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL
    * ANNOS_CSV: annotation csv generated from `pop_data_preprocess.ipynb`
    * DATA: l8, s1, l8s1, l8s1nl

## Evaluations

The above trainings were run for two datasets, firstly single state with 17k villages and then all states with 500k villages. Each trained model is saved as checkpoint and evaluted on val/test dataset partition with different metrics in directory `code/evaluations`.

- notebooks:
    - evaluate_general.ipynb
        - for all regression training models
    - evaluate_general_classification.ipynb
        - specidically for the classification model used in train_l8_classification.py
    - LandScan_comparison.ipynb
        - evalute and compare the estimation performance of LandScan
- evaluatoin methods:
    - map visual eval
        - the results used to produce map visualization on district-level predcition
    - raw log2 density
        - comparison of predicted and true values on per village log2 population density
    - Weighted Average Aggregation comparison
        - comparison on total village population aggregated on subdistrict level. 
