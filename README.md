**Tracking of biological cells in time-lapse microscopy images**

Using different ways to automate the detection and tracking of the cells, as well as to perform subsequent
quantitative analysis of cell motion. The task of computer vision includes image preprocessing, feature extraction, classification, motion detection, tracking and recognition. Also used the popular method “U-net” as a neural network model to predict cell movement.

Set up instruction 

1. Make sure all three data sets:
 
 'DIC-C2DH-HeLa', 'Fluo-N2DL-HeLa', 'PhC-C2DL-PSC'  in the data folder.


2. Download the first U-net pertained model from website address:

 https://drive.google.com/drive/folders/1Jp6qs7rqqkK74LjMr59h1Kb2Jv6Ar8ao
 
 Put it into the model_1 folder, the format inside should be similar to model_3.


3. Running environment

   openCv 3.4.2, 
   
   tensorflow 2.3.0, 
   
   numpy,  
   
   skimage, 
   
   scipy.

4. Run 'run_method.py' to start program.
