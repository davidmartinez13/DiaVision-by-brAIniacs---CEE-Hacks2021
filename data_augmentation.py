# Importing necessary functions
import os
import cv2
import numpy as np

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
   
# Initialising the ImageDataGenerator class.
# We will pass in the augmentation parameters in the constructor.
datagen = ImageDataGenerator(
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        brightness_range = (0.5, 1.5))

path = "c:/Users/David/PythonProjects/Programming/Workspace/hackathon/cropped"
out = "c:/Users/David/PythonProjects/Programming/Workspace/hackathon/augmented"
    
# Loading a sample image 
for image_path in os.listdir(path):

        # create the full input path and read the file
        input_path = os.path.join(path, image_path)
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = load_img('c:/Users/David/PythonProjects/Programming/Workspace/hackathon/phone_aruco_marker.jpg') 
        # Converting the input sample image to an array
        x = img_to_array(img)
        # Reshaping the input image
        x = x.reshape((1, ) + x.shape) 
        
        # Generating and saving 5 augmented samples 
        # using the above defined parameters. 
        i = 0
        for batch in datagen.flow(x, batch_size = 1,
                                  save_to_dir =out, 
                                  save_prefix ='image', save_format ='jpg'):
            i += 1
            if i > 5:
                break