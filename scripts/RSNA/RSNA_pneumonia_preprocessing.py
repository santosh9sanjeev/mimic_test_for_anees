#RUN TWICE FOR TRAIN AND TEST


import os
import cv2
import numpy as np
import pydicom as dicom
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input_dir',default='/l/users/santosh.sanjeev/rsna-18/stage_2_test_images', type=str, help='input directory')
parser.add_argument('--output_dir', default='/home/santosh.sanjeev/rsna_18/test/', type=str, help='output directory')
args = parser.parse_args()



# Create the output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Iterate through the files in the input directory
for filename in os.listdir(args.input_dir):
    if filename.endswith('.dcm'):
        file_path = os.path.join(args.input_dir, filename)

        # Read dcm extension image
        ds = dicom.dcmread(file_path)
        img = np.asarray(ds.pixel_array)
        # Process image
        img = np.expand_dims(img, axis=0)
        img = np.moveaxis(img, -1, 0)
        img = np.moveaxis(img, -1, 0)
        print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Define the output file path (with same name but in jpg format)
        output_file_path = os.path.join(args.output_dir, os.path.splitext(filename)[0] + '.jpg')
        print(output_file_path)
        # Save image in jpg format
        cv2.imwrite(output_file_path, img)
 
print("Conversion completed.")