# IMPORTS
import tensorflow as tf
import tensorflow_datasets as tfds
import csv
import os
import platform
from PIL import Image

# CONSTANTS
TRAINING_SIZE = 60000
TESTING_SIZE = 10000
DATA_IMAGE_WIDTH = 28
DATA_IMAGE_HEIGHT = 28
FINAL_IMAGE_WIDTH = 100
FINAL_IMAGE_HEIGHT = 100
TRAINING_EXPORT_DIR = "MNIST_Converted_Training/"
TRAINING_EXPORT_FILE = "converted_training"
TRAINING_CSV = "training_data.csv"
TESTING_EXPORT_DIR = "MNIST_Converted_Testing/"
TESTING_EXPORT_FILE = "converted_testing"
TESTING_CSV = "test_data.csv"

if platform.system() == 'Windows':
    TRAINING_EXPORT_DIR = "MNIST_Converted_Training\\"
    TESTING_EXPORT_DIR = "MNIST_Converted_Testing\\"

# Convert and upscale function will take an image with DATA_IMAGE_HEIGHT and DATA_IMAGE_WIDTH dimensions
# and create a new image with FINAL_IMAGE_WIDTH and FINAL_IMAGE_HEIGHT dimensions, where the original
# image is embededded within the final at a set of dimensions. It will then return the bounding box
# for where the original image is within the new one
def convert_and_upscale(image, label):
    xmin = tf.random.uniform((), 0, FINAL_IMAGE_WIDTH - DATA_IMAGE_WIDTH, dtype=tf.int32)
    ymin = tf.random.uniform((), 0, FINAL_IMAGE_HEIGHT - DATA_IMAGE_HEIGHT, dtype=tf.int32)
    image = tf.reshape(image, (DATA_IMAGE_WIDTH, DATA_IMAGE_HEIGHT, 1,))
    image = tf.image.pad_to_bounding_box(image, ymin, xmin, FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT)
    xmin = tf.cast(xmin, tf.float32)
    ymin = tf.cast(ymin, tf.float32)
    xmax = (xmin + DATA_IMAGE_WIDTH) / FINAL_IMAGE_WIDTH
    ymax = (ymin + DATA_IMAGE_HEIGHT) / FINAL_IMAGE_HEIGHT
    xmin = xmin / FINAL_IMAGE_WIDTH
    ymin = ymin / FINAL_IMAGE_HEIGHT
    real_image = Image.fromarray(image.numpy().squeeze(axis=2))
    real_image = real_image.convert("L")
    return real_image, ([label.numpy(), xmin.numpy(), ymin.numpy(), xmax.numpy(), ymax.numpy()])


# Do conversion will go through the dataset, reading the images one by one, and converting them
# It will then write out the 'new' images, along with a CSV file containing the label and the bounding box
# details, where the bounding box is xmin, ymin, xmax, ymax
def do_conversion(dataset, export_dir, export_file, csv_name):
    current_item = 0
    if not os.path.isdir(export_dir):
        os.mkdir(export_dir)
    with open(csv_name, 'w', newline='\n') as csv_file:
        csv_data_writer = csv.writer(csv_file, delimiter=',')
        for item in dataset.take(100):
            image, label = convert_and_upscale(item[0], item[1])
            current_item = current_item + 1
            filename = export_file + str(current_item) + ".png"
            image.save(export_dir + filename)
            towrite = []
            towrite.append(filename)
            for i in label:
                towrite.append(str(i))
            csv_data_writer.writerow(towrite)
    csv_file.close()


# Perform the conversion on both the training and test splits
training_dataset = tfds.load("mnist", split="train", as_supervised=True)
testing_dataset = tfds.load("mnist", split="test", as_supervised=True)
do_conversion(training_dataset, TRAINING_EXPORT_DIR, TRAINING_EXPORT_FILE, TRAINING_CSV)
do_conversion(testing_dataset, TESTING_EXPORT_DIR, TESTING_EXPORT_FILE, TESTING_CSV)
