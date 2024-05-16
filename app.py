from flask import Flask, render_template, Response, request, session , redirect
# from transformers import BartForConditionalGeneration, BartTokenizer
import time
import os


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
import cv2
import numpy as np
from IPython.display import display, Image
import ipywidgets as widgets
import threading



solutions_urls = [{"Apple___Apple_scab": "https://en.wikipedia.org/wiki/Apple_scab"},
                  {"Apple___Black_rot": "https://apples.extension.org/black-rot-of-apple/"},
                  {"Apple___Cedar_apple_rust": "https://gardenerspath.com/how-to/disease-and-pests/cedar-apple-rust-control/"},
                  {"Apple___healthy": ""},
                  {"Carpetweeds": "https://extension.umd.edu/resource/carpetweed/"},
                  {"Crabgrass": "https://lawnphix.com/products/crabgrass-killers/"},
                  {"Goosegrass": "https://turf.purdue.edu/goosegrass/"},
                  {"Grape___Black_rot": "https://en.wikipedia.org/wiki/Black_rot_(grape_disease)"},
                  {"Grape___Esca_(Black_Measles)": "https://grapes.extension.org/grapevine-measles/"},
                  {"Tomato_Bacterial_Spot": "https://hort.extension.wisc.edu/articles/bacterial-spot-of-tomato/"},
                  {"Tomato_Early_Blight": "https://www.thespruce.com/early-blight-on-tomato-plants-1402973"},
                  {"Tomato_Leaf_mold": "https://en.wikipedia.org/wiki/Tomato_leaf_mold"},
                  {"Tomato_Septorial_Leaf_Spot": "https://portal.ct.gov/CAES/Fact-Sheets/Plant-Pathology/Septoria-Leaf-Spot-of-Tomato#:~:text=Septoria%20leaf%20spot%20is%20caused%20by%20the%20fungus,can%20occur%20on%20petioles%2C%20stems%2C%20and%20the%20calyx."},
                  {"Tomato_Yellow_Leaf_Curl_Virus": "https://en.wikipedia.org/wiki/Tomato_yellow_leaf_curl_virus"}]

solutions_map = {key: url for solution in solutions_urls for key, url in solution.items()}


import pathlib

import argparse



main_model = None
# selection_model = "NEW"

parser = argparse.ArgumentParser(description="Example Argument Parser")
    
# Adding a string argument with a default value of "OLD" and not required
parser.add_argument('--model', type=str, default='OLD', required=False, help='Specify the model string (default: OLD)')

args = parser.parse_args()

# Accessing the value of the 'model' argument
selection_model = args.model


data_dir = "leaf_photos"
data_dir = pathlib.Path(data_dir)


batch_size = 32
img_height = 180
img_width = 180



train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)



class_names = train_ds.class_names
print(class_names)




def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)




num_classes = len(class_names)

if selection_model == "NEW":
    
    main_model = make_model(input_shape=(180, 180) + (3,), num_classes=num_classes)
    print("new model loaded..")
    # keras.utils.plot_model(model, show_shapes=True)
    main_model.load_weights('checkpoint_newv2\\my_checkpoint_high_acc.ckpt')


else:
    main_model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    main_model.load_weights('checkpoint_oldv2\\my_checkpoint_high_acc.ckpt')


print(main_model.summary())





app = Flask(__name__)

app.secret_key = "dfjklwcnhe45ui672cvn894726c46m23w7845cvyt2u34rv7bfwuhjdf"


def generate_result(file_name):
  

    test_image_path = 'static/uploads/'+file_name  

    img = keras.preprocessing.image.load_img(
        test_image_path, target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = main_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    #PIL.Image.open(test_image_path)

    return "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)), class_names[np.argmax(score)]


def getFileName():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_name = timestr+".JPG"
    # timestr ="asset"
    path = os.path.join(os.getcwd()+'\\static\\uploads\\', (timestr+".JPG"))
    return path, file_name

@app.route("/", methods = ['GET','POST'])
def index():
    return render_template("index.html")



@app.route("/upload_file", methods = ['GET','POST'])
def upload_file():
    if request.method == 'POST':
      f = request.files['file']
      file_path, file_name  = getFileName() 
      print(file_path)
      session['file_name'] = file_name
      f.save(file_path)
      f.close()
      print("file saved")
      return "<script>alert('File upload successful.'); window.open('/preview_file','_self')</script>"
    
    return render_template("upload-file.html")

@app.route("/preview_file", methods = ['GET','POST'])
def preview_file():
    file_name =  session['file_name']

    if file_name is None:
        return redirect('/')
    
    file_path =  'static/uploads/'+file_name  
    if request.method == 'GET':
      
        return render_template("preview-file.html", preview_img=file_path, result_txt=None )
    else:
        
        result_summary, class_name = generate_result(file_name)
       
        soulution_url = solutions_map[class_name]

      


        return render_template("preview-file.html", preview_img=file_path, result_txt=result_summary, soulution_url=soulution_url )





if __name__=='__main__':
    app.run(debug=True, host="0.0.0.0")