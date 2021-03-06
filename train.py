import pandas as pd
import os

from datetime import datetime

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# also consider some scheduled decrease in LR: https://keras.io/api/callbacks/learning_rate_scheduler/

from keras.optimizers import Adam

from keras.models import Model, load_model
# from keras import backend as K

# TODO: try InceptionV3
# from keras.applications.inception_v3 import InceptionV3

from keras.applications.resnet50 import ResNet50

from keras.layers import Dense, GlobalAveragePooling2D

from keras.preprocessing import image_dataset_from_directory

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np

# WeedML working directory containing test.csv
os.chdir('/home/peter/ml/weeds/WeedML')

# start from scratch?
newModel = False

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


# Classification labels
CLASS_NAMES = ['dontspray','spray']
n_classes = len(CLASS_NAMES)

# Global variables
IMG_SIZE = (224, 224) # image size expected by the model
RAW_IMG_SIZE = IMG_SIZE # (256, 256) had previously taken larger images and taken crops in augmentation

INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
BATCH_SIZE = 32
INITIAL_LR = 0.0001

# Create a new model.  If f is not med to be the DeepWeeds model path.
def get_newModel(f,n_classes,trainable=False):   
    if f == 'ResNet50':
        print ('Load pretrained ResNet50')
        model = ResNet50(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
        last_layer = model.output
        last_layer = GlobalAveragePooling2D(name='avg_pool')(last_layer)
    elif f == 'MobileNetV2':        
        print ('Load pretrained MobileNetV2')
        model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
        last_layer = model.output
        last_layer = GlobalAveragePooling2D(name='avg_pool')(last_layer)        
    else:
        print ('Load pretrained model from DeepWeeds paper (ResNet50)')
        model = load_model(f)
        # keep existing av pool
        last_layer = model.get_layer('avg_pool').output

    # freeze model- assume the pretrained model is good (can't do a lot else on laptop anyway)
    model.trainable=trainable

    # experiment with extra dense layer (as we are actually pulling into multiple distinct weeds into the two classes)
    last_layer = Dense(8, activation='relu', name='merge_labels')(last_layer)
        
    # Now dense and softmax
    outputs = Dense(n_classes, activation='softmax', name='prediction')(last_layer)

    # Return the modified model
    model = Model(inputs=model.input, outputs=outputs)
    return model


if newModel:    
    model = get_newModel('/home/peter/ml/weeds/DeepWeeds/resnet.hdf5',n_classes)
    #model = get_newModel('MobileNetV2',n_classes,False)    
else:
    model = load_model('DW_28-07_4.h5')# 26-07epoch_5.h5')

model.summary()


def img_flow(csv_file,base_path):
    datagen = ImageDataGenerator(
                rescale=1. / 255,
                fill_mode="reflect",
                shear_range=0.2,
                # zoom_range=(0.5, 1),
                horizontal_flip=True,
                rotation_range=10,
                channel_shift_range=10,
                brightness_range=(0.85, 1.15))

    return datagen.flow_from_dataframe(
        dataframe=pd.read_csv(csv_file),
        directory=base_path,
        x_col='Filename', 
        y_col='Label', 
        class_mode='categorical',
        target_size=RAW_IMG_SIZE, 
        batch_size=BATCH_SIZE,
        classes=CLASS_NAMES,
        shuffle=True,
        seed=123
        )

# assumes files are train.csv and test.csv
base_path = '/home/peter/ml/weeds/traintestimages'
train_ds = img_flow('train.csv',base_path)
val_ds = img_flow('test.csv',base_path)


print ('{0:.0%} are spray '.format(np.mean(train_ds.labels)) )

demo_batch = val_ds[4]

# Here are the first 9 augmented images
plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(demo_batch[0][i])
    plt.title(CLASS_NAMES[np.argmax(demo_batch[1][i])])
    plt.axis("off")

plt.show()

if newModel:
    print('Compile new model')
    #model.compile(loss='binary_crossentropy', optimizer=Adam(lr=INITIAL_LR), metrics=['categorical_accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=INITIAL_LR), metrics=['categorical_accuracy'])


run_date = datetime.today().strftime('%d-%m')
model_prefix = 'DW_'
callbacks = [keras.callbacks.ModelCheckpoint("/home/peter/ml/weeds/WeedML/" + model_prefix + run_date + "_{epoch}.h5")]

trainhist = model.fit(train_ds, epochs=5, callbacks=callbacks, validation_data=val_ds)
