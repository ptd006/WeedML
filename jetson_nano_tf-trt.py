# Optimise model for TensorFlow TensorRT
# Tested with MobileNet trained with Keras saved in h5

import time
import tensorflow as tf
import numpy as np
import pandas as pd
import os

os.chdir('/home/peter/ml/weeds/WeedML')



# Load Keras model and save in TensorFlow format
from keras.models import load_model
model = load_model('MNv2_224x224_multi.h5')
model.summary()
model.save("./SavedModel1809")


# Convert, build, save (fp16)
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP16)
converter = trt.TrtGraphConverterV2(input_saved_model_dir='./SavedModel1809',conversion_params=conversion_params)
converter.convert()

# Takes a while
# converter.save(output_saved_model_dir='SavedModel1809_fp16')

# Get some real data (although it's not clear how essential this is for the TF-TRT optimisation as the examples use random data)
# Might only be the input shape and batch size?!
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224,3)
BATCH_SIZE = 16 #16 # not efficient but makes indexing images easier

# no augmentation, no shuffle, only test binary labels
base_path = '/home/peter/ml/weeds/traintestimages'

val_labels = pd.read_csv('test.csv')
val_labels.Folder.value_counts(normalize=True,sort=True)
val_labels.mean()

multiclass_labels= [c for c in val_labels.columns if c[0].lower()==c[0]]

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_ds = val_datagen.flow_from_dataframe(
        dataframe=val_labels,
        directory=base_path,
        x_col='Filename', 
        y_col=multiclass_labels, #'Label',        
        class_mode='raw',#'categorical',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True)

from functools import partial
def input_fn(num_iterations):
	for i, (batch_images, _) in enumerate(val_ds):
		if i >= num_iterations:
			break
		yield (batch_images,)
		print(" step %d/%d" % (i+1, num_iterations))
		i += 1

#def input_fn():
#  inp1 = np.random.normal(size=(16, 224, 224, 3)).astype(np.float32)
#  yield inp1

converter.build((input_fn=partial(input_fn,16))
converter.save("./SavedModel1809_fp16_built")


# OK, now load the model and check how fast it is


# Takes a while..
import tensorflow as tf
saved_model_loaded = tf.saved_model.load("./SavedModel1809_fp16_built")
graph_func = saved_model_loaded.signatures['serving_default']
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
frozen_func = convert_variables_to_constants_v2(graph_func)


t0 = time.time()

for i in range(10):
  demo_batch = val_ds[i]
  #print(demo_batch[0].shape)
  input_tensors = tf.cast(demo_batch[0], dtype=tf.float32)
  output = frozen_func(input_tensors)[0].numpy()

print("Time: {:.2f}".format(time.time()-t0) )


# sanity check a batch
pred=pd.DataFrame(output,columns=multiclass_labels)
actual=pd.DataFrame(demo_batch[1],columns=multiclass_labels)#['dontspray','spray'])#
pred.join(actual,rsuffix="_actual")

# After all that it only seems about 20% faster than plain TensorFlow eugh
# Possibly consider pure TensorRT (UFF)
