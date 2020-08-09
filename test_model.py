# More detailed tests on the multiclass model

import time
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np

# GPU setup
# TODO: move all this into separate files
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


os.chdir('/home/peter/ml/weeds/WeedML')
model = load_model('MNv2_224x224_multi.h5')
model.summary()


IMG_SIZE = (224, 224)
BATCH_SIZE = 1 #16 # not efficient but makes indexing images easier

# no augmentation, no shuffle, only test binary labels
base_path = '/home/peter/ml/weeds/traintestimages'

val_labels = pd.read_csv('test.csv')
val_labels.Folder.value_counts(normalize=True,sort=True)
val_labels.mean()


val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_ds = val_datagen.flow_from_dataframe(
        dataframe=val_labels,
        directory=base_path,
        x_col='Filename', 
        y_col='Label',        
        class_mode='categorical',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False)


pred =  model.predict(val_ds)

multiclass_labels= [c for c in val_labels.columns if c[0].lower()==c[0]]

# 
# tflite_convert --output_file=/home/peter/StudioProjects/weed_classifier/app/src/main/assets/MNv2_224x224_multi.tflite --keras_model_file=/home/peter/ml/weeds/WeedML/MNv2_224x224_multi.h5
# with open('/home/peter/StudioProjects/weed_classifier/app/src/main/assets/multiclass_labels.txt', mode='wt', encoding='utf-8') as t: t.write('\n'.join(multiclass_labels))
# 


pred = pd.DataFrame(pred,columns=multiclass_labels) # cols that begin with lower case
plt.hist(pred.spray + pred.dontspray); plt.show() # should be concentrated around 1!


#sanity check!
np.all( val_labels.spray == val_ds.labels )

def acc(p,l,t):
    pred_spray = np.array(p > t).astype(int)
    return np.mean(pred_spray == l)

# acc(pred.spray,val_labels.spray, 0.5) # 95.8

for c in multiclass_labels:
  print(c, "\t{:.1%}".format( acc( pred[c], val_labels[c],0.5 )))

for c in multiclass_labels:#list(val_labels.Folder.unique()):
  s = val_labels[c] == 1
  print(c, "\t{:.1%}".format( acc( pred[s][c], val_labels[s][c],0.5 )))
  print(c, " spray \t{:.1%}".format( acc( pred[s]['spray'], val_labels[s]['spray'],0.5 )))

class_acc = pd.DataFrame( [ len( pred[val_labels[c] == 1][c] ) for c in  multiclass_labels ], columns=['count']).join(
  pd.DataFrame( [ acc( pred[val_labels[c] == 1][c], val_labels[val_labels[c] == 1][c],0.5 ) for c in  multiclass_labels ], columns=['class'])).join(
  pd.DataFrame( [ acc( pred[val_labels[c] == 1]['spray'], val_labels[val_labels[c] == 1]['spray'],0.5 ) for c in  multiclass_labels ], columns=['spray']))

class_acc.index=multiclass_labels

print(class_acc)
#class_acc.to_clipboard()

#sanity check!
np.all( val_labels.spray == val_ds.labels )

# Quick question: can accuracy improve by taking a different threshold?
t_grid = np.linspace(0.1, 0.9)
acc_t = [acc(pred.spray,val_labels.spray,t) for t in t_grid ]
plt.plot(t_grid,acc_t)
plt.axvline(x=t_grid[np.argmax(acc_t)])
plt.show()
np.max(acc_t)


buttercups = val_labels[val_labels.buttercup == 1]
# predictions for buttercups (seemed bad in live testing)
val_labels[pred.buttercup < 0.1]

np.any( (pred.spray > 0.5) & (pred.buttercup < 0.1) )


# low predictions for grass
val_labels[pred.grass < 0.1]

def show_image(i):
  plt.imshow((val_ds[i][0][0] + 1.0)/2.0); plt.show()

for i in val_labels[pred.grass < 0.1].index:
  show_image(i)


labwp = val_labels.join(pred,rsuffix="_p")
labwp[labwp['clover'] ==1]

plt.boxplot(labwp['buttercup_p'][labwp['buttercup'] ==1]);plt.show()
plt.boxplot(labwp['clover_p'][labwp['clover'] ==1]);plt.show()
plt.boxplot(labwp['spray_p'][labwp['clover'] ==1]);plt.show()

show_image(90)

