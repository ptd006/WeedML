# Generates tflite model for the Demo App, builds debug version and installs
# Phone should be connected (by USB) and with dev options enabled!!

# reference https://developer.android.com/studio/build/building-cmdline

import os
import pandas as pd

# App path
app_path = '/home/peter/StudioProjects/weed_classifier'

# Keras model
keras_model = '/home/peter/ml/weeds/WeedML/MNv2_224x224_multi.h5'

# Assume assets is same as demo app
assets_path = app_path+'/app/src/main/assets'

# Get labels and update in app
labels = pd.read_csv('/home/peter/ml/weeds/WeedML/multiclasslabels.csv')
multiclass_labels= [c for c in labels.columns if c[0].lower()==c[0]] # lower case columns are the labels
with open(assets_path +'/multiclass_labels.txt', mode='wt', encoding='utf-8') as t: t.write('\n'.join(multiclass_labels))


# Convert model to tflife and save in assets
os.system( 'tflite_convert --output_file=' + assets_path + '/MNv2_224x224_multi.tflite --keras_model_file=' + keras_model)

os.chdir(app_path)
os.system( app_path + '/gradlew assembleDebug')
os.system( app_path + '/gradlew installDebug')
