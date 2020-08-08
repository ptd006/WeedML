# Simple script just to generate a CSV file listing all the images and the target label
# After this zip the recent changes and upload to Gdrive (so they can be pulled to colab easily):

# find traintestimages -mtime -7 -print | zip traintestimages_`date +"%d%m%y"`.zip -@
# rclone copy traintestimages_`date +"%d%m%y"`.zip gdrive:/WeedML/ -P
# #zip -r traintestimages.zip traintestimages/* WeedML/*.csv

import os
import pandas as pd

base_path = '/home/peter/ml/weeds/traintestimages'

# easier to edit with separate file
labels = pd.read_csv(base_path + '/../WeedML/multiclasslabels.csv')
labels
list(labels.Label.unique())
# NOTE: by MY convention columns with lowercase first letter are multiclass labels, i.e.:
[c for c in labels.columns if c[0].lower()==c[0]]

ignore = ['dandelion'] # ignore dandelions for now; if there's an issue then realistically we'll be blanket spraying

def generate_csv(base_path, trainortest, labels):
    data = []
    for folder in sorted(os.listdir(base_path + '/' + trainortest)):
        if folder in ignore: continue
        for fn in sorted(os.listdir(base_path + '/' + trainortest +'/'+ folder)):
            data.append((folder, fn))

    df = pd.DataFrame(data, columns=['Folder', 'File'])
    df['Filename'] = trainortest+ '/'+ df['Folder'] + '/' + df['File'] # relative to base_path
    df = df.merge(labels) # ,left_on='Folder', right_on='Folder')
    df.to_csv(base_path + '/../WeedML/' + trainortest + '.csv',index=False)
    print(df['Label'].value_counts())

generate_csv(base_path, 'train', labels)
generate_csv(base_path, 'test', labels)

