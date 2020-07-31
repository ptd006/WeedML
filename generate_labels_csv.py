# Simple script just to generate a CSV file listing all the images and the target label
# After this
# zip -r traintestimages.zip traintestimages/* WeedML/*.csv
# rclone copy traintestimages.zip gdrive:/WeedML/ -P

import os
import pandas as pd

base_path = '/home/peter/ml/weeds/traintestimages'
labels = pd.DataFrame.from_dict(
    dict( {'dock':'spray', 'thistle':'spray', 'grass':'dontspray', 'stinger':'spray', 'clover':'dontspray', 'buttercup':'dontspray', 'dandelion':'spray', 'spray':'spray'   } ),
    orient='index',columns=['Label'])

ignore = ['dandelion'] # ignore dandelions for now; if there's an issue then realistically we'll be blanket spraying

def generate_csv(base_path, trainortest, labels):
    data = []
    for folder in sorted(os.listdir(base_path + '/' + trainortest)):
        if folder in ignore: continue
        for fn in sorted(os.listdir(base_path + '/' + trainortest +'/'+ folder)):
            data.append((folder, fn))

    df = pd.DataFrame(data, columns=['Folder', 'File'])
    df['Filename'] = trainortest+ '/'+ df['Folder'] + '/' + df['File'] # relative to base_path
    df = df.merge(labels,left_on='Folder',right_index=True)
    df.to_csv(base_path + '/../WeedML/' + trainortest + '.csv',index=False)
    print(df['Label'].value_counts())

generate_csv(base_path, 'train', labels)
generate_csv(base_path, 'test', labels)
