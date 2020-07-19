# Simple script just to generate a CSV file listing all the images and the target label
import os
import pandas as pd

base_path = '/home/peter/ml/weeds/traintestimages'
labels = pd.DataFrame.from_dict(
    dict( {'dock':'spray', 'thistle':'spray', 'grass':'dontspray', 'stinger':'spray', 'clover':'dontspray', 'buttercup':'dontspray', 'dandelion':'spray'   } ),
    orient='index',columns=['Label'])

ignore = ['dandelion'] # ignore dandelions for now; if there's an issue then realistically we'll be blanket spraying

def generate_csv(base_path, trainortest, labels):
    data = []
    for folder in sorted(os.listdir(base_path + '/' + trainortest)):
        if folder in ignore: continue
        for fn in sorted(os.listdir(base_path + '/' + trainortest +'/'+ folder)):
            data.append((folder, fn))

    df = pd.DataFrame(data, columns=['Folder', 'File'])
    df['Filename'] = base_path +  '/' + trainortest+ '/'+ df['Folder'] + '/' + df['File']
    df = df.merge(labels,left_on='Folder',right_index=True)
    df.to_csv(base_path + '/../WeedML/' + trainortest + '.csv',index=False)

generate_csv(base_path, 'train', labels)
generate_csv(base_path, 'test', labels)
