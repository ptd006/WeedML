# Helper to tag photos fast

# TODO: use XBox controller
# TODO: autoaccept high confidence predictions
# TODO: clean up imports..


import pandas as pd

# timing
from datetime import datetime
import time

from keras.models import Model, load_model

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
import os
import glob

# Existing model
existing_model = '/home/peter/ml/weeds/WeedML/12-07epoch_5.h5'

# In and out dirs (each image will be saved in dest/class)
src = '/home/peter/ml/weeds/photos/1407'
dest = '/home/peter/ml/weeds/photos/1407/clip'

# keys for new classification
key_dict = dict( {'d':'dock', 't':'thistle', 'g':'grass', 's':'stinger', 'c':'clover', 'b':'buttercup', 'b':'buttercup', 'x':'dandelion'   } )


# GPU setup (for existing model)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

# Classification for initial (!) model
CLASS_NAMES = ['dock','negative','thistle']
n_classes = len(CLASS_NAMES)

model = load_model(existing_model)
model.summary()

# Global variables
IMG_SIZE = (224, 224)
BATCH_SIZE = 16 # try making smaller (laptop GPU RAM maybe too small)

# Start here
# Will chop images based on 2 * 8 grid with small overlap, 400 * 1600px

def chop_photo(fn):
    im = Image.open(fn)
    w, h = im.size

    # rotate if not in wide format
    if w < h:
        im = im.rotate(90,expand=True)
        w, h = im.size

    # Crop from centre
    crop_x0, crop_y0 = (w/2.0 - 4 * 200 - 12, h/2.0 - 200 - 12)
    crop_x1, crop_y1 = (w/2.0 + 4 * 200 + 12, h/2.0 + 200 + 12)
    im_crop = im.crop( (crop_x0,crop_y0, min(crop_x1,w), min(crop_y1,h)))

    #plt.imshow(im_crop); plt.show()

    # overlapping clips
    im_clips = []

    # get all the prediction boxes (note this doesn't appear to copy the memory blocks)
    for i in range(0,2):
        for j in range(0,8):
            # print(i*200,j*200
            x0, y0 = (j*200,i*200)
            im_clips.append( np.array( im_crop.crop( ( (x0, y0 , x0 + 224, y0+224) ) ) ) )
            #draw.rectangle([ (x0, y0 ), (x0 + 224, y0+224)])

    im_clips = np.array(im_clips)
    return im_clips


def save_in_correct_folder(e, im, root, fn, i):
    c = e.char
    if c not in key_dict.keys():
        print('Skipping')
        root.destroy()
        return
    
    cl = key_dict[c]
    print('You chose: '+cl)
    if not os.path.exists(cl):
        os.makedirs(cl)
    im.save(os.path.join(cl, fn.replace('.', '_{:03d}.'.format(i)) ))
    root.destroy()


def show_selection_window(im_clip, i, hint,fn):
    root = tk.Tk()
    root.title('Weed tagger')

    pil_im = Image.fromarray(np.uint8(im_clip)).convert('RGB')
    tkpic = ImageTk.PhotoImage(pil_im)

    # hint
    predlabel = tk.Label(root, text=hint)
    predlabel.pack()

    # display the image
    label = tk.Label(root, image=tkpic)
    label.pack()

    root.bind("<Key>", lambda e=None, imc=pil_im, r=root, f=fn, i=index: save_in_correct_folder(e,imc,r, f,i) )
    root.mainloop()




os.chdir(src)
files = glob.glob('*.jpg')

# now switch to output
os.chdir(dest)

# loop over all images (kind of assumes there's a manageable number)


for fn in files[0:1]:
    im_clips = chop_photo(os.path.join(src,fn))
    pred = model.predict(im_clips/255.0,batch_size=BATCH_SIZE)
    maxpred = np.argmax(pred,axis=1)
    
    for i in range(0,2):
        for j in range(0,8):
            index = 8*i + j
            show_selection_window(im_clips[index], index, CLASS_NAMES[maxpred[index]] + " {:.0%}".format( pred[index][maxpred[index]]), fn)


