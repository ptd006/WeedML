# Run model live with webcam
import cv2

import time
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import pickle
import os

# Preprocessing function - must match what was used for training!
from keras.applications.mobilenet_v2 import preprocess_input

# GPU setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


WEBCAM_DEV_ID = 2



CLASS_NAMES = ['dontspray','spray']
n_classes = len(CLASS_NAMES)

model = load_model('/home/peter/ml/weeds/WeedML/MNv2_01-08_47.h5')
# below is needed to stop memory leak which eventually leads to severe performance degradation (TensorFlow/Keras team pls document what is happening there!)
# model.save("/home/peter/ml/weeds/WeedML/MNv2_01-08_47.h5", include_optimizer=False)
model.summary()

# Grid width and height
gW = 6
gH = 3

# TODO: stop hardcoding for 224px (in case later we change to 160 for speed)
cdX = 200 * (gW/2.0) + 12
cdY = 200 * (gH/2.0) + 12

IMG_SIZE = (224, 224)
BATCH_SIZE = min(gW*gH,32)

fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40) # for annotating with predictions

## live model run from capture

dest = '/home/peter/ml/weeds/demo6-nofocus/'
if not os.path.exists(dest):
    os.makedirs(dest)


cap = cv2.VideoCapture(WEBCAM_DEV_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
cap.set(cv2.CAP_PROP_AUTOFOCUS,0) # try disabling autofocus


n_frames = 0
total_time = 0
wall_time_t0 = time.time()

while(n_frames < 100):
    # Capture frame-by-frame
    t0 =  time.time()
    ret, frame = cap.read()
    

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img)

    w, h = im.size

    # rotate if not in wide format
    if w < h:
        im = im.rotate(90,expand=True)
        w, h = im.size

    crop_x0, crop_y0 = (w/2.0 - cdX, h/2.0 - cdY)
    crop_x1, crop_y1 = (w/2.0 + cdX, h/2.0 + cdY)

    im_crop = im.crop( (crop_x0,crop_y0, min(crop_x1,w), min(crop_y1,h)))

    im_clips = []

    # get all the prediction boxes
    for i in range(0,gH):
        for j in range(0,gW):
            x0, y0 = (j*200,i*200)
            im_clips.append( np.array( im_crop.crop( ( (x0, y0 , x0 + 224, y0+224) ) ) ) )


    im_clips = np.array(im_clips)

    
    pickle.dump( im_clips, open(dest + 'frame_{:03d}.pickle.gz'.format(n_frames) , 'wb') )

    t1 = time.time()
    pred = model.predict(preprocess_input(im_clips),batch_size=BATCH_SIZE)
    n_frames=n_frames+1
    

    print("Time (incl. capture): {:.2f}".format(time.time()-t0) )
    print("Prediction time: {:.2f}".format( time.time()-t1) )

    total_time = total_time + (time.time()-t0)
    print("Average FPS: {:.1f}".format( n_frames/total_time) )

    maxpred = np.argmax(pred,axis=1)

    draw = ImageDraw.Draw(im_crop)
    for i in range(0,gH):
        for j in range(0,gW):
            index = gW*i + j
            x0, y0 = (j*200,i*200)        
            draw.rectangle([ (x0, y0 ), (x0 + 224, y0+224)])
            draw.text((x0, y0 ), "{:.0%}".format( pred[index][1] ),font=fnt,stroke_width=1)
    
    im_crop.save(dest + 'frame_{:03d}.png'.format(n_frames) )
    #plt.imshow(im_crop); plt.show()
    im_np = np.asarray(im_crop)

    # Display the resulting frame
    #opencv_image=cv2.cvtColor(im_crop, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame',cv2.cvtColor(np.asarray(im_crop), cv2.COLOR_RGB2BGR) )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# used for constructing video with ffmpeg afterwards!
print("Wall time: {:.2f}".format(time.time()-wall_time_t0) )

# Release capture
cap.release()
cv2.destroyAllWindows()

