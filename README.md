# WeedML

Inspired by recent literature like [https://github.com/AlexOlsen/DeepWeeds] and efforts such as [https://forums.fast.ai/t/grassland-weed-detector/7635/82] we explore deep nets for weed recognition to be deployed on an autonomous spray robot.

Model is currently ResNet50 based on splitting input image from camera into boxes (with small overlap).  Mostly using own photos to reflect local conditions (might be released eventually)).  

Labelling is time consuming.  First commit here is a quick tool to help label tag images.  Photos are cropped to a central window and split into 8x2 224px square overlapping boxes, similar to how the rover would.  The best model so far to generate predictions, which are shown to user.  A single key represents each class.  After key press the image clip is written to appropriate directory.  Tkinter is used.  As such it is fairly specific to this situation but hopefully may be useful to others.

Work flow is still improving.  Photos are taken on a Samsung Galaxy phone (and a Canon G7 for testing) from 3 farms in the UK.
1. Files are pulled to laptop (rclone from Google photos) and rescaled aiming for around 8px/cm.  
1. Pruning is performed.
1. label_tool.py is used.
1. resample_new_photos.py is run.
1. generate_labels_csv.py is run (this code could be combined with previous step).  This creates the labels spray and dontspray (binary classification).
1. The images, train.csv and test.csv are ready to load using Keras `flow_from_dataframe` and train the model.


Rover plan is a Pixhawk PX4 controller [https://px4.io/] running ArduRover [https://ardupilot.org/] with u-Blox M8T RTK (rtklib) GPS but is not complete yet.

