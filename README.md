# WeedML

Inspired by recent literature like [https://github.com/AlexOlsen/DeepWeeds] and efforts such as [https://forums.fast.ai/t/grassland-weed-detector/7635/82] we explore deep nets for weed recognition to be deployed on an autonomous spray robot.

Model is currently ResNet50 based on splitting input image from camera into boxes (with small overlap).  Mostly using own photos to reflect local conditions (might be released eventually)).  

Labelling is time consuming.  First commit here is a quick tool to help label tag images.  Photos are cropped to a central window and split into 8x2 224px square overlapping boxes, similar to how the rover would.  The best model so far to generate predictions, which are shown to user.  A single key represents each class.  After key press the image clip is written to appropriate directory.  Tkinter is used.  As such it is fairly specific to this situation but hopefully may be useful to others.

Rover will likely be based on a Pixhawk PX4 controller [https://px4.io/] running ArduRover [https://ardupilot.org/] with u-Blox RTK GPS but is not complete yet.

