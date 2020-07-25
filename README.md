# WeedML

Inspired by recent literature like [https://github.com/AlexOlsen/DeepWeeds] and efforts such as [https://forums.fast.ai/t/grassland-weed-detector/7635/82] we explore deep nets for weed recognition to be deployed on an autonomous spray robot.

Model is currently ResNet50.  We split input image from camera into 224px square boxes (with small overlap) and classify (softmax final) into spray/dontspray.  The reason for not using detectors like SSD/YOLO is partly speed and also that the spray application is confined to a grid anyway.

Labelling and first commit here is a hacky tool to help label photos faster.  Photos are cropped to a central window and split into 8x2 224px square overlapping boxes, similar to what the rover would do.  An existing model is used to generate predictions, which are shown to user.  A single key represents each class.  After key press the image clip is written to appropriate directory.  Tkinter is used.  As such it is fairly specific to this situation but hopefully might be useful to others.

Work flow is still improving.  Photos are taken on a Samsung Galaxy phone (and a Canon G7 for testing) from 3 farms in the UK.
1. Files are pulled to laptop (rclone from Google photos) and rescaled aiming for around 8px/cm.  
1. Pruning of files.
1. label_tool.py is used.
1. resample_new_photos.py is run.
1. generate_labels_csv.py is run (this code could be combined with previous step).  This creates the labels spray and dontspray (binary classification) in train.csv and test.csv.
1. train.py loads the images and labels specified in train.csv (and test.csv) using Keras `flow_from_dataframe` and does transfer learning on DeepWeeds ResNet50 model.

Rover plan is a Pixhawk PX4 controller [https://px4.io/] running ArduRover [https://ardupilot.org/] with u-Blox M8T RTK (rtklib) GPS but is not complete yet.

Full image dataset might be released eventually along with trained model at http://www.agrovate.co.uk/.
