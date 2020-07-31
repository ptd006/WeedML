# WeedML

Inspired by recent literature like [https://github.com/AlexOlsen/DeepWeeds] and efforts such as [https://forums.fast.ai/t/grassland-weed-detector/7635/] we explore deep nets for weed recognition to be deployed on an autonomous spray robot.

Model is currently ResNet50 (but changes regularly :)).  We split the input photo into 224px square boxes (with small overlap) and classify (softmax) into spray/dontspray.  The reason for not using detectors like SSD/YOLO is speed and also that the spray application is confined to a grid anyway.

## Data preparation
Work flow is still improving.  Photos are taken on a Samsung Galaxy phone (and a Canon G7 for testing) from 3 farms in the UK midlands.
1. Files are pulled to laptop (rclone from Google photos) and rescaled aiming for around 8px/cm.  
1. Pruning of files.
1. label_tool.py is used.  This is a hacky tool to help label photos faster.  Photos are cropped to a central window and split into 8x2 224px square overlapping boxes, similar to what the will would do.  An existing model is used to generate predictions, which are shown to user.  A single key represents each class.  After key press the image clip is written to appropriate directory.  Tkinter provides interface.  It is fairly specific to this situation but hopefully might be useful to others.  XBox controller can be used.
1. resample_new_photos.py is run (70% of each class moved to train and rest to test).
1. generate_labels_csv.py is run (this code could be combined with previous step).  This creates the labels spray and dontspray (binary classification) in train.csv and test.csv.
1. train.py (example) loads the images and labels specified in train.csv (and test.csv) using Keras `flow_from_dataframe` and does transfer learning from DeepWeeds ResNet50 model.  This is for initial experiments on laptop (MX250 GPU).  Google Colab is used for deeper training.  It is convenient to zip images and upload to Gdrive, e.g. with rclone.

    zip -r traintestimages.zip traintestimages/* WeedML/*.csv
    rclone copy traintestimages.zip gdrive:/WeedML/ -P

Full image dataset might be released eventually (along with trained model) at http://www.agrovate.co.uk/.

### Using XBox Controller for label_tool.py

Game controllers are comfortable to hold and can take a beating.  Also, using them just feels more fun.  This makes them ideal for something monotonous like labelling images!  An XBox controller can be used with https://xboxdrv.gitlab.io/.  An example config file with key mappings for label_tool.py is included in the repo.

    sudo apt-get install xboxdrv
    sudo xboxdrv --config xboxdrv.ini

## Training

More details coming!

## Rover

Rover plan is a Pixhawk PX4 controller [https://px4.io/] running ArduRover [https://ardupilot.org/] with u-Blox M8T RTK (rtklib) GPS but is not complete yet.
