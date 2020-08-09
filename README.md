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

## Model architecture and training

More details coming soon!

## Rover

Rover plan is a Pixhawk PX4 controller [https://px4.io/] running ArduRover [https://ardupilot.org/] with u-Blox M8T RTK (rtklib) GPS but is not complete yet.


## Testing on Android Phone

A live version in your pocket is handy for testing.  Quick summary-

1. Convert to TensorFlow lite https://www.tensorflow.org/lite/convert:
`tflite_convert --output_file=WeedML.tflite --keras_model_file=/home/peter/ml/weeds/WeedML/MNv2_01-08_47.h5`

1. Use demo code https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android

Tutorial clicky https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/

## Testing with webcam

Logitech C920 Pro is *just about* good enough (image quality and speed) for real time recognition.  See [run_model_webcam.py].

A few frames can be grabbed with:
`ffmpeg -f video4linux2 -s 1600x896 -i /dev/video2  -frames 10 -q:v 1 /tmp/%03d.jpg`

Use guvcview to play with settings (recommend disabling autofocus in actual usage)


## Multiclass test results
Experiment with binary multiclass.  It actually improves the test results on binary spray/dontspray even though 
training images are not properly labelled that way.  Buttercups are missed a lot but interestingly model still correctly says dontspray!

|           | count | class | spray |
|-----------|-------|-------|-------|
| dock      | 573   | 94.1% | 96.0% |
| thistle   | 214   | 86.9% | 92.5% |
| stinger   | 61    | 78.7% | 93.4% |
| grass     | 2002  | 99.9% | 96.1% |
| buttercup | 89    | 39.3% | 95.5% |
| clover    | 114   | 75.4% | 97.4% |
| spray     | 870   | 94.6% | 94.6% |
| dontspray | 1132  | 96.7% | 97.2% |
