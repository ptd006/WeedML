# Copy random subset of files to test/train folders

import os
import glob
from sklearn.model_selection import train_test_split
from shutil import copy


# Directory structure in both src and dest needs to be same!
src = '/home/peter/ml/weeds/photos/1907/clip' #clipmerge' # "/home/peter/ml/weeds/1207/clip"
dest = "/home/peter/ml/weeds/traintestimages"

# only *.jpg will be considered (with glob)
def copy_random_pc(c,src,dest,pc=0.33):
    src1 = os.path.join(src,c,'*.jpg')    
    files = glob.glob(src1)
    train, test = train_test_split(files, shuffle=True,test_size=pc)
    print("Copying {} images to train".format(len(train)) )
    dest2 = os.path.join(dest,'train',c) #train destination
    if not os.path.exists(dest2):
        os.makedirs(dest2)

    for f in train:
        copy(f, dest2)

    print("Copying {} images to test".format(len(test)) )
    dest2 = os.path.join(dest,'test',c) #train destination
    if not os.path.exists(dest2):
        os.makedirs(dest2)
    for f in test:
        copy(f, dest2)


class_dirs = os.listdir(src)

# ['grass', 'dock', 'dandelion', 'thistle', 'stinger', 'clover', 'buttercup']

for c in class_dirs:
    print(c)
    copy_random_pc(c,src,dest,0.3)
