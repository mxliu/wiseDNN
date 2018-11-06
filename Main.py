"""
Please use the theano matrix format and tensorflow backend.

change the file  $HOME/.keras/keras.json to

{
    "image_data_format": "channels_first",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}


python Main.py

Authors: Dr. Mingxia Liu and Dr. Jun Zhang
"""

import numpy as np
import keras
import argparse
from Generator import MultiInstance_patch
from Model import merged_model
import os
data_format = 'channels_first'
keras.backend.set_image_data_format(data_format)
parser = argparse.ArgumentParser(description='Landmark-based multiscale learning for AD prognosis')
parser.add_argument('--input', type=str, default='Img.nii.gz', required=False, help='Image path')
parser.add_argument('--outfolder',type=str,default='../Result/',required=False,help='Folder for saving results')
parser.add_argument('--landmark',type=int,default=40,required=False,help='number of landmarks')
parser.add_argument('--scale',type=int,default=2,required=False,help='number of scales')
opt = parser.parse_args()
print(opt)
if not os.path.isdir(opt.outfolder):
    os.mkdir(opt.outfolder)

# We set the sizes of patches as 24*24*24 and 48*48*48 in our experiments.
# The patches with the size of 48*48*48 will be first downsampled to 24*24*24, and then concatenated with the small-scale patches
patch_size = 24

# In this code, we attached the pre-trained model trained with 40 landmarks, which can found online ()
landmk_num =opt.landmark
# Since we used multi-scale patches, we need to set the scale to 2
numofscales=opt.scale
# In the training stage, we used the these weights to normalize the range of outputs (from 0 to 1) for four types of clinical scores
weight = np.tile(np.array([10,40,60,30]),4)

# Load the image
image =np.load('../Data/img.npy')

# Load the landmarks
# If you want to test your own data, you may need to generate anatomical landmarks using the code in https://github.com/zhangjun001/AD-Landmark-Prediction
new_landmark_test = np.load('../Data/landmark.npy')


subject_num = 1

# Extract the multi-scale patches
patch_data_test = MultiInstance_patch(image, new_landmark_test, patch_size, subject_num, numofscales)

# Load the model
M = merged_model(patch_size,landmk_num,numofscales)
M.load_weights('../Model/DiagnosisModel{0}_scale_{1}'.format(landmk_num,numofscales))

# Predict the scores
targets = M.predict(patch_data_test,batch_size=1)
score = targets[0]*weight

# Print results
names = ['BL', 'M06', 'M12', 'M24']
for i_time in range(4):
    print('Clinical Scores --> {0}'.format(names[i_time]) )
    print( 'CDR-SB:{0:0.2f}; ADAS-Cog11:{1:0.2f}; ADAS-Cog13:{2:0.2f}; MMSE: {3:0.2f}. '.format(
        score[i_time*4+0],score[i_time*4+1],score[i_time*4+2],score[i_time*4+3]))
# Save the result
np.save(opt.outfolder+'predicted_scores', score)
