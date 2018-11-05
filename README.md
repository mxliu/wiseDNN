Keras implementation for the wiseDNN for brain disease prognosis

The code was written by Dr. Mingxia Liu and Dr. Jun Zhang, Department of Radiology at UNC-CH. 

Applications

We propose a weakly-supervised Densely-connected Neural Network (wiseDNN) for brain disease prognosis using baseline MRI data and incomplete clinical scores. Specifically, we first extract multi-scale image patches (located by anatomical landmarks) from structural MRI to capture local-to-global structural information of images, and then develop a weakly-supervised densely-connected network for task-oriented extraction of imaging features and joint prediction of multiple clinical measures. A weighted loss function is further employed to make full use of all available subjects (even those without ground-truth scores at certain time-points) for network training.


Prerequisites

Linux python 2.7

Keras version 2.0.8

NVIDIA GPU + CUDA CuDNN (CPU mode, untested) Cuda version 8.0.61

Getting Started

Installation

Install Keras and dependencies

Install numpywith pip install numpy

Download the pretrained model from https://drive.google.com/file/d/1GiMgubo6swgIgMBFE-zggcwxyT5ey01U/view?usp=sharing

Copy the model to the folder of Model/

cd to folder Code/ and

Apply our Pre-trained Model with GPU

python Main.py 

Note we use the Keras backend as follows { "image_data_format": "channels_first", "epsilon": 1e-07, "floatx": "float32", "backend": "tensorflow" }

Citation

If you use this code for your research, please cite our papers:

[1] Jun Zhang, Yue Gao, Yaozong Gao, Munsell Brent, and Dinggang Shen. Detecting Anatomical Landmarks for Fast Alzheimer’s Disease Diagnosis. IEEE Trans. on Medical Imaging, 35(12):2524-2533, 2016.

[2] Mingxia Liu, Jun Zhang, Ehsan Adeli, Dinggang Shen. Landmark-based Deep Multi-instance Learning for Brain Disease Diagnosis. Medical Image Analysis, 43: 157-168, 2018. 
