# Viola-Jones-Face-Detector
Viola Jones face detector using Python from scratch. (No OpenCV implementations used).

Face Detection in Python using the Viola-Jones algorithm on the CBCL Face Database published by MIT's Center for Biological and Computational Learning.


# Code
- facedetector.py
  - An implementation of the Viola-Jones algorithm
  - Viola, Paul, and Michael Jones. "Rapid object detection using a boosted cascade of simple features." Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on. Vol. 1. IEEE, 2001. https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf
- cascade.py
  - An implementation of the attentional cascade introduced by Paul Viola and Michael Jones
- face_detection.py
  - Methods to train and test a ViolaJones classifier on the training and test datasets
  - Methods to train and test a CascadeClassifier on the training and test datasets

# Data
The data is described at http://cbcl.mit.edu/software-datasets/FaceData2.html, and I downloaded from www.ai.mit.edu/courses/6.899/lectures/faces.tar.gz and compiled into pickle files.

Each image is 19x19 and greyscale. There are Training set:  2,429 faces, 4,548 non-faces
Test set: 472 faces, 23,573 non-faces 


# Model

- final_classifier.pkl
  - A 10 feature Viola Jones classifier


# Results
The hyperparameter T for the ViolaJones class represents how many weak classifiers it uses. 

For T=10, the model achieved 85.5% accuracy on the training set and 78% accuracy on the test set.

