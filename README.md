# Face Detection using AdaBoost

## Overview

Inspired by the face detection technology used in modern cameras, this project was developed as a lab assignment for the course **TBMI26: Neural Networks**. The goal was to implement the **AdaBoost algorithm** from scratch to detect faces in images.

The model utilizes **Haar-like features** to train weak classifiers, which are then combined into a strong classifier using AdaBoost to improve detection accuracy. The effectiveness of the approach was evaluated using the famous **1927 Solvay Conference on Physics** image, demonstrating promising results in accurately recognizing faces.

## Repository Structure
```bash
|- data
   |- faces.mat            # Training data for faces
   |- non_faces.mat        # Training data for non-faces
   |- solvay.mat           # Image data from the Solvay Conference, for final testing
|- util.py                 # Utility functions for feature extraction and classification
|- FaceDetection.ipynb     # Jupyter Notebook implementing face detection using AdaBoost
|- README.md               # Documentation with brief overview, key features and skills demonstrated.
```


## Key Features

- **AdaBoost Algorithm**: Implemented from scratch to classify face and non-face images.
- **Haar-like Features**: Extracted from training images to construct weak classifiers.
- **Ensemble Learning**: Combining multiple weak classifiers to form a strong classifier.
- **Performance Evaluation**: Tested on the historical **1927 Solvay Conference image**.

## Skills Demonstrated

- **Ensemble Learning**
- **Python (Programming Language)**
- **AdaBoost Algorithm**
- **Matplotlib for Visualization**
- **SciPy for Scientific Computing**

## License

This project is intended for educational and research purposes only.

## Acknowledgments

Developed as part of the **TBMI26: Neural Networks** course.
