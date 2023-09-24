# Real-Time Face Mask Detection with MobileNetV2

A real-time face mask detection project using TensorFlow and OpenCV, which utilizes a pre-trained MobileNetV2 model to identify faces and determine whether a person is wearing a mask or not. This project is designed to help in enforcing safety measures during the COVID-19 pandemic.

![Example](example.png)

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [How it Works](#how-it-works)
- [Training Your Model](#training-your-model)
- [Acknowledgments](#acknowledgments)

## Introduction

The Real-Time Face Mask Detection project is a practical application of computer vision and deep learning. It combines face detection and mask classification to determine whether individuals are wearing masks properly. This can be utilized in various settings, including airports, healthcare facilities, and public spaces.

## Project Overview

- **Face Detection**: The project employs a pre-trained deep learning model (SSD-based) to detect faces within a video stream. This enables accurate face tracking even in crowded or dynamic environments.

- **Mask Classification**: A pre-trained MobileNetV2 model is used to classify faces into two categories: "With Mask" and "Without Mask." This classification helps in identifying individuals who are not following mask-wearing guidelines.

- **Real-Time Processing**: The system operates in real-time, making it suitable for monitoring purposes where immediate action is required.

## Key Features

- Real-time face mask detection with a webcam or video source.
- Accurate face detection and tracking using a pre-trained SSD-based model.
- Classification of individuals into "With Mask" or "Without Mask" categories.
- Visual annotations on the video stream to highlight mask-wearing status.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Yomna-moustafa/Face_Mask_Detection-.git
