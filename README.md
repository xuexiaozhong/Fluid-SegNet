# Fluid-SegNet
This repository contains the official implementation of Fluid-SegNet: Multi-Dimensional Loss-Driven Y-Net with Dilated Convolutions for OCT B-scan Fluid Segmentation. https://www.sciencedirect.com/science/article/abs/pii/S0895611125001223
We propose Fluid-SegNet, a deep learning model specifically designed to segment fluid regions from OCT B-scan images. Our approach addresses several key challenges in fluid segmentation, including:

Difficulty in capturing fine-grained fluid structures,

High heterogeneity of fluid regions,

Computational overhead when incorporating rich contextual information.

By introducing a multi-dimensional loss function and leveraging dilated convolutions within a Y-Net architecture, Fluid-SegNet achieves state-of-the-art performance in fluid region segmentation.

Below is the architecture diagram of Fluid-SegNet:

![Figm1_](https://github.com/user-attachments/assets/bacd4f9a-8b8e-4830-84b3-8cf0d5d69738)

# How to run it

Step 1:
Please download the datasets (UMN, OIMHS, and AROI)

Step 2:
For UMN dataset, you need run ROI_detection_by_Unet to get the ROI Images firstly.
For AROI dataset, we do not do any preprocessing.
For OIMHS dataset, we extracted ROI by GT directly.

Step 3:
Run main.py to segment.
