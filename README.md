# Lane Detection using Computer Vision

A simple lane detection system using classical computer vision, demonstrated on the TuSimple dataset.

## 🎯 Project Goal

Build a lightweight lane detection model that:
- Detects lane boundaries in road images
- Draws predicted lane lines
- Demonstrates understanding of perception systems in autonomous vehicles

## 🚗 Motivation

Lane detection is a fundamental perception task in autonomous driving. This project explores:
- Computer vision techniques for road scene understanding
- Real-time processing constraints

## 📊 Dataset

**TuSimple Lane Detection Challenge**
- Highway driving scenarios
- Annotated lane markings
- 3,626 training images
- 358 validation images

## 🛠️ Approach

**Phase 1: Classical CV Baseline**
- Edge detection (Canny)
- Hough transform for line detection
- Region of interest masking

**Phase 2: Deep Learning Model** *(planned)*
- Lightweight CNN architecture
- Lane segmentation
- Post-processing for lane fitting

**Phase 3: Optimization** *(future)*
- Model quantization for embedded deployment
- Inference time optimization
- Discussion of TensorRT/ONNX deployment

## 🚀 Status

🔨 **Work in Progress**

Currently implementing classical CV baseline.

## 💡 Next Steps

- [ ] Download and preprocess TuSimple dataset
- [ ] Implement classical CV pipeline
- [ ] Train initial deep learning model
- [ ] Evaluate and visualize results
- [ ] Document optimization strategies for automotive deployment

## 🎓 Learning Goals

- Understanding perception pipelines
- Balancing accuracy vs. real-time performance
- Understanding the basics of autonomous driving perception

---
