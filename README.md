# Lane Detection using Computer Vision

A simple lane detection system using classical computer vision, demonstrated on a set of dashcam videos and sample images from the TuSimple dataset. Built to understand classical CV fundamentals before learning deep learning approaches.
## Demo
### Video Results
<p align="center">
  <img src="Classical%20CV/videos/output/demo.gif" width="600" alt="Lane Detection Demo">
</p>

### Image Results
<p align="center">
  <img src="Classical%20CV/visualization1.png"/>
    <br>
  <img src="Classical%20CV/visualization2.png" />
</p>

## 🚗 How It Works

### Pipeline Overview
The system processes frames through 8 sequential steps:
<p align="center">
<img src="Classical%20CV/pipeline.png"  alt ="Visualization of the Pipeline"/>
</p>

1. **Color Filtering** - Isolate white and yellow lane markings using HSL color space
2. **Grayscale Conversion** - Reduce computational complexity
3. **Gaussian Blur** - Reduce noise for cleaner edge detection
4. **Canny Edge Detection** - Identify sharp intensity gradients
5. **ROI Masking** - Focus on road area, ignore sky/trees
6. **Hough Transform** - Detect straight lines in edge image
7. **Slope Filtering** - Separate left/right lanes, reject noise
8. **Temporal Smoothing** - Average across frames for stability

### Key Algorithms

**Canny Edge Detection**
- Applies Sobel operators to find intensity gradients
- Uses non-maximum suppression to thin edges
- Applies hysteresis thresholding for robust edge selection

**Hough Transform**
- Converts image space (x, y) to parameter space (ρ, θ)
- Each edge pixel votes for possible lines through it
- Lines in image = peaks in accumulator array

## Results

**Works well on:**
- Straight highways with clear markings
- Well-lit conditions
- High-contrast lane lines

**Limitations:**
- Struggles with sharp curves (Hough detects straight lines)
- Sensitive to shadows and worn paint
- Requires manual ROI tuning for different camera angles

## Usage
```bash
# Process single video
python main.py

# Process image
# Edit main.py to specify input/output paths

# Visualize pipeline steps
python visualize.py
```

## Requirements
```bash
pip install opencv-python numpy matplotlib
```

## Future Work

- Implement polynomial lane fitting for curved roads
- Compare with CNN-based segmentation (U-Net, ENet)
- Adaptive thresholding for varying lighting conditions
- Real-time optimization for embedded deployment

## Learning Goals

This project explores:
- Classical computer vision techniques
- Understanding why deep learning is needed (limitations of geometric approaches)
- Real-time processing constraints
- Tradeoffs between accuracy and computational cost
- 
