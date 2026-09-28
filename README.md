# 🛣️ Road Lane Detection using OpenCV

This project detects road lane lines in real-time using classical computer vision techniques in Python and OpenCV.

## 📽️ How it works

1. Extracts frames from a video
2. Converts to grayscale and applies Gaussian blur
3. Detects edges using the Canny algorithm
4. Focuses only on the region of interest
5. Applies Hough Line Transform to detect lines
6. Draws the average lane lines on the original video

## 🧪 Files

- `LaneDetection.py` – main Python code
- `test_video.mp4` – input dashcam video
- `output.mp4` – final output with lane lines
- `output.png` – example output frame
- `test_image.jpg` – sample road image

## 💡 What's next?

Planning to upgrade this with Deep Learning (e.g. CNN-based segmentation) for better performance in tough conditions (night, rain, curves).

## 🛠️ Tools Used

- Python
- OpenCV
- NumPy

## 🎥 Output Preview


![Output Frame](Road%20Lane%20Detection/output.png)

---
## 👤 Author

**Manthan Patel**
- Linkedin: [Manthan Patel](https://www.linkedin.com/in/manthan-patel18)
- Portfolio: [Manthan Patel](https://manthan-portfolio-opal.vercel.app/)

---
🔗 Feel free to clone, learn, and modify!
