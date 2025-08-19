# 🔍 RT-DETR Object Detection

This project implements real-time object detection using the RT-DETR model from Peking University. It supports detection from both webcam feeds and image folders, leveraging Python, OpenCV, and the Transformers library.

---

## 🚀 Features

- Real-time object detection using a webcam
- Batch processing of images in a folder
- Visualizes detected objects with bounding boxes and labels
- Uses the RT-DETR model (`PekingU/rtdetr_r50vd`) for high-performance detection
- Potential use cases: surveillance, autonomous systems, image analysis

---

## 📝 Tutorial
Dive into the code with my Medium tutorial: [Understanding Real-Time Object Detection with RT-DETR](https://medium.com/@sharavanan.mathivanan/understanding-real-time-object-detection-with-rt-detr-a-code-deep-dive-03f8bb2daafe).

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DiamondDolby/rtdetr-object-detection.git
cd rtdetr-object-detection
```

### 2. Install Required Packages

Make sure you have Python 3.7 or later installed.

#### Option A: If you have `requirements.txt`

```bash
pip install -r requirements.txt
```

#### Option B: Install manually

```bash
pip install opencv-python torch transformers
```

### 3. Download and Save the RT-DETR Model

To use the pre-trained RT-DETR model, run the provided `local_save.py` script to download and save the model locally.

```bash
python src/local_save.py
```

This will download the `PekingU/rtdetr_r50vd` model and save it to the `src/rtdetr_model` directory.

---

## ▶️ How to Run

Run the main script with:

```bash
python src/main.py
```

- **Choose mode**:
  - Enter `1` for real-time webcam detection (press `q` to quit the webcam feed).
  - Enter `2` for processing images in the `images` folder (output saved to the `output` folder).

---

## 📁 Project Structure

```perl
rtdetr object-detection/
├── src/
│   ├── local_save.py       # Script to download and save the RT-DETR model
│   ├── main.py             # Main script for object detection
│   ├── utils.py            # Utility functions for model loading and visualization
│   └── rtdetr_model/       # Directory for saved model
├── images/                 # Input folder for image processing
├── output/                 # Output folder for processed images
├── requirements.txt        # Dependency file
├── .gitignore             # Files/folders to be excluded from Git tracking
└── README.md              # This file
```

---

## 🖼️ Detection Logic

The RT-DETR model processes input images or webcam frames to detect objects. The `src/utils.py` script handles:

- Loading the pre-trained model and processor.
- Drawing bounding boxes and labels on detected objects with a confidence threshold of 0.3.
- Outputs include class labels and confidence scores overlaid on the images.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙋‍♂️ Author

### Sharavanan Mathivanan

#### [GitHub](https://github.com/DiamondDolby)

#### [LinkedIn](https://www.linkedin.com/in/sharkyca)
