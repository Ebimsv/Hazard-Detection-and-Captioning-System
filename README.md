# **Hazard Detection and Captioning System**

## **Table of Contents**

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Prepare Input Data](#1-prepare-input-data)
  - [Run the System](#2-run-the-system)
- [Outputs](#outputs)
  - [Results File](#results-file)
  - [Example Output](#example-output)
- [Core Components](#core-components)
- [Our Key Contributions](#Our-Key-Contributions)
- [Testing](#testing)
  - [Test YOLO Detection](#test-yolo-detection)
  - [Test Captioning](#test-captioning)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## **Overview**

This project implements a hazard detection and captioning system for driver monitoring using videos. The system combines **YOLO** for object detection, **BLIP-based models** for caption generation, and a custom **state change detection** algorithm to evaluate driver behavior. It processes videos frame-by-frame, identifies hazards, generates descriptions for them, and outputs results in a **CSV format** compatible with competition scoring requirements.

---

## **Project Structure**

```graphql
├── data
│   ├── annotations
│   │   └── annotations.pkl               # Annotations for videos
│   └── videos
│       ├── video_0001.mp4                # Sample video file
│       └── video_0200.mp4                # Additional video files
├── models
│   ├── blip-image-captioning-base        # BLIP captioning model files
│   │   ├── config.json
│   │   ├── preprocessor_config.json
│   │   ├── pytorch_model.bin
│   │   ├── README.md
│   │   └── ...
│   └── YOLO_models                       # Pre-trained YOLO models
│       ├── yolo11n.pt
│       └── yolov8n.pt
├── pics                                  # Sample images for testing
│   ├── car.jpeg
├── README.md                             # Documentation file (this file)
├── requirements.txt                      # Dependencies for the project
├── results
│   └── results.csv                       # Output file for detection results
├── src                                   # Source code
│   ├── __init__.py
│   ├── main.py                           # Main script for video processing
│   └── utils                             # Utility modules
│       ├── captioning_utils.py           # Captioning-related utilities
│       ├── detection_utils.py            # Detection-related utilities
│       ├── state_change_utils.py         # State change detection logic
│       └── video_utils.py                # Video handling utilities
```

---

## **Installation**

### **Prerequisites**

- Python 3.9 or later.
- A machine with GPU support for efficient video processing (optional but recommended).

### **Setup**

1. **Clone the repository**:

   ```bash
   git clone git@github.com:Ebimsv/Hazard-Detection-and-Captioning-System.git
   cd Hazard-Detection-and-Captioning-System
   ```

2. **Install dependencies**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Models**:

   - **YOLO Models**:

     - Download and place pre-trained YOLO models (e.g., `yolov8n.pt` or `yolo11n.pt`) in the `models/YOLO_models` directory.
     - Alternatively, you can set `model_name = "yolo11n.pt"` or `model_name = "yolov8n.pt"` in `detection_utils.py` to reference the downloaded models.

   - **BLIP Captioning Model**:
     - Place the BLIP captioning model in `models/blip-image-captioning-base`.
     - Alternatively, set `model_name = "Salesforce/blip-image-captioning-base"` in `captioning_utils.py` to use the model from the Hugging Face repository.

## **Usage**

### **1. Prepare Input Data**

- **Annotations**: Ensure `annotations.pkl` is in `data/annotations`.
- **Videos**: Place video files in `data/videos`.

### **2. Run the System**

Execute the main script to process videos and generate results:

```bash
python src/main.py --annotations data/annotations/annotations.pkl --video_root data/videos --caption_model blip_base
```

### **Arguments**:

- `--annotations`: Path to the annotations file.
- `--video_root`: Directory containing video files.
- `--caption_model`: Captioning model (`blip_base`, `instruct_blip`, or `vit_g`).

---

## **Outputs**

### **Results File**

The system generates a `results.csv` file in the `results` directory. It contains:

- `ID`: Frame identifier (e.g., `video_0001_0` for the first frame of `video_0001.mp4`).
- `Driver_State_Changed`: Boolean flag (`True`/`False`) for state change detection.
- `Hazard_Track_X` and `Hazard_Name_X`: Tracks and descriptions of detected hazards, up to 22 slots.

### **Example Output**

```csv
ID,Driver_State_Changed,Hazard_Track_1,Hazard_Name_1,...,Hazard_Track_22,Hazard_Name_22
video_0001_0,False,1,"car detected",,,,,,,,,,,,,,,,,
video_0001_28,True,3,"bicycle detected",,,,,,,,,,,,,,,,,
```

### **Core Components**

1. **Driver State Change Detection**:

   - **Purpose**: Analyze driver behavior to identify changes in state (e.g., slowing down, reacting to hazards).
   - **Implementation**: A robust algorithm evaluates movement trends in detected hazards using median distances across frames. A custom threshold mechanism determines state changes.
   - **Contribution**:
     - Improved detection logic for accurately identifying driver state changes.
     - Integration of temporal filtering to minimize noise and false positives.

2. **Hazard Detection and Description**:

   - **Purpose**: Identify hazards in each video frame and provide meaningful descriptions.
   - **Implementation**:
     - **Object Detection**: Utilizes the **YOLOv8 model** for bounding box detection, class identification, and confidence scoring.
     - **Captioning**: Employs state-of-the-art captioning models (e.g., BLIP) to generate descriptions of detected objects.
     - **Dynamic Class Filtering**: Introduced a filtering mechanism for relevant YOLO classes (e.g., cars, pedestrians) to focus on impactful hazards.
   - **Contribution**:
     - Dynamically retrieved YOLO class names, removing the need for hardcoded labels.
     - Combined class names with captions for improved hazard descriptions.

3. **CSV Output Formatting**:

   - **Purpose**: Ensure output adheres to competition guidelines with consistent and structured data.
   - **Implementation**:
     - Automatically initializes a `results.csv` file with appropriate headers.
     - Records up to 22 hazards per frame, ensuring correct alignment of `Hazard_Track` and `Hazard_Name`.
     - Handles frames with no hazards gracefully by filling empty slots.
   - **Contribution**:
     - Streamlined hazard recording with unique identifiers and descriptive captions.
     - Adherence to competition requirements for structured CSV output.

4. **Video Processing Pipeline**:
   - **Purpose**: Efficiently process multiple videos for hazard detection and driver state analysis.
   - **Implementation**:
     - Frame-by-frame analysis using OpenCV.
     - Skips non-relevant frames for performance optimization.
     - Detects and tracks hazards dynamically across frames.
   - **Contribution**:
     - Developed a modular and extensible pipeline for large-scale video processing.
     - Optimized performance by integrating filtering and temporal consistency checks.

---

### **Our Key Contributions**

1. **Enhanced Hazard Detection**:

   - Implemented a **class filtering mechanism** to prioritize relevant YOLO classes.
   - Introduced a global counter for generating unique `Hazard_Track` values, ensuring meaningful hazard tracking.

2. **Dynamic Caption Generation**:

   - Combined YOLO-detected class names with state-of-the-art captioning models to produce accurate and interpretable hazard descriptions.
   - Improved interpretability of results by dynamically retrieving class names from the model.

3. **Robust Driver State Detection**:

   - Developed a novel approach to detect driver state changes using median movement trends.
   - Added a cool-down mechanism to prevent rapid toggling of state changes.

4. **Flexible and Modular Design**:

   - Designed a **YOLODetector** class that dynamically retrieves class names and processes detections efficiently.
   - Modularized the pipeline into distinct components (video processing, detection, captioning), making it adaptable for future improvements.

5. **Competition-Compliant Output**:
   - Ensured `results.csv` aligns with the competition format, including up to 22 hazards per frame with structured track-name pairs.
   - Addressed issues with missing or redundant hazards by integrating validation checks.

---

## **Testing**

### **Test YOLO Detection**

Run the YOLO detection module on a sample image:

```bash
python src/utils/detection_utils.py --image pics/car.jpeg --model YOLO_models/yolov8n.pt
```

### **Test Captioning**

Generate captions for an image:

```python
from PIL import Image
from src.utils.captioning_utils import get_captioner

image = Image.open("pics/car.jpeg")
captioner = get_captioner("blip_base")
caption = captioner.get_caption(image)
print("Generated Caption:", caption)
```

---

## **Future Work**

1. **Support for Advanced Models**:

   - Incorporate cutting-edge detection models (e.g., **YOLO11x**, **Detectron2**) for enhanced accuracy and robustness.
   - ⚠️ **High Computational Requirement**: These models may require significant GPU memory for inference or fine-tuning. For my case, with a laptop having only 2GB of VRAM, this is not feasible for training or real-time applications without a high-performance GPU.

2. **Improved Driver State Detection**:

   - Use advanced temporal algorithms, such as **LSTMs** or **Transformer-based models**, to better capture driver reactions and state changes.
   - ⚠️ **High Computational Requirement**: Training Transformer-based models on large datasets is GPU-intensive.

3. **Real-Time Hazard Monitoring**:

   - Adapt the system for **real-time hazard detection**, ensuring low latency for live monitoring applications.
   - Optimize for edge devices using lightweight models (e.g., **YOLO-Nano**, **MobileNet**).

4. **Hazard Severity Estimation**:

   - Implement a scoring mechanism to prioritize detected hazards based on their proximity, size, and potential impact.

5. **Multi-Hazard Scenarios**:
   - Enhance the pipeline to handle complex scenes with multiple overlapping hazards using advanced tracking or segmentation techniques (e.g., **Mask R-CNN**, **Panoptic Segmentation**).
   - ⚠️ **High Computational Requirement**: Segmentation models like Mask R-CNN are resource-intensive, requiring high VRAM for training.

---

## **Acknowledgments**

- **YOLO Models**: Powered by [Ultralytics YOLO](https://github.com/ultralytics/yolov5).
- **BLIP Captioning Models**: Provided by [Hugging Face Transformers](https://huggingface.co).
- Thanks to contributors and open-source communities for their tools and resources.

---

## **License**

This project is open-source and licensed under the MIT License.
