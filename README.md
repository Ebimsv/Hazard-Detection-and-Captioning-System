# **Hazard Detection and Captioning System**

## **Overview**

This project implements a hazard detection and captioning system for driver monitoring using videos. It utilizes YOLO for object detection, BLIP-based models for caption generation, and state change detection logic to evaluate driver behavior. The system processes videos frame-by-frame, detects hazards, generates captions for them, and outputs results in a specified CSV format for scoring.

---

## **Project Structure**

```graphql
├── data
│   ├── annotations
│   │   └── annotations.pkl               # Annotations for videos
│   └── videos
│       ├── video_0001.mp4                # Sample video file
|       |── .....
│       └── video_0200.mp4                # Additional video files
|── models
│   ├── blip-image-captioning-base        # BLIP captioning model files
│   │   ├── config.json
│   │   ├── preprocessor_config.json
│   │   ├── pytorch_model.bin
│   │   ├── README.md
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   └── vocab.txt
│   └── YOLO_models                       # Pre-trained YOLO models
│       ├── yolo11n.pt
│       └── yolov8n.pt
├── pics                                  # Sample images for testing
│   ├── dog.png
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

- Python 3.9 or later
- A machine with GPU support for efficient video processing (optional but recommended)

### **Setup**

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-repository/hazard-detection.git
   cd hazard-detection
   ```

2. **Install dependencies**:
   Create a virtual environment and install the required Python libraries:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Models**:
   Place pre-trained YOLO models (`yolov8n.pt`, `yolo11n.pt`) in the `YOLO_models` directory and BLIP models in `blip-image-captioning-base`.

---

## **Usage**

### **1. Prepare Input Data**

- **Annotations**: Provide an `annotations.pkl` file in `data/annotations` for video annotations.
- **Videos**: Place video files in the `data/videos` directory.

### **2. Process Videos**

Run the main script to process videos and generate results:

```bash
python src/main.py --annotations data/annotations/annotations.pkl --video_root data/videos --caption_model blip_base
```

### **Arguments**:

- `--annotations`: Path to the annotations file.
- `--video_root`: Path to the directory containing video files.
- `--caption_model`: Captioning model to use (`blip_base`, `instruct_blip`, or `vit_g`).

---

## **Outputs**

### **Results File**

The system outputs a `results.csv` file in the `results` directory. The CSV contains the following columns:

- `ID`: Frame ID (e.g., `video_0001_0` for the first frame of `video_0001.mp4`).
- `Driver_State_Changed`: Boolean flag (`True` or `False`).
- `Hazard_Track_X` and `Hazard_Name_X` (X = 1 to 22): Tracks and captions for detected hazards, padded with empty strings if fewer than 22 hazards are detected.

### **Example Output**

```csv
ID,Driver_State_Changed,Hazard_Track_1,Hazard_Name_1,...,Hazard_Track_22,Hazard_Name_22
video_0001_0,False,1,"car detected",,,,,,,,,,,,,,,,,
video_0001_1,False,2,"pedestrian detected",,,,,,,,,,,,,,,,,
video_0001_28,True,3,"bicycle detected",,,,,,,,,,,,,,,,,
```

---

## **Code Functionality**

### **1. Hazard Detection**

- The system uses YOLO models for object detection (`detection_utils.py`).
- Each detected object is tracked and paired with a centroid for movement analysis.

### **2. Caption Generation**

- BLIP-based models generate captions for each detected hazard (`captioning_utils.py`).
- Commas in captions are removed for CSV compatibility.

### **3. Driver State Change**

- Calculates driver state change based on movement patterns of detected objects (`state_change_utils.py`).
- Changes the flag `Driver_State_Changed` to `True` only once per video.

### **4. Video Handling**

- Reads videos frame-by-frame and processes each frame for hazards and captions (`video_utils.py`).

---

## **Testing**

### **Test YOLO Detection**

Use the `pics` directory to test YOLO detection on sample images:

```bash
python src/utils/detection_utils.py --image pics/car.jpeg --model YOLO_models/yolov8n.pt
```

### **Test Captioning**

Generate captions for a sample image:

```python
from PIL import Image
from src.utils.captioning_utils import get_captioner

image = Image.open("pics/car.jpeg")
captioner = get_captioner("blip_base")
caption = captioner.get_caption(image)
print("Generated Caption:", caption)
```

---

## **License**

This project is open-source and licensed under the MIT License.

---

## **Acknowledgments**

- **YOLO Models**: Powered by [Ultralytics YOLO](https://github.com/ultralytics/yolov5).
- **BLIP Captioning Models**: Provided by [Hugging Face Transformers](https://huggingface.co).
- Special thanks to contributors and open-source communities for their tools and resources.

---

## **Future Work**

- Support for more YOLO and captioning models.
- Improved driver state detection using advanced temporal algorithms.
- Enhanced scoring and evaluation metrics for hazard predictions.

Feel free to contribute to the project by submitting pull requests or issues. 🚗🔍
