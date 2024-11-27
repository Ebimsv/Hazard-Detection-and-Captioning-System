from ultralytics import YOLO
import numpy as np
import cv2


class YOLODetector:
    def __init__(
        self,
        model_name="models/YOLO_models/yolov8n.pt",
        conf_threshold=0.5,
        input_size=640,
    ):
        """
        Initialize the YOLO detector with a specified model, confidence threshold, and input size.

        Parameters:
        - model_name: Path to the YOLO model file.
        - conf_threshold: Confidence threshold for detections.
        - input_size: Size to resize input images for the model.
        """
        try:
            self.model = YOLO(model_name)  # Initialize YOLO model
            self.conf_threshold = conf_threshold
            self.input_size = input_size  # Model's expected input size (e.g., 640x640)
            self.class_labels = (
                self.get_class_labels()
            )  # Fetch class labels from the model
        except Exception as e:
            raise RuntimeError(f"Failed to initialize YOLO detector: {e}")

    def get_class_labels(self):
        """
        Retrieve class labels from the YOLO model.

        Returns:
        - class_labels: List of class labels.
        """
        try:
            class_labels = (
                self.model.names
            )  # This returns a dictionary {class_id: class_name}
            return [class_labels[i] for i in range(len(class_labels))]
        except Exception as e:
            raise RuntimeError(f"Failed to load class labels from YOLO model: {e}")

    def get_bboxes(self, image):
        """
        Detect bounding boxes in an image using the YOLO model.

        Parameters:
        - image: Input image (BGR format).

        Returns:
        - bboxes: List of bounding boxes in [x1, y1, x2, y2] format.
        - track_ids: List of track IDs (if available, otherwise -1).
        - class_ids: List of class IDs for each detection.
        - confidences: List of confidence scores for each detection.
        """
        if image is None:
            raise ValueError("Input image is None. Please provide a valid image.")

        # Resize the image to the model's input size
        original_height, original_width = image.shape[:2]
        resized_image = cv2.resize(image, (self.input_size, self.input_size))

        try:
            results = self.model(
                resized_image, conf=self.conf_threshold
            )  # Run detection
        except Exception as e:
            raise RuntimeError(f"Error during YOLO inference: {e}")

        bboxes = []
        track_ids = []
        class_ids = []
        confidences = []

        try:
            for result in results:
                for box in result.boxes:  # Iterate through detected boxes
                    xyxy = box.xyxy[0].cpu().numpy()  # Extract [x1, y1, x2, y2]
                    conf = float(box.conf[0])  # Confidence score
                    cls = int(box.cls[0])  # Class ID
                    track_id = box.id  # Track ID (None if tracking not enabled)

                    if conf >= self.conf_threshold:  # Filter detections by confidence
                        # Scale bounding box coordinates back to original image size
                        scale_x = original_width / self.input_size
                        scale_y = original_height / self.input_size
                        x1, y1, x2, y2 = xyxy
                        x1, x2 = x1 * scale_x, x2 * scale_x
                        y1, y2 = y1 * scale_y, y2 * scale_y

                        bboxes.append([x1, y1, x2, y2])
                        confidences.append(conf)
                        class_ids.append(cls)
                        track_ids.append(track_id if track_id is not None else -1)

        except AttributeError as e:
            raise RuntimeError(f"Error accessing box attributes: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during bounding box processing: {e}")

        return np.array(bboxes), track_ids, class_ids, confidences


def load_image(image_path):
    """
    Load an image from the given path.

    Parameters:
    - image_path: Path to the image.

    Returns:
    - image: Loaded image in BGR format.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    return image


def draw_bounding_boxes(image, bboxes, class_ids, confidences, class_labels):
    """
    Draw bounding boxes on an image.

    Parameters:
    - image: Input image (BGR format).
    - bboxes: List of bounding boxes in [x1, y1, x2, y2] format.
    - class_ids: List of class IDs corresponding to each bounding box.
    - confidences: List of confidence scores for each detection.
    - class_labels: List of class names for each class ID.

    Returns:
    - image: Image with bounding boxes drawn.
    """
    for bbox, cls, conf in zip(bboxes, class_ids, confidences):
        x1, y1, x2, y2 = map(int, bbox)
        label = f"{class_labels[cls]}: {conf:.2f}"  # Class label with confidence
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return image


def visualize_and_save_image(
    image, window_name="Detections", output_path="pics/output_with_bboxes.jpg"
):
    """
    Visualize the image with bounding boxes and save it to a file.

    Parameters:
    - image: Image with bounding boxes drawn.
    - window_name: Name of the display window.
    - output_path: Path to save the image with bounding boxes.
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(output_path, image)
    print(f"Saved result image to {output_path}")


def test_yolo_detector(image_path, model_name="yolov8n.pt"):
    """
    Test the YOLODetector by detecting objects in an image and visualizing the results.

    Parameters:
    - image_path: Path to the test image.
    - model_name: Name of the YOLO model to use.
    """
    try:
        # Initialize the detector
        detector = YOLODetector(model_name=model_name, conf_threshold=0.5)

        # Load the test image
        image = load_image(image_path)

        # Detect bounding boxes
        bboxes, track_ids, class_ids, confidences = detector.get_bboxes(image)

        # Draw bounding boxes on the image
        image_with_bboxes = draw_bounding_boxes(
            image, bboxes, class_ids, confidences, detector.class_labels
        )

        # Visualize and save the image
        visualize_and_save_image(image_with_bboxes)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except RuntimeError as e:
        print(f"YOLO detection failed: {e}")
    except Exception as e:
        print(f"Unexpected error during testing: {e}")


if __name__ == "__main__":
    test_image_path = "pics/car.jpeg"
    try:
        test_yolo_detector(test_image_path, model_name="models/YOLO_models/yolo11n.pt")
    except Exception as e:
        print(f"Failed to test YOLO detector: {e}")
