import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection

def load_model():
    processor = AutoImageProcessor.from_pretrained("src/rtdetr_model")
    model = AutoModelForObjectDetection.from_pretrained("src/rtdetr_model")
    return processor, model

def draw_boxes(image, outputs, processor, model, threshold=0.3):
    target_sizes = [image.shape[:2][::-1]]  # (height, width) → (width, height)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
    id2label = model.config.id2label

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(i) for i in box.tolist()]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(image, f"{id2label[label.item()]}: {score:.2f}", 
                    (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return image
