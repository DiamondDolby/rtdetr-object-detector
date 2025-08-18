import os
import cv2
import torch
from utils import load_model, draw_boxes

# Load model
processor, model = load_model()
model.eval()


def detect_from_webcam():
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue  # Skip every other frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        annotated = draw_boxes(frame, outputs, processor, model)
        cv2.imshow("RT-DETR Live Feed", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_from_images():
    input_folder = "images"
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(input_folder, filename)
            image = cv2.imread(path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            annotated = draw_boxes(image, outputs, processor, model)
            cv2.imwrite(os.path.join(output_folder, filename), annotated)
            print(f"Saved: {filename}")

if __name__ == "__main__":
    mode = input("Choose mode (1 = webcam, 2 = image folder): ")
    if mode == "1":
        detect_from_webcam()
    elif mode == "2":
        detect_from_images()
    else:
        print("Invalid mode.")
