import os
import cv2
import torch
from ultralytics import YOLO

from utils import get_files

MODE = "IMAGE"  # Options: "IMAGE", "VIDEO", "REALTIME"


def main():
    model = YOLO("yolo12m.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if MODE == "IMAGE":
        image_path = "images"
        output_path = "output"

        images = get_files(image_path)
        for img_path in images:
            results = model(img_path)

            for result in results:
                annotated_img = result.plot()
                output_file = f"{output_path}/annotated_{os.path.basename(img_path)}"
                cv2.imwrite(output_file, annotated_img)

    elif MODE == "VIDEO":
        pass
    elif MODE == "REALTIME":
        cap = cv2.VideoCapture("http://192.168.0.53:4747/video")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv12 Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
