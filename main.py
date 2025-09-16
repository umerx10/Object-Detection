from ultralytics import YOLO
import sys

def run_detection(source):
    # Load YOLOv8 small model 
    model = YOLO("yolov8s.pt")

    # Run detection
    results = model(source, show=True, save=True)

    # Save results automatically 
    print(f"Results saved to: {results[0].save_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_or_video_path_or_0>")
    else:
        source = sys.argv[1]                   # can be image path or video 
        run_detection(source)
