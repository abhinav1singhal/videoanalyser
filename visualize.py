from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")

results = model("editedswim.mp4",save=True)
results[0].show()
