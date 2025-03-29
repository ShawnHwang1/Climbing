from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import matplotlib.pyplot as plt

# Setup config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16  # Change to match your dataset
cfg.MODEL.WEIGHTS = "output/model_final.pth"  # Path to your trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
cfg.DATASETS.TEST = ("my_dataset",)  # Make sure this matches what you registered

predictor = DefaultPredictor(cfg)
# Load image (make sure path is correct)
image = cv2.imread("train/IMG_8508.jpg")

# Run inference
outputs = predictor(image)
metadata = MetadataCatalog.get("my_dataset")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Visualize
v = Visualizer(image_rgb, metadata=metadata, scale=1.0)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Display using matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(out.get_image())
plt.axis("off")
plt.show()