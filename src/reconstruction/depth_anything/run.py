import sys
import pickle

import torch
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def run(img_path: str, output_path: str) -> None:
    image_processor: AutoImageProcessor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
    model: AutoModelForDepthEstimation = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")

    image: Image = Image.open(img_path)
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    depth_map: np.ndarray = prediction.squeeze().cpu().numpy()

    with open(output_path, "wb") as f:
        pickle.dump(depth_map, f)


if __name__ == "__main__":
    run(img_path=sys.argv[1], output_path=sys.argv[2])
