
import os
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
from PIL import ImageDraw

COLOR_MAP = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
    "orange": (255, 165, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "black": (0, 0, 0),
    "white": (255, 255, 255)
}

class PolygonColorDataset(Dataset):
    def __init__(self, root_dir, split="training", transform=None):
        self.root = os.path.join(root_dir, split)
        self.input_dir = os.path.join(self.root, "inputs")
        self.output_dir = os.path.join(self.root, "outputs")
        self.transform = transform or T.ToTensor()
        
        with open(os.path.join(self.root, "data.json"), "r") as f:
            self.pairs = json.load(f)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
      entry = self.pairs[idx]
    
      # Load polygon image and output
      input_path = os.path.join(self.input_dir, entry["input_polygon"])
      output_path = os.path.join(self.output_dir, entry["output_image"])
    
      input_img = Image.open(input_path).convert("RGB")
      output_img = Image.open(output_path).convert("RGB")
      color = entry["colour"]

      # Convert color name to solid fill image
      color_rgb = COLOR_MAP[color]
      color_img = Image.new("RGB", input_img.size, color=color_rgb)

      # Generate polygon mask from input outline
      mask_img = self.get_polygon_mask(input_img)  # new method below

      # Convert all to tensors
      input_tensor = self.transform(input_img)      # [3, H, W]
      color_tensor = self.transform(color_img)      # [3, H, W]
      mask_tensor = self.transform(mask_img)        # [1, H, W]
      output_tensor = self.transform(output_img)    # [3, H, W]

      # Combine everything into one input tensor
      combined_input = torch.cat([input_tensor, color_tensor, mask_tensor], dim=0)  # [7, H, W]

      return combined_input, output_tensor


    def get_polygon_mask(self, img):
    
      gray = img.convert("L")  # Grayscale
      mask = Image.new("L", img.size, 0)
      draw = ImageDraw.Draw(mask)

    # Use bounding box of non-white pixels as approximate shape
      bbox = gray.getbbox()
      if bbox:
          draw.polygon([
              (bbox[0], bbox[1]),
              (bbox[2], bbox[1]),
              (bbox[2], bbox[3]),
              (bbox[0], bbox[3])
          ], fill=255)

      return mask



