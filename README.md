# 🧠 Polygon Color Filling with UNet (Conditioned on Color Name)

This project implements a deep learning model using a custom UNet architecture to generate filled polygon images. The model takes as input:
- A polygon outline image (e.g., triangle, octagon),
- A color name (e.g., “red”, “cyan”),

and outputs an RGB image of the polygon filled with the specified color.

---

## 📦 Dataset

- Structure:
  - `inputs/`: outline images of shapes.
  - `outputs/`: same shapes filled with specified colors.
  - `data.json`: mappings of input, color name, and expected output.
- Color options included: red, green, blue, yellow, cyan, magenta, purple, orange, white, black.
- Custom masks generated from the polygon outlines were used to guide training.

---

## ⚙️ Hyperparameters

| Parameter         | Value                          | Rationale                               |
|------------------|---------------------------------|------------------------------------------|
| Batch size        | 8                              | Fits comfortably in Colab GPU memory     |
| Learning rate     | 1e-3                           | Stable convergence with Adam             |
| Epochs            | 75                             | Longer training to allow mask-guided learning |
| Optimizer         | Adam                           | Adaptive and fast                        |
| Loss Function     | MSELoss (initial), then L1Loss | L1 used for sharper image generation     |
| Image Size        | 128x128                        | Trade-off between speed and quality      |

---

## 🧠 UNet Architecture and Conditioning

- Input shape: **[7, H, W]**
  - 3 channels: Polygon outline image
  - 3 channels: RGB image of the color to fill
  - 1 channel: **Binary mask of polygon region**
- Output shape: **[3, H, W]** RGB image of filled polygon

**Conditioning Method:**  
Color was injected as a full RGB image and concatenated along with the polygon image and the binary mask — a simple but effective way to spatially align the color intent with the shape.

**Ablations Tried:**
- Conditioning using only outline + color failed to fill regions correctly.
- Adding binary mask (generated from the outline itself) enabled the model to correctly learn polygon interiors.

---

## 📉 Training Dynamics

- Initial epochs: Model colored only the outlines.
- After ~25 epochs: It began to fill partial interiors.
- After adding the **mask channel** and training longer (75 epochs), the output quality improved significantly.

### 🔍 Typical Failure Modes:
| Mode | Description | Fix |
|------|-------------|-----|
| Outline-only coloring | Model followed edges but left interiors white | Added binary mask |
| Full-image color fill | Entire image became filled with color | Adjusted mask generation logic |
| Blurry outputs | Caused by MSE loss | Switched to `L1Loss()` for sharper boundaries |

---

## 📊 wandb Logging

- Project: [polygon-coloring-unet](https://wandb.ai/aashigupta-1509-mahindra-university/polygon-coloring-unet)
- Logs include train/val loss and model checkpoints.



---

## 🔑 Key Learnings

- **Mask supervision was essential** — model cannot infer fill regions from outline alone.
- Color conditioning via full RGB image worked better than one-hot vector injection.
- Adding visual logging (via wandb and inference notebooks) made debugging and evaluation intuitive.
- Even simple approximations (like using bounding boxes for mask) can greatly enhance training if aligned with the task objective.

---


