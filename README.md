# Clothing Color Changer

Upload a photo of any clothing item and recolor it with a single click.

## How It Works

1. **Upload** a photo of a shirt, dress, jacket, or any garment
2. **Select** a target color from 20 options
3. Click **Recolor** — the app automatically detects the garment's dominant color, builds a pixel mask, and replaces the hue

## Technique
- Convert image to **HSV color space**
- Find dominant clothing color via **k-means clustering**
- Build a precise mask using **hue-range thresholding**
- Replace hue + saturation while **preserving brightness variation** (shadows, highlights stay natural)
- Morphological cleanup to remove noise

## Tips
- Works best with **solid-color garments** on a contrasting background
- Increase **tolerance** if the mask misses parts of the fabric
- For patterned clothing, the dominant color is recolored

## Live Demo
[Try it on Hugging Face Spaces](https://huggingface.co/spaces/Dnshitobu/clothing-color-changer)

## Run Locally
```bash
pip install -r requirements.txt
python app.py
```

---
Built by [Dnshitobu](https://github.com/Dnshitobu)
