"""
Clothing Color Changer — Gradio App
======================================
Upload a photo of clothing (shirt, dress, jacket, etc.),
select a target color, and the app recolors the garment.

Technique:
  1. Convert image to HSV
  2. Detect the dominant clothing color using k-means
  3. Create a mask for that color range
  4. Replace the hue + saturation with the target color
  5. Return the recolored image + color mask
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image


# ── Color palette ─────────────────────────────────────────────────────────────

TARGET_COLORS = {
    "Red":        (0,   200, 200),
    "Crimson":    (0,   230, 180),
    "Orange":     (15,  220, 220),
    "Yellow":     (28,  230, 230),
    "Lime Green": (60,  200, 200),
    "Green":      (70,  200, 180),
    "Teal":       (88,  210, 180),
    "Cyan":       (95,  220, 210),
    "Sky Blue":   (105, 210, 220),
    "Blue":       (115, 230, 210),
    "Navy":       (115, 230, 120),
    "Purple":     (140, 220, 200),
    "Magenta":    (155, 220, 220),
    "Pink":       (165, 180, 220),
    "White":      (0,   0,   250),
    "Light Gray": (0,   0,   200),
    "Dark Gray":  (0,   0,   100),
    "Black":      (0,   0,   30),
    "Beige":      (20,  80,  220),
    "Brown":      (12,  180, 130),
}


def get_dominant_color_hsv(hsv_img: np.ndarray, mask: np.ndarray | None = None):
    """Find dominant HSV color using k-means on hue-saturation channel."""
    pixels = hsv_img.reshape(-1, 3).astype(np.float32)
    if mask is not None:
        mask_flat = mask.reshape(-1)
        pixels = pixels[mask_flat > 0]

    # Filter out very dark/light (likely background) pixels
    sat  = pixels[:, 1]
    val  = pixels[:, 2]
    keep = (sat > 30) & (val > 40) & (val < 240)
    pixels = pixels[keep]

    if len(pixels) < 50:
        return None

    # K-means to find dominant cluster
    n_clusters = min(4, len(pixels) // 20)
    criteria   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, n_clusters, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
    )
    counts  = np.bincount(labels.flatten())
    dominant = centers[np.argmax(counts)]
    return dominant  # (H, S, V)


def build_color_mask(hsv_img: np.ndarray, dominant_hsv, tolerance: int = 25):
    """Build a mask for pixels close to the dominant hue."""
    h = int(dominant_hsv[0])
    s = int(dominant_hsv[1])
    v = int(dominant_hsv[2])

    # Handle hue wrap-around (red straddles 0/180)
    lo1 = np.array([max(h - tolerance, 0),   max(s - 60, 20),  max(v - 80, 20)],  dtype=np.uint8)
    hi1 = np.array([min(h + tolerance, 180),  min(s + 60, 255), min(v + 80, 255)], dtype=np.uint8)
    mask = cv2.inRange(hsv_img, lo1, hi1)

    if h - tolerance < 0:
        lo2 = np.array([h - tolerance + 180, max(s - 60, 20), max(v - 80, 20)], dtype=np.uint8)
        hi2 = np.array([180,                  min(s + 60, 255), min(v + 80, 255)], dtype=np.uint8)
        mask |= cv2.inRange(hsv_img, lo2, hi2)
    elif h + tolerance > 180:
        lo2 = np.array([0,                    max(s - 60, 20), max(v - 80, 20)], dtype=np.uint8)
        hi2 = np.array([h + tolerance - 180,  min(s + 60, 255), min(v + 80, 255)], dtype=np.uint8)
        mask |= cv2.inRange(hsv_img, lo2, hi2)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask


def recolor(image: Image.Image, target_color_name: str, tolerance: int):
    if image is None:
        return None, None, "Please upload an image."

    # Convert PIL to OpenCV BGR then HSV
    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    dominant = get_dominant_color_hsv(hsv)
    if dominant is None:
        return image, None, "Could not detect dominant clothing color. Try a photo with more fabric visible."

    mask = build_color_mask(hsv, dominant, tolerance=tolerance)

    # Apply target color in HSV
    target_h, target_s, target_v = TARGET_COLORS[target_color_name]
    result_hsv = hsv.copy()
    result_hsv[mask > 0, 0] = target_h
    result_hsv[mask > 0, 1] = target_s

    # Preserve original brightness variation (multiply value by ratio)
    if target_v > 0:
        orig_v = hsv[:, :, 2].astype(np.float32)
        max_v  = float(dominant[2]) if dominant[2] > 10 else 150.0
        ratio  = orig_v / max_v
        new_v  = np.clip(ratio * target_v, 0, 255).astype(np.uint8)
        result_hsv[:, :, 2] = np.where(mask > 0, new_v, hsv[:, :, 2])
    else:
        result_hsv[mask > 0, 2] = target_v

    result_bgr = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_rgb)

    # Visualize mask
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    mask_pil = Image.fromarray(mask_vis)

    covered_pct = 100 * np.sum(mask > 0) / mask.size
    status = (
        f"Detected dominant hue: {int(dominant[0])} | "
        f"Mask coverage: {covered_pct:.1f}% | "
        f"Target: {target_color_name}"
    )
    return result_pil, mask_pil, status


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Clothing Color Changer") as demo:
    gr.Markdown("# Clothing Color Changer")
    gr.Markdown(
        "Upload a photo of clothing (shirt, dress, jacket, hoodie...), "
        "choose a new color, and click **Recolor** to see the result."
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_input  = gr.Image(type="pil", label="Upload Clothing Photo")
            color_choice = gr.Dropdown(
                choices=list(TARGET_COLORS.keys()),
                value="Blue",
                label="Target Color",
            )
            tolerance_slider = gr.Slider(
                10, 50, value=25, step=5,
                label="Color Detection Tolerance (higher = broader mask)",
            )
            recolor_btn = gr.Button("Recolor", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Recolored Image")
            mask_image   = gr.Image(type="pil", label="Detected Clothing Mask")
            status_text  = gr.Textbox(label="Status", interactive=False)

    recolor_btn.click(
        fn=recolor,
        inputs=[image_input, color_choice, tolerance_slider],
        outputs=[output_image, mask_image, status_text],
    )

    gr.Markdown(
        "**Tips:** Works best with solid-color garments on a contrasting background. "
        "Increase tolerance if the mask misses parts of the clothing."
    )

if __name__ == "__main__":
    demo.launch()
