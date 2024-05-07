from PIL import Image
import os

# -----------------------------------------------------------------------------
GIF_FOLDER = "./dataset/images/08/"
PNG_FOLDER = "./dataset/images/08_png/"

GIFS = sorted([gif for gif in os.listdir(GIF_FOLDER) if gif.endswith(".gif")])


# -----------------------------------------------------------------------------
def gif_to_png(gif_path, png_path):
    with Image.open(gif_path) as img:
        rgba_img = img.convert("RGBA")
        rgba_img.save(png_path, format="PNG")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    for i, gif in enumerate(GIFS):
        gif_to_png(f"{GIF_FOLDER}{gif}", f"{PNG_FOLDER}{i+1:03d}.png")
