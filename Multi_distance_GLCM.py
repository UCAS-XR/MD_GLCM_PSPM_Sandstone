import os
import numpy as np
import pandas as pd
from skimage import io, color
from skimage.util import img_as_ubyte
from skimage.feature import graycomatrix


def calculate_glcms(image, distances=None, angles=None, levels=None):
    glcm = graycomatrix(image,
                        distances=distances,
                        angles=angles,
                        levels=levels,
                        symmetric=True,
                        normed=True)
    glcm_avg = np.mean(glcm, axis=3)
    return glcm_avg


def save_to_excel(glcms, output_folder, image_name, distances):
    os.makedirs(output_folder, exist_ok=True)
    for d in range(max(distances)):
        distance = d + 1
        output_path = os.path.join(output_folder, f"{image_name}_Distance_{distance}.xlsx")
        glcm_matrix = glcms[:, :, d]
        df = pd.DataFrame(glcm_matrix)
        df.to_excel(output_path, index=False, header=False)
        print(f"Saved: {output_path}")


def process_image(image_path, output_folder, distances, angles, levels):
    try:
        image = io.imread(image_path)
        if len(image.shape) == 3:
            image = color.rgb2gray(image)
        image = img_as_ubyte(image)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"Processing: {image_name}")
        glcms = calculate_glcms(image, distances, angles, levels)
        save_to_excel(glcms, output_folder, image_name, distances)
    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {str(e)}")


def process_folder(input_folder, output_folder, distances, angles, levels):
    if not os.path.isdir(input_folder):
        print(f"Error: Input path is not a directory - {input_folder}")
        return
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(input_folder, filename)
            process_image(image_path, output_folder, distances, angles, levels)


if __name__ == "__main__":
    input_folder = r"D:\GLCM\results\img_256"
    output_folder = r"D:\GLCM\results\GLCM_data"
    distances = list(range(1, 301))
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    levels = 256
    process_folder(input_folder, output_folder, distances, angles, levels)
    print("All images processed!")
