import os
from PIL import Image
import glob


def center_crop(input_dir, output_dir, target_size=(3072, 3072)):
    os.makedirs(output_dir, exist_ok=True)
    tiff_files = glob.glob(os.path.join(input_dir, "*.bmp"))
    for tiff_path in tiff_files:
        try:
            with Image.open(tiff_path) as img:
                orig_width, orig_height = img.size
                target_width, target_height = target_size
                left = (orig_width - target_width) // 2
                top = (orig_height - target_height) // 2
                right = left + target_width
                bottom = top + target_height
                cropped_img = img.crop((left, top, right, bottom))
                filename = os.path.basename(tiff_path)
                output_path = os.path.join(output_dir, filename)
                cropped_img.save(output_path, "TIFF")
                print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {tiff_path}: {str(e)}")


input_folder = r"D:\GLCM\img"
output_folder = r"D:\GLCM\img_RGB"
center_crop(input_folder, output_folder)
