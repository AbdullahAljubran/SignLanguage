import os
from PIL import Image

# Path to the main folder
main_dir = "/Users/abdullahaljubran/Downloads/Sign_Language/data"

# Target size
target_size = (640, 480)

# Loop through each subfolder
for subdir in os.listdir(main_dir):
    subfolder_path = os.path.join(main_dir, subdir)

    # Ensure it's a directory
    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)

            # Check for image files
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                try:
                    img = Image.open(file_path)
                    img_resized = img.resize(target_size)
                    img_resized.save(file_path)  # Overwrite the original
                    print(f"Resized: {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")
