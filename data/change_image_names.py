import os

# Path to the main directory containing the 8 folders
main_dir = '/Users/abdullahaljubran/Downloads/Sign_Language/data'  # <-- Replace this with the actual path

# Loop through each folder in the main directory
for folder_name in os.listdir(main_dir):
    folder_path = os.path.join(main_dir, folder_name)
    
    # Skip if not a folder
    if not os.path.isdir(folder_path):
        continue

    # Get list of image files in the folder
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    # Rename each image
    for idx, image_name in enumerate(image_files):
        ext = os.path.splitext(image_name)[1]
        new_name = f"{folder_name}_{idx:02d}{ext}"
        src = os.path.join(folder_path, image_name)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
        print(f"Renamed: {src} -> {dst}")
