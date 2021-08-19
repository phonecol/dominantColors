import os
import shutil
import glob
from natsort import natsorted

root_dir = '/home/pi/Desktop/dominantColors/captured_images'
dest_dir = '/home/pi/Desktop/dominantColors/images_of_papersensors'

print("hello")

folders = os.listdir(root_dir)
folders = natsorted(folders)
print(folders)
for folder in folders:
#     print(folder)
    folder_path = os.path.join(root_dir, folder)
    
    print(folder_path)
    if os.path.isdir(folder_path):
        print('yes')
        subfolders = os.listdir(root_dir)
        subfolders = natsorted(subfolders)
        for subfolder in os.listdir(folder_path):
            print(subfolder)
            
            if subfolder == '0ppm':
                print("ypw")
                for filename in os.listdir(subfolder_path):
                    file_path = os.path.join(root_dir, folder, subfolder, filename)
                    dest_path = os.path.join(dest_dir, filename)
                    shutil.copy(file_path, dest_path)
                    print("Copied ", file_path, "to", dest_path)