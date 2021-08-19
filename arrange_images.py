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
    print(folder)
    folder_path = os.path.join(root_dir, folder)
    
#     print(folder_path)
    if os.path.isdir(folder_path):
        print('yes')
        subfolders = os.listdir(folder_path)
        subfolders = natsorted(subfolders)
#         print(subfolders)
        for subfolder in subfolders:
#             print(subfolder)
#             time = os.listdir(folder_path)
            subfolder_path = os.path.join(root_dir,folder)
            
            if os.path.isdir(subfolder_path):
                print('yesss')
#                 print(subfolder_path)
                sub2folders = os.listdir(subfolder_path)
                sub2folders = natsorted(sub2folders)
                print(sub2folders)
                images_path = os.path.join(root_dir,folder,subfolder)
                print("low")
                print(images_path)
                for sub2folder in sub2folders:
                
                    imagess = os.listdir(images_path)
                    print(imagess)
   

            
        
     
           
