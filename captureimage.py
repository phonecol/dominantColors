from time import sleep
from picamera import PiCamera
import argparse
import datetime
import os

ap = argparse.ArgumentParser()
ap.add_argument("-f","--filename", required=True,
    help ="filename of the images")
ap.add_argument("-i", "--images", type = int, default = 5,
    help="number of images to be captured")
ap.add_argument("-e", "--exposure", type = int, default = 320,
    help="exposure value of the camera")
args = vars(ap.parse_args())
print(args)

hiRes = (2592, 1944)
lowRes = (640,480)
camera = PiCamera()
camera.resolution = lowRes
camera.iso = args['exposure']
camera.shutter_speed = camera.exposure_speed
#camera.exposure_mode = 'off'

camera.start_preview()
# Camera warm-up time
sleep(2)
camera.annotate_text = "Press ENTER to start the image acquisition"
# takePhoto = input("Press 'Y' then 'Enter' to take photos: ")
# print(takePhoto)
try:
    input("Press ENTER to start the image acquisition")
except SyntaxError:
    pass

print('taking photos')


sleep(2)
ts = datetime.datetime.now()
print(ts)
#         filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H"))
ts = ts.strftime("%Y-%m-%d_%H-%M")
directory = args["filename"]+'/'+str(ts) +'/'
parent_dir = '/home/pi/Desktop/dominantColors/captured_images/'
path = os.path.join(parent_dir, directory)
os.mkdir(path)
for i in range(args["images"]):
    #capture image every 30 seconds
    
    filename =path + args["filename"]+','+str(i*30)+',seconds.png'
    
    camera.annotate_text = str(filename)
    camera.capture(filename)
    print('Image captured and saved as ', filename )
#     sleep(5)
#     camera.stop_preview()
#     sleep(20)
#     print('pause')
#     camera.start_preview()
    sleep(1)
    
camera.stop_preview()
print("Done taking photos")
    
    