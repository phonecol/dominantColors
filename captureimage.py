from time import sleep
from picamera import PiCamera
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-f","--filename", required=True,
    help ="filename of the images")
ap.add_argument("-i", "--images", type = int, default = 2,
    help="number of images to be captured")
args = vars(ap.parse_args())
print(args)


camera = PiCamera()
camera.resolution = (1024, 768)
camera.iso = 547
camera.shutter_speed = camera.exposure_speed


#camera.start_preview()
# Camera warm-up time
sleep(2)

# takePhoto = input("Press 'Y' then 'Enter' to take photos: ")
# print(takePhoto)
try:
    input("Press ENTER to start the image acquisition")
except SyntaxError:
    pass

print('taking photos')
camera.start_preview()
sleep(2)

for i in range(args["images"]):
    #capture image every 30 seconds
    filename ='/home/pi/Desktop/dominantColors/captured_images/'+args["filename"]+'/'+args["filename"]+','+str(i*30)+',seconds.png'
    camera.capture(filename)
    print('Image captured and saved as ', filename )
    sleep(5)
    camera.stop_preview()
    sleep(20)
    print('pause')
    camera.start_preview()
    sleep(3.5)
    
camera.stop_preview()
print("Done taking photos")
    
    