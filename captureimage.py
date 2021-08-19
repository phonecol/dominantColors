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
ap.add_argument("-r", "--reso", default = "l",
    help="resolution of the image captured( 'L' for low, 'M' for medium, 'H' for high)")
args = vars(ap.parse_args())
print(args)

hiRes = (2592, 1944)
# midRes = (1920,1080)
midRes = (2048,1080)
lowRes = (640,480)
if args["reso"] == "H":
    res = hiRes
elif args["reso"] == "M":
    res = medRes
elif args["reso"] == "L":
    res = lowRes

camera = PiCamera()
camera.resolution = res
camera.iso = args['exposure']
camera.shutter_speed = camera.exposure_speed
camera.rotation = 180


camera.brightness = 40
camera.saturation = 40
camera.contrast = 100
#camera.exposure_mode = 'off'

camera.start_preview()
# Camera warm-up time
sleep(2)
camera.annotate_text = "Press ENTER to start the image acquisition for the 1st image or CTRL-D to Quit"

# takePhoto = input("Press 'Y' then 'Enter' to take photos: ")
# print(takePhoto)
ts = datetime.datetime.now()
print(ts)
ts = ts.strftime("%Y-%m-%d-%H-%M")
directory = args["filename"]+'/'+str(ts) +','+str(args["exposure"])+'/'
parent_dir = '/home/pi/Desktop/dominantColors/captured_images/optimizationA/'
path = os.path.join(parent_dir, directory)
os.mkdir(path)
try:
    input("Press ENTER to start the image acquisition")
except SyntaxError:
    pass

print('taking photo')
tss1 = datetime.datetime.now()
tss1 = tss1.strftime("%S")     
print(tss1)
    #path + image number + PPM concentration + seconds in realtime + exposure value + time interval
filename1 = path + args["filename"]+','+str(tss1)+','+'1stphoto.png'

message = "First photo"
camera.annotate_text = str(message)
camera.capture(filename1)
camera.stop_preview()
sleep(2)

print('Image captured and saved as ', filename1 )
camera.start_preview()
# Camera warm-up time
sleep(2)
camera.annotate_text = "Press ENTER to start the image acquisition or CTRL-D to Quit"

try:
    input("Press ENTER to start the image acquisition")
except SyntaxError:
    pass

print('taking photos')



sleep(2)
#         filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H"))

for i in range(args["images"]):
    #capture image every 30 seconds
    tss = datetime.datetime.now()
    tss = tss.strftime("%S")
    print(tss)
    
    #path + image number + PPM concentration + seconds in realtime + exposure value + time interval
    filename = path +str(i).zfill(2) +','+ args["filename"]+','+str(tss)+','+str(i*30)+',seconds.png'
    
    camera.annotate_text = str(filename)
    camera.capture(filename)
    print('Image captured and saved as ', filename )
#     sleep(5)
#     camera.stop_preview()
#     sleep(20)
#     print('pause')
#     camera.start_preview()
    sleep(16)
    
camera.stop_preview()
print("Done taking photos")
    
    