from time import sleep
from picamera import PiCamera
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-f","--filename", required=True,
    help ="filename of the images")
ap.add_argument("-i", "--images", type = int, default = 5,
    help="number of images to be captured")
args = vars(ap.parse_args())
print(args)


camera = PiCamera(resolution=(1280, 720), framerate=30)
# Set ISO to the desired value
camera.iso = 500
# Wait for the automatic gain control to settle
sleep(2)
# Now fix the values
camera.shutter_speed = camera.exposure_speed
camera.exposure_mode = 'off'
g = camera.awb_gains
camera.awb_mode = 'off'
camera.awb_gains = g
# Finally, take several photos with the fixed settings
camera.capture_sequence(['image%02d.jpg' % i for i in range(10)])
