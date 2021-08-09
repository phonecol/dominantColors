import time
import picamera
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-f","--filename", required=True,
    help ="filename of the images")
ap.add_argument("-i", "--images", type = int, default = 5,
    help="number of images to be captured")
args = vars(ap.parse_args())
print(args)


try:
    input("Press ENTER to start the image acquisition")
except SyntaxError:
    pass

print('taking photos')



with picamera.PiCamera() as camera:
    camera.start_preview()
    try:
        for i, filename in enumerate(
                camera.capture_continuous('/home/pi/Desktop/dominantColors/captured_images/'+args["filename"]+'/'+args["filename"]+','+'{timestamp:%H-%M-%S-%f}'+',seconds.png' )):
            camera.annotate_text = str(filename)
            print(filename)
            time.sleep(5)
            if i == 10:
                break
    finally:
        camera.stop_preview()
        
        