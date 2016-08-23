import numpy as np
import time
import cv2
import flycapture2 as fc2
from rx.subjects import Subject
import rx
from enum import Enum
import tkinter as tk
from tkinter.colorchooser import askcolor
from PIL import Image, ImageTk

class LogType(Enum):
    EVENT = 1
    SET = 2

class Camera(object):

    def __init__(self, camera_index):
        self.log_stream = Subject()
        self.image_stream = Subject()
        self.camera = cv2.VideoCapture(camera_index)

    def log_set(self, propset):
        self.stream.on_next(dict(type = LogType.SET,
                                 val = propset))
        
    def set(self, propset):
        self.context.set_property(**propset)
        self.log_set(propset)

    def log_event(self, msg):
        self.log_stream.on_next(dict(type = LogType.EVENT,
                                     val = msg))

    def start_capture(self):
        pass #self.context.start_capture()
        self.log_event("Start Capture")

    def stop_capture(self):
        pass #self.context.stop_capture()
        self.log_event("Stop Capture")

    def capture_image(self):
        ret, im = self.camera.read()
        hack = time.time()
        self.image_stream.on_next(dict(t=hack, image=im))

THUMBWIDTH=640
THUMBHEIGHT=480
THUMB_SIZE = (THUMBWIDTH, THUMBHEIGHT)
BLUR_SIZE = (7,7)
WHITE=(255,255,255)

# A circlefinder function maps (image, t) into
# (image, t, thumbs, position) where
# thumbs is a record of intermediate steps.
# this creates such a circlefinder function based on HSV
# masks
def make_hsv_circlefinder(hsv_low, hsv_high, thumbsize):

    def result(m):
        result = m.copy()
        im = m["image"]
        thumbs = []
        

class TrackerGUI(object):

    def __init__(self, master, camera):
        self.frame = tk.Frame(master)
        self.frame.pack()
        self.update_button = tk.Button(self.frame,
                                       text = "Set Background",
                                       command = self.acquire_background)
        self.update_button.pack()

        self.start_feed_button = tk.Button(self.frame,
                                           text = "Start Feed",
                                           command = self.start_feed)
        self.start_feed_button.pack()

        self.stop_feed_button = tk.Button(self.frame,
                                          text = "Stop Feed",
                                          command = self.stop_feed)
        self.stop_feed_button.pack()

        self.color_low_button = tk.Button(self.frame, text = "Lower Limit", command = self.ask_lower_match)
        self.color_low_button.pack()
        self.color_high_button = tk.Button(self.frame, text = "Upper Limit", command = self.ask_upper_match)
        self.color_high_button.pack()

        self.lower_match = (35,59,54)
        self.upper_match = (91,255,255)

        blackarr = np.zeros((THUMBWIDTH,THUMBHEIGHT))
        self.thumb = ImageTk.PhotoImage(Image.fromarray(blackarr))
        self.monitor = tk.Label(self.frame, image=self.thumb)
        self.monitor.pack()

        self.outline = tk.Label(self.frame, image=self.thumb)
        self.outline.pack()

        self.background = None
        self.background_thumb = None
        self.latest_image = None
        self.previous_image = None
        self.previous_thumb = None
        self.latest_thumb = None


        self.camera = camera
        self.camera.image_stream.subscribe(self.on_new_image)

        self.scheduler = rx.concurrency.TkinterScheduler(master)

        self.feed_sub = None

    def ask_lower_match(self):
        self.lower_match = askcolor()[0]
        print(self.lower_match)
        
    def ask_upper_match(self):
        self.upper_match = askcolor()[0]
        print(self.upper_match)
        
    def on_new_image(self, d):
        im = d["image"]
        hack = d['t']
        if self.background is None:
            self.background = im
            self.background_thumb = cv2.resize(im, THUMB_SIZE)
        else :
            self.previous_image = self.latest_image
            self.previous_thumb = self.latest_thumb
            self.latest_image = im
            self.set_monitor()

    def acquire_background(self):
        self.background = None
        self.background_thumb = None
        self.camera.capture_image()
    
    def acquire(self, val=0):
        self.camera.capture_image()
        return val

    def start_feed(self):
        if self.feed_sub is None:
            self.feed_sub = self.scheduler.schedule_periodic(50, self.acquire)

    def stop_feed(self):
        if self.feed_sub is not None:
            self.feed_sub.dispose()
            self.feed_sub = None
        
    def set_monitor(self):
        # im should be a numpy array
        print("mon")
        resized = cv2.resize(self.latest_image, THUMB_SIZE)
        thumb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        #edged = cv2.Canny(resized, 50, 100)
        if (self.previous_thumb is not None):
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_match, self.upper_match)
            mask = cv2.erode( mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

            if len(contours) > 0 :
                # from adrian, find the largest contour and min
                # enclosing circle and centroid
                c = max(contours, key=cv2.contourArea)
                ((x,y),radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                # learn more about this
                center = (int(M["m10"]/M["m00"]),
                          int(M["m01"]/M["m00"]))
                if radius > 10 :
                    cv2.circle(thumb, (int(x), int(y)), int(radius), (0,255,255), 2)
                    cv2.circle(thumb, center, 5, (0,255,255), 2)
        else:
            mask = np.zeros(resized.shape, dtype=resized.dtype)
        im = Image.fromarray(thumb)
        dim = Image.fromarray(mask)
        self.latest_thumb = resized
        self.thumb = ImageTk.PhotoImage(im)
        self.monitor.configure(image=self.thumb)

        self.diff_thumb = ImageTk.PhotoImage(dim)
        self.outline.configure(image = self.diff_thumb)
        
    
if __name__ == "__main__":
    misc_log = Subject()
    print_log = misc_log.subscribe(lambda x: print(str(x)))

    cam = Camera(0)
    joint_log = misc_log.merge(cam.log_stream)
    print_log.dispose()
    print_log = joint_log.subscribe(lambda x: print(str(x)))

    cam.start_capture()

    root = tk.Tk()
    app = TrackerGUI(root, cam)
    root.mainloop()

    cam.stop_capture()
    
    
