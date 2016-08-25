import numpy as np
import time
import cv2
import flycapture2 as fc2
from rx.subjects import Subject
import rx
from enum import Enum
import tkinter as tk
from tkinter import ttk
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


def largest_contour_circle(contours):
    """
    Finds the largest contour from a list of contours, and computes the 
    minimum enclosing circle and centroid.  Returns (circle, centroid)
    where circle is the min enclosing circle ((x,y), radius) and centroid
    is the (x,y) position of the center of the blob
    From Adrian's blog (pyimagesearch)
    """
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        circle = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"]/M["m00"]),
                  int(M["m01"]/M["m00"]))

        return circle, center
    else :
        return None, None

# A circlefinder function maps (image, t) into
# (image, t, steps, position) where
# steps is a record of intermediate steps.
# this creates such a circlefinder function based on HSV
# masks
def make_hsv_circlefinder(hsv_low, hsv_high, resized_size=None):

    def finder(m):
        result = m.copy()

        im = m["image"] if resized_size is None else cv2.resize(m["image"], resized_size)

        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_low, hsv_high)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # get the contours
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        result["contours"] = contours

        circle, centroid_center = largest_contour_circle(contours)

        result["position"] = centroid_center
        CIRCLE_COLOR = (0,255,255)
        CIRCLE_THICKNESS = 2
        CENTER_RADIUS = 5
        if circle is not None:
            ((x,y),radius) = circle
            if circle[1] > 10 :
                cv2.circle(im, (int(x), int(y)), int (radius),
                           CIRCLE_COLOR, CIRCLE_THICKNESS)
                cv2.circle(im, centroid_center, CENTER_RADIUS, CIRCLE_THICKNESS)
                
        result["steps"] = [im, mask]

        return result

    return finder

class HSVWidget(tk.Frame):

    def __init__(self, master, text, init_val=(0,0,0)):

        tk.Frame.__init__(self, master)
        self.pack()
        self.label = tk.Label(self, text=text)
        self.label.pack()
        self.slider_frame = tk.Frame(self)
        self.slider_frame.pack()
        self.h=tk.IntVar(value=init_val[0])
        self.h_label = tk.Label(self.slider_frame, text="H")
        self.h_label.grid(row=0, column=0)
        self.h_slider = tk.Scale(self.slider_frame, from_=0, to=255,
                                 variable=self.h,
                                 command=self.update_val, orient=tk.HORIZONTAL)
        self.h_slider.grid(row=0, column=1)
        self.s=tk.IntVar(value=init_val[1])
        self.s_label = tk.Label(self.slider_frame, text="S")
        self.s_label.grid(row=1, column=0)
        self.s_slider = tk.Scale(self.slider_frame, from_=0, to=255,
                                 variable=self.s,
                                 command=self.update_val, orient=tk.HORIZONTAL)
        self.s_slider.grid(row=1, column=1)
        self.v=tk.IntVar(value=init_val[2])
        self.v_label = tk.Label(self.slider_frame, text="V")
        self.v_label.grid(row=2, column=0)
        self.v_slider = tk.Scale(self.slider_frame, from_=0, to=255,
                                 variable=self.v,
                                 command=self.update_val, orient=tk.HORIZONTAL)
        self.v_slider.grid(row=2, column=1)
        self.slider_frame.pack()

        self.thumbsize = 100
        self.thumb_array = np.zeros((self.thumbsize, self.thumbsize,3), dtype=np.uint8)
        self.thumb_image = Image.fromarray(self.thumb_array, mode='HSV')
        self.thumb_tkimage = ImageTk.PhotoImage(self.thumb_image)
        self.thumb = tk.Label(self.slider_frame, image=self.thumb_tkimage)
        self.thumb.grid(row=0, column=2, rowspan=3, pady=5)
        self.update_thumb()
        
        self.val_stream = Subject()

    def update_thumb(self):
        self.thumb_image = Image.fromarray(self.thumb_array, mode='HSV')
        self.thumb_tkimage = ImageTk.PhotoImage(self.thumb_image)
        self.thumb.configure(image=self.thumb_tkimage)

        
    def update_val(self, _):
        # update thumb
        hsv = [self.h.get(), self.s.get(), self.v.get()]
        self.thumb_array[:,:,0] = hsv[0]
        self.thumb_array[:,:,1] = hsv[1]
        self.thumb_array[:,:,2] = hsv[2]
        self.update_thumb()
        self.val_stream.on_next({'h':hsv[0], 's':hsv[1], 'v':hsv[2]})
        
class FrameTab(tk.Frame):

    def __init__(self, master, text):

        tk.Frame.__init__(self, master)
        self.text = text

    def is_selected(self):
        return (self.master.tab(self.master.select(), "text") == self.text)
        
class ColorChooserFrame(FrameTab):

    def __init__(self, master, text):

        FrameTab.__init__(self, master, text)
        self.pack()

        self.THUMBWIDTH = 320
        self.THUMBHEIGHT = 240
        self.THUMB_SIZE = (self.THUMBHEIGHT, self.THUMBWIDTH)

        self.instructions = tk.Label(self, text=\
"""Change the low and high settings for H,S,V such that the mask in the
right-hand frame shows only the colored ping-pong ball""",
                                     wraplength=640)
        self.instructions.pack()
        
        self.bars_frame = tk.Frame(self)
        self.bars_frame.pack()
        self.low_hsv = HSVWidget(self.bars_frame, "Low value", (35,59,54))
        self.low_hsv.pack(side=tk.LEFT)
        self.low_subscription = self.low_hsv.val_stream.subscribe(self.set_low)
        self.high_hsv = HSVWidget(self.bars_frame, "High value", (91,255,255))
        self.high_hsv.pack(side=tk.LEFT)
        self.high_subscription = self.high_hsv.val_stream.subscribe(self.set_high)
        self.set_high({'h':0, 's':0, 'v':0})
        self.set_low({'h':0, 's':0, 'v':0})

        self.thumb_frame = tk.Frame(self)
        self.thumb_frame.pack()
        black = np.zeros(self.THUMB_SIZE)
        self.black_thumb = ImageTk.PhotoImage(Image.fromarray(black))
        self.live_thumb = tk.Label(self.thumb_frame, image=self.black_thumb)
        self.live_thumb.pack(side=tk.LEFT, padx=10)

        self.mask_thumb = tk.Label(self.thumb_frame, image=self.black_thumb)
        self.mask_thumb.pack(side=tk.LEFT, padx=10)

        self.cam_subscription = None

        self.bind("<Destroy>", self.dispose_subscriptions)

    def subscribe_to_camera_stream(self, camera_stream, delay, scheduler):
        self.throttled_camera = camera_stream#.throttle_last(delay, scheduler)
        self.cam_subscription = self.throttled_camera.subscribe(self.on_new_image)

    def dispose_subscriptions(self, event=None):
        #self.throttled_camera.dispose()
        print("disposing color mask subscriptions")
        self.cam_subscription.dispose()
    
    def set_low(self, d):
        """d should be a dict keyed by hsv"""
        self.low_match = (d['h'],d['s'],d['v'])

    def set_high(self, d):
        """d should be a dict keyed by hsv"""
        self.high_match = (d['h'],d['s'],d['v'])

    def on_new_image(self, d):
        # probably only update this if focused
        if self.is_selected():
            im = d['image']
            resized = cv2.resize(im, (self.THUMBWIDTH, self.THUMBHEIGHT))
            thumb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self.thumb_image = ImageTk.PhotoImage(Image.fromarray(thumb))
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.low_match, self.high_match)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            print (self.low_match, self.high_match)
            self.mask_image = ImageTk.PhotoImage(Image.fromarray(mask))
            self.live_thumb.config(image = self.thumb_image)
            self.mask_thumb.config(image = self.mask_image)

class LiveVideoFrame(FrameTab):

    def __init__(self, master, text):

        FrameTab.__init__(self, master, text)
        self.pack()
        self.update_button = tk.Button(self,
                                       text = "Set Background",
                                       command = self.acquire_background)
        self.update_button.pack()

        self.lower_match = (35,59,54)
        self.upper_match = (91,255,255)

        blackarr = np.zeros((THUMBHEIGHT,THUMBWIDTH))
        self.thumb = ImageTk.PhotoImage(Image.fromarray(blackarr))
        self.monitor = tk.Label(self, image=self.thumb)
        self.monitor.pack()

        self.outline = tk.Label(self, image=self.thumb)
        self.outline.pack()

        self.background = None
        self.background_thumb = None
        self.latest_image = None
        self.previous_image = None
        self.previous_thumb = None
        self.latest_thumb = None

        self.feed_sub = None

        self.bind("<Destroy>", self.dispose_subscriptions)

    def subscribe_to_camera_stream(self, stream, delay, scheduler):
        self.throttled_camera = stream#.throttle_last(delay, scheduler)
        self.camera_subscription = self.throttled_camera.subscribe(self.on_new_image)

    def dispose_subscriptions(self, event=None):
        print("disposing live view subscriptions")
        self.camera_subscription.dispose()
        
    def on_new_image(self, d):
        self.last_step = d
        if self.is_selected():
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

        
    def set_monitor(self):
        # im should be a numpy array
        finder = make_hsv_circlefinder(self.lower_match, self.upper_match)

        this_step = finder(self.last_step)

        thumb = cv2.resize(this_step["steps"][0], THUMB_SIZE)
        self.latest_thumb = thumb
        im = Image.fromarray(thumb)
        self.thumb = ImageTk.PhotoImage(im)
        self.monitor.configure(image=self.thumb)

        mask_thumb = cv2.resize(this_step["steps"][1], THUMB_SIZE)
        dim = Image.fromarray(mask_thumb)
        self.diff_thumb = ImageTk.PhotoImage(dim)
        self.outline.configure(image = self.diff_thumb)


class TrackerGUI(object):

    def __init__(self, master, camera):

        self.notebook = ttk.Notebook(master)
        self.scheduler = rx.concurrency.TkinterScheduler(master)
        self.FPS = 100
        self.camera = camera
        self.feed_sub = None

        self.color_chooser_frame = ColorChooserFrame(self.notebook, "Color Chooser")
        

        self.live_video_frame = LiveVideoFrame(self.notebook, "Live Video")

        self.wire_subscriptions(camera)
        self.notebook.add(self.color_chooser_frame, text=self.color_chooser_frame.text)
        self.notebook.add(self.live_video_frame, text=self.live_video_frame.text)
        self.notebook.pack()
        self.notebook.bind("<Destroy>", self.on_destroy)
        self.start_feed()

    def wire_subscriptions(self, camera):
        self.live_video_frame.subscribe_to_camera_stream(camera.image_stream, 200, self.scheduler)
        self.color_chooser_frame.subscribe_to_camera_stream(camera.image_stream, 500, self.scheduler)

    def on_destroy(self, event):
        print("stopping feed")
        self.stop_feed()
        
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
    
    
