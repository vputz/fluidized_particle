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

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

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

def empty_acc():
    return dict(t=[], x=[], y=[], vx=[], vy=[])

def accumulate_steps(acc, current):
    """
    Accumulate steps in the tracking of the ball.  While "current" is the most
    recent output of the circlefinder function (which contains time hack, image, contours,
    position, and intermediate steps), "acc" only tracks t, x, y, vx, and vy.
    t, x, and y are taken from the "current" entry, but vx and vy are calculated
    using the last entries in acc.
    """
#    print("Curr: ", current)
    # if we're the first one, do a dummy
    acc['t'].append(current['t'])
    pos = current['position']
    acc['x'].append(pos[0] if pos is not None else np.nan)
    acc['y'].append(pos[1] if pos is not None else np.nan)

    if len(acc['t']) == 1:
        acc['vx'].append(np.nan)
        acc['vy'].append(np.nan)
    else:
        dt = acc['t'][-2] - acc['t'][-1]
        dx = acc['x'][-2] - acc['x'][-1]
        dy = acc['y'][-2] - acc['y'][-1]

        acc['vx'].append(dx/dt)
        acc['vy'].append(dy/dt)

    return acc

    

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

    def __init__(self, master, text, low, high):

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

        self.stream = Subject()
        self.low_match = low
        self.high_match = high

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

    def broadcast_range(self):
        self.stream.on_next(dict(low=self.low_match, high=self.high_match))
        
    def set_low(self, d):
        """d should be a dict keyed by hsv"""
        self.low_match = (d['h'],d['s'],d['v'])
        self.broadcast_range()

    def set_high(self, d):
        """d should be a dict keyed by hsv"""
        self.high_match = (d['h'],d['s'],d['v'])
        self.broadcast_range()

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
            self.mask_image = ImageTk.PhotoImage(Image.fromarray(mask))
            self.live_thumb.config(image = self.thumb_image)
            self.mask_thumb.config(image = self.mask_image)

class LiveVideoFrame(FrameTab):

    def __init__(self, master, text, low, high):

        FrameTab.__init__(self, master, text)
        self.pack()
        self.update_button = tk.Button(self,
                                       text = "Set Background",
                                       command = self.acquire_background)
        self.update_button.pack()


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

        self.circlefinder = make_hsv_circlefinder(low, high)

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

    def on_new_limits(self, limits):
        self.circlefinder = make_hsv_circlefinder(limits['low'], limits['high'])
                
    def acquire_background(self):
        self.background = None
        self.background_thumb = None
        self.camera.capture_image()

        
    def set_monitor(self):
        # im should be a numpy array

        this_step = self.circlefinder(self.last_step)

        thumb = cv2.resize(this_step["steps"][0], THUMB_SIZE)
        self.latest_thumb = thumb
        im = Image.fromarray(thumb)
        self.thumb = ImageTk.PhotoImage(im)
        self.monitor.configure(image=self.thumb)

        mask_thumb = cv2.resize(this_step["steps"][1], THUMB_SIZE)
        dim = Image.fromarray(mask_thumb)
        self.diff_thumb = ImageTk.PhotoImage(dim)
        self.outline.configure(image = self.diff_thumb)

class DataAcquisitionFrame(FrameTab):

    def __init__(self, master, text, low, high):
        FrameTab.__init__(self, master, text)

        self.start_button = tk.Button(self, text="Start Acquisition",
                                      command=self.start_acquisition)
        self.start_button.grid(row=0,column=0)
        self.stop_button = tk.Button(self, text="Stop Acquisition",
                                     command=self.stop_acquisition)
        self.stop_button.grid(row=0,column=1)

        self.thumb_width = 320
        self.thumb_height = 240

        blackarr = np.zeros((self.thumb_height, self.thumb_width))
        self.monitor_image = ImageTk.PhotoImage(Image.fromarray(blackarr))
        self.monitor = tk.Label(self, image=self.monitor_image)
        self.monitor.grid(row=1,column=0)

        self.stats_frame = tk.Frame(self)
        self.num_points = tk.Label(self.stats_frame, text="Num Points:")
        self.num_points.pack()
        self.stats_frame.grid(row=1,column=1)

        self.v_fig = Figure(figsize=(3,2), dpi=100)
        self.v_hist = FigureCanvasTkAgg(self.v_fig, master=self)
        self.v_hist.show()
        self.v_hist.get_tk_widget().grid(row=2,column=0)

        self.x_fig = Figure(figsize=(3,2), dpi=100)
        self.x_hist = FigureCanvasTkAgg(self.x_fig, master=self)
        self.x_hist.show()
        self.x_hist.get_tk_widget().grid(row=2,column=1)

        self.data_subscription = None
        self.camera_stream = None
        self.step_stream = None
        self.step_subscription = None

        self.circlefinder = make_hsv_circlefinder(low,high)

    def on_new_limits(self, limits):
        self.circlefinder = make_hsv_circlefinder(limits['low'], limits['high'])
    
    def start_acquisition(self):
        self.dispose_subscriptions()

        self.step_stream = self.camera_stream.map(self.circlefinder)
        self.step_subscription = self.step_stream.subscribe(self.on_new_step)
        self.data_subscription = self.step_stream.scan(accumulate_steps, seed=empty_acc()).subscribe(self.on_new_data)

    def on_new_step(self, step):
        if self.is_selected() :
            thumb = cv2.resize(step['steps'][0], (self.thumb_width, self.thumb_height))
            self.image = ImageTk.PhotoImage(Image.fromarray(thumb))
            self.monitor.config(image=self.image)

    def update_v_hist(self, acc):
        a = self.v_fig.gca()
        a.clear()
        a = self.v_fig.add_subplot(111)
        v_components = np.array([x for x in zip(acc['vx'], acc['vy']) if (not np.isnan(x[0])) and (not np.isnan(x[1]))])
        vs = np.sqrt((v_components*v_components).sum(1))
        n, bins, patches = a.hist(vs, normed=1)
        a.set_title("Normalized hist of |v|")
        self.v_fig.tight_layout()
        self.v_hist.draw()

    def update_x_hist(self, acc):
        a = self.x_fig.gca()
        
    def on_new_data(self, acc):
        self.num_points.config(text="Num Points: {0}".format(len(acc['t'])))
        EVERY=10
        if len(acc['t']) % EVERY == 0:
            self.update_v_hist(acc)

    def dispose_subscriptions(self):
        if self.step_stream is not None:
            self.step_stream = None
        if self.step_subscription is not None:
            self.step_subscription.dispose()
            self.step_subscription = None
        if self.data_subscription is not None:
            self.data_subscription.dispose()
            self.data_subscription = None
        
    def stop_acquisition(self):
        self.dispose_subscriptions()
        # ask to save data
        
        
class TrackerGUI(object):

    def __init__(self, master, camera, low, high):

        self.notebook = ttk.Notebook(master)
        self.scheduler = rx.concurrency.TkinterScheduler(master)
        self.FPS = 100
        self.camera = camera
        self.feed_sub = None

        self.color_chooser_frame = ColorChooserFrame(self.notebook, "Color Chooser", low, high)
        self.live_video_frame = LiveVideoFrame(self.notebook, "Live Video", low, high)
        self.data_acquisition_frame = DataAcquisitionFrame(self.notebook, "Data Acquisition", low, high)

        self.wire_subscriptions(camera)
        self.notebook.add(self.color_chooser_frame, text=self.color_chooser_frame.text)
        self.notebook.add(self.live_video_frame, text=self.live_video_frame.text)
        self.notebook.add(self.data_acquisition_frame, text = self.data_acquisition_frame.text)
        
        self.notebook.pack()
        self.notebook.bind("<Destroy>", self.on_destroy)
        self.start_feed()

    def wire_subscriptions(self, camera):
        self.live_video_frame.subscribe_to_camera_stream(camera.image_stream, 200, self.scheduler)
        self.color_chooser_frame.subscribe_to_camera_stream(camera.image_stream, 500, self.scheduler)
        self.color_chooser_frame.stream.subscribe(self.live_video_frame.on_new_limits)

        self.data_acquisition_frame.camera_stream = camera.image_stream

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
    low = (35,59,54)
    high = (91,255,255)
    app = TrackerGUI(root, cam, low, high)
    root.mainloop()

    cam.stop_capture()
    
    
