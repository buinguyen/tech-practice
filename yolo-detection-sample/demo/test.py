#!/usr/bin/env python3

# Import for caffe
import os
import sys
import argparse
import cv2

import matplotlib
matplotlib.use('Agg')

import socket
import threading
from threading import Thread

sys.path.append(os.path.join(os.getcwd(),'socket/'))
from socket_server import SocketServer

sys.path.append(os.path.join(os.getcwd(),'darknet/binary/'))
import darknet as dn 

sys.path.append(os.path.join(os.getcwd(),'caffe/binary/python/'))
import gender_detection as gd 

#---------------------------------------------------------------------------
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

#---------------------------------------------------------------------------
cap = cv2.VideoCapture(1)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 240);

YOLO_CFG_FILE_PATH = os.path.join(os.getcwd(), 'darknet/models/tiny_yolo_face_ub_8box.cfg')
YOLO_WEIGHTS_FILE_PATH = os.path.join(os.getcwd(), 'darknet/models/tiny_yolo_face_ub_8box.weights')
YOLO_DATA__FILEPATH = os.path.join(os.getcwd(), 'darknet/models/yolo_face_person.data')

net = dn.load_net(YOLO_CFG_FILE_PATH, YOLO_WEIGHTS_FILE_PATH, 0)
meta = dn.load_meta(YOLO_DATA__FILEPATH)

#########################################################################

buffer_frames = []
buffer_index = 0
r = None
frame = None

gender = ""

def read_frame():
    global frame
    ret, frame = cap.read()
    im = dn.array_to_image(frame)
    dn.rgbgr_image(im)
    buffer_frames[(buffer_index + 1) % 3] = im

def process_frame():
    global r
    r = dn.detect2(net, meta, buffer_frames[(buffer_index + 2) % 3])

class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)
        global frame, buffer_frames, buffer_index
        ret, frame = cap.read()
        im = dn.array_to_image(frame)
        dn.rgbgr_image(im)
        buffer_frames = [im] * 3
        buffer_index = 0

        self.number_frames = 0
        self.fps = 60
        self.duration = (1.0 / self.fps) * Gst.SECOND  # duration of a frame in nanoseconds
        self.launch_string = ('appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width=320,height=240,framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96').format(self.fps)

    def on_need_data(self, src, lenght):
        if cap.isOpened():
            global frame, buffer_frames, buffer_index, gender
            buffer_index = (buffer_index + 1) % 3
            thread_read_frame = Thread(target = read_frame, args = ())
            thread_read_frame.start()

            thread_process_frame = Thread(target = process_frame, args = ())
            thread_process_frame.start()

            thread_read_frame.join() 
            thread_process_frame.join()

            #process opencv 
            for i in range(len(r)):
                obj, rate, pos = r[i]
                x,y,w,h = pos
                if obj == 'body':
                    cv2.rectangle(frame,(int(x-w/2),int(y-h/2)), (int(x+w/2),int(y+h/2)),(0,255,0),3)
                elif obj == 'face':
                    cv2.rectangle(frame,(int(x-w/2),int(y-h/2)), (int(x+w/2),int(y+h/2)),(0,0,255),3)   
                    face_img = frame[y-h/2:y+h/2, x-w/2:x+w/2]
                    gender = gd.detect_gender(face_img)         

            #continue
            data = frame.tostring()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            buf.duration = self.duration
            timestamp = self.number_frames * self.duration
            buf.pts = buf.dts = int(timestamp)
            buf.offset = timestamp
            self.number_frames += 1
            retval = src.emit('push-buffer', buf)

            if retval != Gst.FlowReturn.OK:
                print(retval)

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)


class GstServer(GstRtspServer.RTSPServer):
    def __init__(self, **properties):
        super(GstServer, self).__init__(**properties)
	print "init server"
        self.factory = SensorFactory()
        self.factory.set_shared(True)
        self.get_mount_points().add_factory("/test", self.factory)
        self.attach(None)

#-------------------------------------------------------------------------------

def getIpAddress():
    return [l for l in ([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1], [[(s.connect(('8.8.8.8', 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) if l][0][0]

#---------------------------------------------------------------------------
###################################### MAIN ###########################################

if __name__ == "__main__":

    # Thread for socket server
    socket_host = getIpAddress()
    socket_port = "8889"
    print "Start MAIN - host: ", socket_host, ", port: ", socket_port

    socketServer = SocketServer(socket_host, socket_port)
    socketThread = Thread(target = socketServer.communicate, args = (gender,))
    socketThread.setDaemon(True)
    socketThread.start()
    print "SocketServer is running"

    # Streaming thread
    GObject.threads_init()
    Gst.init(None)

    server = GstServer()

    loop = GObject.MainLoop()
    loop.run()

    # End
    socketServer.close()
    print "Close SocketServer"
