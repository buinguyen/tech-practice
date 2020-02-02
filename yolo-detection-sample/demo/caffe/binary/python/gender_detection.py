import os
import sys
import os
import sys
import argparse
import cv2
import numpy as np
#import matplotlib
#matplotlib.use('Agg')

sys.path.append(os.path.join(os.getcwd(),'.'))
import caffe
from caffe.proto import caffe_pb2

DEFAULT_PROTOTXT =  os.path.join(os.getcwd(), 'caffe/models/deploy_gender.prototxt')
DEFAULT_MODEL    =  os.path.join(os.getcwd(), 'caffe/models/gender_net.caffemodel')
DEFAULT_LABELS   =  os.path.join(os.getcwd(), 'caffe/models/label.txt')
DEFAULT_MEAN     =  os.path.join(os.getcwd(), 'caffe/models/mean.binaryproto')

caffe_output = "prob"

# Functions for using caffe
def get_caffe_mean(filename):
    mean_blob = caffe_pb2.BlobProto()
    with open(filename, "rb") as f:
        mean_blob.ParseFromString(f.read())
    mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
                 (mean_blob.channels, mean_blob.height, mean_blob.width))
    return mean_array.mean(1).mean(1)

def show_top_preds(img, top_probs, top_labels):
    x = 10
    y = 40
    font = cv2.FONT_HERSHEY_PLAIN

    print "GENDER DETECTION"
    pred = ""
    for i in range(len(top_probs)):
        pred = "{:.4f} {:20s}".format(top_probs[i], top_labels[i])
        cv2.putText(img, pred, (x, y), font, 1, (0,0,240))
        y += 20
        print "GENDER: ", pred
    return pred

def detect(img, net, transformer, labels, caffe_output, crop):
    if crop:
        height, width, channels = img.shape
        if height < width:
            img_crop = img[:, ((width-height)//2):((width+height)//2), :]
        else:
            img_crop = img[((height-width)//2):((height+width)//2), :, :]
    else:
        img_crop = img;

    caffe.set_device(0)
    caffe.set_mode_gpu()

    net.blobs["data"].data[...] = transformer.preprocess("data", img_crop)
    output = net.forward()
    output_prob = output[caffe_output][0]
    top_inds = output_prob.argsort()[::-1][:3]
    top_probs = output_prob[top_inds]
    top_labels = labels[top_inds]
    return show_top_preds(img, top_probs, top_labels)

def detect_gender(frame):
    if frame is None:
        print "gender_detection frame is None"
        return
    
    return detect(frame, caffeNet, transformer, labels, caffe_output, None)


caffeNet = caffe.Net(DEFAULT_PROTOTXT, caffe.TEST, weights=DEFAULT_MODEL)
mu = get_caffe_mean(DEFAULT_MEAN)
print("Mean-subtracted values:", zip('BGR', mu))
transformer = caffe.io.Transformer({'data': caffeNet.blobs['data'].data.shape})
transformer.set_transpose("data", (2,0,1))
transformer.set_mean("data", mu)
labels = np.loadtxt(DEFAULT_LABELS, str, delimiter='\t')
