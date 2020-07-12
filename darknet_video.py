from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import math
import random
import os
import cv2
import numpy as np
import time


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(img, detections, classes):
    # Draw bounding boxes and labels of detections
    for xmin, ymin, xmax, ymax, conf, cls_conf, cls_pred in detections:
        label = classes[int(cls_pred)]
        score = cls_conf.item()
        print("\t+ Label: %s, Conf: %.5f" % (label, score))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    label +
                    " [" + str(round(score * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


def YOLO():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--video", type=str, required=True, help="input video")
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--output", default="./output", help="output dir")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(opt.video)
    cap.set(3, 1280)
    cap.set(4, 720)
    # out = cv2.VideoWriter(
    #     "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
    #     (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")

    while True:
        try:
            prev_time = time.time()
            ret, frame_read = cap.read()

            frame = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)

            # Extract image as PyTorch tensor
            img = transforms.ToTensor()(frame)

            # Pad to square resolution
            img, _ = pad_to_square(img, 0)
            # Resize
            img = resize(img, opt.img_size)
            img = img.unsqueeze(0)
            # Configure input
            input_imgs = Variable(img.type(Tensor))
            # Get detections
            with torch.no_grad():
                detections = model(input_imgs)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            detections = list(filter(lambda x: x is not None, detections))
            if detections is not None and len(detections) > 0:
                # Rescale boxes to original image
                detections = rescale_boxes(detections[0], opt.img_size, frame.shape[:2])
                frame = cvDrawBoxes(frame, detections, classes)
                current_time = datetime.datetime.now()
                if int(time.time()*10) % 10 == 0:
                    str_date = datetime.datetime.strftime(current_time, "%Y%m%d")
                    str_time = datetime.datetime.strftime(current_time, "%Y%m%d%H%M%S")
                    os.makedirs(os.path.join(opt.output, str_date), exist_ok=True)
                    cv2.imwrite(os.path.join(opt.output, str_date, str_time + ".jpg"), frame)
            # print(1/(time.time()-prev_time))
            if opt.display:
                cv2.imshow('Demo', frame)
                cv2.waitKey(3)
        except Exception as e:
            print("fail to detect", e)
    cap.release()
    out.release()


if __name__ == "__main__":
    YOLO()
