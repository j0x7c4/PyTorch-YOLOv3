from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
from torch2trt import torch2trt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model_path", type=str, required=True, help="path to input model")
    parser.add_argument("--output_model_path", type=str, required=True, help="path to output model")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.input_model_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.input_model_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.input_model_path))

    model.eval()  # Set in evaluation mode
    x = torch.ones((1, 3, 224, 224)).cuda()
    model_trt = torch2trt(model, [x])

    torch.save(model_trt.state_dict(), opt.output_model_path)
