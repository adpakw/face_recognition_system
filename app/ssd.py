import torch
# from src.detect import detect
import torch
import torchvision
import cv2
import numpy as np
import json
from PIL import Image
import os
import time

model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()