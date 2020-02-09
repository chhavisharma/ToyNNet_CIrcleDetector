import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets

# _________Globals__________

# loss depends on mac row, col, radius
maxlength = torch.from_numpy(np.array([200.0,200.0,50.0])) 
torch.manual_seed(99)
WORK_FOLDER = './'


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def find_circle(img, device, model): 
    # Fill in this function
    img = img.astype(np.float32)
    img  = torch.from_numpy(img).unsqueeze(0) # add channel dimension
    data = img.to(device)

    with torch.no_grad():
      data = data.unsqueeze(0) # batch size 1
      output = model(data)
    
    output = output.cpu() * maxlength
    result = output.numpy().flatten()
    return result


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


class CircleDetector(torch.nn.Module):
  def __init__(self):
    super(CircleDetector, self).__init__()

    # Backbone
    self.conv0    = nn.Conv2d(1, 4, 3, padding=0)
    self.conv0_bn = nn.BatchNorm2d(4, track_running_stats=True)

    self.conv1    = nn.Conv2d(4, 16, 5, padding=0)
    self.conv1_bn = nn.BatchNorm2d(16, track_running_stats=True)

    self.conv2    = nn.Conv2d(16, 32, 5, padding=0) 
    self.conv2_bn = nn.BatchNorm2d(32, track_running_stats=True)
    
    self.conv3    = nn.Conv2d(32,64, 5, padding=0) 
    self.conv3_bn = nn.BatchNorm2d(64, track_running_stats=True)

    self.conv4    = nn.Conv2d(64,128, 5, padding=0) 
    self.conv4_bn = nn.BatchNorm2d(128, track_running_stats=True)

    self.pool1    = nn.MaxPool2d(2, stride=2, padding=0)

    #Down
    self.convInterm    = nn.Conv2d(128, 64, 3, padding=1) 
    self.convInterm_bn = nn.BatchNorm2d(64, track_running_stats=True)

    #Projecting down to Y_hat dims
    self.conv1x4       = nn.Conv2d(64, 3, 2)

  def forward(self, x):

    #Backbone
    x = self.conv0(x) 
    x = self.conv0_bn(x)
    F.relu(x, inplace=True)
    x = self.pool1(x)

 
    x = self.conv1(x) 
    x = self.conv1_bn(x)
    F.relu(x, inplace=True)
    x = self.pool1(x)


    x = self.conv2(x) 
    x = self.conv2_bn(x)
    F.relu(x, inplace=True)
    x = self.pool1(x)

    x = self.conv3(x) 
    x = self.conv3_bn(x)
    F.relu(x, inplace=True)
    x = self.pool1(x)

    x = self.conv4(x) 
    x = self.conv4_bn(x)
    F.relu(x, inplace=True)
    x = self.pool1(x)

    #Intermediate
    x = self.convInterm(x) 
    x = self.convInterm_bn(x)
    F.relu(x, inplace=True)

    # Prediction
    out =  self.conv1x4(x) 
    
    # predict row,col,rad side-normalised
    out = out.squeeze(-1).squeeze(-1)

    return torch.sigmoid(out)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("_____Circle Detection_____")

    #Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = WORK_FOLDER + '/model_mse_20k_5e-3_50epochs.pth' #'/model_5e-3_50epochs.pth'
    model = CircleDetector()
    print("Device :",device)
    print("Model  : Circle Detector")
    print("Parms  :",count_parameters(model))    
    
    #Load weights
    if(not torch.cuda.is_available()):
        model.load_state_dict(torch.load(trained_model, map_location=lambda storage, loc: storage))
    else:
        model = model.to(device)
        model.load_state_dict(torch.load(trained_model))

    model.eval()

    # Run tests
    results = []
    for _ in range(1000):
        params, img = noisy_circle(200, 50, 2)
        detected = find_circle(img, device, model)
        results.append(iou(params, detected))
    results = np.array(results)

    print("AP over 1000 runs:",(results > 0.7).mean())
    del model

main()
