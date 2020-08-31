import os
import sys
sys.path.append('./')
from glob import glob

import random
import time
import argparse

import torch
import torchvision
from torchvision import models, transforms
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import skimage
import tkinter
import sort as SORT

from collections import defaultdict, OrderedDict

#python demo.py --video_name MOT17/train/MOT17-02-FRCNN --writeout True

#Set up the argument parser
parser = argparse.ArgumentParser(description='mot demo')
parser.add_argument('--detector', type=str, default='fasterrcnn',
                    help='Model for detection')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
#parser.add_argument('--bbox', default='', type=str, help='Init bounding box for consistent user input init box')
parser.add_argument('--gt', default='', type=str, help='bounding box ground truth')
parser.add_argument('--writeout', default=False, type=bool,
                    help='write to a file if True')
args = parser.parse_args()

# gt format 
# <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
# https://github.com/STVIR/pysot/blob/master/tools/demo.py
def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        print('reading image')
        print(os.path.join(video_name, '*.jp*'))
        images = glob(os.path.join(video_name, '*.jp*'))
        #print(images)
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))[0:5]
        for img in images:
            frame = cv2.imread(img)
            yield frame

# compute IoU between prediction and ground truth
def compute_iou(prediction, gt):
    #ensure the bounding boxes exist
    assert(prediction[0] <= prediction[2])
    assert(prediction[1] <= prediction[3])
    assert(gt[0] <= gt[2])
    assert(gt[1] <= gt[3])

    #intersection rectangule
    xA = max(prediction[0], gt[0])
    yA = max(prediction[1], gt[1])
    xB = min(prediction[2], gt[2])
    yB = min(prediction[3], gt[3])

    #compute area of intersection
    interArea = max(0, xB-xA + 1) * max(0, yB - yA + 1)

    #compute the area of the prection and gt
    predictionArea = (prediction[2] - prediction[0] +1) * (prediction[3] - prediction[1] +1)
    gtArea = (gt[2] - gt[0] + 1) * (gt[3]-gt[1]+1)

    #computer intersection over union
    iou = interArea / float(predictionArea+gtArea-interArea)
    return iou

#compute mean IoU in a frame
def mean_iou(iou_list):
    assert(len(iou_list) >0)
    return mean(iou_list)

#get color for different id
#https://github.com/ifzhang/FairMOT/blob/master/src/lib/tracking_utils/visualization.py
def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color



# This is to compute i
def compute_accuracy(prediction, gt):
    pass
   # return tp, tn, fp, fn, switch

# get mask R-CNN inference from model
def get_prediction(model, frame, device, threshold=0.9):
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    transform = transforms.Compose([transforms.ToTensor()])
    frame = transform(frame)
    frame = frame.to(device)
    pred = model([frame])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].detach(). numpy())]
    pred_boxes = [[i[0], i[1], i[2], i[3]] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    pred_score = pred_score[:pred_t+1]
    
    prediction_dicts = []
    for i in range(len(pred_boxes)):
        temp_dict ={}
        temp_dict['bbox'] = pred_boxes[i]
        temp_dict['labels'] = pred_class[i]
        temp_dict['scores'] = pred_score[i]
        prediction_dicts.append(temp_dict)  
    return prediction_dicts

#plot detection results
def get_detection(model, path, device, threshold=0.5, rect_th=2, text_size=1.5, text_th=2):
    pred_boxes, pred_class = get_predection(model, path, device, threshold)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(pred_boxes)):
        cv2.rectangle(img, pred_boxes[i][0], pred_boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
        cv2.putText(img, pred_class[i], pred_boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
    
    plt.figure(figsize=(15,20)) # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current device is', device)

    images = glob(os.path.join(args.video_name, '*.jp*'))
    images = sorted([item.split('/')[-1] for item in images])

    if args.gt:
        print('ground truth file is', args.gt)
        #ground truth represented as dictionary list by the first frame
        gt = defaultdict(list)
        with open(args.gt) as file:
            for line in file:
                #print(line)
                #<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <class>, <visibility>
                print(eval(line)[0])
                gt[images[eval(line)[0]-1]].append(list(eval(line)[1:]))

    # get faster RCNN 
    #https://github.com/mlvlab/COSE474/blob/master/3_Object_Detection_and_MOT_tutorial.ipynb
    #load faster R-CNN model
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model = model.to(device)

    print(model)
    odata = OrderedDict()
    i =0
    for frame in get_frames(args.video_name):
        print(images[i])
        start = time.time()
        #change different selection threhold here
        odata[images[i]]=get_prediction(model, frame, device, 0.8)
        i+=1
        print('process time is', time.time()- start)

    #tracker starts here
    save_path= './results/'
    mot_tracker = SORT.Sort()

    img_array = []

    for key in odata.keys():   
        arrlist = []
        det_img = cv2.imread(os.path.join(args.video_name, key))
        det_result = odata[key] 
        #read in the detection results
        for info in det_result:
            bbox = info['bbox']
            labels = info['labels']
            scores = info['scores']
            templist = bbox+[scores]
            
            if labels == 'person': # label 1 is a person in MS COCO Dataset
                arrlist.append(templist)
        
        start = time.time()        
        track_bbs_ids = mot_tracker.update(np.array(arrlist))
        print(track_bbs_ids.shape)
        print('Association update time is {} second'.format(round(time.time() - start, 4)))
        
        newname = save_path + key
        print(newname)

        for j in range(track_bbs_ids.shape[0]):  
            ele = track_bbs_ids[j, :]
            x = int(ele[0])
            y = int(ele[1])
            x2 = int(ele[2])
            y2 = int(ele[3])
            track_label = str(int(ele[4]))
            cv2.rectangle(det_img, (x, y), (x2, y2), get_color(int(ele[4])), 4)
            cv2.putText(det_img, track_label, (x, y+20), 0,0.6, (255,255,255),thickness=2)
        
        img_array.append(det_img)
        if args.writeout:
            height, width, _ = img_array[0].shape
            size = (width, height)
            out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()
        else:
            cv2.imwrite(newname,det_img, [cv2.IMWRITE_JPEG_QUALITY, 70])


if __name__ == '__main__':
	main()



