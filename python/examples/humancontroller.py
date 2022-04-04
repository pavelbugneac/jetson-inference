#!/usr/bin/python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import argparse
import sys
import time

# import pyautogui

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)
import pyautogui

nose_y = 0
jump = 0

# process frames until the user exits
while True:
    # capture the next image
    img = input.Capture()

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=opt.overlay)
    # print the pose results
    print("detected {:d} objects in image".format(len(poses)))

    for pose in poses:
        print(pose)
        print(pose.Keypoints)
        print('Links', pose.Links)
        for point in pose.Keypoints:
        # find the keypoint index from the list of detected keypoints
        # you can find these keypoint names in the model's JSON file, 
        # or with net.GetKeypointName() / net.GetNumKeypoints()

            # print('test0')
            # left_elbow_idx = pose.FindKeypoint(('left_elbow'))
            # print('test1')
            # left_shoulder_idx = pose.FindKeypoint('left_shoulder')
            # print('test2')
            if point.ID == 7:
                left_elbow = point
                print(left_elbow)
                print('test3')
            if point.ID == 5:
                left_shoulder = point
                print(left_shoulder)
                print('test4')
            if point.ID == 8:
                right_elbow = point
                print(right_elbow)
            if point.ID == 6:
                right_shoulder = point
                print(right_shoulder)
            if point.ID == 0:
                nose = point
                print(nose)

                #pyautogui.press('d')
        try:
            if (left_elbow.x - left_shoulder.x) > 100:
                print("<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>.")
                pyautogui.keyDown('a')
            else:
                pyautogui.keyUp('a')
            del left_elbow
            del left_shoulder
        except:
            pyautogui.keyUp('a')

        try:
            if (right_elbow.x - right_shoulder.x) < 100:
                print("<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>.")
                pyautogui.keyDown('d')
            else:
                pyautogui.keyUp('d')
            del right_elbow
            del right_shoulder
        except:
            pyautogui.keyUp('d')

        try:
            if (nose.y - nose_y) > 20:
                print("<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>.")
                pyautogui.keyDown('w')
                # jump=0
            elif jump>300:
                pyautogui.keyUp('w')
                jump=0
        except:
            # pyautogui.keyUp('w')
            pass

    nose_y = nose.y
    jump+=1

        # # if the keypoint index is < 0, it means it wasn't found in the image
        # if left_wrist_idx < 0 or left_shoulder_idx < 0:
        #     continue
        
        # left_wrist = pose.Keypoints[left_wrist_idx]
        # left_shoulder = pose.Keypoints[left_shoulder_idx]

        # point_x = left_shoulder.x - left_wrist.x
        # point_y = left_shoulder.y - left_wrist.y

        # print(f"person {pose.ID} is pointing towards ({point_x}, {point_y})")

    # pose = poses[0]
    # print(pose.Keypoints)

    # try:
    #     left_elbow = pose.FindKeypoint('left_elbow')
    #     left_wrist = pose.FindKeypoint('left_shoulder')

    #     print(left_elbow)
    #     print(left_wrist)
    #     if (left_wrist.x - left_elbow.x) > 10:
    #         print('go to left')
        
    #     break
        
    # except:
    #     print('elbow or wrist not found')

    # if point_left:
    #     pyautogui.press('left')

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
