#from persons import window2
import tkinter as tk
from tkinter import *
from yolov3 import PeopleDetector
from process import PostProcessor
from output import video
import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pygame import mixer 
from PIL import ImageTk, Image

# init yolo network , postprocessor and visualization mode
window = tk.Tk()
window.title("Social distancing monitoring App")
window.geometry('1350x750+0+0')
image = Image.open("corona1.png")
#image1=Image.open("soc.png")
#image.paste(image1)
#image.save('com.png', quality=95)
background_image = ImageTk.PhotoImage(image)
background = Label(window, image=background_image)    
background.pack(side='top')


    
def feed():
    global l2
    l2 = txt2.get() 
    window.destroy()
    net = PeopleDetector()
    net.load_network()
    
    # Process inputs
    parser = argparse.ArgumentParser(
        description='Run social distancing meter')
    parser.add_argument('--image', help='Path to image file.')
    parser.add_argument('--video', help='Path to video file.')
    args = parser.parse_args()
    winName = 'predicted people'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

    outputFile = "yolo_out_py.avi"
    if (args.image):
        # Open the image file
        if not os.path.isfile(args.image):
            print("Input image file ", args.image, " doesn't exist")
            sys.exit(1)
        cap = cv2.VideoCapture(args.image)
        outputFile = args.image[:-4]+'_yolo_out_py.jpg'
    elif (args.video):
        # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        outputFile = args.video[:-4]+'_yolo_out_py.avi'
    else:
        # Webcam input
        cap = cv2.VideoCapture(0)

    # Get the video writer initialized to save the output video
    if (not args.image):
        vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (round(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cv2.waitKey(1) < 0:

        # get frame from the video
        hasFrame, frame = cap.read()

        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv2.waitKey(3000)
            # Release device
            cap.release()
            break
        outs = net.predict(frame)
        pp = PostProcessor()
        indices, boxes, ids, confs, centers,count = pp.process_preds(frame, outs)
        cameraviz = video(indices, frame, ids, confs, boxes, centers,count)
        
        cameraviz.draw_pred()
        
        #print(count)
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.net.getPerfProfile()
        label = "Persons:"+str(count)
        cv2.putText(frame, label, (100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
        #l2 = txt2.get()            
        if str(count) > l2:
            label = "Warning:Person limit exceeded "+str(count)
            cv2.putText(frame, label, (200, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0,255))
            mixer.init()
            #for producing alarm when critical distance
            mixer.music.load('beep.ogg')
            mixer.music.play()
        # Write the frame with the detection boxes
        if (args.image):
            cv2.imwrite(outputFile, frame.astype(np.uint8))
        else:
            vid_writer.write(frame.astype(np.uint8))

        cv2.imshow(winName, frame)

message = tk.Label(window, text="Social Distancing Monitoring App", bg="GhostWhite", fg="black", width=40,
                   height=2, font=('times', 20, 'italic bold'))

message.place(x=400, y=130)

lbl2 = tk.Label(window, text="Enter max person limit allowed", width=35, height=2, fg="black", bg="GhostWhite", font=('times', 15, ' bold '))
lbl2.place(x=480, y=240)
txt2 = tk.Entry(window, width=25,bg="GhostWhite", fg="black", font=('times', 20, ' bold '))
txt2.place(x=480, y=330)
quitWindow = tk.Button(window, text="submit", command=feed  ,fg="black"  ,bg="Teal"  ,width=30  ,height=2, activebackground = "LightSeaGreen" ,font=('times', 15, ' bold '))
quitWindow.place(x=490, y=400)

window.mainloop()
