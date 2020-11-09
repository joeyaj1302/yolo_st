#Using transfer learning to import pretrained yolo model and its weights to predict/detect objects on video in real time
import streamlit as st
import cv2
import numpy as np
import os
import time
import io
from PIL import Image
#import shutil

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Welcome to the Object detection Web App")
image = Image.open("street_output.png")
st.image(image,caption = 'Example of object detection by this web-app',width = 650)
if st.checkbox("Click here for a short description of how the yolo algorithm works"):
    image = Image.open("yolo_image.PNG")
    st.markdown("**The image below shows the working of the _YOLO_ algorithm**")
    
    st.markdown('''### **YOLO** or  you only look once is a state of the art object detection algorithm made by Joseph Redmon and trained by the darknet team. In simple words the algorithm:
    1. Divides the image or the frame of the video in a grid of SxS squares 
    2. Detects and gives the bounding boxes(x & y co-ordinates of detections with h & w)
        and confidence level of detected class 
    3. Non-max supression is done to find the best bounding box covering the detected
        object properly
    4. And lastly a score is generated which shows the probabilty of the prediction
        in  the bounding box  ''')
    st.image(image,width = 650)

st.write("For more detailed information on YOLO algorithm and its uses check the urls in the sidebar")
st.sidebar.markdown('''### Click on the following URLs for detailed explaination:
1. https://arxiv.org/pdf/1506.02640.pdf
2. https://www.coursera.org/lecture/convolutional-neural-networks/yolo-algorithm-fF3O0 
3. https://manalelaidouni.github.io/manalelaidouni.github.io/Understanding%20YOLO%20and%20YOLOv2.html''')
st.subheader("Select the file you want to upload")
option = st.selectbox("Select your choice of file :",['IMAGE',"VIDEO"])

def load_file(option):
    if option=="IMAGE":
        file1 = st.file_uploader("Upload an image",type = ['png','jpg','jpeg'])       
        return file1
    elif option =="VIDEO":
        folder_path='.'
        filenames = os.listdir(folder_path)
        file1 = st.file_uploader("Upload a video",type = ['mp4'])
        g = io.BytesIO(file1.read())  ## BytesIO Object
        temporary_location = "testout_sample.mp4"
        with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
            out.write(g.read())  ## Read bytes into file
            # close file
            out.close()
        return os.path.join(folder_path,temporary_location)

file1 = load_file(option)
#os.mkdir("HOME/temp")
folder_path = "."
temp_loc = "sample_video1.mp4"
#with open(temp_loc,"wb") as f:
    #print("the temp file is being created")
    #f.close()
#file2 = os.path.join(folder_path,temp_loc)

main_path = "main_path"
output_path = "output_path"
Labels_path = os.path.join(main_path,'coco.names')
LABELS = open(Labels_path).read().strip().split("\n")
st.write(" The following items can be classified by the ml model :",LABELS)

weights_path = os.path.join(main_path,'yolov3-tiny.weights')
config_path = os.path.join(main_path,'yolov3-tiny.cfg')

np.random.seed(32)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8") 


model = cv2.dnn.readNetFromDarknet(config_path,weights_path)
ln = model.getLayerNames()
ln = [ln[i[0]-1] for i in model.getUnconnectedOutLayers()]

if option=="IMAGE":    
    # load our input image and grab its spatial dimensions
    image = np.array(Image.open(file1))
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    #creating blob from image and passing it into the yolo model
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=False, crop=False)
    model.setInput(blob)
    start = time.time()
    layerOutputs = model.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence>0.3:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    indexs = cv2.dnn.NMSBoxes(boxes,confidences,0.3,0.1) #Non max supression 

    if len(indexs)>0:
        for i in indexs.flatten():
            (x,y) = (boxes[i][0],boxes[i][1])
            (w,h) = (boxes[i][2],boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)                 #drawing the bounding boxes over the detections
            text = "{}: {}%".format(LABELS[classIDs[i]], int(confidences[i]*100))  #Annotating the text of the class label
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)
    st.image(file1,caption = "This was your selected image")
    st.write("==============================================================")
    st.image(image,"The detections on your selected image")

elif option=="VIDEO":
    st.write("==============================================================")
    st.video(file1)
    #video_path = output_path+file1
    #instantiating the writer object that will later help in writing the predicted output video to disk
    writer = None
    (W,H) = (None,None)
    #video_path = os.path.join(main_path,a)
    vid = cv2.VideoCapture(file1)
    st.write("---------------------------------------------------------------")
    st.header("Your video is  being fed to the Machine learning model")
    st.write("---------------------------------------------------------------")
    #os.mkdir('out_path4')
    #temp = "sample.mp4"
    #video_out_path = os.path.join('out_path4',temp)
    #st.write(file1)
    st.markdown("""Note : Due to Heroku's ephemeral file system the written output video file 
                    does not get played and is erased from temp memory. So instead i,ve provided 
                    the option to view the frames while its being processed in real time.
                    It will keep populating the webpage with processed frames. 
                """)
    option2 = st.checkbox("Click here to view the frames being processed in real time")
    
    while True:
        i=0
        (confirmed , frame) = vid.read() #getting frames from video stream
        if not confirmed:
            break
        if W is None and H is None:
            (H,W) = frame.shape[:2]
         
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)  #BLob is similar to Image augmentation present in image data generators                                                                                         
        model.setInput(blob)                                                                 #of tensorflow
        start = time.time()
        layer_outputs = model.forward(ln)         #The captured frame is being passsed into the yolo model and the output is stored in layer_outputs
        end = time.time()

        #Declaring the lists that will contain data about class detections,co-ordinates of bounding boxes and the predicted classID's
        boxes = []
        confidences = []
        classIDs = []
        
        for outputs in layer_outputs:
            for detection in outputs:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence>0.3:       #Confidence values can be tuned from the optional args to get desired outputs
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerx,centery,width,height) = box.astype('int')
                    x = int(centerx - (width/2))         #getting the left most starting point from the center points for drawing bounding box later
                    y = int(centery - (height/2))
                    boxes.append([x,y,int(width),int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.3,0.3) #Non max supression to remove over lapping boxes

        #now to draw the bounding rectangles over the images
        #Make sure atleast one object is detected
        if len(indexes)>0:
            for i in indexes.flatten():
                (x,y) = (boxes[i][0],boxes[i][1])
                (w,h) = (boxes[i][2],boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,3)
                text = "{}: {}%".format(LABELS[classIDs[i]], int(confidences[i]*100))
                cv2.putText(frame,text,(x, y - 5),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 2)
                #cv2.imshow("video",frame)  #For displaying the frame being passed through the model and showing real time predictions
	        #st.image("video",frame)
            #Check if the video writer is None
               # i+=1
            if option2 == True:
                frame1 = cv2.resize(frame,(720,400))
                frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB)
                st.image(frame1,"Object detection on this frame")
            #cv2.imwrite("out_path4\img{}.jpg".format(i),frame)       
        #if writer is None:
            #Initialize our video writer to write the output video with predictions to output path specified on disk
            #out_path = "."
            #video_out_path = os.path.join(out_path,file1)
            #vid_out_path = os.path.join(output_path,file1)
            #fourcc = cv2.VideoWriter_fourcc(*'x246')
            #writer = cv2.VideoWriter(file2, fourcc, 30, (frame.shape[1], frame.shape[0]), True) # write the output frame to disk
        #writer.write(frame)
        
        
    #writer.release()
    #vid.release()
    #f.close()
    #st.video(file2)
    #st.write("The folder path is ",os.listdir("."))
    st.write("=========================Done====================================")
        
    #shutil.rmtree('out_path4', ignore_errors=True)           
