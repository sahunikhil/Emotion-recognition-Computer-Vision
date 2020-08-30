
import cv2
import uuid
from datetime import datetime
from face_emocpython37 import predict

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

#predefines
faces = []
predList = []


while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_color = frame[y:y + h, x:x + w] 
        #print("[INFO] Object found. Saving locally.") 
        cv2.imwrite("others/faces/"+str(w) + str(h) + '_faces.jpg', roi_color)
        
        p = predict("others/faces/"+str(w) + str(h) + '_faces.jpg')
        
        cv2.putText(frame,text=p,org=(x+5, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,color=(0, 255, 0),thickness = 2)
        
        ID = uuid.uuid1()
        cv2.putText(frame,text=str(ID),org=(x, y+h+13),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,color=(0, 255, 0),thickness = 2)

        dateTimeObj = datetime.now()
        timeObj = dateTimeObj.time()

        cv2.putText(frame,text=str(timeObj),org=(x, y+h+30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,color=(0, 255, 0),thickness = 2)
           

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
