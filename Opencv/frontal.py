
import cv2
import numpy as np
capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('data/frontal.xml')
#face_alt_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')
#face_alt2_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
#face_alt_tree_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt_tree.xml')
while 1:
    ret,frame = capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    
    
    #cv2.namedWindow("CAM", cv2.WND_PROP_FULLSCREEN)
    #cv2.setWindowProperty("CAM",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

    
    #circles = cv2.HoughCircles(frame1, cv2.HOUGH_GRADIENT, dp=2, minDist=frame1.shape[0]/4, param1=200, param2=170)
# dp: amostragem / centro (2 pixels), minDist: menor raio, param1: threshold2 do Canny (o threshold1 vale a metade)
# param2: threshold p/ número de votos
   # print(np.squeeze(circles))
   # print(np.squeeze(circles).shape)
   # print(np.squeeze(circles))
   # if(np.squeeze(circles).shape!=(3,) and np.squeeze(circles).shape!=()):
    #    for x,y,r in np.squeeze(circles):
    #        cv2.circle(frame, (x,y), r, (0,0,0), 5)
   # elif(np.squeeze(circles).shape ==(3,)):
    #    x,y,r = np.squeeze(circles)
        #print(circles[0])
            
     #   cv2.circle(frame, (x,y), r, (0,0,0), 5)
        #plt.imshow(shapes,'gray'); plt.show()
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    #faces2 = face_alt_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    #faces3 = face_alt2_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    #faces4 = face_alt_tree_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    
    #Organiza numa as classificações numa lista para loop
    classifiers = [faces]
    
    # Coloca os quadrados nas faces
    for classifier in classifiers:
        for (x,y,w,h) in classifier:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            #print("oi")
            roi_gray = gray[y:y+h, x:x+w]
            #roi_color = img[y:y+h, x:x+w]
    cv2.imshow('CAM',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
capture.release()        
cv2.destroyAllWindows()

