
import cv2
import numpy as np
capture = cv2.VideoCapture(0)
filter_derivate = np.array([[0,0,0],[0,-1,1],[0,0,0]])
while 1:
    ret,frame = capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(gray,(5,5),20)
    #img_filterX = cv2.filter2D(img,-1,filter_derivate)
    img_filterX = cv2.Canny(img,50,150)
    lines = cv2.HoughLines(img_filterX,2,np.pi/180,200)
    if np.any(lines == None):
        lines =0
      
    elif(len(lines) == 1):
        cosseno, seno = np.cos(lines[0,0,1]), np.sin(lines[0,0,1])
        x0,y0 = lines[0,0,0]*cosseno, lines[0,0,0]*seno
        delta = 1000
        x1,y1 = int(x0 + delta*seno), int(y0 - delta*cosseno)
        x2,y2 = int(x0 - delta*seno), int(y0 + delta*cosseno)
        cv2.line(frame,(x1,y1),(x2,y2),(50,25,70),5)
     
        print("")
    elif(len(lines)>1): 
        for d,theta in np.squeeze(lines):
            #print(d,theta)
            cosseno, seno = np.cos(theta), np.sin(theta)
            x0,y0 = d*cosseno, d*seno
            delta = 1000
            x1,y1 = int(x0 + delta*seno), int(y0 - delta*cosseno)
            x2,y2 = int(x0 - delta*seno), int(y0 + delta*cosseno)
            cv2.line(frame,(x1,y1),(x2,y2),(50,25,70),5)
     
    
    

    cv2.imshow('CAM',img_filterX)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
capture.release()        
cv2.destroyAllWindows()
