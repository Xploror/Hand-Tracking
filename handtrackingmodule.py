# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:29:16 2021

@author: surya
"""

import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, max_hands=2, det_confidence=0.5, track_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.det_confidence = det_confidence
        self.track_confidence = track_confidence

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.max_hands, self.det_confidence, self.track_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        

#To display the connected landmark drawing hence tracking hand
    def findhands(self, img, draw=True):
        colorimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(colorimg)
        print(self.results.multi_hand_landmarks)
    
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mphands.HAND_CONNECTIONS)
                    
        return img

#To store position of landmarks in a list
    def findpos(self, img, handnum=0, draw=True):
        
        lmlist = []
        
        if self.results.multi_hand_landmarks:
            Myhand = self.results.multi_hand_landmarks[handnum]
            
            for id, lm in enumerate(Myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id,cx,cy)
                lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (0,255,255), cv2.FILLED)    
    
        return lmlist
    
def tracking_index(lmlist, lmlist2, img, draw=True):
    if draw:
        if len(lmlist) !=0:
            cv2.circle(img, (lmlist[8][1], lmlist[8][2]), 10, (255,255,255), cv2.FILLED)
        if len(lmlist2) !=0:
            cv2.circle(img, (lmlist2[8][1], lmlist2[8][2]), 10, (255,255,255), cv2.FILLED)
                
        y,x,c = img.shape
        center_x, center_y = int(x/2), int(y/2)
        #Draw arrow from any hand index to center of image for cam tracking
        if len(lmlist) !=0:
            cv2.line(img, (center_x, center_y), (lmlist[8][1], lmlist[8][2]), (255,0,0), 1)
    
def main():
    
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findhands(img, draw=False)
        lmlist = detector.findpos(img, draw=False)
        try:
            lmlist2 = detector.findpos(img, 1, draw=False)  #For second hand
        except:
            pass
        
        tracking_index(lmlist, lmlist2, img)  #Tracking one index finger
        
        cTime = time.time()
        try:
            fps = 1/(cTime-pTime)
        except:
            pass
        pTime = cTime
    
        cv2.putText(img, str(int(fps)), (10,60), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    
        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
        if key == 27:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
        
if __name__ == '__main__':
    main()
