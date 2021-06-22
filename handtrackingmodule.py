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
        #print(results.multi_hand_landmarks)
    
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
    
def main():
    
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findhands(img)
        lmlist = detector.findpos(img)
        if len(lmlist) !=0:
            print(lmlist[4])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
    
        cv2.putText(img, str(int(fps)), (10,60), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    
        cv2.imshow('Image', img)
        cv2.waitKey(1)
        
    cap.release()
    cv2.destroyAllWindows()
    
        
if __name__ == '__main__':
    main()