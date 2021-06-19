import cv2 
import numpy as np 
import math 

class Tracker(object):
    """Encapsulates all functionalities related to keypoint motion tracking"""
    # params for ShiTomasi corner detection
    def __init__(self):
        self._feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )
        self._lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self._color = np.random.randint(0, 255, (100, 3))
        self._prev_gray_frame = None 
        self._gray_frame = None 
        self._prev_points = None
        self._points = None 
        self._mask = None
        self._good_prev = None
        self._good_new = None
        self._detect_interval = 100
        self._update_interval = 10
        self._total_dx = 0
        self._total_dy = 0 
        self._angle = 0 
     

    def getFirstKeyPoints(self, frame):
        """Takes first frame and find corners in it"""
        # find N strongest corners in the image by Shi-Tomasi method, unpack feature_params as arguments 
        self._prev_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._prev_points = cv2.goodFeaturesToTrack(self._prev_gray_frame, mask=None, **self._feature_params)
        self._mask = np.zeros_like(frame)

    def registerGrayFrame(self, frame):
        self._gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def updatePrevIteration(self):
        self._prev_gray_frame = self._gray_frame.copy()
        self._prev_points = self._good_new.reshape(-1, 1, 2)

    def getOpticalFlowPoints(self):
        """Obtain results of optical flow tracked points"""
        self._points, st, err = cv2.calcOpticalFlowPyrLK(self._prev_gray_frame, self._gray_frame, self._prev_points, None, **self._lk_params)
        # select good points 
        if self._points is not None:
            self._good_new = self._points[st==1]
            self._good_prev = self._prev_points[st==1]
            return True 
        else:
            print("No detection, redetect")
            return False 

    def processMotionDirections(self, frame, framesElapsed, draw = False):
        """Draw motion lines based on optical flow tracking"""
        drawn_img = None
        average_angle = 0
        if self._good_new is not None:
            counter = 0 
            for i, (new, old) in enumerate(zip(self._good_new, self._good_prev)): 
                #if counter > 10:
                #    break
                a, b = new.ravel()
                c, d = old.ravel()
                self._total_dx += (a-c) 
                self._total_dy += (b-d) 
                if draw:
                    #self._mask = cv2.line(self._mask, (int(a),int(b)),(int(c),int(d)), self._color[i].tolist(), 2)
                    frame = cv2.circle(frame,(int(a),int(b)),5, self._color[i].tolist(),-1)
            if draw:
                drawn_img = cv2.add(frame, self._mask)

            if self._total_dx > 5 or self._total_dy > 5:
                self.drawArrow(drawn_img, self._angle)
            
            if framesElapsed % self._update_interval == 0:
                if self._total_dx != 0:
                    self._angle = math.atan(self._total_dy/self._total_dx)
                self._total_dx = 0 
                self._total_dy = 0 

        else:
            print("good_new is None")

        return drawn_img

    def getDetectInterval(self):
        return self._detect_interval

    def drawArrow(self, frame, average_angle):
        arrow_length = 100
        start_point = (int(frame.shape[1]/2), int(frame.shape[0]/2))   
        if average_angle > 0:
            end_point = (start_point[0] - int(arrow_length*math.cos(average_angle)), start_point[1]-int(arrow_length*math.sin(average_angle)))  
        else:
            end_point = (start_point[0] + int(arrow_length*math.cos(average_angle)), start_point[1]-int(arrow_length*math.sin(average_angle)))  

        #print(end_point)
        color = (0, 255, 0)   
        thickness = 6
        frame = cv2.arrowedLine(frame, start_point, end_point, color, thickness) 

        

