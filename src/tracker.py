# Referenced: Programming Computer Vision with Python: Tools and Algorithms for Analyzing Images by Jan Erik Solem

import cv2 
import numpy as np 
from scipy import stats
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
        self._angle_multiple = 30 
        #self._possible_angles = np.arange(-180, 210, self._angle_multiple)
     
    def registerGrayFrame(self, frame):
        self._prev_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def updatePrevIteration(self):
        self._prev_gray_frame = self._gray_frame

    def getOpticalFlowPoints(self):
        """Obtain results of optical flow"""
        # return a 2 channel image called 'flow'
        flow = cv2.calcOpticalFlowFarneback(self._prev_gray_frame, self._gray_frame, None, 0.5, 2, 40, 2, 5, 1.5, 0)
        return flow

    def processMotionDirections(self, frame, framesElapsed, draw = False):
        """Draw motion lines based on optical flow tracking"""
        self._gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #if framesElapsed % self._update_interval == 0:
        average_angle, lines = self.getTrajectoryAngle(self.getOpticalFlowPoints())  
        print(average_angle)
        if draw:
            self.drawPoints(frame, lines)
        self.drawArrow(frame, average_angle)
        return frame

    def getTrajectoryAngle(self, flow):
        angles = []
        step = 32
        h, w = self._gray_frame.shape[:2]
        y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1).astype(int)
        fx,fy = flow[y,x].T

        # lines between all optical flow points, defined by their endpoints 
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)

        for (x1, y1), (x2, y2) in lines:
            if y2!=y1 or x2!=x1:
                angles.append(round(math.atan2(- int(y2) + int(y1), int(x2) - int(x1)) * 180 / np.pi))
        angles = np.array(angles)
        vals, counts = np.unique(angles, return_counts=True)
        index = np.argmax(counts)
        mode_angle = angles[index]
        mode_angle = self._roundAngle(mode_angle)
        #return np.mean(angles), lines
        return mode_angle, lines

    def _roundAngle(self, angle):
        if self._angle_multiple == 0:
            return angle
        remainder = abs(angle) % self._angle_multiple
        if remainder == 0:
            return angle
        if angle < 0:
            return -(abs(angle) - self._angle_multiple - remainder)
        else:
            return angle + self._angle_multiple - remainder 

    def getDetectInterval(self):
        return self._detect_interval

    def drawPoints(self, frame, lines):
        for (x1, y1), (x2, y2) in lines:
            cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),1)
            cv2.circle(frame, (x1, y1), 1, (255, 0, 0), -1)

    def drawArrow(self, frame, average_angle):
        average_angle = math.radians(average_angle)
        arrow_length = 100
        start_point = (int(frame.shape[1]/2), int(frame.shape[0]/2))   
        try:
            if average_angle > 0:
                end_point = (start_point[0] + int(arrow_length*math.cos(average_angle)), start_point[1]+int(arrow_length*math.sin(average_angle)))  
            else:
                end_point = (start_point[0] - int(arrow_length*math.cos(average_angle)), start_point[1]+int(arrow_length*math.sin(average_angle)))  
        except:
            print("NaN error?")
            return

        #print(end_point)
        color = (0, 255, 0)   
        thickness = 6
        frame = cv2.arrowedLine(frame, start_point, end_point, color, thickness) 

        

