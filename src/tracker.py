import cv2 
import numpy as np 

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
        else:
            print("No points detected at all")

    def drawMotionLines(self, frame):
        """Draw motion lines based on optical flow tracking"""
        drawn_img = None
        if self._good_new is not None:
            for i, (new, old) in enumerate(zip(self._good_new, self._good_prev)): 
                a, b = new.ravel()
                c, d = old.ravel()
                self._mask = cv2.line(self._mask, (int(a),int(b)),(int(c),int(d)), self._color[i].tolist(), 2)
                frame = cv2.circle(frame,(int(a),int(b)),5, self._color[i].tolist(),-1)
            drawn_img = cv2.add(frame, self._mask)
        return drawn_img

