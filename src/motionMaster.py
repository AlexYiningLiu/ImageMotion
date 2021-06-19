import cv2
import keyboard
from src.managers import WindowManager, CaptureManager
from src.tracker import Tracker

class MotionMaster(object):
    """Abstraction that encapsulates everything needed to run the program"""
    def __init__(self):
        self._windowManager = WindowManager('ImageMotion', self.onKeyPress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0, cv2.CAP_DSHOW), self._windowManager, shouldMirrorPreview = True)
        self._isRecording = False 
        self._isTracking = False 
        self._firstTrackingFrame = True 
        self._tracker = Tracker()

    def run(self):
        """Main loop of the MotionMaster object"""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame()
            #############################################################
            if frame is not None:
                if self._isTracking:
                    self._tracker.registerGrayFrame(frame)
                    if self._firstTrackingFrame:
                        self._firstTrackingFrame = False 
                        self._tracker.getFirstKeyPoints(frame)                   
                    else:
                        r = self._tracker.getOpticalFlowPoints()
                        if not r:
                            self._firstTrackingFrame = True
                            self._captureManager.exitFrame()
                            continue
                        if self._captureManager.framesElapsed() % 50 == 0:
                            drawn_img = self._tracker.processMotionDirections(frame.copy(), self._captureManager.framesElapsed(), True)
                            self._tracker.updatePrevIteration()
                        else:
                            drawn_img = frame.copy()
                        if drawn_img is not None:
                            drawn_img = cv2.flip(drawn_img, 1)
                            cv2.imshow('drawn image', drawn_img)

                        #if self._captureManager.framesElapsed() % self._tracker.getDetectInterval() == 0: # update the tracking points 
                        #    self._firstTrackingFrame = True
            #############################################################
            self._captureManager.exitFrame()
            cv2.waitKey(1)
            if not self.onKeyPress():
                print("Done")
                break
                
    def onKeyPress(self):
        """Used to handle a key press"""
        if keyboard.is_pressed('q'): # q to exit 
            self._windowManager.destroyWindow()
            return False 
        if keyboard.is_pressed('r'): # r to record 
            if not self._isRecording:
                self._isRecording = True 
                print("Recording video...")
                self._captureManager.startWritingVideo('recording.avi')
        if keyboard.is_pressed('s'): # s to stop recording 
            print("Stop recording...")
            self._isRecording = False  
            self._captureManager.stopWritingVideo()
        if keyboard.is_pressed('t'): # t to start motion tracking 
            if not self._isTracking:
                print("Start tracking...")
                self._isTracking = True 
        return True 
        # TODO: more control options 