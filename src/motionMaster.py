import cv2
from src.managers import WindowManager, CaptureManager
import keyboard

class MotionMaster(object):
    """Abstraction that encapsulates everything needed to run the program"""
    def __init__(self):
        self._windowManager = WindowManager('ImageMotion', self.onKeyPress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0, cv2.CAP_DSHOW), self._windowManager, False)
    
    def run(self):
        """Main loop of the MotionMaster object"""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            #if frame is not None:
                # TODO: image processing functions 
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
        if keyboard.is_pressed('space'): # space to record/stop recording video of the captured frames 
            if not self._captureManager.isWritingVideo:
                print("Recording video...")
                self._captureManager.startWritingVideo('recording.avi')
            else:
                print("isWritingVideo is true")
                self._captureManager.stopWritingVideo()
        return True 
        # TODO: more control options 