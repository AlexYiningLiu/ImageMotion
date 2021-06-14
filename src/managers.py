
# Referenced: Learning OpenCV4 Computer Vision with Python by Joseph Howse, Joe Minichino

import cv2 
import numpy
import time 

class CaptureManager(object):
	"""An abstraction used to manage the capture and processing of image frames"""
	def __init__(self, capture, previewWindowManager = None, shouldMirrorPreview = False):
		self.previewWindowManager = previewWindowManager
		self.shouldMirrorPreview = shouldMirrorPreview
		self._capture = capture
		self._channel = 0
		self._enteredFrame = False
		self._frame = None
		self._videoFilename = None
		self._videoEncoding = None
		self._videoWriter = None
		self._startTime = None
		self._framesElapsed = 0
		self._fpsEstimate = None
	
	def frame(self):
		return self._frame

	def channel(self):
		return self._channel

	def channel(self, value):
		if self._channel != value:
			self._channel = value
			self._frame = None

	def enterFrame(self):
		"""Used to obtain a new frame"""
		assert not self._enteredFrame, 'no exitFrame for enterFrame'
		if self._capture is not None:
			self._enteredFrame, self._frame = self._capture.read()

	def exitFrame(self):			
		"""Perform work with the current frame, then release it"""

		# Basic implemenation of estimating FPS 
		if self._framesElapsed == 0:
			self._startTime = time.time()
		else:
			timeElapsed = time.time() - self._startTime
			self._fpsEstimate = self._framesElapsed / timeElapsed
			self._framesElapsed += 1

		if self._videoFilename != None:
			self.writeVideoFrame()

		# Display the window (mirrored or not) 
		if self.previewWindowManager is not None:
			if self.shouldMirrorPreview:
				mirroredFrame = cv2.flip(self._frame, 1)
				self.previewWindowManager.show(mirroredFrame)
			else:
				self.previewWindowManager.show(self._frame)

		# Release the frame for next use 
		self._frame = None
		self._enteredFrame = False

	def startWritingVideo(self, filename, encoding = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')):
		"""simplying sets the filename and encoding"""
		self._videoFilename = filename
		self._videoEncoding = encoding
		size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		self._videoWriter = cv2.VideoWriter(self._videoFilename, self._videoEncoding, 30, size)

	def stopWritingVideo(self):
		self._videoFilename = None
		self._videoEncoding = None
		self._videoWriter = None 

	def writeVideoFrame(self):
		self._videoWriter.write(self._frame)
		
class WindowManager(object):
	"""An abstraction used to manage the creation, display, keyboard controls, and termination of a cv2 window"""
	def __init__(self, windowName, keypressCallback = None):
		"""Takes a string name and a callback function that determines what happens during key press"""
		self.keypressCallback = keypressCallback
		self._windowName = windowName
		self._isWindowCreated = False		

	@property
	def isWindowCreated(self):
		return self._isWindowCreated

	def createWindow(self):
		cv2.namedWindow(self._windowName)
		self._isWindowCreated = True

	def show(self, frame):
		cv2.imshow(self._windowName, frame)

	def destroyWindow(self):
		cv2.destroyWindow(self._windowName)
		self._isWindowCreated = False




