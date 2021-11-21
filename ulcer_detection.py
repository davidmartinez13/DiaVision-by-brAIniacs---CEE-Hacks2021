import datetime
import time
import sys
import os
import json
import cv2 
from opcua import ua
import pyrealsense2
import numpy as np
import random
import matplotlib as mpl
import scipy.signal
from scipy import ndimage
from scipy.spatial import distance as dist
from collections import OrderedDict
import tensorflow as tf
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
class processing_apriltag:
	
	def __init__(self,intrinsic,color_image,depth_frame):
		self.color_image = color_image
		self.intrinsic = intrinsic
		self.depth_frame = depth_frame
		self.radius = 20
		self.axis = 0
		self.image_points = {}
		self.world_points = {}
		self.world_points_detect = []
		self.image_points_detect = []
		self.homography= None


	def detect_tags(self):
		rect1 = []
		pixel_cm_ratio = None
		image = self.color_image.copy()
		# self.load_original_points()
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		(corners, ids, rejected) = cv2.aruco.detectMarkers( image, cv2.aruco.Dictionary_get(
                                    cv2.aruco.DICT_APRILTAG_36h11),
                                    parameters=cv2.aruco.DetectorParameters_create())
		ids = ids.flatten()
		int_corners = np.int0(corners)
		cv2.polylines(image, int_corners, True, (0, 255, 0), 2)
        
		for (tag_corner, tag_id) in zip(corners, ids):
			# get (x,y) corners of tag
			aruco_perimeter = cv2.arcLength(corners[0], True)
			pixel_cm_ratio = aruco_perimeter / 11.2
			corners = tag_corner.reshape((4, 2))
            
			(top_left, top_right, bottom_right, bottom_left) = corners
			top_right, bottom_right = (int(top_right[0]), int(top_right[1])),\
									  (int(bottom_right[0]), int(bottom_right[1]))
			bottom_left, top_left = (int(bottom_left[0]), int(bottom_left[1])),\
									(int(top_left[0]), int(top_left[1]))
			# compute centroid
			cX = int((top_left[0] + bottom_right[0]) / 2.0)
			cY = int((top_left[1] + bottom_right[1]) / 2.0)


			self.image_points[str(int(tag_id))] = [cX,cY]
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
			# draw ID on frame
			cv2.putText(image, str(tag_id),(top_left[0], top_left[1] - 15),
						cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
            
		return image,pixel_cm_ratio

class CentroidTracker:
	def __init__(self, maxDisappeared=20):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		# is box empty
		if len(rects) == 0:
			# loop overobjects and mark them as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
			#deregister if maximum number of consecutive frames where missing
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
			# return early as there are no centroids or tracking info
			# to update
			return self.objects
		# array of input centroids at current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")
		# inputCentroids = centroid
		# # loop over the bounding box rectangles
		for i in range(0,len(rects)):
			# # use the bounding box coordinates to derive the centroid
			inputCentroids[i] = rects[i][1]
		# if not tracking any objects take input centroids, register them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])
		# else, while currently tracking objects try match the input centroids to existing centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())
			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()
			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]
			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()
			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				# val
				if row in usedRows or col in usedCols:
					continue
				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0
				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)
			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)
			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
				for row in unusedRows:
					# grab the object ID for the corresponding row
					# index and increment the disappeared counter
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					# check to see if the number of consecutive
					# frames the object has been marked "disappeared"
					# for warrants deregistering the object
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])
		# return the set of trackable objects
		return self.objects

class ulcer_detector:
    def __init__(self,paths,files,checkpt):
        self.paths = paths
        self.files = files
        self.checkpt = checkpt
        self.world_centroid = None
        self.category_index = label_map_util.create_category_index_from_labelmap(self.files['LABELMAP'])
        configs = config_util.get_configs_from_pipeline_file(self.files['PIPELINE_CONFIG'])
        self.detection_model = model_builder.build(model_config=configs['model'], is_training=False)
        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model= self.detection_model)
        ckpt.restore(os.path.join(self.paths['CHECKPOINT_PATH'], self.checkpt)).expect_partial()

    @tf.function
    def detect_fn(self, image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    def detect_corners(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img,(x, y),4 , (255,0,0),-1)
        cv2.imshow("corners",img)

    def find_ulcer_contours(self, img, ymin, ymax, xmin, xmax, centroid,pixel_cm_ratio):
        box = np.int64(np.array([[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]))
        w = xmax - xmin
        h = ymax - ymin
        angle = 0
        # box = None
        crop = img[int(ymin):int(ymax),int(xmin):int(xmax),:]
        
        # gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # ret, mask = cv2.threshold(gray, 60, 255, 0)

        # lower = np.array([0,70,70])
        # upper = np.array([170,170,170])
        # hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        # mask = cv2.inRange(hsv, lower, upper)
        # mask = cv2.GaussianBlur(mask, (5, 5), 50)

        # lower boundary RED color range values; Hue (0 - 10)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
        
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160,100,20])
        upper2 = np.array([179,255,255])
        
        lower_mask = cv2.inRange(hsv, lower1, upper1)
        upper_mask = cv2.inRange(hsv, lower2, upper2)
        
        mask = lower_mask + upper_mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
            # if area > 25000 and area < 110000:
                # cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect
                cx , cy = x+xmin, y+ymin
                centroid = (int(cx), int(cy))
                box = cv2.boxPoints(((cx,cy),(w, h), angle))
                box = np.int0(box)
            crop = cv2.drawContours(crop, contours, -1, (0,255,0), 3,lineType = cv2.LINE_AA)

        if pixel_cm_ratio is not None:
            object_width = w / pixel_cm_ratio
            object_height = h / pixel_cm_ratio
            # print(object_height)
            cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (centroid[0], centroid[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (centroid[0], centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # cv2.imshow("ulcer contours", crop)
        # cv2.imshow("ulcer mask", mask)
            
        return img, box, angle, centroid

    def compute_mask(self, img, box_mask, box_array):
        is_box_empty = len(box_array) == 0
        if is_box_empty:
            return img, box_mask
        else:
            cv2.fillPoly(box_mask, box_array,(255, 255, 255))
            box_mask = cv2.GaussianBlur(box_mask, (5, 5), 0)
            cv2.polylines(img, box_array, True, (255, 0, 0), 3)
            return img, box_mask

    def deep_detector(self, color_frame, pixel_cm_ratio, bnd_box = True):
        box_array = []
        rects = []
        box_mask = np.zeros_like(color_frame)
        image_np = np.array(color_frame)
        height, width, depth = image_np.shape[0],image_np.shape[1],image_np.shape[2]
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections
        # detection_classes should be ints
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        label_id_offset = 1
        img_np_detect = image_np.copy()

        boxes = detections['detection_boxes']
        # get all boxes from an array
        # max_boxes_to_draw = boxes.shape[0]
        max_boxes_to_draw = 1
        # get scores to get a threshold
        scores = detections['detection_scores']
        # set as a default but free to adjust it to your needs
        min_score_thresh=.3
        # iterate over all objects found
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            
            if scores is None or scores[i] > min_score_thresh:
                # boxes[i] is the box which will be drawn
                # print ("This box is gonna get used", boxes[i], detections['detection_classes'][i])
                ymin, xmin = boxes[i][0]*height, boxes[i][1]*width
                ymax, xmax = boxes[i][2]*height, boxes[i][3]*width
                cx,cy = (xmax+xmin)/2,(ymax+ymin)/2
                centroid = (int(cx),int(cy))
                img_np_detect, box, angle, centroid = self.find_ulcer_contours(image_np, ymin, ymax, xmin, xmax, centroid, pixel_cm_ratio)

                box_array.append(box)

                cv2.circle(img_np_detect, centroid, 4, (255, 0, 0),5)

                cv2.putText(img_np_detect, "{} deg".format(round(angle, 1)), (centroid[0], centroid[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                rects.append([box, centroid, self.world_centroid, angle, detections['detection_classes'][i]])
        img_np_detect, box_mask = self.compute_mask(img_np_detect,box_mask, box_array)
                     
        if bnd_box:
            viz_utils.visualize_boxes_and_labels_on_image_array(
                        img_np_detect,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        self.category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=1,
                        min_score_thresh=.3,
                        agnostic_mode=False, 
                        line_thickness=1)
        detection_result = np.bitwise_and(color_frame,box_mask)
        # detection_result = box_mask
        return img_np_detect, detection_result, rects
def objects_update(objects,image, color_frame):
    gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    # heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(image, text, (centroid[0] , centroid[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.circle(image, (centroid[0], centroid[1]), 4, (255, 255, 0), -1)

        cv2.putText(heatmap, text, (centroid[0] , centroid[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(heatmap, 'Temperature:'+str(random.randint(22,38)), (centroid[0] , centroid[1] - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.circle(heatmap, (centroid[0], centroid[1]), 4, (0, 0, 0), -1)
        
    cv2.imshow('heatmap', heatmap)
CUSTOM_MODEL_NAME = 'my_ssd_mobnet_ulcers' 
check_point ='ckpt-6'
LABEL_MAP_NAME = 'label_map.pbtxt'
paths = {
    'ANNOTATION_PATH': os.path.join('hackathon','Tensorflow', 'workspace','annotations'),
    'CHECKPOINT_PATH': os.path.join('hackathon','Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME) 
}
files = {
    'PIPELINE_CONFIG':os.path.join('hackathon','Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}
if __name__ == '__main__':

    warn_count = 0
    a = 0
    b = 0
    d = 2.61
    pixel_cm_ratio = None
    bbox = True
    ct = CentroidTracker()    
    ulcer_detect = ulcer_detector(paths,files,check_point)
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(2)
    path = "c:/Users/David/PythonProjects/Programming/Workspace/hackathon/cropped/abscess"
    # out = "c:/Users/David/PythonProjects/Programming/Workspace/hackathon/augmented"
    
# Loading a sample image 
    # for image_path in os.listdir(path):
    #     # create the full input path and read the file
    #     input_path = os.path.join(path, image_path)
    #     img = cv2.imread(input_path)
    #     color_frame = cv2.resize(img, (600,600))
    #     time.sleep(0.5) 
    while True:
        _, img = cap.read()
        color_frame = img

        height, width, depth = color_frame.shape[0],color_frame.shape[1],color_frame.shape[2]
        
        apriltag = processing_apriltag(None, color_frame, None)
        try:
            color_frame, pixel_cm_ratio = apriltag.detect_tags()
                
        except:
		#Triggered when no markers are in the frame:
            warn_count += 1
            if warn_count == 1:
                print("[INFO]: Markers out of frame or moving.")
            pass

        img_np_detect, result, rects = ulcer_detect.deep_detector(color_frame, pixel_cm_ratio, bnd_box = bbox)
            
        
        objects = ct.update(rects)
        objects_update(objects, img_np_detect, color_frame)
        pixel_cm_ratio = None
        cv2.imshow('object detection', img_np_detect)
        # cv2.imshow('object detection', cv2.resize(img_np_detect, (720,720)))
        
        k = cv2.waitKey(1)
        if k == ord('w'):
            a+=0.1
        if k == ord('s'):
            a-=0.1
        if k == ord('a'):
            b+=1
        if k == ord('d'):
            b-=1
        if k == ord('z'):
            d+=2
        if k == ord('x'):
            d-=2
        if k == ord('l'):
            if bbox == False:
                bbox = True
            else:
                bbox = False
        if k == 27:
            cv2.destroyAllWindows()
            break