# _*_coding : utf-8_*_
# @Author   : 浪淘沙1230
# @Time     : 2022/2/25 10:09

from typing_extensions import runtime
import numpy as np
import cv2
import onnxruntime
import os

class ONNXModelYolov5:
	def __init__(self, onnx_path,img_size,confThreshold=0.5, nmsThreshold=0.5):
		"""
		:param onnx_path:
		"""
		self.img_size = img_size
		self.threshold = confThreshold
		self.iou_thres = nmsThreshold
		self.stride = 1
		self.sess = onnxruntime.InferenceSession(onnx_path)
		self.input_name = self.sess.get_inputs()[0].name
		self.output_name = self.sess.get_outputs()[0].name


	def letterbox(self,img, new_shape=(416, 416), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
		shape = img.shape[:2]  # current shape [height, width]
		if isinstance(new_shape, int):
			new_shape = (new_shape, new_shape)

		r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
		if not scaleup:
			r = min(r, 1.0)

		ratio = r, r  # width, height ratios
		new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
		dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
		if auto:  # minimum rectangle
			dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
		elif scaleFill:  # stretch
			dw, dh = 0.0, 0.0
			new_unpad = (new_shape[1], new_shape[0])
			ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

		dw /= 2  # divide padding into 2 sides
		dh /= 2
		if shape[::-1] != new_unpad:  # resize
			img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
		top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
		left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
		img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
		return img, ratio, (dw, dh)

	def clip_coords(self,boxes, img_shape):
		# Clip bounding xyxy bounding boxes to image shape (height, width)
		boxes[:, 0].clip(0, img_shape[1])  # x1
		boxes[:, 1].clip(0, img_shape[0])  # y1
		boxes[:, 2].clip(0, img_shape[1])  # x2
		boxes[:, 3].clip(0, img_shape[0])  # y2

	def scale_coords(self,img1_shape, coords, img0_shape, ratio_pad=None):
		if ratio_pad is None:  # calculate from img0_shape
			gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
			pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
		else:
			gain = ratio_pad[0][0]
			pad = ratio_pad[1]

		coords[:, [0, 2]] -= pad[0]  # x padding
		coords[:, [1, 3]] -= pad[1]  # y padding
		coords[:, :4] /= gain
		self.clip_coords(coords, img0_shape)
		return coords


	def preprocess(self, img):
		img0 = img.copy()
		img = self.letterbox(img, new_shape=self.img_size)[0]
		img = img[:, :, ::-1].transpose(2, 0, 1)
		img = np.ascontiguousarray(img).astype(np.float32)
		img /= 255.0  # 图像归一化
		img = np.expand_dims(img, axis=0)
		assert len(img.shape) == 4

		return img0, img


	def postprocess(self, pred):
		pred = pred.astype(np.float32)
		pred = np.squeeze(pred, axis=0)

		index = (pred[..., 4] > self.threshold)
		pred = pred[index]
		classids = np.argmax(pred[..., 5:], axis=1)
		pred_box = pred[..., :4].astype(np.float)
		pred_box[..., [0, 1]] -= pred_box[..., [2, 3]] / 2
		pred_box_ = pred_box.astype(np.int)
		boxes = pred_box_.tolist()
		confidences = pred[..., 4].tolist()

		return boxes,classids,confidences


	def detect(self, im):
		im0, img = self.preprocess(im) 
		frame_height = im0.shape[0]
		frame_width = im0.shape[1]
		pred = self.sess.run(None, {self.input_name: img})[0]

		boxes, classids, confidences=self.postprocess(pred)
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.threshold, self.iou_thres)
		pred_result=[]
		if len(idxs) > 0:
			img0_shape=im0.shape
			for i in idxs.flatten():
				confidence = confidences[i]
				if confidence >= self.threshold:
					left, top, width, height = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
					box = (left, top, left + width, top + height)
					box = np.squeeze(self.scale_coords((self.img_size, self.img_size),\
													   np.expand_dims(box, axis=0).astype("float"),\
										  				img0_shape[:2]).round(),axis=0).astype("int")

					pred_result.append([int(box[0]), int(box[1]), int(box[2]), int(box[3]),confidence])
		
		table_box = []
		pred_result_wh = np.array(pred_result)[...,:4]
		range_array = np.array([frame_width, frame_height, frame_width, frame_height])
		if len(pred_result) > 0:
			table_box_bool = pred_result_wh < range_array
			for i in range(len(table_box_bool)):
				if table_box_bool[i].all() == np.array([True, True, True, True]).all():
					table_box.append((pred_result[i]))

		return table_box


