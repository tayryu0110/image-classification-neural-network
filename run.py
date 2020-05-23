import numpy as np
import cv2
from tensorflow.keras.models import load_model
from scipy import io
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

pretrain_1 = load_model('pretrain/VGGPreTrained_2_round.hdf5')
pretrain_2 = load_model('pretrain/VGG.classifier.hdf5')

def find_1(image, stepSize, windowSize, ratio):
	for y in range(0, image.shape[0], int(stepSize/ratio)):
		for x in range(0, image.shape[1], int(stepSize*3/ratio)):
			if y + windowSize[0] < image.shape[0] and \
						x + windowSize[1] < image.shape[1] :
				yield (x, y, image[y:y + windowSize[0], x:x + windowSize[1]])

def find_2(image, stepSize, windowSize, y_1, x_1, y_2, x_2, ratio, x_ini):
	for y in range(y_1 + int(10*ratio), y_2-int(25*ratio), int(stepSize*ratio)):
		for x in range(x_1+int(x_ini*ratio), x_2-int(25*ratio), int(stepSize*ratio/2) if int(stepSize*ratio/2) > 0 else 1):
			if y + windowSize < image.shape[0] and \
						x + windowSize < image.shape[1] :
				yield (x, y, image[y:y + windowSize, x:x + windowSize])

def iou(box1, box2):
    yi1 = max(box1[0], box2[0])
    xi1 = max(box1[1], box2[1])
    yi2 = min(box1[2], box2[2])
    xi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1)*(yi2 - yi1) 
    box1_area = (box1[3] - box1[1])*(box1[2]- box1[0])
    box2_area = (box2[3] - box2[1])*(box2[2]- box2[0])
    union_area = (box1_area + box2_area) - inter_area
    iou = inter_area / union_area
    return iou

def overlap_box(box_list, threshold = 0.01):
	remain_box = []
	while len(box_list) != 0:
		hist_box = box_list [0]
		remain_box.append(hist_box)
		del box_list[0]
		box_list_temp = list(box_list)
		for box in box_list_temp:
			ele_1 = hist_box[1:]
			ele_2 = box[1:]
			iou_score = iou(ele_1, ele_2)
			if iou_score > threshold :
				box_list.remove(box)
	return remain_box



for i in range(5):
	print('detecting image' + str(i+1) + '.....')
	image = cv2.imread('InputImages/input_' + str(i+1) + '.png')
	
	output = np.copy(image)
	output_final = np.copy(image)
	
	box_list = []
	print('scale 1')
	if i == 2 or i == 4:
		steptemp = 30
	elif i == 0:
		steptemp = 20
	elif i == 3:
		steptemp = 5
	else: 
		steptemp = 10
	for (x, y, window) in find_1(image, stepSize=steptemp, windowSize=(100, 100), ratio = 1.0):
		clone = image.copy()
		img_parts = np.reshape(window/255, (1,100,100,3))
		yOut = pretrain_1.predict(img_parts)
		cv2.rectangle(clone, (x, y), (x + 100, y + 100), (0, 255, 0), 1)
	
		if yOut[0][0] > 0.99:
			cv2.rectangle(image, (x, y), (x + 100, y + 100), (0, 255, 0), 1)
			font = cv2.LINE_AA
			center = (x, y)
			cv2.putText(image, str(round(yOut[0][0],2)), center, font, 0.5, (0, 255, 0), 2)
			box_info = [yOut[0][0], y, x, y + 100, x + 100, 1]
			box_list.append(box_info)
	
		cv2.imshow("Window", clone)
		cv2.waitKey(1)
	
	box_list = sorted(box_list, key=lambda x: x[0], reverse=True)	
	if len(box_list) == 0:
		print('scale 2')
		image_2 = cv2.resize(image, (0,0), fx=0.7, fy=0.7) 
		for (x, y, window) in find_1(image_2, stepSize=5, windowSize=(100, 100), ratio = 0.7):
			clone = image_2.copy()
			img_parts = np.reshape(window/255, (1,100,100,3))
			yOut = pretrain_1.predict(img_parts)
			cv2.rectangle(clone, (x, y), (x + 100, y + 100), (0, 255, 0), 1)
		
			if yOut[0][0] > 0.99:
				cv2.rectangle(image_2, (x, y), (x + 100, y + 100), (0, 255, 0), 1)
				font = cv2.LINE_AA
				center = (x, y)
				cv2.putText(image_2, str(round(yOut[0][0],2)), center, font, 0.5, (0, 255, 0), 2)
				x_1,y_1 = int(x/0.7),int(y/0.7)
				x_2,y_2 = int((x+100)/0.7),int((y+100)/0.7)
				cv2.rectangle(image, (x_1, y_1), (x_2, y_2), (0, 255, 0), 1)
				box_info = [yOut[0][0], y_1, x_1, y_2, x_2, 0.7]
				box_list.append(box_info)
		
			cv2.imshow("Window", clone)
			cv2.waitKey(1)
	
	box_list = sorted(box_list, key=lambda x: x[0], reverse=True)
	
	remain_box = overlap_box(box_list)
	count = 0
	for box in remain_box:
		y_1, x_1, y_2, x_2 = box[1], box[2], box[3], box[4]
		cv2.rectangle(output, (x_1, y_1), (x_2, y_2), (0, 255, 0), 1)
	
		output = cv2.resize(output, (0,0), fx=box[5], fy=box[5]) 
		x_1,y_1 = int(x_1*box[5]),int(y_1*box[5])
		x_2,y_2 = int(x_2*box[5]),int(y_2*box[5])
		x_ini = -1.0
	
		if i == 1: 
			ratio_scale = 0.85
		elif i == 3:
			ratio_scale = 0.65
			x_ini = 20
		elif i == 4:
			ratio_scale =0.5
			x_ini = -15
		elif i ==2:
			ratio_scale = 0.9
			x_ini = -3
		else :
			ratio_scale = 0.9
		output = cv2.resize(output, (0,0), fx=ratio_scale, fy=ratio_scale) 
	
		x_1,y_1 = int(x_1*ratio_scale),int(y_1*ratio_scale)
		x_2,y_2 = int(x_2*ratio_scale),int(y_2*ratio_scale)
	
		Map = {}
		output_3 = np.copy(output)
		
		if i >= 3:
			ss = 3
		elif i == 1:
			ss = 7
		else:
			ss = 8
		for (x, y, window) in find_2(output, ss, 32, y_1, x_1, y_2, x_2, box[5],x_ini):
			output_2 = np.copy(output)
			output_3 = np.copy(output_3)
			img_parts = np.reshape(window/255, (1,32,32,3))
			yOut = pretrain_2.predict(img_parts)
			pmax = np.max(yOut)
			pmax_index = np.argmax(yOut)
		
			font = cv2.LINE_AA
			center = (x, y)
			testLoc = (x, y)
			string = str(pmax_index) + ' :  ' "'"+str(round(pmax,2))
			cv2.rectangle(output_2, (x, y), (x + 32, y + 32), (0, 0, 255), 1)
			if pmax > 0.952: 
				cv2.putText(output_2, string, testLoc, font, 0.7, (0, 255, 0), 2)
				box_info = [pmax, pmax_index, y, x, box[5]]
				if pmax_index in Map:
					Map[pmax_index].append(box_info)
				else:
					Map[pmax_index] = [box_info]
			cv2.imshow("Window", output_2)
			cv2.waitKey(1)
	
		new_map = {}
		for key,value in Map.items():
			value = sorted(value, key=lambda x: x[0], reverse=True)
			remain_box = overlap_box(value, threshold = 0.8)
			new_map[key] = remain_box
		
		new_map = sorted(new_map.items(), key = lambda x: (x[1][0][3], x[1][0][2]))
		temp = 0.0
		for key,remain_box in new_map:
			for box in remain_box:
				pmax = box[0]
				pmax_index = box[1]
				if box[1] == 4: 
					temp = box[3]
		y_min,x_min = 5000,5000
		y_max,x_max = -1,-1
		res = ''
		for key, remain_box in new_map:
			for box in remain_box:
				pmax = box[0]
				pmax_index = box[1]
				if (box[1]==1 and np.abs(box[3] - temp) > 5) or box[1] != 1:
					res = res + str(pmax_index)
					y = int(box[2]/(box[4]*ratio_scale)) + 4
					x = int(box[3]/(box[4]*ratio_scale)) + 4
					testLoc = (x, y)
					win_size = int(32/(box[4]*ratio_scale))
					y_min = y if y < y_min else y_min
					x_min = x if x < x_min else x_min
					y_max = y+win_size if y+win_size > y_max else y_max
					x_max = x+win_size if x+win_size > x_max else x_max
					PLoc = (x, y + win_size + 10)
		
		cv2.rectangle(output_final, (x_min - 20, y_min - 20), (x_max + 20 , y_max+ 20), (255, 0, 0), 2)
		cv2.putText(output_final, res, (x_min - 20, y_min - 30), font, 1, (255, 0, 0), 2)
		cv2.imwrite('graded_images /'+ str(i+1) + '.png', output_final)