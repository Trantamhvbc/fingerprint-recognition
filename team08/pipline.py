
from model import *
from data import *
import json
import re
import os

# tìm minuntiae cho 1 ảnh
def main1(path):
	w = 16
	image = read_image_rgb(path)
	print(image.shape)
	show_image(image,'image')
	image = normalize(image,m0 = float(100), v0 = float(100))
	image_segment,norm_img,mask = create_segmented_and_variance_images(image, w = w, threshold=.4)
	image_oriented = calculate_angles(image_segment,w)
	show_image(visualize_angles(image, mask, image_oriented, W = w),'huong')
	print(image.shape)
	freq = ridge_freq(norm_img, mask, image_oriented, w, kernel_size = 5, minWaveLength = 5, maxWaveLength = 15)
	gabor_img = gabor_filter(norm_img, image_oriented, freq)
	show_image(gabor_img,'gabor')
	image_thinning = skeletonize(gabor_img)
	show_image(image_thinning,'image thining')
	list_point_minunate,result_im = get_minunatiaes_point(image_thinning,image_oriented)
	# for i  in list_point_minunate:
	# 	point1 = [i[1][0], i[1][1], i[2]]
	# 	point2 = rest_point(point1)
	# 	cv.line(result_im, (point1[1], point1[0]), (point2[1], point2[0]), (255,255,0), 1)
	show_image(result_im,'dd')
	result = []
	for i in list_point_minunate:
		result.append([i[0], i[1][0], i[1][1], i[2]])
	return result

# ./data/dataset/test/DB1/101_3.tif
# tìm minuntiae cho toàn bộ data
def main(path):
	w = 16
	image = read_image_rgb(path)
	print(image.shape)
	image = normalize(image,m0 = float(100), v0 = float(100))
	image_segment,norm_img,mask = create_segmented_and_variance_images(image, w = w, threshold=.4)
	image_oriented = calculate_angles(image_segment,w)
	print(image.shape)
	freq = ridge_freq(norm_img, mask, image_oriented, w, kernel_size = 5, minWaveLength = 5, maxWaveLength = 15)
	gabor_img = gabor_filter(norm_img, image_oriented, freq)
	image_thinning = skeletonize(gabor_img)
	list_point_minunate,result_im = get_minunatiaes_point(image_thinning,image_oriented)
	for i  in list_point_minunate:
		print(i)
	result = []
	for i in list_point_minunate:
		result.append([i[0], i[1][0], i[1][1], i[2]])
	return result

def search_image(I, data):
	no_max = 0
	point = []
	for T in data:
		check = np.zeros(len(T['points']))
		count = 0
		for mi in I:
			for i, mt in enumerate(T['points']):
				(sd, dd) = calculate_distance(mt, mi)
				if check[i] == 0 and mt[0] == mi[0] and sd < 50 and dd < pi/24:
					count += 1
					check[i] = 1
		if(no_max < count):
			no_max = count
			point = T
	return (point, no_max)


def path_tokenizer(source, rel):
	source_token = source.split('/')[-2:]
	rel_token = rel.split('/')[-2:]
	if(source_token[0] == rel_token[0] and source_token[1].split('_')[0] == rel_token[1].split('_')[0]):
		return True
	return False

# result = '['
# main_path = 'D:/My Document/HK8/HeCSDLDPT/BTL/fingerprint-recognization/f/data/dataset/train/DB'
# result_data = './data/dataset/db_data.json'
# with open(result_data, mode='w') as f:
# 	for i in range(1,5):
# 		tmp_path = main_path + str(i)
# 		list_imgs = os.listdir(tmp_path)
# 		for j in list_imgs:
# 			img_path = tmp_path + '/' + j
# 			print(img_path)
# 			list_points = main(img_path)
# 			obj = {
# 				'img': 'DB' + str(i) + '/' + str(j),
# 				'points': list_points
# 			}
# 			result += str(obj) + ','
# 	result += ']'
# 	f.write(result)	


# test_path = 'D:/My Document/HK8/HeCSDLDPT/BTL/fingerprint-recognization/f/data/dataset/test/DB'
# data_path = './data/dataset/db_data.json'
# data = []
# with open(data_path, mode='r') as f:
# 	data = json.load(f)
# result = './data/dataset/result.txt'
# rel_file = open(result, mode='w')
# count = 0
# hit_point = 0
# for i in range(1, 5):
# 	tmp_path = test_path + str(i)
# 	list_test = os.listdir(tmp_path)
# 	for test in list_test:
# 		count += 1
# 		img_path = tmp_path + "/" + test
# 		print(img_path)
# 		I = main(img_path)
# 		(point, no_max) = search_image(I, data)
# 		rel_file.write(img_path + " - " + str(point) + ' - ' + str(no_max) + '\n')
# 		if(path_tokenizer(img_path, point['img'])):
# 			hit_point += 1
# print(hit_point, count)
# rel_file.close()


print('Mời nhập link ảnh: ')
input_path = input()

data_path = './data/dataset/db_data.json'
data = []
with open(data_path, mode='r') as f:
	data = json.load(f)

I = main1(input_path)
(point, no_max) = search_image(I, data)
img = read_image_rgb('./data/dataset/train/' + point['img'])
show_image(img, 'matched image')