import os
import sys
os.system('rm label.txt label_val.txt')
import cv2
import imutils

from file_helper import read_lines, write

global img
global point1, point2
global point_list
global cur_box_list
global cur_img_path


def on_mouse(event, x, y, flags, param):
    global img, point1, point2, cur_img_path
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (255, 0, 0), 5)
        for i in range(len(cur_box_list)):
            cur_point1, cur_point2 = cur_box_list[i]
            cv2.rectangle(img2, cur_point1, cur_point2, (255, 0, 0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        for i in range(len(cur_box_list)):
            cur_point1, cur_point2 = cur_box_list[i]
            cv2.rectangle(img2, cur_point1, cur_point2, (255, 0, 0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:
        point2 = (x, y)
        min_x = min(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        max_x = max(point1[0], point2[0])
        max_y = max(point1[1], point2[1])
        cur_box_list.append([(min_x, min_y), (max_x, max_y)])
        for i in range(len(cur_box_list)):
            cur_point1, cur_point2 = cur_box_list[i]
            cv2.rectangle(img2, cur_point1, cur_point2, (255, 0, 0), 5)
        cv2.imshow('image', img2)



def label_animal_boxes():
    img_pathes = read_lines('data/strange.txt')
    global cur_box_list
    for img_name in img_pathes:
        cur_box_list = []
        global img, cur_img_path
        cur_img_path = 'data/strange_animal/' + img_name.strip()
        img = cv2.imread(cur_img_path)
        img = imutils.resize(img, width=400)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', on_mouse)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        for i in range(len(cur_box_list)):
            cur_line = '%s\t%4f\t%4f\t%4f\t%4f\n' % (
                cur_img_path,
                float(cur_box_list[i][0][0])/img.shape[1], float(cur_box_list[i][0][1])/img.shape[0],
                float(cur_box_list[i][1][0])/img.shape[1], float(cur_box_list[i][1][1])/img.shape[0])
            write('animal_box.txt', cur_line)


def label_cattle_boxes():
    img_pathes = read_lines('data/cattle.txt')
    global cur_box_list
    for img_name in img_pathes:
        cur_box_list = []
        global img, cur_img_path
        cur_img_path = 'data/hard_cattle/' + img_name.strip()
        img = cv2.imread(cur_img_path)
        img = imutils.resize(img, width=400)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', on_mouse)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        for i in range(len(cur_box_list)):
            cur_line = '%s\t%4f\t%4f\t%4f\t%4f\n' % (
                cur_img_path,
                float(cur_box_list[i][0][0])/img.shape[1], float(cur_box_list[i][0][1])/img.shape[0],
                float(cur_box_list[i][1][0])/img.shape[1], float(cur_box_list[i][1][1])/img.shape[0])
            write('body_box.txt', cur_line)

def label_boxes(input_path, output_box_path):
    img_pathes = read_lines(input_path)
    global cur_box_list
    for img_name in img_pathes:
        cur_box_list = []
        global img, cur_img_path
        cur_img_path = img_name.strip()
        img = cv2.imread(cur_img_path)
        img = imutils.resize(img, width=400)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', on_mouse)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        for i in range(len(cur_box_list)):
            cur_line = '%s\t%4f\t%4f\t%4f\t%4f\n' % (
                cur_img_path,
                float(cur_box_list[i][0][0])/img.shape[1], float(cur_box_list[i][0][1])/img.shape[0],
                float(cur_box_list[i][1][0])/img.shape[1], float(cur_box_list[i][1][1])/img.shape[0])
            write(output_box_path, cur_line)

if __name__ == '__main__':
    # label_animal_boxes()
    # label_cattle_boxes()
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    # label_boxes('no_target.txt', 'behind_box.txt')
    label_boxes(input_path, output_path)
