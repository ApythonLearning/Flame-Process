import cv2 as cv
import numpy as np
import os
import math


def image_augment(img, ctr, bri):  # 亮度bright 对比度contrast
    blank = np.zeros(img.shape, img.dtype)
    out = cv.addWeighted(img, ctr, blank, 1 - ctr, bri)

    return out


def mask_make(img_binary, center_point_mask, mask_radius):
    """
    Simply making a mask image.
    :param img_binary:
    :param center_point_mask:
    :param mask_radius:
    :return:
    """
    dst = np.zeros(img_binary.shape, dtype=np.uint8)
    cv.circle(dst, center_point_mask, mask_radius, 255, -1)

    return dst


def draw_obj(img, win_name):
    """
    :param img:
    :param win_name:
    :return: image_contours, contour_points, eight_point
    """
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    h, w = img.shape
    dst = np.zeros((h, w, 3), dtype=np.uint8)

    max_area = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv.contourArea(cnt)
        if area >= max_area:
            max_area = area
            index = i
    epsilon = 0.02*cv.arcLength(contours[index], True)
    poly_approx = cv.approxPolyDP(contours[index], epsilon, True)
    cv.drawContours(dst, contours, index, (0, 0, 255), 1, 8)
    cv.circle(dst, (x_ctr, y_ctr), 2, (0, 255, 255), 2)
    cv.circle(dst, (x_end, y_end), 2, (0, 255, 255), 2)
    for point in poly_approx:
        cv.circle(dst, (point[0][0], point[0][1]), 2, (0, 255, 0), 2)

    cv.imshow(win_name, dst)
    return dst, contours[index], poly_approx


def draw_center(win_name, input_image):
    """
     Selecting center point.
    """
    cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
    (x, y, w, h) = cv.selectROI(win_name, input_image, fromCenter=True)
    center_x, center_y = (int(x+w/2), int(y+h/2))
    cv.circle(input_image, (center_x, center_y), 1, (0, 0, 255), -1)
    cv.imshow(win_name, input_image)

    return center_x, center_y


def nothing(x):
    pass


def thresh_track_bar(win_name, initial_thresh=50):  # 参数可调 initial_threshold
    cv.createTrackbar('Threshold', win_name, initial_thresh, 255, nothing)
    param = cv.getTrackbarPos('Threshold', win_name)

    return param


def coordinate_system(center_point, radius, angle_initial):
    """
    Calculating the all coordinate system.
    :param center_point:
    :param radius:
    :param angle_initial:
    :return:
    """
    (x0, y0) = center_point
    angle = [angle_initial+45*i for i in range(8)]
    angle_correct = [angle_initial+45*i+90 for i in range(8)]
    print('angle normal is :\n', angle_correct)
    print('original angle is:\n', angle)
    radian = []
    coordinates = []
    for a in angle:
        radian.append(a/180*math.pi)

    for i in range(len(radian)):
        coordinates.append((int(x0+radius*math.cos(radian[i])), int(y0-radius*math.sin(radian[i]))))

    return coordinates, angle_correct


def polar_coordinate(center_point, end_point):
    """
    Calculating the polar coordinate.
    :param center_point:
    :param end_point:
    :return: the range of angle is (-pi/2, pi/2).
    """
    (x0, y0) = center_point
    (x1, y1) = end_point

    radius = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    divide_evade = x1 - x0
    if divide_evade == 0:
        divide_evade = 0.001
    angle = math.atan(-(y1 - y0)/divide_evade) * 180 / math.pi  # radian convert to angle
    return radius, angle


def polar2cartesian(polar_coordinate_input, center_point):
    """
    convert the polar to cartesian
    :param polar_coordinate_input:the form like this :(radius, angle)
    :param center_point:
    :return:the cartesian coordinates like:(x, y)
    """
    (x_0, y_0) = center_point
    (radius, angle) = polar_coordinate_input
    x_car = radius*math.cos(angle/180*math.pi)
    y_car = radius*math.sin(angle/180*math.pi)

    return x_car+x_0, -y_car+y_0


def draw_coordinates(coordinates_input, image_assigned):
    """
    Simply draw the normalized coordinate system in the image which you assigned.
    :param coordinates_input:
    :param image_assigned:
    :return:
    """
    for i in range(len(coordinates_input)):
        cv.line(image_assigned, (x_ctr, y_ctr), coordinates_input[i], (0, 255, 255), 2)

    return


def find_nearest(obj_num, obj_list):
    abs_subtract = []
    for i in range(len(obj_list)):
        abs_subtract.append(abs(obj_list[i] - obj_num))

    min_l = min(abs_subtract)
    index_min = abs_subtract.index(min_l)
    metric = abs_subtract[index_min]

    return index_min, metric


def judge_overlap(obj_list, pick_flag=True, position=0):
    """
    Note: the obj_list is a two dimensional list like:[(1, 2),......]
    :param obj_list:
    :param position:
    :param pick_flag:
    :return:
    """
    def unique_list(list_obj):
        unique_li = []
        repeat_record = []
        ret = 0

        for id_ls, x in enumerate(list_obj):
            if x not in unique_li:
                unique_li.append(x)
            else:
                ret = 1
                repeat_record.append((x, id_ls))

        return ret, repeat_record

    if pick_flag is True:
        pick_list = []
        for i in range(len(obj_list)):
            pick_list.append(obj_list[i][position])
    else:
        pick_list = obj_list

    ret_out, repeat_elements = unique_list(pick_list)

    return ret_out, repeat_elements


def tuple_split(list_tuple):
    li_1 = []
    li_2 = []
    for element in list_tuple:
        li_1.append(element[0])
        li_2.append(element[1])

    return li_1, li_2


def calculate_angle(input_point, angle_sys, center_point):
    (x0, y0) = center_point
    polar_cor = []
    r_a_cor = []
    r_sum = 0
    # According to the quadrant refine the polar coordinates
    for id_p, point in enumerate(input_point):
        (x_p, y_p) = (point[0][0], point[0][1])
        r, a = polar_coordinate(center_point, (x_p, y_p))
        r_sum += r
        if y_p < y0:
            if x_p < x0:
                a += 180
        else:
            if x_p < x0:
                a -= 180
        if a < 0:
            a += 360
        polar_cor.append((a, r, id_p))
        r_a_cor.append((r, a, id_p))
    # Using the angle of every point to sort the coordinates.
    polar_cor.sort()
    r_a_cor.sort()
    r_a_cor.reverse()
    print('Radius_Angle_COR is :\n', r_a_cor)
    # Selecting the eight point fitting the contour.
    r_mean = r_sum/(len(input_point))
    record_near = []
    for p_i in range(len(r_a_cor)):
        if r_a_cor[p_i][0] > r_mean:
            record_near.append((find_nearest(r_a_cor[p_i][1], angle_sys)))
        else:
            if p_i < 8:
                record_near.append((find_nearest(r_a_cor[p_i][1], angle_sys)))
            else:
                break

    l1, l2 = tuple_split(record_near)
    print('record nearest is :\n', record_near)
    remove_id = []
    if len(record_near) > 8:
        r_a_p = r_a_cor[:len(record_near)]

        ret_outs, elements = judge_overlap(record_near)
        for element in elements:
            before_id = l1.index(element[0])
            rear_id = element[1]
            if l2[before_id] >= l2[rear_id]:
                # record_near.remove((element[0], l2[before_id]))
                remove_id.append(before_id)
            else:
                # record_near.remove((element[0], l2[rear_id]))
                remove_id.append(rear_id)
    print('removed id is :', remove_id)
    match_r_a = []
    for i in range(len(l1)):
        if i in remove_id:
            continue
        match_r_a.append((r_a_cor[i][0]/2, angle_sys[l1[i]]))
    print('end match is :\n', match_r_a)
    return match_r_a


def contours_tree(image_tree):
    """
    Returning the maximum contour and it's child contours.
    :param image_tree:
    :return:
    """
    contours, hierarchy = cv.findContours(image_tree, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    dst = np.zeros((image_tree.shape[0], image_tree.shape[1], 3), dtype=np.uint8)
    contours_filter = []

    max_area = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv.contourArea(cnt)
        if area >= max_area:
            max_area = area
            index = i
    contours_filter.append(index)

    for i in range(len(contours)):
        cnt = contours[i]
        area = cv.contourArea(cnt)

        if area <= 100:
            continue
        if hierarchy[0][i][3] == index:
            contours_filter.append(i)

    print('filtered id of contours :\n', contours_filter)
    for index_c in contours_filter:
        cv.drawContours(dst, contours, index_c, (0, 100, 255), 1, 8)
    # cv.imshow('Tree Contours', dst)

    return contours, contours_filter


def calculate_min_distance(point_cal, list_dub):
    x0 = point_cal[0]
    y0 = point_cal[1]
    distance_list = []
    for i in range(len(list_dub)):
        x_ls = list_dub[i][0][0]
        y_ls = list_dub[i][0][1]
        distance_list.append(math.sqrt((x_ls - x0)**2 + (y_ls - y0)**2))
    minimum_d = min(distance_list)
    index_m = distance_list.index(minimum_d)

    return minimum_d, index_m


def find_point(center_point_find, target_point, contours_find):
    (x0, y0) = center_point_find
    (xt, yt) = target_point
    ra_find, angle_find = polar_coordinate(target_point, center_point_find)
    if y0 < yt:
        if x0 < xt:
            angle_find += 180
    else:
        if x0 < xt:
            angle_find -= 180
    if angle_find < 0:
        angle_find += 360
    vertical_angle_positive = angle_find + 90
    vertical_angle_negative = angle_find - 90
    # print(vertical_angle_positive, vertical_angle_negative)

    extend_find = []
    flag_1 = False
    for detect_r in range(1000):
        if flag_1:
            break
        positive_extend = (detect_r, vertical_angle_positive)
        x_find, y_find = polar2cartesian(positive_extend, target_point)
        for i in range(len(contours_find)):
            contours_find_list = contours_find[i].tolist()
            min_dis, min_index = calculate_min_distance((x_find, y_find), contours_find_list)
            if min_dis < 3:
                flag_1 = True
                extend_find.append((int(x_find), int(y_find)))
                break
    if flag_1 is False:
        extent.append(target_point)

    flag_2 = False
    for detect_r in range(1000):
        if flag_2:
            break
        negative_extend = (detect_r, vertical_angle_negative)
        x_find2, y_find2 = polar2cartesian(negative_extend, target_point)
        for i in range(len(contours_find)):
            contours_find_list2 = contours_find[i].tolist()
            min_dis, min_index = calculate_min_distance((x_find2, y_find2), contours_find_list2)
            if min_dis < 3:
                flag_2 = True
                extend_find.append((int(x_find2), int(y_find2)))
                break
    if flag_2 is False:
        extend_find.append(target_point)

    return extend_find


def angle_cone(base_point, center_point_cone, boundary_point):
    """
    Calculating the cone angle.
    :param base_point:
    :param center_point_cone:
    :param boundary_point:
    :return:
    """
    def triangle(point_c, point_base, point_boundary):
        (x_c, y_c) = point_c
        (x_b, y_b) = point_base
        (x_bd, y_bd) = point_boundary
        c2b = math.sqrt((x_c - x_b)**2 + (y_c - y_b)**2)
        c2bd = math.sqrt((x_c - x_bd)**2 + (y_c - y_bd)**2)
        b2bd = math.sqrt((x_bd - x_b)**2 + (y_bd - y_b)**2)
        cos_angle = abs(c2b**2 + c2bd**2 - b2bd**2) / (2 * c2b * c2bd)

        radian = math.acos(cos_angle)
        angle_vertex = radian * 180/math.pi

        return angle_vertex

    eight_angle = []
    for i in range(len(base_point)):
        point_base_out = base_point[i]
        point_bounder_p = boundary_point[i][0]
        point_bounder_n = boundary_point[i][1]
        angle_cone_p = triangle(center_point_cone, point_base_out, point_bounder_p)
        angle_cone_n = triangle(center_point_cone, point_base_out, point_bounder_n)

        if angle_cone_n < angle_cone_p:
            eight_angle.append(2*angle_cone_p)
        else:
            eight_angle.append(2*angle_cone_n)

    return eight_angle


'''
Selecting the background, object and coordinate reference picture.
'''
background = cv.imread("E:/cvdata/spray-3bar/3-110-1.1(1)/3-110-1.1-1_C001H001S0002002575.bmp")  # 背景图像
background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
image = cv.imread("E:/cvdata/spray-3bar/3-110-1.1(1)/3-110-1.1-1_C001H001S0002002600.bmp")  # 目标图像
coordinate_image = image.copy()
image1 = cv.imread("E:/cvdata/spray-3bar/3-110-1.1(1)/3-110-1.1-1_C001H001S0002002586.bmp")  # 坐标系参考图

'''
Global variable center point.
'''
(x_ctr, y_ctr) = draw_center('Select Center Point', image1)  # center point
(x_end, y_end) = draw_center('Select Center Point', image1)  # axis end point
(x_max, y_max) = draw_center('Select Center Point', image1)  # maximum circle
print('Base line :\n', x_ctr, x_end, y_ctr, y_end)

'''
Calculating the initial polar coordinates.
'''
radius_out, angle_out = polar_coordinate((x_ctr, y_ctr), (x_end, y_end))

'''
Output the angle of the normalized coordinate system.
'''
coordinates_out, angle_sys_out = coordinate_system((x_ctr, y_ctr), radius_out, angle_out)
draw_coordinates(coordinates_out, coordinate_image)
cv.imshow('coordinate line', coordinate_image)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # 转灰度
# gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
cv.imshow('gary', gray)
sub_frame1 = cv.absdiff(background, gray)  # 减背景帧
cv.imshow('origin_sub_frame', sub_frame1)
# sub_frame2 = cv.absdiff(background, gray1)
# sub_frame_joint = cv.absdiff(gray1, gray)
# sub_frame1 = cv.medianBlur(sub_frame1, 3)
# ret3, prime_binary = cv.threshold(sub_frame1, 30, 255, cv.THRESH_BINARY)
# cv.imshow('origin_binary', prime_binary)
# cv.imshow('sub_origin_background', sub_frame1)
# cv.imshow('sub1', sub_frame2)
# cv.imshow('joint', sub_frame_joint)
# aug1 = image_augment(sub_frame_joint, 3, 3)
aug2 = image_augment(sub_frame1, 3, 3)  # 图片增强 提高对比度 参数可调
# cv.imshow('augment', aug1)
# cv.imshow('augment2', aug2)
# smooth1 = cv.medianBlur(aug1, 3)
# smooth2 = cv.medianBlur(aug2, 3)
smooth2 = cv.GaussianBlur(aug2, (3, 3), 15)  # 图像平滑 高斯滤波
# cv.imshow('smooth1', smooth1)
cv.imshow('smooth_background', smooth2)

'''
Threshold track bar
'''
thresh_low = thresh_track_bar('smooth_background')
# ret1, binary1 = cv.threshold(smooth1, 50, 255, cv.THRESH_BINARY)
ret2, binary2 = cv.threshold(smooth2, thresh_low, 255, cv.THRESH_BINARY)  # 二值化
cv.imshow('binary_origin', binary2)

'''
Using maximum circle and mask to filter reflective interference.
'''
maxi_radius = int(math.sqrt((x_max - x_ctr)**2 + (y_max - y_ctr)**2))
mask_circle = mask_make(binary2, (x_ctr, y_ctr), maxi_radius)
binary2 = cv.bitwise_and(binary2, mask_circle)
cv.imshow('binary_background', binary2)

'''
Morphological manipulation to filter noise.
'''
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))  # 膨胀参数 （11， 11）
morph = cv.morphologyEx(binary2, cv.MORPH_OPEN, kernel)  # 形态学开操作
morph = cv.morphologyEx(morph, cv.MORPH_DILATE, kernel1)  # 膨胀
cv.imshow("morph", morph)

'''
Returning hierarchy tree
'''
contours_out_tree, child_ids = contours_tree(morph)  # 画轮廓 树状

'''
Output the eight intermediate points.
'''
black_dst, contours_out, points_out = draw_obj(morph, 'poly_approx')  # 画轮廓 最外轮廓
match_result = calculate_angle(points_out, angle_sys_out, (x_ctr, y_ctr))
all_test = []
contours_tree_filters = []
for match in match_result:
    x_caro, y_caro = polar2cartesian(match, (x_ctr, y_ctr))
    all_test.append((x_caro, y_caro))
    cv.circle(black_dst, (int(x_caro), int(y_caro)), 2, (0, 255, 255), -1)
for child_id in range(1, len(child_ids)):
    cv.drawContours(black_dst, contours_out_tree, child_ids[child_id], (255, 0, 0), 1, 8)
for chi_index in range(len(child_ids)):
    contours_tree_filters.append(contours_out_tree[child_ids[chi_index]])

print('Final coordinates is :\n', all_test)

all_extent = []
for cor_coordinate in all_test:
    all_extent.append(find_point((x_ctr, y_ctr), cor_coordinate, contours_tree_filters))

print('the minimum distance points:\n ', all_extent)
for extent in all_extent:
    if len(extent) == 2:
        p_p = extent[0]
        n_p = extent[1]
        cv.line(black_dst, p_p, n_p, (0, 255, 255), 1)
    else:
        continue

angle_list = angle_cone(all_test, (x_ctr, y_ctr), all_extent)
print('angle of every cone is:\n', angle_list)
cv.imshow('new dst', black_dst)
cv.waitKey(0)
cv.destroyAllWindows()
