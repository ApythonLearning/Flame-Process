from flame_debug import *

if __name__ == "__main__":

    file_p = pick_out(file_name)
    print(file_p)

    """ Background Image"""
    background = cv.imread(path + file_p[0])
    background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)

    x_o, y_o, w_o, h_o = select_roi(background, 'bac')
    background = background[y_o:y_o + h_o, x_o:x_o + w_o]
    cv.imshow('bac', background)

    obj_frame = cv.imread(path + file_p[200])
    obj_frame = cv.cvtColor(obj_frame, cv.COLOR_BGR2GRAY)
    obj_frame = obj_frame[y_o:y_o + h_o, x_o:x_o + w_o]

    kernel_1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1))
    kernel_3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    kernel_2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    # optimizer_obj = cv.morphologyEx(obj_frame, cv.MORPH_CLOSE, kernel2)
    # cv.imshow('optimizer', optimizer_obj)
    # obj_frame = cv.medianBlur(obj_frame, 3)
    # obj_frame = cv.GaussianBlur(obj_frame, (3, 3), 0)
    cv.imshow('Maximum crack', obj_frame)

    # hist = cv.calcHist([obj_frame], [0], None, [256], [0, 256])
    # draw_hist(hist)
    ret, binary_ori = cv.threshold(obj_frame, 68, 255, cv.THRESH_BINARY)

    binary_mask = cv.morphologyEx(binary_ori, cv.MORPH_DILATE, kernel_3)
    cv.imshow('binary', binary_ori)
    contours_m, hierarchy_m = cv.findContours(binary_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    mask_maximum = np.zeros(obj_frame.shape, dtype=np.uint8)
    area_max = 0
    max_id = 0
    for i_m in range(len(contours_m)):
        cnt = contours_m[i_m]
        area = cv.contourArea(cnt)

        if area > area_max:
            area_max = area
            max_id = i_m

    cv.drawContours(mask_maximum, contours_m, max_id, (255, 255, 255), -1)
    cv.imshow('mask max', mask_maximum)
    mask_crack = cv.bitwise_and(mask_maximum, binary_ori)
    # mask_crack = cv.subtract(mask_maximum, mask_crack)

    # mask_maximum = cv.morphologyEx(mask_maximum, cv.MORPH_CLOSE, kernel_2)

    # mask_crack = cv.morphologyEx(mask_crack, cv.MORPH_OPEN, kernel_2)
    cv.imshow('mask', mask_crack)

    cv.waitKey(0)
    cv.destroyAllWindows()