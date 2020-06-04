import cv2 as cv
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

dir_ind = 1
root = 'F:/wangqianwen/flame_img/'
os.chdir(root)
dire_name = os.listdir()
path = root+dire_name[dir_ind]+'/'
os.chdir(path)
file_name = os.listdir()


def pick_out(all_f):
    """
    pick up .tif all_f
    :param all_f:
    :return:
    """
    del_id = 0
    length = len(all_f)
    for i in range(length):
        c_name = all_f[i]
        if c_name[-3:] == 'chd':
            del_id = i

    if del_id:
        del all_f[del_id]

    return all_f


def find_index(file_n):
    """
    input file name
    :param file_n:
    :return:
    """
    file_n = file_n.lstrip(file_n[0:11])
    try:
        if file_n[-3:] == 'tif':
            file_n = file_n.rstrip('.tif')
            i = int(file_n)

            return i
    except ValueError as e:

        return False


def select_roi(set_img, window_name):
    # cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    roi = cv.selectROI(window_name, set_img)

    return roi


def draw_hist(hist_g):
    with plt.style.context('seaborn'):
        plt.plot(hist_g)
        plt.show()

    return


def image_augment(img, ctr, bgt):
    """
    image augment
    :param img: image
    :param ctr: contrast
    :param bgt: bright
    :return: output image
    """
    blank = np.zeros(img.shape, img.dtype)
    out = cv.addWeighted(img, ctr, blank, 1 - ctr, bgt)

    return out


def find_mask(img_i):

    contours, hierarchy = cv.findContours(img_i, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    dst_in = np.zeros((img_i.shape[0], img_i.shape[1], 3), dtype=np.uint8)
    max_area = 0
    max_id = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv.contourArea(cnt)
        if area <= 20:
            continue
        else:
            if max_area < area:
                max_area = area
                max_id = i
    ellipse_in = cv.fitEllipse(contours[max_id])
    # print(ellipse_in)
    # cv.drawContours(dst_in, contours, max_id, (0, 255, 0))
    # cv.ellipse(dst_in, ellipse_in, (0, 255, 0), 2)
    # cv.imshow('mask_in', dst_in)
    mask = np.zeros_like(img_i, dtype=np.uint8)
    cv.ellipse(mask, ellipse_in, (255, 255, 255), -1)

    return mask, ellipse_in


def edge_cnt(input_img, ig_s):
    """
    :param input_img:
    :param ig_s: ignore size
    :return:
    """
    contours_list = []
    contours, hierarchy = cv.findContours(input_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    dst_in = np.zeros((input_img.shape[0], input_img.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv.contourArea(cnt)
        if area <= ig_s:
            continue
        cv.drawContours(dst_in, contours, i, (0, 255, 255), 1)

        contours_list.append(contours[i])

    dst_in = cv.resize(dst_in, (4*dst_in.shape[0], 4*dst_in.shape[1]))
    cv.imshow('edge', dst_in)

    return contours_list, hierarchy


def cell_info(contours_in):

    area_all_in = []
    length_all = []
    cell_num = len(contours_in)
    equ_d = []
    for i in range(cell_num):
        cnt = contours_in[i]
        cnt_area = cv.contourArea(cnt)
        if cnt_area <= 1:
            continue
        cnt_length = cv.arcLength(cnt, False)
        if cnt_area*cnt_length == 0:
            continue
        area_all_in.append(cnt_area)
        length_all.append(cnt_length)
        try:
            equ_d.append(cnt_area/cnt_length)
        except ZeroDivisionError as e:
            print(i)

    return area_all_in, length_all, equ_d, cell_num


def three_d(cnt_pint, ellipse_in):

    c_in = max(ellipse_in[1])
    a_in = ellipse_in[1][0]
    b_in = ellipse_in[1][1]
    x_in = cnt_pint[0]
    y_in = cnt_pint[1]
    z_in = c_in*(1-(x_in**2)/(a_in**2)-(y_in**2)/(b_in**2))**0.5

    return x_in, y_in, z_in


if __name__ == '__main__':

    file_p = pick_out(file_name)

    """ Input Image"""
    background = cv.imread(path+file_p[0])
    background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
    cv.imshow('bac', background)

    x_o, y_o, w_o, h_o = select_roi(background, 'bac')
    background = background[y_o:y_o + h_o, x_o:x_o + w_o]

    cur_frame = cv.imread(path+file_p[900])                  # interface of the object picture######################
    cur_frame = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)
    cur_frame = cur_frame[y_o:y_o+h_o, x_o:x_o+w_o]

    sub_frame = cv.absdiff(background, cur_frame)

    """ Image Augment"""
    ctr_o = 2.5
    bgt_o = 2.5
    equ = cv.equalizeHist(sub_frame)
    equ_my = image_augment(sub_frame, ctr_o, bgt_o)  # INTERFACE OF IMAGE AUGMENT
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equ_cl = clahe.apply(sub_frame)

    """The performance of different method """
    black = np.zeros((sub_frame.shape[0] + 10, sub_frame.shape[1]*5), dtype=np.uint8)
    black[10:sub_frame.shape[0]+10, 0:sub_frame.shape[1]] = cur_frame
    black[10:sub_frame.shape[0]+10, sub_frame.shape[1]:sub_frame.shape[1]*2] = sub_frame
    black[10:sub_frame.shape[0]+10, sub_frame.shape[1]*2:sub_frame.shape[1] * 3] = equ
    black[10:sub_frame.shape[0] + 10, sub_frame.shape[1] * 3:sub_frame.shape[1] * 4] = equ_my
    black[10:sub_frame.shape[0] + 10, sub_frame.shape[1] * 4:sub_frame.shape[1] * 5] = equ_cl
    cv.putText(black, 'Origin', (10, 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(black, 'sub', (10+sub_frame.shape[1], 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(black, 'equalization', (10+sub_frame.shape[1]*2, 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(black, 'simple'+str(ctr_o)+"_"+str(bgt_o), (10+sub_frame.shape[1]*3, 10), cv.FONT_HERSHEY_SIMPLEX, 0.5,
               (255, 255, 255), 1)
    cv.putText(black, 'equ_cl', (10 + sub_frame.shape[1] * 4, 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.imwrite('F:/wangqianwen/transient/'+str(dir_ind)+"_"+str(ctr_o)+"_"+str(bgt_o)+'augment.jpg', black)
    cv.imshow('comp1', black)

    """ Threshold segment """
    k_size = 1  # filter k_size #####################
    aug_inter = cv.medianBlur(equ_cl, k_size)  # Interface of threshold segment using previous step outcome
    hist = cv.calcHist([aug_inter], [0], None, [256], [0, 256])
    draw_hist(hist)

    ret, binary_otsu = cv.threshold(aug_inter, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    binary_adapt = cv.adaptiveThreshold(aug_inter, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    binary_adapt_inv = cv.adaptiveThreshold(aug_inter, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

    """Find Mask"""
    # ret1, binary_sub = cv.threshold(equ_my, 10, 255, cv.THRESH_BINARY)
    stride = (3, 3)                         # stride ####################
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, stride)
    binary_otsu = cv.dilate(binary_otsu, kernel)
    # cv.imshow('mask_Out', binary_otsu)
    mask_out, ellipsis_out = find_mask(binary_otsu)  # 火焰最小拟合椭圆 ellipsis_out
    mask_adapt = cv.bitwise_and(mask_out, binary_adapt)

    """The performance of different  threshold method """
    board = np.zeros((sub_frame.shape[0] + 10, sub_frame.shape[1] * 4), dtype=np.uint8)
    board[10:sub_frame.shape[0] + 10, 0:sub_frame.shape[1]] = binary_otsu
    board[10:sub_frame.shape[0] + 10, sub_frame.shape[1]:sub_frame.shape[1] * 2] = binary_adapt
    board[10:sub_frame.shape[0] + 10, sub_frame.shape[1]*2:sub_frame.shape[1] * 3] = mask_adapt
    board[10:sub_frame.shape[0] + 10, sub_frame.shape[1] * 3:sub_frame.shape[1] * 4] = aug_inter
    cv.putText(board, 'otsu', (10, 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(board, 'adapt', (10 + sub_frame.shape[1], 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(board, 'mask_and', (10 + sub_frame.shape[1]*2, 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(board, 'origin_KS'+str(k_size)+'_s'+str(stride), (10 + sub_frame.shape[1] * 3, 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.imshow('binary_compare', board)
    cv.imwrite('F:/wangqianwen/transient/'+str(dir_ind)+"_"+str(ctr_o)+"_"+str(bgt_o)+'thresh.jpg', board)

    """Finding edge of cells"""
    contours_out, hierarchy_out = edge_cnt(mask_adapt, 5)

    """Cell information"""
    print('The planer area is: ', cell_info(contours_out)[0])
    print('The planer length is: ', cell_info(contours_out)[1])
    print('The planer equ_diameter is: ', cell_info(contours_out)[2])
    print('The planer numbers of cells is: ', cell_info(contours_out)[3])
    # print(contours_out[10], type(contours_out), contours_out[10].shape)

    """3D Reconstruction"""
    x_center, y_center = ellipsis_out[0]
    print(ellipsis_out[0])

    for ind in range(len(contours_out)):
        if ind == 0:
            prim_n = contours_out[ind]
            continue
        cur_n = np.vstack((prim_n, contours_out[ind]))
        prim_n = cur_n

    contours_points = (prim_n - np.array([int(x_center), int(y_center)]))*np.array([1, -1])
    print(contours_points, contours_points.shape)

    c_3d = max(ellipsis_out[1]) / 2
    a_3d = ellipsis_out[1][0] / 2
    b_3d = ellipsis_out[1][1] / 2

    Z_3 = c_3d*(np.abs(1-contours_points[:, :, 1]**2/b_3d**2-contours_points[:, :, 0]**2/a_3d**2))**0.5
    print(Z_3, Z_3.shape, Z_3.min(), Z_3.max())
    # # #############绘图##############
    with plt.style.context('ggplot'):
        fig = plt.figure()
        ax = Axes3D(fig)
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_3 = a_3d * np.outer(np.cos(u), np.sin(v))
        y_3 = b_3d * np.outer(np.sin(u), np.sin(v))
        z_3 = c_3d * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_3, y_3, z_3, cmap=cm.gray)
        ax.scatter(contours_points[:, :, 0].ravel(), contours_points[:, :, 1].ravel(), Z_3.ravel(), s=1)
        fig.savefig('F:/wangqianwen/transient/'+str(dir_ind)+"_"+str(ctr_o)+"_"+str(bgt_o)+'3d.png')
        fig2 = plt.figure()
        plt.plot(contours_points[:, :, 0].ravel(), contours_points[:, :, 1].ravel(), linestyle='dotted')
        plt.show()
    # area_all = []
    # for i_o in range(len(contours_out)):
    #     mask_board = np.zeros((sub_frame.shape[0], sub_frame.shape[1]), np.uint8)
    #     cv.drawContours(mask_board, contours_out, i_o, (255, 255, 255), -1)
    #
    #     area_sum = 0
    #     for row in range(mask_board.shape[0]):
    #         for cols in range(mask_board.shape[1]):
    #             if mask_board[row][cols] == 255:
    #                 x_3d, y_3d = coordinates_trans(ellipsis_out, [(cols, row)])
    #                 z_3d = c_3d * (1 - (x_3d ** 2) / (a_3d ** 2) - (y_3d ** 2) / (b_3d ** 2)) ** 0.5
    #                 area_sum = area_sum+(1+((c_3d**2)*(x_3d ** 2))/((a_3d ** 4)*(z_3d**2))+((c_3d**2)*(y_3d ** 2))/(
    #                         (b_3d ** 4)*(z_3d**2)))**0.5
    #
    #     area_all.append(area_sum)

    cv.waitKey(0)
    cv.destroyAllWindows()

