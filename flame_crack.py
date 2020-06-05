from flame_debug import *

if __name__ == '__main__':
    file_p = pick_out(file_name)
    """ Background Image"""
    background = cv.imread(path + file_p[0])
    background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
    cv.imshow('bac', background)

    x_o, y_o, w_o, h_o = select_roi(background, 'bac')
    background = background[y_o:y_o + h_o, x_o:x_o + w_o]
    print(background.shape)

    out = cv.VideoWriter('F:/wangqianwen/transient/'+str(dir_ind)+'_output.avi',
                         cv.VideoWriter_fourcc('D', 'I', 'V', 'X'), 20.0,
                         (int(background.shape[1]*5), int(background.shape[0]+10)), False)
    out2 = cv.VideoWriter('F:/wangqianwen/transient/'+str(dir_ind)+'_output2.avi',
                          cv.VideoWriter_fourcc('D', 'I', 'V', 'X'), 20.0,
                          (int(background.shape[1] * 4), int(background.shape[0] + 10)), False)

    for i_d in range(len(file_p)):
        if i_d == 0:
            continue
        cur_frame = cv.imread(path + file_p[i_d])  # interface of the object picture######################
        cur_frame = cv.cvtColor(cur_frame, cv.COLOR_BGR2GRAY)
        cur_frame = cur_frame[y_o:y_o + h_o, x_o:x_o + w_o]

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
        out.write(black)

        """ Threshold segment """
        k_size = 1  # filter k_size #####################
        aug_inter = cv.medianBlur(equ_cl, k_size)  # Interface of threshold segment using previous step outcome

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
        # cv.imshow('binary_compare', board)
        out2.write(board)

        if cv.waitKey(1) == ord('q'):
            break

    out.release()
    out2.release()