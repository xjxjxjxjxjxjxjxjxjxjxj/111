from sklearn.cluster import KMeans
import numpy as np
import cv2
from enum import Enum
import base64
import math
import serial
import struct
import time
from auto_exposure import AutoExposureController
cap = cv2.VideoCapture(0)

class State(Enum):
    left = 1
    right = 2
    center = 3
    left_to_center = 4 
    right_to_center = 5 

class Bike_args:
    def __init__(self):
        self.blue_low  = np.array([65, 66, 60])
        self.blue_upper = np.array([125, 254, 255])

        self.find_current_x_delta_x = 130
        self.shifted_x = 18 
        # 避障参数
        self.block_detect_y = 255
        self.block_detect_delta_y = 80 
        self.block_h_upper = 20 
     
        # 通信数据
        self.block_avoid_data = struct.pack("!BB",0xA5,0X01)
        self.crosswalk_data = struct.pack("!BB", 0xA5, 0x05)

        self.change_road_difx = 20


class Bike_class:
    def __init__(self):
        #此模块内许多常数皆为需要调整的参数
        self.kmeans = KMeans(n_clusters=2)
        self.bike_args = Bike_args()
        self.M = None
        self.M_inverse = None
        self.M_inverse_list = None
        self.get_warp_M()

        # 串口初始化
        self.ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0)
    
        self.state = State.center
        self.dynamic_center_x = 320

        self.left_fit = None
        self.right_fit = None

        # 过度线
        self.skip_frame_times = 0
        self.initial_skip_frame_times = 60
        self.shift_to_center_x = 0          # 需要在每次从中线切换到边线时记录此值
        self.to_center_x = 0                # 需要在每次从中线切换到边线时记录此值
        self.dynamic_shift_x = 0            # 过度线时使用，动态变化

        self.task = 0
        self.skip_block_nums = 0
        self.last_block_state = State.center 
        self.catch_block_times = 0 
        self.wait_back_center_flag = 0      # 避障（切换寻线后）需要打开此flag，等待积分位移达到一定值后 将寻线置为过度线。
        self.block_nums = 3

        #左右变道
        self.change_road_method = 3
        self.arrow_detect_upy = 270            
        self.arrow_detect_downy = 480
        self.arrow_detect_deltax = 22
        self.last_change_road_direction = 0    
        self.catch_arrow_time = 0
        self.arrow_wait = 0                 
        self.change_index = 0
        self.near_dis = 5                    #x_error绝对值小于near_dis，则从本道内靠边一侧，准备变道


        #人行横道
        self.crossroad_method = 2
        self.last_crossroad_sign = 0
        self.road_detect_ratio = 1 / 7    
        self.crossroad_detect_upy = 295  
        self.crossroad_detect_downy = 445
        self.catch_crossroad_time = 0
        self.crossroad_detect_deltax = 25
        self.crossroad_mission_complete = 0


        # 周期控制
        self.block_detect_times = 0
        self.stop_line_times = 0

        # 全局变量
        self.img = None
        self.warp_img = None
        self.edges = None
        self.stop_skip_times = 20

        self.line_points_x = []    
        self.left_line_x = []    
        self.right_line_x = [] 

        self.leftx_mean = 0
        self.left_point_source = None
        self.rightx_mean = 0
        self.right_point_source = None
        self.catch_point_source = None

        self.rightx_mean_list = []
        self.leftx_mean_list = []
        self.x_mean_list = []
        self.x_mean = None  
        self.x_error = None
        self.last_x_error = None

        self.error_flag = 0

        self.img_size = (640, 480)
        self.y = 160

        self.wait_back_center_flag_debug = 0
        self.record_flag = 0
        self.error_x_save_list = []

        # ====== 自动曝光控制器（强光场景） ======
        self.ae = AutoExposureController(
            cap=cap,                     # 全局的 cv2.VideoCapture(0)
            device_id=0,
            target_brightness=115,
            roi_top_ratio=0.0,               # 全图测光
            roi_bottom_ratio=1.0,
            enable_hardware=True,
            enable_clahe=True,
            debug=False,
        )


    def get_warp_M(self):
        #本模块内的参数需要根据摄像头安装位置和角度自行调整
        objdx = 200
        objdy = 230
        imgdx = 220
        imgdy = 250
        list_pst = [[170, 330], [465, 330], [70, 473], [544, 475]]
        pts1 = np.float32(list_pst) 
        pts2 = np.float32([[imgdx, imgdy], [imgdx + objdx, imgdy], [imgdx, imgdy + objdy], [imgdx + objdx, imgdy + objdy]])
        self.M  = cv2.getPerspectiveTransform(pts1, pts2)
        self.M_inverse = cv2.getPerspectiveTransform(pts2, pts1)
        self.M_inverse_list = self.M_inverse.flatten()
    
    def point_reverse_perspective(self,point):
        x, y = point
        denom = self.M_inverse_list[6] * x + self.M_inverse_list[7] * y + 1
        x_transformed = (self.M_inverse_list[0] * x + self.M_inverse_list[1] * y + self.M_inverse_list[2]) / denom
        y_transformed = (self.M_inverse_list[3] * x + self.M_inverse_list[4] * y + self.M_inverse_list[5]) / denom
        return (int(x_transformed),int(y_transformed))
    
    def interpolate_value(self,start_value, end_value, initial_times, current_times):
        step = (end_value - start_value) / initial_times
        current_value = start_value + step * current_times 
        return int(current_value)
    
    def img_preprocess(self):
        self.img = cv2.medianBlur(self.img, 9)
        self.warp_img = cv2.warpPerspective(self.img,self.M,self.img_size)
        warp_img_hsv = cv2.cvtColor(self.warp_img, cv2.COLOR_BGR2HSV)
        blue_mask_white = cv2.inRange(warp_img_hsv, self.bike_args.blue_low,self.bike_args.blue_upper)
        blue_mask_white = cv2.bitwise_not(blue_mask_white)
        kernel = np.ones((5, 5), np.uint8)
        blue_mask_white = cv2.erode(blue_mask_white, kernel, iterations=1)
        edges = cv2.Canny(self.warp_img, 50, 40, apertureSize=3)
        edges = cv2.bitwise_and(edges, edges, mask=blue_mask_white)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2) 
        edges_mask = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.uint8)
        cv2.rectangle(edges_mask, (160, 0), (480, 480), 255, thickness=cv2.FILLED)
        self.edges = cv2.bitwise_and(edges, edges, mask=edges_mask)


    def img_HoughLines(self):
        self.line_points_x.clear()
        lines = cv2.HoughLines(self.edges, 1, np.pi / 180, threshold=260)
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                theta_degree = np.degrees(theta)
                if theta_degree > 90:
                    theta_degree = 180 - theta_degree
                if np.abs(theta_degree) > 35:
                    continue
                elif np.abs(theta) == 0:
                    b = rho
                    self.line_points_x.append(int(b))
                else:
                    m = -1 / np.tan(theta)
                    b = rho / np.sin(theta)
                    self.line_points_x.append(int((self.y-b)/m))


    def img_HoughLines_filter(self):
        self.left_line_x.clear()
        self.right_line_x.clear()
        if len(self.line_points_x) != 0:
            for point_x in self.line_points_x:
                if point_x < self.dynamic_center_x and point_x > (self.dynamic_center_x - self.bike_args.find_current_x_delta_x):
                    self.left_line_x.append(point_x)
                    cv2.circle(self.warp_img, (point_x,self.y), radius=5, color=(255, 255, 255), thickness=-1)
                elif point_x > self.dynamic_center_x and point_x < (self.dynamic_center_x + self.bike_args.find_current_x_delta_x):
                    self.right_line_x.append(point_x)
                    cv2.circle(self.warp_img, (point_x,self.y), radius=5, color=(255, 255, 255), thickness=-1)
            
            if self.state == State.left or self.state == State.left_to_center or self.state == State.center:
                if len(self.left_line_x) != 0:
                    self.leftx_mean = int(np.mean(self.left_line_x))
                    cv2.line(self.warp_img, (self.leftx_mean, 0), (self.leftx_mean, 480), (255, 0, 0), 3)
                    self.error_flag = 0
                else:
                    self.error_flag = 1
                
            if self.state == State.right or self.state == State.right_to_center or self.state == State.center:
                if len(self.right_line_x) != 0:
                    self.rightx_mean = int(np.mean(self.right_line_x))
                    cv2.line(self.warp_img, (self.rightx_mean, 0), (self.rightx_mean, 480), (255, 0, 0), 3)
                    self.error_flag = 0
                else:
                    self.error_flag = 1
        else:
            self.error_flag = 1


    def img_swap_windows(self):
        margin = 28   # 滑动窗宽度
        minpix = 33   # 窗口内最小像素值
        try:
            if self.error_flag != 1:
                left_lane_inds = []
                last_good_left_inds_len = 0
                right_lane_inds = []
                last_good_right_inds_len = 0
                nwindows = 8
                window_height = np.int32(self.img_size[1] / nwindows)
                nonzero = self.edges.nonzero()
                nonzeroy = np.array(nonzero[0])
                nonzerox = np.array(nonzero[1])
                for window in range(nwindows):
                    win_y_low = self.img_size[1] - (window + 1) * window_height
                    win_y_high = self.img_size[1] - window * window_height
                    if self.state == State.left or self.state == State.center or self.state == State.left_to_center:
                        win_xleft_low = self.leftx_mean - margin
                        win_xleft_high = self.leftx_mean + margin
                        
                    if self.state == State.right or self.state == State.center or self.state == State.right_to_center:
                        win_xright_low = self.rightx_mean - margin
                        win_xright_high = self.rightx_mean + margin
                        
                    if self.state == State.left or self.state == State.center or self.state == State.left_to_center:
                        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                        left_lane_inds.append(good_left_inds)
                        last_good_left_inds_len = len(good_left_inds)
                    
                    if self.state == State.right or self.state == State.center or self.state == State.right_to_center:
                        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                    (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                        right_lane_inds.append(good_right_inds)
                        last_good_right_inds_len = len(good_right_inds)
                    
                    if self.state == State.left or self.state == State.center or self.state == State.left_to_center:
                        if last_good_left_inds_len > minpix:
                            self.leftx_mean = np.int32(np.mean(nonzerox[good_left_inds]))
                    
                    if self.state == State.right or self.state == State.center or self.state == State.right_to_center:
                        if last_good_right_inds_len > minpix:
                            self.rightx_mean = np.int32(np.mean(nonzerox[good_right_inds]))
                
                if self.state == State.left or self.state == State.center or self.state == State.left_to_center:
                    left_lane_inds = np.concatenate(left_lane_inds)
                    leftx = nonzerox[left_lane_inds]
                    lefty = nonzeroy[left_lane_inds]
                
                if self.state == State.right or self.state == State.center or self.state == State.right_to_center:
                    right_lane_inds = np.concatenate(right_lane_inds)
                    rightx = nonzerox[right_lane_inds]
                    righty = nonzeroy[right_lane_inds]
 
                if self.state == State.left or self.state == State.left_to_center:
                    self.left_fit = np.polyfit(lefty, leftx, 2)
                    self.left_fit[2] = self.left_fit[2] + 5
                    _x = self.left_fit[0] * self.y ** 2 + self.left_fit[1] * self.y + self.left_fit[2]
                    self.left_point_source = self.point_reverse_perspective((_x,self.y))
                elif self.state == State.right or self.state == State.right_to_center:
                    self.right_fit = np.polyfit(righty, rightx, 2)
                    self.right_fit[2] = self.right_fit[2] - 5
                    _x = self.right_fit[0] * self.y ** 2 + self.right_fit[1] * self.y + self.right_fit[2]
                    self.right_point_source = self.point_reverse_perspective((_x,self.y))
                elif self.state == State.center:
                    self.left_fit = np.polyfit(lefty, leftx, 2)
                    self.right_fit = np.polyfit(righty, rightx, 2)
                    self.left_fit[2] = self.left_fit[2] + 5
                    _x = self.left_fit[0] * self.y ** 2 + self.left_fit[1] * self.y + self.left_fit[2]
                    self.left_point_source = self.point_reverse_perspective((_x,self.y))
                    _x = self.right_fit[0] * self.y ** 2 + self.right_fit[1] * self.y + self.right_fit[2]
                    self.right_point_source = self.point_reverse_perspective((_x,self.y))
        except:
            self.error_flag = 1


    def img_get_error_x(self):
        if self.error_flag != 1:
            try:
                if self.state == State.left or self.state == State.left_to_center:
                    shifted_left_fit = np.copy(self.left_fit)
                    shifted_left_fit[2] = shifted_left_fit[2] + self.bike_args.shifted_x + self.dynamic_shift_x
                    self.x_mean = shifted_left_fit[0] * self.y ** 2 + shifted_left_fit[1] * self.y + shifted_left_fit[2]
                    
                elif self.state == State.right or self.state == State.right_to_center:
                    shifted_right_fit = np.copy(self.right_fit)
                    shifted_right_fit[2] = shifted_right_fit[2] - self.bike_args.shifted_x - self.dynamic_shift_x
                    self.x_mean = shifted_right_fit[0] * self.y ** 2 + shifted_right_fit[1] * self.y + shifted_right_fit[2]
                    
                else:
                    mid_fit = (self.left_fit+self.right_fit) / 2
                    self.x_mean = mid_fit[0] * self.y ** 2 + mid_fit[1] * self.y + mid_fit[2]
                    
                self.catch_point_source = self.point_reverse_perspective((self.x_mean,self.y))
                self.x_error = self.x_mean - 320
                if self.last_x_error != None:
                    self.x_error = int(self.last_x_error * 0.4 + self.x_error * 0.6)
                self.last_x_error = self.x_error
                if self.record_flag:
                    self.error_x_save_list.append(self.x_error)              
            except:
                self.error_flag = 1


    def img_transition_line_task(self):
        if self.state == State.left_to_center or self.state == State.right_to_center:
            self.skip_frame_times += 1
            self.dynamic_shift_x = self.interpolate_value(0, self.shift_to_center_x, self.initial_skip_frame_times, self.skip_frame_times)
            if self.skip_frame_times > self.initial_skip_frame_times:
                self.state = State.center
                self.dynamic_shift_x = 0 
                # 绕过3个障碍物后不检测障碍物
                self.skip_block_nums += 1
                if self.skip_block_nums == self.block_nums and self.task == 0:
                    self.task = 1
                self.skip_frame_times = 0

                if self.task == 4:
                    self.task = 5
                

    def img_dynamic_center_task(self):
        if self.state == State.right or self.state == State.right_to_center:
            if len(self.rightx_mean_list) != 10:
                if self.error_flag != 1:
                    self.rightx_mean_list.append(self.rightx_mean)
            else:
                rightx_mean_data = np.array(self.rightx_mean_list[5:]).reshape(-1,1)
                self.kmeans.fit(rightx_mean_data)
                labels = self.kmeans.labels_
                count = np.bincount(labels)
                if len(count) == 2:
                    if count[0] > count[1]:
                        kmeans_rightx_mean = int(np.mean(rightx_mean_data[labels == 0]))
                    else:
                        kmeans_rightx_mean = int(np.mean(rightx_mean_data[labels == 1]))
                    if abs(self.dynamic_center_x - kmeans_rightx_mean) < int(self.to_center_x) or abs(self.dynamic_center_x - kmeans_rightx_mean) > int(self.to_center_x):
                        self.dynamic_center_x = kmeans_rightx_mean - self.to_center_x + 13
                self.rightx_mean_list.clear()
                
        elif self.state == State.left or self.state == State.left_to_center:
            if len(self.leftx_mean_list) != 10:
                if self.error_flag != 1:
                    self.leftx_mean_list.append(self.leftx_mean)
            else:
                leftx_mean_data = np.array(self.leftx_mean_list[5:]).reshape(-1,1)
                self.kmeans.fit(leftx_mean_data)
                labels = self.kmeans.labels_
                count = np.bincount(labels)
                if len(count) == 2:
                    if count[0] > count[1]:
                        kmeans_leftx_mean = int(np.mean(leftx_mean_data[labels == 0]))
                    else:
                        kmeans_leftx_mean = int(np.mean(leftx_mean_data[labels == 1]))
                    if abs(self.dynamic_center_x - kmeans_leftx_mean) < int(self.to_center_x) or abs(self.dynamic_center_x - kmeans_leftx_mean) > int(self.to_center_x):
                        self.dynamic_center_x = kmeans_leftx_mean + self.to_center_x - 13
                self.leftx_mean_list.clear()  
        else:
            self.dynamic_center_x = 320


    def serial_send_x(self):      #本部分所乘常数，根据舵机实际表现进行调整
        if self.error_flag != 1:
            try:
                self.x_error1 = self.x_error * (-1)
                if self.x_error1 < 0:
                    self.x_error1 *= 2.5
                else:
                    self.x_error1 *= 1.5
                self.x_error1 = int(self.x_error1)

                if self.task == 2 or self.task == 3:
                    self.x_error1 = int(self.x_error1 * 1.3)

                error_x_packdata = struct.pack("!BBBB",0xA5,0x00,(self.x_error1 >> 8) & 0xFF,self.x_error1 & 0xFF)
                self.ser.write(error_x_packdata)
                self.ser.flushOutput()
            except:
                print("usb_error!")


    def change_road_detect_task(self):     
        if self.task == 2 and self.state == State.center and self.change_index == 0:
            cv2.rectangle(self.warp_img, (int(self.leftx_mean + self.arrow_detect_deltax), self.arrow_detect_upy), (int(self.rightx_mean - self.arrow_detect_deltax), self.arrow_detect_downy), (0, 255, 255), 3)
            arrow_detect_img = self.warp_img[self.arrow_detect_upy:self.arrow_detect_downy,int(self.leftx_mean + self.arrow_detect_deltax):int(self.rightx_mean - self.arrow_detect_deltax)]
            arrow_hsv = cv2.cvtColor(arrow_detect_img, cv2.COLOR_BGR2HSV)
            arrow_mask = cv2.inRange(arrow_hsv, self.bike_args.blue_low, self.bike_args.blue_upper)

            arrow_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            arrow_mask = cv2.erode(arrow_mask, arrow_element, iterations=2)

            contours, hierarchy = cv2.findContours(arrow_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            x_coor = []
            y_coor = []
            w_coor = []
            h_coor = []
            arc_len_record = []
            if len(contours) == 2:
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    x_coor.append(x)
                    y_coor.append(y)
                    w_coor.append(w)
                    h_coor.append(h)
                    arc_len = cv2.arcLength(cnt, True)
                    arc_len_record.append(arc_len)
                    cv2.rectangle(arrow_detect_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    cv2.rectangle(self.warp_img, (int(self.leftx_mean + self.arrow_detect_deltax) + x, self.arrow_detect_upy + y), (int(self.leftx_mean + self.arrow_detect_deltax) + x + w, self.arrow_detect_upy + y + h), (0, 255, 0), 2)
                
                if self.change_road_method == 1:
                    print("not this")
                elif self.change_road_method == 2 or self.change_road_method == 3:
                    if arc_len_record[0] > arc_len_record[1]:
                        arrow_arc_index = 1
                    else:
                        arrow_arc_index = 0
                    arrow_M = cv2.moments(contours[arrow_arc_index])
                    if arrow_M["m00"] != 0:
                        if self.change_road_method == 2:
                            print("not this")
                        elif self.change_road_method == 3:
                            arrow_check_deltax = 6
                            arrow_mask_inv = cv2.bitwise_not(arrow_mask)
                            arrow_kernel = np.ones((3, 3), np.uint8)
                            arrow_frame1 = cv2.morphologyEx(arrow_mask_inv, cv2.MORPH_CLOSE, arrow_kernel)
                            detect_center_x = int(x_coor[arrow_arc_index] + w_coor[arrow_arc_index] / 2)
                            detect_center_y = y_coor[arrow_arc_index] + 3
                            arrow_detect_value_left = arrow_frame1.item(detect_center_y, detect_center_x - arrow_check_deltax)
                            arrow_detect_value_right = arrow_frame1.item(detect_center_y, detect_center_x + arrow_check_deltax)
                            while arrow_detect_value_left == arrow_detect_value_right and detect_center_y <= y_coor[arrow_arc_index] + h_coor[arrow_arc_index] / 2:
                                detect_center_y = detect_center_y + 1
                                arrow_detect_value_left = arrow_frame1.item(detect_center_y, detect_center_x - arrow_check_deltax)
                                arrow_detect_value_right = arrow_frame1.item(detect_center_y, detect_center_x + arrow_check_deltax)
                            if arrow_detect_value_left == 0 and arrow_detect_value_right == 255:
                                change_road_direction = 2
                            else:
                                change_road_direction = 1
                
                if self.last_change_road_direction == change_road_direction and y_coor[0] >= 2 and y_coor[1] >= 2 and h_coor[1] >= 80:
                    self.catch_arrow_time = self.catch_arrow_time + 1
                    if self.catch_arrow_time >= 2:
                        self.catch_arrow_time = 0
                        self.change_index = 1
                else:
                    self.last_change_road_direction = change_road_direction
                    self.catch_arrow_time = 0

        if self.task == 2 and self.arrow_wait > 0:
            self.arrow_wait = self.arrow_wait + 1
            if self.arrow_wait >= 10:
                self.arrow_wait = 0
                if self.last_change_road_direction == 1:
                    self.state = State.right
                elif self.last_change_road_direction == 2:
                    self.state = State.left
                self.task = 3
                self.shift_to_center_x = int((self.rightx_mean-self.leftx_mean)/2) - self.bike_args.shifted_x
                self.to_center_x = self.shift_to_center_x + self.bike_args.shifted_x
                #下面这一行除以的系数4，根据实际情况修改
                self.shift_to_center_x = int((self.rightx_mean-self.leftx_mean)/4) - self.bike_args.shifted_x


        if self.task == 3:
            if abs(self.x_error) <= self.near_dis:
                self.task = 4
                self.bike_args.shifted_x = 28
                if self.state == State.right:
                    self.state = State.left
                    self.dynamic_center_x = self.rightx_mean + self.to_center_x - 20
                elif self.state == State.left:
                    self.state = State.right
                    self.dynamic_center_x = self.leftx_mean - self.to_center_x + 20
                

        if self.task == 4 and (self.state == State.right or self.state == State.left):
            if abs(self.x_error) <= self.near_dis:
                if self.state == State.right:
                    self.state = State.right_to_center
                elif self.state == State.left:
                    self.state = State.left_to_center
                

        if self.task == 2 and self.wait_back_center_flag == 1:
            if self.get_back_center_flag() == 1:
                self.arrow_wait = 1
                self.wait_back_center_flag = 0


    def crosswalk_detect_task(self):    
        if self.task == 1 and self.state == State.center:
            detect_left = self.leftx_mean + self.crossroad_detect_deltax 
            detect_right = self.rightx_mean - self.crossroad_detect_deltax 
            crosswalk_detect_img = self.warp_img[self.crossroad_detect_upy:self.crossroad_detect_downy, detect_left:detect_right]
            crosswalk_gray = cv2.cvtColor(crosswalk_detect_img, cv2.COLOR_BGR2GRAY)
            ret, crosswalk_thresh = cv2.threshold(crosswalk_gray, 170, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            crosswalk_thresh_pro = cv2.morphologyEx(crosswalk_thresh, cv2.MORPH_OPEN, kernel)
            contours, hierarchy = cv2.findContours(crosswalk_thresh_pro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.rectangle(self.warp_img, (detect_left, self.crossroad_detect_upy), (detect_right, self.crossroad_detect_downy), (0, 255, 0), 3)

            road_sign = 0
            road_detect_bar_area = []
            if self.crossroad_method == 1:
                print("not this")
            elif self.crossroad_method == 2:     #面积检测法
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    road_detect_bar_area.append(w * h)
                total_area = np.sum(np.array(road_detect_bar_area))
                sum_area = (self.crossroad_detect_downy - self.crossroad_detect_upy) * (detect_right - detect_left)
                if total_area / sum_area >= self.road_detect_ratio:
                    road_sign = 1
                else:
                    road_sign = 0

            
            if road_sign == self.last_crossroad_sign and road_sign == 1:
                self.catch_crossroad_time = self.catch_crossroad_time + 1
                if self.catch_crossroad_time >= 3:
                    self.ser.write(self.bike_args.crosswalk_data)
                    self.ser.flushOutput()
                    self.wait_back_center_flag = 1
                    self.catch_crossroad_time = 0
                    self.last_crossroad_sign = 0
                    self.crossroad_mission_complete = 1
                    self.task = 2
            else:
                self.last_crossroad_sign = road_sign
                self.catch_crossroad_time = 0

    def main_task(self):
        ret, self.img = cap.read()

        # ====== 自动曝光处理（在 img_preprocess 之前） ======
        if ret:
            self.img = self.ae.apply(self.img)

        self.img_preprocess() 
        self.img_HoughLines() 
        self.img_HoughLines_filter()       
        self.img_swap_windows()

        self.img_get_error_x() 
        self.serial_send_x()
        self.img_dynamic_center_task()
        self.img_transition_line_task()

        self.detect_block_task()
        self.crosswalk_detect_task()     
        self.change_road_detect_task()


    
    def detect_block_task(self):
        if self.task == 0:
            self.block_detect_times += 1
            if self.block_detect_times >= 1:
                
                if self.state == State.center and self.error_flag != 1:
                    block_detect_leftx = self.left_point_source[0] + 20
                    block_detect_rightx = self.right_point_source[0] - 70

                    cv2.line(self.img, (block_detect_leftx, 0), (block_detect_leftx, 480), (125, 125, 125), 3)
                    cv2.line(self.img, (block_detect_rightx, 0), (block_detect_rightx, 480), (125, 125, 125), 3)
                    if block_detect_rightx > block_detect_leftx:
                        detect_center = int((block_detect_leftx + block_detect_rightx)/2)
                        
                        block_detect_image = self.img[self.bike_args.block_detect_y - self.bike_args.block_detect_delta_y:self.bike_args.block_detect_y,int(block_detect_leftx):int(block_detect_rightx)]
                        block_hsv_img = cv2.cvtColor(block_detect_image, cv2.COLOR_BGR2HSV)
                        blue_mask = cv2.inRange(block_hsv_img, self.bike_args.blue_low,self.bike_args.blue_upper)
                       
                        block_nonzero = blue_mask.nonzero()
                        block_nonzerox = np.array(block_nonzero[1])
                        block_meanx = np.int32(np.mean(block_nonzerox))
                        contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        block_list = []
                        for item in contours:
                            x, y, w, h = cv2.boundingRect(item)
                            if h < self.bike_args.block_h_upper or w < 10:
                                continue
                            else:
                                block_list.append((x,y,w,h))
                        if (len(block_list) == 1):
                            x, y, w, h = block_list[0]
                            if (block_detect_leftx + block_meanx < detect_center):
                                block_state = State.left
                            else:
                                block_state = State.right
                            
                            if block_state == self.last_block_state:
                                self.catch_block_times += 1
                                if self.catch_block_times >= 2:

                                    self.shift_to_center_x = int((self.rightx_mean-self.leftx_mean)/2) - self.bike_args.shifted_x
                                    self.to_center_x = self.shift_to_center_x + self.bike_args.shifted_x
                                    #下面这一行除以的常数3.1，根据实际情况调整
                                    self.shift_to_center_x = int(self.shift_to_center_x / 3.1)

                                    # 锥桶接近中线 避障线偏移要少一些 更加靠近边线
                                    if abs(block_detect_leftx+block_meanx-detect_center) < 10:
                                        self.bike_args.shifted_x = 13
                                    # 锥桶靠边 避障线偏移多一些
                                    else:
                                        self.bike_args.shifted_x = 28
                                        
                                    if block_state == State.left:
                                        self.state = State.right
                                    elif block_state == State.right:
                                        self.state = State.left
                    
                                    #发送串口消息
                                    self.ser.write(self.bike_args.block_avoid_data)
                                    self.ser.flushOutput()
                                    self.wait_back_center_flag = 1
                                    self.catch_block_times = 0
                            else:
                                self.catch_block_times = 0
                                self.last_block_state = block_state
            if self.wait_back_center_flag:
                if self.wait_back_center_flag_debug or self.get_back_center_flag():
                    if self.state == State.left:
                        self.state = State.left_to_center
                    elif self.state == State.right:
                        self.state = State.right_to_center
                    self.wait_back_center_flag = 0
                    self.last_block_state = State.center
                    self.wait_back_center_flag_debug = 0
                self.block_detect_times = 0

    def get_back_center_flag(self):
        if self.ser.in_waiting >= 5:
            whether_done_data = self.ser.read(5)
            whether_done_data = struct.unpack(">5B", whether_done_data)
            self.ser.flushInput()
            if whether_done_data[1] == 1 and self.task == 0:
                return 1
            elif whether_done_data[1] == 4 and self.task == 2:
                return 1
            else:
                return 0
        else:
            return 0


bike = Bike_class()
while True:
    bike.main_task()