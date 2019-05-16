import cv2
import numpy as np
import math
from os import listdir
from os.path import join
import pickle
from .HoughBundler import HoughBundler


class LineDetect():
    def __init__(self):
        self.config = {
                        'enable_gaussian_blur': True,
                        'blur_kernal_size': 5,
                        'canny_low_threshold': 0,
                        'canny_high_threshold': 100,
                        'hough_rho': 1, # distance resolution in pixels of the Hough grid
                        'hough_threshold': 5, # minimum number of votes (intersections in Hough grid cell)
                        'hough_min_line_length': 25, # minimum number of pixels making up a line
                        'hough_max_line_gap': 4, # maximum gap in pixels between connectable line segments
                        'angle_limit': 15, # only lines which angle with horizontal or vertical lines are within this limit will be considered
                        'max_num_lines': 5, # the max number of lines for outside each boundary of the bounding boxes, totoal number of lines will be max_num_lines*4
                      }

    # bbox = [x1, y1, x2, y2]
    def get_lines(self, img, bbox):

        img_h, img_w, _ = img.shape

        # find lines
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        if self.config['enable_gaussian_blur']:
            blur_gray = cv2.GaussianBlur(gray,(self.config['blur_kernal_size'], self.config['blur_kernal_size']),0)
        else:
            blur_gray = gray

        edges = cv2.Canny(  blur_gray, 
                            self.config['canny_low_threshold'], 
                            self.config['canny_high_threshold'])

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, 
                                self.config['hough_rho'], 
                                np.pi / 180, 
                                self.config['hough_threshold'], 
                                np.array([]),
                                self.config['hough_min_line_length'], 
                                self.config['hough_max_line_gap'])
        
        #print('orig lines: ', lines)

        ##a = HoughBundler()
        ##lines1 = a.process_lines(lines, img)

        ##lines = np.array(lines1).reshape([-1,1,4])
        
        # remove the lines inside the bbox
        
        if type(lines)!=type(None):
            lines, _ = self.rm_lines_inside_bbox(lines, bbox)
        else:
            return None

        #print('rm inside lines: ', lines)

        # remove the lines intersect the bbox
        if lines.shape[0]:
            lines, _ = self.rm_lines_intersect_bbox(lines, bbox)
        else:
            return None
        
        #print('rm intersect lines: ', lines)
        
        if lines.shape[0]:
            lines_h, lines_v = self.get_horizontal_vertical_lines(lines, bbox)
        else:
            return None
        #print('lines_h: ', lines_h)
        #print('lines_v: ', lines_v)
        
        
        if lines_h.shape[0]:
            lines_h, _ = self.rm_lines_out_of_angle(lines_h, self.config['angle_limit'], orientation='x')
        if lines_v.shape[0]:
            lines_v, _ = self.rm_lines_out_of_angle(lines_v, self.config['angle_limit'], orientation='y')
        

        #print('rm angle lines_h: ', lines_h)
        #print('rm angle lines_v: ', lines_v)

        tmpData = self.prepare_lines_input(  lines_h, 
                                             lines_v, 
                                             self.config['max_num_lines'], 
                                             bbox, 
                                             img_w, 
                                             img_h)  
        return tmpData

# 
    def rm_lines_inside_bbox(self, lines, bbox): 
        idx_to_rm = []
        bbx_x1, bbx_y1, bbx_x2, bbx_y2 = bbox
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]

            if  (x1 >= bbx_x1 and x1 <= bbx_x2) and \
                (x2 >= bbx_x1 and x2 <= bbx_x2) and \
                (y1 >= bbx_y1 and y1 <= bbx_y2) and \
                (y2 >= bbx_y1 and y2 <= bbx_y2):
                idx_to_rm += [i]
        return np.delete(lines,idx_to_rm,axis=0), lines[idx_to_rm,:,:]


    def one_segment(self, p, q, r):
        px,py = p 
        qx,qy = q 
        rx,ry = r 
        return qx<=max(px,rx) and qx>=min(px,rx) and qy<=max(py,ry) and qy>=min(py,ry)

    def orientation(self, p, q, r):
        px,py = p 
        qx,qy = q 
        rx,ry = r 
        val = int((qy-py)*(rx-qx)-(qx-px)*(ry-qy))
        if val ==0:
            return 0
        elif val >0:
            return 1
        else:
            return 2

    def do_intersect(self, p1,q1,p2,q2):
        o1 = self.orientation(p1,q1,p2)
        o2 = self.orientation(p1,q1,q2)
        o3 = self.orientation(p2,q2,p1)
        o4 = self.orientation(p2,q2,q1)

        if o1!=o2 and o3!=o4:
            return True

        if  (o1==0 and self.one_segment(p1,p2,q1)) or \
            (o2==0 and self.one_segment(p1,q2,q1)) or \
            (o3==0 and self.one_segment(p2,p1,q2)) or \
            (o4==0 and self.one_segment(p2,q1,q2)):
            return True
        else:
            return False


    def rm_lines_intersect_bbox(self, lines, bbox):
        idx_to_rm = []
        bbx_x1, bbx_y1, bbx_x2, bbx_y2 = bbox
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]

            if  self.do_intersect((x1,y1),(x2,y2),(bbx_x1,bbx_y1),(bbx_x2,bbx_y1)) or \
                self.do_intersect((x1,y1),(x2,y2),(bbx_x1,bbx_y1),(bbx_x1,bbx_y2)) or \
                self.do_intersect((x1,y1),(x2,y2),(bbx_x2,bbx_y2),(bbx_x2,bbx_y1)) or \
                self.do_intersect((x1,y1),(x2,y2),(bbx_x2,bbx_y2),(bbx_x1,bbx_y2)):
                idx_to_rm += [i]
        return np.delete(lines,idx_to_rm,axis=0), lines[idx_to_rm,:,:]


    # divide all the lines into horizontal and vertial lines
    # lines directly above and below bbox are horizontal lines
    # lines directly on the left and right of bbox are vertical lines
    # one line can be horizontal and vertical at the same time
    def get_horizontal_vertical_lines(self, lines, bbox):
        idx_h = []
        idx_v = []
        bbx_x1, bbx_y1, bbx_x2, bbx_y2 = bbox
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            if (x1 >= bbx_x1 and x1 <= bbx_x2) or (x2 >= bbx_x1 and x2 <= bbx_x2):
                idx_h += [i]
            if (y1 >= bbx_y1 and y1 <= bbx_y2) or (y2 >= bbx_y1 and y2 <= bbx_y2):
                idx_v += [i]
        return lines[idx_h,:,:], lines[idx_v,:,:]


    # return angle range is in (-90, 90)
    def angle(self, p, q):
        px,py = p 
        qx,qy = q 
        if px==qx and py==qy:
            assert False, 'this is a point not line!!!'
        if px==qx:
            return 90.0
        if py==qy:
            return 0.0
        if px > qx:
            return np.arctan(1.0*(py-qy)/(px-qx))/np.pi*180
        if px < qx:
            return np.arctan(1.0*(qy-py)/(qx-px))/np.pi*180

    def rm_lines_out_of_angle(self, lines, angle_limit, orientation='x'):
        idx_to_rm = []
            
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]

            tmp_angle = self.angle((x1,y1),(x2,y2))

            #print('tmp_angle: ', tmp_angle)

            if orientation == 'x':
                if (tmp_angle < -angle_limit) or (tmp_angle > angle_limit):
                    idx_to_rm += [i]
            elif orientation == 'y':
                if (tmp_angle > -90.0+angle_limit) and (tmp_angle < 90.0-angle_limit):
                    idx_to_rm += [i]
            else:
                assert False, 'wrong orientation!!'

        return np.delete(lines,idx_to_rm,axis=0), lines[idx_to_rm,:,:]

    # process the lines to get inputs for machine learning
    def prepare_lines_input(self, lines_h, lines_v, max_num_lines, bbox, img_w, img_h):
        bbx_x1, bbx_y1, bbx_x2, bbx_y2 = bbox
        rst = np.zeros([4*max_num_lines,4]).astype(np.float32)
        lines_h_t = lines_h[lines_h[:,0,1]<bbx_y1,...].astype(np.float32)
        lines_h_b = lines_h[lines_h[:,0,1]>bbx_y2,...].astype(np.float32)
        lines_v_l = lines_v[lines_v[:,0,0]<bbx_x1,...].astype(np.float32)
        lines_v_r = lines_v[lines_v[:,0,0]>bbx_x2,...].astype(np.float32)
        if lines_h_t.shape[0]:
            for ri, i in enumerate(np.argsort(lines_h_t[:,0,1]+lines_h_t[:,0,3])[::-1][0:max_num_lines]):
                #print('lines_h_t sort: {} {}'.format(i, lines_h_t[i,0,1]+lines_h_t[i,0,3]))
                rst[ri,:] = lines_h_t[i,0,:]/[img_w,img_h,img_w,img_h]*2-1
        if lines_h_b.shape[0]:
            for ri, i in enumerate(np.argsort(lines_h_b[:,0,1]+lines_h_b[:,0,3])[0:max_num_lines]):
                #print('lines_h_b sort: {} {}'.format(i, lines_h_b[i,0,1]+lines_h_b[i,0,3]))
                rst[ri+max_num_lines,:] = lines_h_b[i,0,:]/[img_w,img_h,img_w,img_h]*2-1
        if lines_v_l.shape[0]:
            for ri, i in enumerate(np.argsort(lines_v_l[:,0,0]+lines_v_l[:,0,2])[::-1][0:max_num_lines]):
                #print('lines_v_l sort: {} {}'.format(i, lines_v_l[i,0,0]+lines_v_l[i,0,2]))
                if lines_v_l[i,0,1] < lines_v_l[i,0,3]:
                    rst[ri+max_num_lines*2,:] = lines_v_l[i,0,:]/[img_w,img_h,img_w,img_h]*2-1
                else:
                    rst[ri+max_num_lines*2,:] = lines_v_l[i,0,[2,3,0,1]]/[img_w,img_h,img_w,img_h]*2-1
        if lines_v_r.shape[0]:
            for ri, i in enumerate(np.argsort(lines_v_r[:,0,0]+lines_v_r[:,0,2])[0:max_num_lines])  :
                #print('lines_v_r sort: {} {}'.format(i, lines_v_r[i,0,0]+lines_v_r[i,0,2]))
                if lines_v_r[i,0,1] < lines_v_r[i,0,3]:
                    rst[ri+max_num_lines*3,:] = lines_v_r[i,0,:]/[img_w,img_h,img_w,img_h]*2-1
                else:
                    rst[ri+max_num_lines*3,:] = lines_v_r[i,0,[2,3,0,1]]/[img_w,img_h,img_w,img_h]*2-1
        #print('img_w: ', img_w)
        #print('img_h: ', img_h)
        #print('lines_h_t: ', lines_h_t)
        #print('lines_h_b: ', lines_h_b)
        #print('lines_v_l: ', lines_v_l)
        #print('lines_v_r: ', lines_v_r)
        #print ('line feature: ', rst)
        return rst.reshape([-1])
