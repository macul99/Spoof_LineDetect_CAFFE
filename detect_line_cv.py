import cv2
import numpy as np
import math
from os import listdir
from os.path import join
import pickle
from HoughBundler import HoughBundler
import caffe

#import isect_segments_bentley_ottmann.poly_point_isect as bot

test_list = [    '/home/macul/record_2018_09_11_13_47_51'
                ]

pos_list = [    '/media/macul/black/spoof_db/casia/real',
                '/media/macul/black/spoof_db/iim/iim_real_jul_2018',
                '/media/macul/black/spoof_db/iim/toukong_real_jul_2018',
                '/media/macul/black/spoof_db/NUAA/ClientRaw'
                ]
neg_list = [    '/media/macul/black/spoof_db/collected/original/screen_attack/tablet_small',                
                '/media/macul/black/spoof_db/casia/image_attack',
                '/media/macul/black/spoof_db/casia/screen_attack',
                '/media/macul/black/spoof_db/collected/original/image_attack/printed',
                '/media/macul/black/spoof_db/collected/original/screen_attack/tablet',              
                '/media/macul/black/spoof_db/iim/iim_image_attack',
                '/media/macul/black/spoof_db/NUAA/ImposterRaw',
                #'/media/macul/black/spoof_db/record_2018_08_13_17_31_27'
                ]

LineDetectCfg  = {
                    'enable_gaussian_blur': True,
                    'blur_kernal_size': 5,
                    'canny_low_threshold': 0,
                    'canny_high_threshold': 100,
                    'hough_rho': 1, # distance resolution in pixels of the Hough grid
                    'hough_threshold': 5, # minimum number of votes (intersections in Hough grid cell)
                    'hough_min_line_length': 25, # minimum number of pixels making up a line
                    'hough_max_line_gap': 4, # maximum gap in pixels between connectable line segments
                    'tgt_bbox_width': 90, # the input image will be resize to this width
                    'angle_limit': 15, # only lines which angle with horizontal or vertical lines are within this limit will be considered
                    'max_num_lines': 5, # the max number of lines for outside each boundary of the bounding boxes, totoal number of lines will be max_num_lines*4
                    'crop_scale_to_bbox': 2.2 # crop scale compared to bbox size
};

rst_dir = '/media/macul/black/MK/Projects/spoofing_ld'

model = '/media/macul/black/MK/Projects/spoofing_ld/clf.prototxt'
weights = '/media/macul/black/MK/Projects/spoofing_ld/snapshot_line_detect_1/mySolverSpoofingLineDetect_iter_100000.caffemodel'

# 
def rm_lines_inside_bbox(lines, bbox): 
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


def one_segment(p, q, r):
    px,py = p 
    qx,qy = q 
    rx,ry = r 
    return qx<=max(px,rx) and qx>=min(px,rx) and qy<=max(py,ry) and qy>=min(py,ry)

def orientation(p, q, r):
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

def do_intersect(p1,q1,p2,q2):
    o1 = orientation(p1,q1,p2)
    o2 = orientation(p1,q1,q2)
    o3 = orientation(p2,q2,p1)
    o4 = orientation(p2,q2,q1)

    if o1!=o2 and o3!=o4:
        return True

    if  (o1==0 and one_segment(p1,p2,q1)) or \
        (o2==0 and one_segment(p1,q2,q1)) or \
        (o3==0 and one_segment(p2,p1,q2)) or \
        (o4==0 and one_segment(p2,q1,q2)):
        return True
    else:
        return False


def rm_lines_intersect_bbox(lines, bbox):
    idx_to_rm = []
    bbx_x1, bbx_y1, bbx_x2, bbx_y2 = bbox
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]

        if  do_intersect((x1,y1),(x2,y2),(bbx_x1,bbx_y1),(bbx_x2,bbx_y1)) or \
            do_intersect((x1,y1),(x2,y2),(bbx_x1,bbx_y1),(bbx_x1,bbx_y2)) or \
            do_intersect((x1,y1),(x2,y2),(bbx_x2,bbx_y2),(bbx_x2,bbx_y1)) or \
            do_intersect((x1,y1),(x2,y2),(bbx_x2,bbx_y2),(bbx_x1,bbx_y2)):
            idx_to_rm += [i]
    return np.delete(lines,idx_to_rm,axis=0), lines[idx_to_rm,:,:]


# divide all the lines into horizontal and vertial lines
# lines directly above and below bbox are horizontal lines
# lines directly on the left and right of bbox are vertical lines
# one line can be horizontal and vertical at the same time
def get_horizontal_vertical_lines(lines, bbox):
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
def angle(p, q):
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

def rm_lines_out_of_angle(lines, angle_limit, orientation='x'):
    idx_to_rm = []
        
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]

        tmp_angle = angle((x1,y1),(x2,y2))

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
def prepare_lines_input(lines_h, lines_v, max_num_lines, bbox, img_w, img_h):
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

def processing_bk(folder_list):
    train_data = []
    test_data = []
    np.random.seed(42)
    train_test_split = 0.7
    for pl in folder_list:
        with open(join(pl+'_processed', 'bbox.txt'), 'rb') as bbox_f:
            bbox_f.readline()
            for bbox_l in bbox_f.readlines():
                fn, bbx_x, bbx_y, bbx_w, bbx_h = bbox_l.strip().split()
                bbx_w = int(bbx_w)
                bbx_h = int(bbx_h)
                bbx_x = int(bbx_x)
                bbx_y = int(bbx_y)
                tmpScale = 1.0*LineDetectCfg['tgt_bbox_width']/bbx_w

                print(join(pl, fn))
                img = cv2.imread(join(pl,fn))
                img_h, img_w, _ = img.shape
                new_img_w = int(tmpScale*img_w)
                new_img_h = int(tmpScale*img_h)
                img = cv2.resize(img,(new_img_w,new_img_h))

                bbx_x = int(1.0*bbx_x/img_w*new_img_w)
                bbx_y = int(1.0*bbx_y/img_h*new_img_h)          
                bbx_w = int(bbx_w*tmpScale)
                bbx_h = int(bbx_h*tmpScale)

                crp_w = int(bbx_w*LineDetectCfg['crop_scale_to_bbox'])
                crp_h = int(bbx_h*LineDetectCfg['crop_scale_to_bbox'])
                crp_x1 = max(0, int(bbx_x-(crp_w-bbx_w)/2.0))
                crp_y1 = max(0, int(bbx_y-(crp_h-bbx_h)/2.0))
                crp_x2 = min(crp_x1+crp_w-1, new_img_w-1)
                crp_y2 = min(crp_y1+crp_h-1, new_img_h-1)
                crp_w = crp_x2-crp_x1+1
                crp_h = crp_y2-crp_y1+1

                # crop image
                img = img[crp_y1:crp_y2+1,crp_x1:crp_x2+1,:]

                bbx_x1 = bbx_x - crp_x1
                bbx_x2 = bbx_x - crp_x1 + bbx_w -1
                bbx_y1 = bbx_y - crp_y1
                bbx_y2 = bbx_y - crp_y1 + bbx_h -1

                print('bbx_x1: ', bbx_x1)
                print('bbx_x2: ', bbx_x2)
                print('bbx_y1: ', bbx_y1)
                print('bbx_y2: ', bbx_y2)

                # find lines
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                if LineDetectCfg['enable_gaussian_blur']:
                    blur_gray = cv2.GaussianBlur(gray,(LineDetectCfg['blur_kernal_size'], LineDetectCfg['blur_kernal_size']),0)
                else:
                    blur_gray = gray

                edges = cv2.Canny(blur_gray, LineDetectCfg['canny_low_threshold'], LineDetectCfg['canny_high_threshold'])

                rho = 1  # distance resolution in pixels of the Hough grid
                theta = np.pi / 180  # angular resolution in radians of the Hough grid
                threshold = 10  # minimum number of votes (intersections in Hough grid cell)
                min_line_length = 25  # minimum number of pixels making up a line
                max_line_gap = 4  # maximum gap in pixels between connectable line segments
                line_image = np.copy(img) * 0  # creating a blank to draw lines on

                # Run Hough on edge detected image
                # Output "lines" is an array containing endpoints of detected line segments
                lines = cv2.HoughLinesP(edges, 
                                        LineDetectCfg['hough_rho'], 
                                        theta, 
                                        LineDetectCfg['hough_threshold'], 
                                        np.array([]),
                                        LineDetectCfg['hough_min_line_length'], 
                                        LineDetectCfg['hough_max_line_gap'])

                #a = HoughBundler()
                #lines1 = a.process_lines(lines, img)

                #lines = np.array(lines1).reshape([-1,1,4])

                # remove the lines inside the bbox
                if type(lines)!=type(None):
                    lines, _ = rm_lines_inside_bbox(lines, [bbx_x1,bbx_y1,bbx_x2,bbx_y2])
                else:
                    continue
                # remove the lines intersect the bbox
                if lines.shape[0]:
                    lines, _ = rm_lines_intersect_bbox(lines, [bbx_x1,bbx_y1,bbx_x2,bbx_y2])
                
                if lines.shape[0]:
                    lines_h, lines_v = get_horizontal_vertical_lines(lines, [bbx_x1,bbx_y1,bbx_x2,bbx_y2])

                if lines_h.shape[0]:
                    lines_h, _ = rm_lines_out_of_angle(lines_h, LineDetectCfg['angle_limit'], orientation='x')
                if lines_v.shape[0]:
                    lines_v, _ = rm_lines_out_of_angle(lines_v, LineDetectCfg['angle_limit'], orientation='y')

                tmpData = prepare_lines_input(  lines_h, 
                                                lines_v, 
                                                LineDetectCfg['max_num_lines'], 
                                                [bbx_x1,bbx_y1,bbx_x2,bbx_y2], 
                                                crp_w, 
                                                crp_h)
                if np.random.random() > train_test_split:
                    test_data += [tmpData]
                else:
                    train_data += [tmpData]

                if True:
                    if lines_h.shape[0]:
                        print('lines_h shape: ',lines_h.shape)
                        mylines = lines_h.copy()

                        max_x_idx = np.argsort(lines_h[:,0,2]-lines_h[:,0,0])[:]
                        for i, line in enumerate(lines_h):
                            for x1, y1, x2, y2 in line:
                                if i in max_x_idx:
                                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

                    if lines_v.shape[0]:
                        print('lines_v shape: ',lines_v.shape)
                        mylines = lines_v.copy()

                        max_y_idx = np.argsort(lines_v[:,0,3]-lines_v[:,0,1])[:]
                        for i, line in enumerate(lines_v):
                            for x1, y1, x2, y2 in line:
                                if i in max_y_idx:
                                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    cv2.line(line_image, (bbx_x1, bbx_y1), (bbx_x2, bbx_y1), (0, 0, 255), 3)
                    cv2.line(line_image, (bbx_x1, bbx_y1), (bbx_x1, bbx_y2), (0, 0, 255), 3)
                    cv2.line(line_image, (bbx_x1, bbx_y2), (bbx_x2, bbx_y2), (0, 0, 255), 3)
                    cv2.line(line_image, (bbx_x2, bbx_y1), (bbx_x2, bbx_y2), (0, 0, 255), 3)

                    #cv2.line(line_image, (crp_x, crp_y), (crp_x+crp_w-1, crp_y), (0, 255, 255), 3)
                    #cv2.line(line_image, (crp_x, crp_y), (crp_x, crp_y+crp_h-1), (0, 255, 255), 3)
                    #cv2.line(line_image, (crp_x, crp_y+crp_h-1), (crp_x+crp_w-1, crp_y+crp_h-1), (0, 255, 255), 3)
                    #cv2.line(line_image, (crp_x+crp_w-1, crp_y), (crp_x+crp_w-1, crp_y+crp_h-1), (0, 255, 255), 3)

                    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

                    cv2.imshow("res",lines_edges)
                    if cv2.waitKey(0)==']':
                        exit(1)

    return np.array(train_data), np.array(test_data)
        
def test_score(net, line_vec):    
    net.blobs['clf_data'].data[0,:,0,0] = np.array(line_vec)
    net.forward()
    score = net.blobs['clf_prob'].data[0,1]
    print('score: ', score)

def processing(folder_list, plot_flag=False):    
    caffe.set_device(0);
    caffe.set_mode_gpu();
    net = caffe.Net(model, weights, caffe.TEST)

    train_data = []
    test_data = []
    np.random.seed(42)
    train_test_split = 0.7
    for pl in folder_list:
        with open(join(pl+'_processed', 'bbox.txt'), 'rb') as bbox_f:
            bbox_f.readline()
            for bbox_l in bbox_f.readlines():
                fn, bbx_x, bbx_y, bbx_w, bbx_h = bbox_l.strip().split()
                bbx_w = int(bbx_w)
                bbx_h = int(bbx_h)
                bbx_x = int(bbx_x)
                bbx_y = int(bbx_y)

                #print('orig_bbx_x: ', bbx_x)
                #print('orig_bbx_y: ', bbx_y)
                #print('orig_bbx_width: ', bbx_w)
                #print('orig_bbx_height: ', bbx_h)                

                print(join(pl, fn))
                img = cv2.imread(join(pl,fn))
                img_h, img_w, _ = img.shape

                #print('orig_img_shape: ', img.shape)

                crp_w = int(bbx_w*LineDetectCfg['crop_scale_to_bbox'])
                crp_h = int(bbx_h*LineDetectCfg['crop_scale_to_bbox'])
                crp_x1 = max(0, int(bbx_x-(crp_w-bbx_w)/2.0))
                crp_y1 = max(0, int(bbx_y-(crp_h-bbx_h)/2.0))
                crp_x2 = min(crp_x1+crp_w-1, img_w-1)
                crp_y2 = min(crp_y1+crp_h-1, img_h-1)
                crp_w = crp_x2-crp_x1+1
                crp_h = crp_y2-crp_y1+1

                #print('crop_x: ', crp_x1)
                #print('crop_y: ', crp_y1)
                #print('crop_w: ', crp_w)
                #print('crop_h: ', crp_h)

                img = img[crp_y1:crp_y2+1,crp_x1:crp_x2+1,:]
                img_h, img_w, _ = img.shape
                #print('crop_img_shape: ', img.shape)

                tmpScale = 1.0*LineDetectCfg['tgt_bbox_width']/bbx_w
                bbx_x = bbx_x - crp_x1
                bbx_y = bbx_y - crp_y1

                #print('crop_bbx_x: ', bbx_x)
                #print('crop_bbx_y: ', bbx_y)

                new_img_w = int(tmpScale*img_w)
                new_img_h = int(tmpScale*img_h)
                img = cv2.resize(img,(new_img_w,new_img_h))
                img_h, img_w, _ = img.shape

                #print('resized_img_shape: ', img.shape)

                bbx_x = int(bbx_x*tmpScale)
                bbx_y = int(bbx_y*tmpScale)          
                bbx_w = int(bbx_w*tmpScale)
                bbx_h = int(bbx_h*tmpScale)

                #print('resize_bbx_x: ', bbx_x)
                #print('resize_bbx_y: ', bbx_y)
                #print('resize_bbx_width: ', bbx_w)
                #print('resize_bbx_height: ', bbx_h)

                # crop image
                

                bbx_x1 = bbx_x
                bbx_x2 = bbx_x1 + bbx_w -1
                bbx_y1 = bbx_y
                bbx_y2 = bbx_y1 + bbx_h -1

                #print('bbx_x1: ', bbx_x1)
                #print('bbx_x2: ', bbx_x2)
                #print('bbx_y1: ', bbx_y1)
                #print('bbx_y2: ', bbx_y2)

                # find lines
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                if LineDetectCfg['enable_gaussian_blur']:
                    blur_gray = cv2.GaussianBlur(gray,(LineDetectCfg['blur_kernal_size'], LineDetectCfg['blur_kernal_size']),0)
                else:
                    blur_gray = gray

                edges = cv2.Canny(  blur_gray, 
                                    LineDetectCfg['canny_low_threshold'], 
                                    LineDetectCfg['canny_high_threshold'])

                rho = 1  # distance resolution in pixels of the Hough grid
                theta = np.pi / 180  # angular resolution in radians of the Hough grid
                threshold = 10  # minimum number of votes (intersections in Hough grid cell)
                min_line_length = 25  # minimum number of pixels making up a line
                max_line_gap = 4  # maximum gap in pixels between connectable line segments
                line_image = np.copy(img) * 0  # creating a blank to draw lines on

                # Run Hough on edge detected image
                # Output "lines" is an array containing endpoints of detected line segments
                lines = cv2.HoughLinesP(edges, 
                                        LineDetectCfg['hough_rho'], 
                                        theta, 
                                        LineDetectCfg['hough_threshold'], 
                                        np.array([]),
                                        LineDetectCfg['hough_min_line_length'], 
                                        LineDetectCfg['hough_max_line_gap'])
                
                #print('orig lines: ', lines)

                ##a = HoughBundler()
                ##lines1 = a.process_lines(lines, img)

                ##lines = np.array(lines1).reshape([-1,1,4])
                
                # remove the lines inside the bbox
                
                if type(lines)!=type(None):
                    lines, _ = rm_lines_inside_bbox(lines, [bbx_x1,bbx_y1,bbx_x2,bbx_y2])
                else:
                    continue

                #print('rm inside lines: ', lines)

                # remove the lines intersect the bbox
                if lines.shape[0]:
                    lines, _ = rm_lines_intersect_bbox(lines, [bbx_x1,bbx_y1,bbx_x2,bbx_y2])
                
                #print('rm intersect lines: ', lines)
                
                if lines.shape[0]:
                    lines_h, lines_v = get_horizontal_vertical_lines(lines, [bbx_x1,bbx_y1,bbx_x2,bbx_y2])
                #print('lines_h: ', lines_h)
                #print('lines_v: ', lines_v)
                
                
                if lines_h.shape[0]:
                    lines_h, _ = rm_lines_out_of_angle(lines_h, LineDetectCfg['angle_limit'], orientation='x')
                if lines_v.shape[0]:
                    lines_v, _ = rm_lines_out_of_angle(lines_v, LineDetectCfg['angle_limit'], orientation='y')
                

                #print('rm angle lines_h: ', lines_h)
                #print('rm angle lines_v: ', lines_v)

                tmpData = prepare_lines_input(  lines_h, 
                                                lines_v, 
                                                LineDetectCfg['max_num_lines'], 
                                                [bbx_x1,bbx_y1,bbx_x2,bbx_y2], 
                                                img_w, 
                                                img_h)                

                if np.random.random() > train_test_split:
                    test_data += [tmpData]
                else:
                    train_data += [tmpData]

                if plot_flag:
                    test_score(net, tmpData)

                    if lines_h.shape[0]:
                        print('lines_h shape: ',lines_h.shape)
                        mylines = lines_h.copy()

                        max_x_idx = np.argsort(lines_h[:,0,2]-lines_h[:,0,0])[:]
                        for i, line in enumerate(lines_h):
                            for x1, y1, x2, y2 in line:
                                if i in max_x_idx:
                                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    if lines_v.shape[0]:
                        print('lines_v shape: ',lines_v.shape)
                        mylines = lines_v.copy()

                        max_y_idx = np.argsort(lines_v[:,0,3]-lines_v[:,0,1])[:]
                        for i, line in enumerate(lines_v):
                            for x1, y1, x2, y2 in line:
                                if i in max_y_idx:
                                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    cv2.line(line_image, (bbx_x1, bbx_y1), (bbx_x2, bbx_y1), (0, 0, 255), 3)
                    cv2.line(line_image, (bbx_x1, bbx_y1), (bbx_x1, bbx_y2), (0, 0, 255), 3)
                    cv2.line(line_image, (bbx_x1, bbx_y2), (bbx_x2, bbx_y2), (0, 0, 255), 3)
                    cv2.line(line_image, (bbx_x2, bbx_y1), (bbx_x2, bbx_y2), (0, 0, 255), 3)

                    #cv2.line(line_image, (crp_x, crp_y), (crp_x+crp_w-1, crp_y), (0, 255, 255), 3)
                    #cv2.line(line_image, (crp_x, crp_y), (crp_x, crp_y+crp_h-1), (0, 255, 255), 3)
                    #cv2.line(line_image, (crp_x, crp_y+crp_h-1), (crp_x+crp_w-1, crp_y+crp_h-1), (0, 255, 255), 3)
                    #cv2.line(line_image, (crp_x+crp_w-1, crp_y), (crp_x+crp_w-1, crp_y+crp_h-1), (0, 255, 255), 3)

                    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

                    cv2.imshow("res",lines_edges)
                    if cv2.waitKey(0)==']':
                        exit(1)

                    #break

    return np.array(train_data), np.array(test_data)

print('Processing test data ...')
train_test, test_test = processing(test_list, True)
exit(1)

print('Processing negative data ...')
train_neg, test_neg = processing(neg_list, False)

print('Processing positive data ...')
train_pos, test_pos = processing(pos_list, False)


with open(join(rst_dir, 'train_pos.pkl'), 'wb') as f:
        pickle.dump(train_pos, f, protocol = pickle.HIGHEST_PROTOCOL)
with open(join(rst_dir, 'test_pos.pkl'), 'wb') as f:
        pickle.dump(test_pos, f, protocol = pickle.HIGHEST_PROTOCOL) 
with open(join(rst_dir, 'train_neg.pkl'), 'wb') as f:
        pickle.dump(train_neg, f, protocol = pickle.HIGHEST_PROTOCOL) 
with open(join(rst_dir, 'test_neg.pkl'), 'wb') as f:
        pickle.dump(test_neg, f, protocol = pickle.HIGHEST_PROTOCOL) 

'''
#f_path = '/home/macul/Projects/spoofing/data/iim_real_jul_2018'
#f_path = '/home/macul/Projects/spoofing/data/iim_image_attack_1'
f_path = '/media/macul/black/spoof_db/collected/original/screen_attack/tablet_small'

new_img_w = 200

for f in listdir(f_path):
    print(f)

    img = cv2.imread(join(f_path,f))
    #img = cv2.imread('/home/macul/Projects/spoofing/data/iim_real_jul_2018/0f73b690aad04a679cebdb6c49ebfad2_2018_04_04_09_50_41_0.jpg')
    #img = cv2.imread('/home/macul/Projects/spoofing/data/iim_image_attack_1/1a365a023f2499aa2ffbf12828f4_2018_05_09_14_12_52_0.jpg')
    #img = cv2.imread('/home/macul/Projects/spoofing/data/nuaa_real/0001_00_00_01_0.jpg')
    img_h, img_w, _ = img.shape
    new_img_h = int(1.0*new_img_w/img_w*img_h)
    img = cv2.resize(img,(new_img_w,new_img_h))

    bbox_h = int(img_h/2.2*new_img_h/img_h)
    bbox_w = int(img_w/2.2*new_img_w/img_w)
    bbox_x1 = int((new_img_w-bbox_w)/2.0)
    bbox_x2 = bbox_x1 + bbox_w - 1
    bbox_y1 = int((new_img_h-bbox_h)/2.0)
    bbox_y2 = bbox_y1 + bbox_h - 1

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    #blur_gray = gray

    low_threshold = 0
    high_threshold = 100
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 25  # minimum number of pixels making up a line
    max_line_gap = 4  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    #a = HoughBundler()
    #lines1 = a.process_lines(lines, img)

    #lines = np.array(lines1).reshape([-1,1,4])
    if type(lines) != type(None):
        print('lines shape: ',lines.shape)
        mylines = lines.copy()

        points = []

        max_x_idx = np.argsort(lines[:,0,2]-lines[:,0,0])[:]
        max_y_idx = np.argsort(lines[:,0,3]-lines[:,0,1])[:]
        for i, line in enumerate(lines):
            for x1, y1, x2, y2 in line:
                points.append(((x1 + 0.0, y1 + 0.0), (x2 + 0.0, y2 + 0.0)))
                if i in max_x_idx:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                if i in max_y_idx:
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.line(line_image, (bbox_x1, bbox_y1), (bbox_x2, bbox_y1), (0, 0, 255), 3)
    cv2.line(line_image, (bbox_x1, bbox_y1), (bbox_x1, bbox_y2), (0, 0, 255), 3)
    cv2.line(line_image, (bbox_x1, bbox_y2), (bbox_x2, bbox_y2), (0, 0, 255), 3)
    cv2.line(line_image, (bbox_x2, bbox_y1), (bbox_x2, bbox_y2), (0, 0, 255), 3)

    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    print(lines_edges.shape)
    #cv2.imwrite('line_parking.png', lines_edges)

    
    #print points
    #intersections = bot.isect_segments(points)
    #print intersections

    #for inter in intersections:
    #    a, b = inter
    #    for i in range(3):
    #        for j in range(3):
    #            lines_edges[int(b) + i, int(a) + j] = [0, 255, 0]
    
    cv2.imwrite('result.png', lines_edges)

    cv2.imshow("res",lines_edges)
    if cv2.waitKey(0)==']':
        exit(1)
'''