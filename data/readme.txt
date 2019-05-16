pos_list = [    '/media/macul/black/spoof_db/casia/real',
                '/media/macul/black/spoof_db/iim/iim_real_jul_2018',
                '/media/macul/black/spoof_db/iim/toukong_real_jul_2018',
                '/media/macul/black/spoof_db/NUAA/ClientRaw']
neg_list = [    '/media/macul/black/spoof_db/collected/original/screen_attack/tablet_small',
                '/media/macul/black/spoof_db/casia/image_attack',
                '/media/macul/black/spoof_db/casia/screen_attack',
                '/media/macul/black/spoof_db/collected/original/image_attack/printed',
                '/media/macul/black/spoof_db/collected/original/screen_attack/tablet',              
                '/media/macul/black/spoof_db/iim/iim_image_attack',
                '/media/macul/black/spoof_db/NUAA/ImposterRaw']

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
