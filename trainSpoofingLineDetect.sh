#!/usr/bin/env sh

# testcase 1
#python -m /home/macul/Projects/spoofing/trainMySqueezeNet 3
#rm my_log_1.log
#caffe train -solver /media/macul/black/MK/Projects/spoofing_ld/mySolverSpoofingLineDetect.prototxt -gpu 1 2>&1 | tee -a my_log_1.log

# testcase 1
#python -m /home/macul/Projects/spoofing/trainMySqueezeNet 3
#rm my_log_2.log
caffe train -solver /media/macul/black/MK/Projects/spoofing_ld/mySolverSpoofingLineDetect.prototxt -weights /media/macul/black/MK/Projects/spoofing_ld/snapshot_line_detect_1/mySolverSpoofingLineDetect_iter_1000000.caffemodel -gpu 1 2>&1 | tee -a my_log_2.log
#resume training
#caffe train -solver /media/macul/black/MK/Projects/online_classify/mySolver.prototxt -snapshot /media/macul/black/MK/Projects/online_classify/snapshot_1/mySolver_iter_20000.solverstate -gpu 1 2>&1 | tee -a my_log_1.log

