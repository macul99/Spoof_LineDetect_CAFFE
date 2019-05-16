import cv2
import numpy as np
import math
from os import listdir
from os.path import join
import pickle
import caffe


rst_file = '/media/macul/black/MK/Projects/spoofing_ld/test_result.pkl'


pos_list = [    '/media/macul/black/MK/Projects/spoofing_ld/test_pos.pkl']
neg_list = [    '/media/macul/black/MK/Projects/spoofing_ld/test_neg.pkl']

model = '/media/macul/black/MK/Projects/spoofing_ld/clf.prototxt'
weights = '/media/macul/black/MK/Projects/spoofing_ld/snapshot_line_detect_1/mySolverSpoofingLineDetect_iter_100000.caffemodel'

caffe.set_device(0);
caffe.set_mode_gpu();


net = caffe.Net(model, weights, caffe.TEST)

rst = []
for fl in pos_list:
    with open(fl,'rb') as f:
        data = pickle.load(f)

    for d in data:
        net.blobs['clf_data'].data[0,:,0,0] = np.array(d)
        net.forward()
        score = net.blobs['clf_prob'].data[0,1]
        rst +=[[0,score]]

for fl in neg_list:
    with open(fl,'rb') as f:
        data = pickle.load(f)

    for d in data:
        net.blobs['clf_data'].data[0,:,0,0] = np.array(d)
        net.forward()
        score = net.blobs['clf_prob'].data[0,1]
        rst +=[[1,score]]

print(rst)
with open(rst_file,'wb') as f:
    pickle.dump(rst, f, protocol = pickle.HIGHEST_PROTOCOL)



from sklearn import metrics
import matplotlib.pyplot as plt

def myScore(y, pred, proba_, plot=True):
    print(zip(y,pred))
    score = {}
    score['accuracy']  = metrics.accuracy_score(y, pred)
    score['kappa']     = metrics.cohen_kappa_score(y, pred)
    score['f1']        = metrics.f1_score(y, pred)
    score['precision'] = metrics.precision_score(y, pred)
    score['recall']    = metrics.recall_score(y, pred)
    score['confusion_matrix'] = metrics.confusion_matrix(y, pred)
    score['tpr'] = 1.0*score['confusion_matrix'][1,1]/(score['confusion_matrix'][1,1]+score['confusion_matrix'][1,0])
    score['fpr'] = 1.0*score['confusion_matrix'][0,1]/(score['confusion_matrix'][0,1]+score['confusion_matrix'][0,0])
    score['roc_auc'] = metrics.roc_auc_score(y, proba_)
    roc_crv = {}
    roc_crv['fpr'], roc_crv['tpr'], roc_crv['thresholds'] = metrics.roc_curve(y, proba_)
    score['roc_crv'] = roc_crv
    print 'accuracy_score: ', score['accuracy']
    print 'kappa_score:', score['kappa']
    print 'f1_score: ', score['f1']
    print 'precision: ', score['precision']
    print 'recall: ', score['recall']
    print 'confusion_matrix: ', score['confusion_matrix']
    print 'tpr: ', score['tpr']
    print 'fpr: ', score['fpr']
    print 'roc_auc: ', score['roc_auc']
    if plot:
        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
        plt.plot(roc_crv['fpr'], roc_crv['tpr'], 'k--', label='Mean ROC (area = %0.4f)' % score['roc_auc'], lw=2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    return score

threshold = 0.5
with open(rst_file,'rb') as f:
    rst = pickle.load(f)
    rst = np.array(rst)
    rst_label = np.append(rst[:,0].reshape([-1,1]), (rst[:,1]>threshold).astype(int).reshape([-1,1]),axis=1)
    print(rst_label.shape)
    print(1.0*sum(rst_label[:,0]==rst_label[:,1])/rst_label.shape[0])

    myScore(rst_label[:,0],rst_label[:,1],rst[:,1])