import sklearn.externals
import hypercolumn
import geometry_utils
import matplotlib.pylab as plt
import vis_utils
import joblib
import cv2
from datetime import datetime as dt
from skimage.segmentation import slic,mark_boundaries
from skimage import color
import numpy as np
class Detector(object):
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.dh = hypercolumn.hypercolumn()

    def detect(self, img):
        """
        A little bit inefficient if one just needs the rectangle and not the probability image.
        """
        start = dt.now()
        self.dh.do_hypercolumn(img)
        
        print ('finsihed hypercolumn in', (dt.now() - start))
        Xtest = self.dh.hcf.reshape(self.dh.hcf.shape[0] * self.dh.hcf.shape[1], self.dh.hcf.shape[2])
        preds = self.model.predict(Xtest)
        print ('finsihed predict in', (dt.now() - start))
        preds_prob = self.model.predict_proba(Xtest)
        print ('finsihed predict_prob in', (dt.now() - start))
        '''
        pos = []
        for i in preds:
            if i <3:
                pos.append(0)
            else:
                pos.append(1)
        np.asarray(pos)
        '''
        preds_img = preds.reshape(self.dh.hcf.shape[0], self.dh.hcf.shape[1])
        
        preds_prob_image = preds_prob[:, 1].reshape(self.dh.hcf.shape[0], self.dh.hcf.shape[1])
        
        '''
        start = dt.now()
        img2 = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
        segments = slic(img2, n_segments=300, compactness=10)
        print ('分割時間:', (dt.now() - start))
        start = dt.now()
        preds_prob_image = color.label2rgb(segments, preds_prob_image, kind='avg')
        print ('塗色時間:', (dt.now() - start))
        
        plt.imshow(mark_boundaries(preds_prob_image,segments))
        plt.show()
        '''
        
        pred_box = geometry_utils.Box.find_rect_from_preds(geometry_utils.Box.post_process_preds(preds_img),img)
        print ('finsihed in', (dt.now() - start))
        return pred_box, preds_prob_image
        
    def draw(self, img, part_detected, part_probability):
        fig = plt.figure(figsize=(10, 20))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        ax1.imshow(part_detected.draw_box(img, color=(255, 0, 0)))
        vis_utils.vis(part_probability, img, ax=ax2, fig=fig)
        plt.show()
