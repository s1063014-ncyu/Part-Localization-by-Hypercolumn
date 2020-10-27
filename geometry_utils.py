#from fast_bird_part_localization import poisson_disk
import numpy as np
import cv2
from PIL import Image
from poisson_disk import PoissonDiskSampler
import skimage.morphology
import skimage.measure
import scipy.stats
import matplotlib.pyplot as plt

class Box(object):
    """
    This class represents a box in an image. This could be a bounding box of an object or part.
    Internally each box is represented by a tuple of 4 integers: (xmin, xmax, ymin, ymax)
    """

    POINT_GENERATION_POLECIES = ['poisson_disk']

    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def __repr__(self):
        return "%d - %d - %d - %d" % (self.xmin, self.xmax, self.ymin, self.ymax)

    def is_valid(self):
        return int(self.xmin) != -1

    @staticmethod
    def box_from_img(img):
        """
        Creats a box from the image
        """
        
        height, width,domain = img.shape
        return Box(0, height, 0, width)

    @staticmethod
    def box_from_cendim(cen, dim):
        """
        Create a box from a pair of center and dimension. Each center or dimension is a tuple. For short we call the center and dimension the `cendim`
        Center: (cenX, cenY)
        Dimension: (height, width)
        """
        cenX, cenY = cen
        height, width = dim
        height_2 = height / 2.
        width_2 = width / 2.
        xmin = int(round(cenX - height_2))
        xmax = int(round(cenX + height_2))
        ymin = int(round(cenY - width_2))
        ymax = int(round(cenY + width_2))
        return Box(xmin, xmax, ymin, ymax)

    def cendim(self):
        """
        Convert the box into cendim format. In cendim format the center and dimension are stored as floating point numbers.
        """
        cenX = float((self.xmin + self.xmax) / 2)
        cenY = float((self.ymin + self.ymax) / 2)
        height = float(self.xmax - self.xmin)
        width = float(self.ymax - self.ymin)

        cen = (cenX, cenY)
        dim = (height, width)
        return cen, dim

    def trim_to_borders(self, img_shape):
        """
        Trims the box with respect to the image provided.
        """
        img_h, img_w,domain = img_shape.shape
        if(self.xmin<0):
            self.xmin=0
            
        print(self.xmax)
        if (self.xmax>img_h):
            self.xmax=(img_h - 1)
            
        if(self.ymin<0):
            self.ymin=0
            
        if (self.ymax>img_w):
            self.ymax=(img_w - 1)

        return self
    '''
    def draw_box(self, img, color=(255, 0, 0), width=2):
        """
        Annotate the `img` with this Box. This returns a new image with the box annotated on it.
        """
        
        new_img = img.copy()
        
        cv2.rectangle(new_img, (self.ymin, self.xmin), (self.ymax, self.xmax), color, width)
        #cv2.rectangle(new_img, (self.ymin, self.xmin), (self.ymax, self.xmax), color, width)
        return new_img
    '''
    
    def draw_box(self, img, color=(255, 0, 0), width=4):
        """
        Annotate the `img` with this Box. This returns a new image with the box annotated on it.
        """
        sheight, swidth, sdomain = img.shape
        new_img = img.copy()
        
        feature_size=50
        
        cv2.rectangle(new_img, (int(self.ymin/feature_size*swidth), int(self.xmin/feature_size*sheight) ), ( int(self.ymax/feature_size*swidth), int(self.xmax/feature_size*sheight)), color, width)
        #cv2.rectangle(new_img, (self.ymin, self.xmin), (self.ymax, self.xmax), color, width)
        return new_img
    
    def get_sub_image(self, img):
        """
        Return a sub-image only containing information inside this Box.
        """
        self.trim_to_borders(img.shape)

        return img[self.xmin:self.xmax, self.ymin:self.ymax]

    @staticmethod
    def expand_cendim(cen, dim, alpha):
        height, width = dim

        height = (2 * alpha) * height
        width = (2 * alpha) * width

        dim = (height, width)
        return cen, dim

    def expand(self, alpha=0.666):
        cen, dim = self.cendim()
        cen, dim = Box.expand_cendim(cen, dim, alpha)
        new_box = Box.box_from_cendim(cen, dim)
        self.xmin = new_box.xmin
        self.xmax = new_box.xmax
        self.ymin = new_box.ymin
        self.ymax = new_box.ymax

        return self
    
    def filter_points(neg_points , box, img):
        new_nge_points_ind = []
        for i, p in enumerate(neg_points):
            #print(p[1],p[0])
            #print(img.shape)
            if ((box.xmin <= p[1] <= box.xmax) and (box.ymin <= p[0] <= box.ymax) and (img[p[1],p[0]]!=0) ):
                continue
            else:
                new_nge_points_ind.append(i)
                
        return neg_points[new_nge_points_ind, :]
    
    def filter_points_negtive(neg_points, pos_points, img):
        new_pos_points_ind = []
        new_neg_points_ind = []
        for i, p in enumerate(pos_points):
            if img[p[1],p[0]]==0:
                new_neg_points_ind.append(p)
            else:
                new_pos_points_ind.append(p)
                
        
        for i, p in enumerate(neg_points):
            new_neg_points_ind.append(p)
                
        return np.asarray(new_neg_points_ind), np.asarray(new_pos_points_ind)
    
    def filter_points_positive(points, img):
        new_points_ind = []
        for i, p in enumerate(points):
            if img[p[1],p[0]]!=0:
                continue
            else:
                new_points_ind.append(i)
                
        return points[new_points_ind, :]
    
    def evalIOU(self, gt_box, source_shape):
        # TODO
        # making sure not to generate errors further down the line
        self.trim_to_borders(source_shape)
        gt_box.trim_to_borders(source_shape)

        height, width,domain = source_shape.shape

        gt_part = np.zeros((height, width), np.uint8)
        gt_part[gt_box.xmin:gt_box.xmax, gt_box.ymin:gt_box.ymax] = 1

        sl_part = np.zeros((height, width), np.uint8)
        sl_part[self.xmin:self.xmax, self.ymin:self.ymax] = 1

        intersection = (gt_part & sl_part).sum()
        union = (gt_part | sl_part).sum()

        return intersection / float(union)

    def evalPCP(self, gt_box, source_shape, thresh=0.5):
        iou = self.evalIOU(gt_box, source_shape)
        print('IOU:',iou)
        if iou >= thresh:
            return 1
        else:
            return 0

    def generate_points_inside(self, policy='poisson_disk', param=None, img=None):
        """
       This function generates points inside this rectangle. It uses the poisson disk to do it by default. But there is a policy option that is configurable.
       There is an optional `param` parameter that specifies the parameters of the generation policy.

                                Different Policies:
                                    - `poisson_disk`:
                                        The param is expected to be the radius. The radius is the parameter of the poisson disk sampler.
                                        By default radius is set to be average of 1/10 of width and height of the box.

                                Each point is a row vector [x, y]. A set of `n` points will be represented as a numpy array of shape (n,2). The dtype is numpy.int.

                                There can be an optional img option. We can use the image's shape to further prune points that are located outside the boundary of the image.
        """
        assert(policy in self.POINT_GENERATION_POLECIES)
        cen, dim = self.cendim()
        height, width = dim
        if policy == 'poisson_disk':
            if param is None:
                radius = ((height / 10.) + (width / 10.)) / 2.
            else:
                radius = param
            # please note that PoissonDiskSampler does use a flipped version of the axis
            # also the algorithm generates points in the range [0, height] but we want [0, height) that is
            # the reason behind the "-1".
            pds = PoissonDiskSampler(width - 1, height - 1, radius)
            samples = pds.get_sample()
            points = np.zeros((len(samples), 2), dtype=np.int)
            for i, s in enumerate(samples):
                points[i, :] = [int(round(s[0]))-1, int(round(s[1]))-1]

            points += np.array([self.ymin, self.xmin])

        return points


    def draw_points(points, ax, color=None):
        if color is None:
            color = 'red'
        for p in points:
            # Notice that in plt the axis are different from what we work with
            # namely in plt the horizontal axis is x and vertical axis is y
            # whereas in numpy and images that we work with the vertical axis is x
            # this is the reason behind the flipping of points here.
            ax.plot(p[1], p[0], 'o', color=color)


    


    def post_process_preds(preds):
        preds = skimage.morphology.closing(preds, skimage.morphology.square(1)) #透過縮放
        preds = skimage.morphology.remove_small_objects(preds, min_size=10, connectivity=2)#移除小的區域
        return preds


    def find_rect_from_preds(preds,img):
        L, N = skimage.measure.label(preds, return_num=True, background=0) #連通區域標記(同一個連通區域的所有像素點，都用同一個數值來進行標記)
        if N > 0:
            L_no_bg = L[L != 0].flatten()
            vals, counts = scipy.stats.mode(L_no_bg)#找出次數最多的區域
            part_label = int(vals[0])
            
            '''
            print(part_label)
            print(N)
            i=0
            sheight, swidth, sdomain = img.shape
            while i<N:
                i=i+1
                
                indices = np.where(L == i)
                xmin = indices[0].min()
                xmax = indices[0].max()
                ymin = indices[1].min()
                ymax = indices[1].max()
                #print(len(indices))
                #print(indices)
                size = 50
                if len(indices[0])>20:
                    cv2.rectangle(img, (int(ymin/size*swidth), int(xmin/size*sheight) ), ( int(ymax/size*swidth), int(xmax/size*sheight)), (255,0,0), 4)  
                    
            fig = plt.figure(figsize=(10, 20))
            ax1 = fig.add_subplot(211)
            ax1.imshow(img)
            plt.show()
            print('Picture Out')
            '''
            
            indices = np.where(L == part_label)
            
            xmin = indices[0].min()
            xmax = indices[0].max()
            ymin = indices[1].min()
            ymax = indices[1].max()

            return Box(xmin, xmax, ymin, ymax)
        else:
            return Box(-1, -1, -1, -1)
