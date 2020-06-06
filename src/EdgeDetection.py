from PIL import Image
import math

def grayscale(r,g,b):
    return ( (0.3 * r) + (0.59 * g) + (0.11 * b) )

class EdgeDetector:
    '''
    A class to handle edge detection of images.
    '''
    SOBEL_KX = [[-1,0,1],
                [-2,0,2],
                [-1,0,1]]

    SOBEL_KY = [[-1,-2,-1],
                [ 0, 0, 0],
                [ 1, 2, 1]]

    NO_EDGE     = 0
    WEAK_EDGE   = 1
    STRONG_EDGE = 2

    ALPHA_IGNORE_LIMIT = 100

    def __init__(self,img:Image,*,GAUSSIAN_KERNEL_SIGMA:float=1.4,GAUSSIAN_KERNEL_SIZE:int=5):
        '''
        Create a new Edge Detector using the given Image.

            Parameters:
                img : PIL.Image
                    The initial image to detect edges on.

                GAUSSIAN_KERNEL_SIGMA : float = 1.4, positional
                    The Sigma to use in Gaussian Filter Kernel creation. This can affect Edge Detection.
                
                GAUSSIAN_KERNEL_SIZE : int = 5, positional
                    The Kernel Size to use in Gaussian Filter Kernel creation. This can affect Edge Detection.

            Returns:
                EdgeDetector : EdgeDetector
                    An instance of the EdgeDetector class
        '''

        if not GAUSSIAN_KERNEL_SIGMA is float:
            try:
                GAUSSIAN_KERNEL_SIGMA = float(GAUSSIAN_KERNEL_SIGMA)
            except:
                raise TypeError('GAUSSIAN_KERNEL_SIGMA should be type float, not %' % type(GAUSSIAN_KERNEL_SIGMA))
        if not GAUSSIAN_KERNEL_SIZE is int:
            try:
                GAUSSIAN_KERNEL_SIZE = int(GAUSSIAN_KERNEL_SIZE)
            except:
                raise TypeError('GAUSSIAN_KERNEL_SIZE should be type int, not %' % type(GAUSSIAN_KERNEL_SIZE))


        self.sigma = GAUSSIAN_KERNEL_SIGMA
        self.ksize = GAUSSIAN_KERNEL_SIZE

        self.load_new_image(img)

    def load_new_image(self,img:Image):
        '''
        Loads a new image into the Edge Detector, and does all neccessary preprocessing to said image.

            Parameters:
                img : PIL.Image
                    The new image to load
    
            Returns:
                None
        '''
        img = img.convert('RGBA')
        size = img.size
        self.w = size[0]
        self.h = size[1]

        self.__matrix = [ [{'v': 0, 'a': 255} for _ in range(self.w)] for _ in range(self.h) ]

        for y in range(self.h):
            for x in range(self.w):
                px = img.getpixel((x,y))
                self.__matrix[y][x]['v'] = min(255,round(grayscale(px[0],px[1],px[2])))
                self.__matrix[y][x]['a'] = px[3]
        

        gaussian_matrix = self.__gaussian_filter_matrix()

        self.__matrix = self.__convolute(gaussian_matrix)
        self.__matrix = self.__sobel()
        self.__suppress()
        self.__calculate_iqr()

    def __calculate_iqr(self):
        value_list = []
        for row in self.matrix:
            for value in row:
                if value['a'] > EdgeDetector.ALPHA_IGNORE_LIMIT:
                    value_list.append(value['g'])
        value_list = sorted(value_list)
        median_index = (len(value_list)/2)
        if round(median_index) == median_index:
            self.__median_value = value_list[median_index-1]
        else:
            self.__median_value = (value_list[math.floor(median_index)-1] + value_list[math.ceil(median_index)-1])/2
        
        q1_index = (median_index) / 2
        if round(q1_index) == q1_index:
            self.__q1 = value_list[q1_index-1]
        else:
            self.__q1 = (value_list[math.floor(q1_index)-1] + value_list[math.ceil(q1_index)-1])/2

        q3_index = len(value_list) * .75
        if round(q3_index) == q3_index:
            self.__q3 = value_list[q3_index-1]
        else:
            self.__q3 = (value_list[math.floor(q3_index)-1] + value_list[math.ceil(q3_index)-1])/2

    @property
    def median_value(self):
        return self.__median_value

    @property
    def q1(self):
        return self.__q1
    
    @property
    def q3(self):
        return self.__q3

    @property
    def iqr(self):
        return self.__q3 - self.__q1

    @property
    def low_outlier(self):
        return min(0,self.__q1 - (1.5*self.iqr))

    @property
    def high_outlier(self):
        return max(255,self.__q3 + (1.5*self.iqr))

    def __gaussian_filter_matrix(self):
        
        # computations that dont need to be done more than once

        sigma_squared = self.sigma ** 2
        double_sigma_squared = sigma_squared * 2
        fraction = 1/(math.pi*double_sigma_squared)
        k = ((self.ksize - 1)/2)+1 # we only use k as (k+1) so we can just add one to k here

        def H(i,j):
            a = (i-k)**2
            b = (j-k)**2
            exp = 0 - ((a+b)/double_sigma_squared)
            return fraction*math.exp(exp)
        
        gaus = [ [0 for _ in range(self.ksize)] for _ in range(self.ksize) ]
        for j,row in enumerate(gaus):
            for i in range(len(row)):
                row[i] = H(i+1,j+1)
        return gaus

    def __convolute_at_point(self,x,y,kernel,normalize=True):
        v = 0
        normalizer = 0
        kernal_size = len(kernel)
        half_kernal_size = round(kernal_size/2)

        for j in range(kernal_size):
            nY = y - j + half_kernal_size
            if nY >= 0 and nY < self.h:
                for i in range(kernal_size):
                    nX = x - i + half_kernal_size
                    if nX >= 0 and nX < self.w:
                        kern = kernel[j][i]
                        v += kern*self.__matrix[nY][nX]['v']
                        normalizer += kern
        
        if not normalize:
            normalizer = 1
        
        return {'v': math.floor(v/normalizer), 'a': self.__matrix[y][x]['a']}

    def __convolute(self,kernel,normalize=True):
        new_matrix = [ [{'v': 0, 'a': 255} for _ in range(self.w)] for _ in range(self.h) ]

        for y in range(self.h):
            for x in range(self.w):
                new_matrix[y][x] = self.__convolute_at_point(x,y,kernel,normalize=normalize)
        
        return new_matrix

    def __sobel(self):
        sobel_result = [ [{'g': 0, 'theta': 0, 'a': 255} for _ in range(self.w)] for _ in range(self.h) ]

        for y in range(self.h):
            for x in range(self.w):
                gx = self.__convolute_at_point(x,y,EdgeDetector.SOBEL_KX,normalize=False)
                gy = self.__convolute_at_point(x,y,EdgeDetector.SOBEL_KY,normalize=False)
                sobel_result[y][x]['g'] = math.hypot(gx['v'],gy['v'])
                sobel_result[y][x]['theta'] = math.atan2(gy['v'],gx['v']) * (180/math.pi)
                sobel_result[y][x]['a'] = gx['a']

        return sobel_result

    def __suppress(self):

        x = 0
        y = 0
        theta = 0
        g = 0

        def determine_g(dx1,dy1,dx2,dy2):
            isMax = True
            if(x + dx1 >= 0 and x + dx1 < self.w and y + dy1 >= 0 and y + dy1 < self.h):
                isMax = isMax and (self.__matrix[y+dy1][x+dx1]['g'] < g)
            if(x + dx2 >= 0 and x + dx2 < self.w and y + dy2 >= 0 and y + dy2 < self.h):
                isMax = isMax and (self.__matrix[y+dy2][x+dx2]['g'] < g)

            if not isMax:
                self.__matrix[y][x]['g'] = 0
        
        for y in range(self.h):
            for x in range(self.w):
                center = self.__matrix[y][x]
                theta = center['theta']
                g = center['g']

                if(g > 0):
                    if(theta > 157.5 or theta <= 22.5):
                        determine_g(-1,0,1,0)
                    elif(theta > 22.5 and theta <= 67.5):
                        determine_g(1,1,-1,-1)
                    elif(theta > 67.5 and theta <= 112.5):
                        determine_g(0,-1,0,1)
                    else:
                        determine_g(-1,1,1,-1)
                
    def __double_threshold(self,no_edge_threshold,strong_edge_threshold):
        classifications = [ [EdgeDetector.NO_EDGE for _ in range(self.w)] for _ in range(self.h) ]
        strongs = []
        g = 0

        def connect_weak_edges(x,y):
            for j in (-1,0,1):
                if y+j >= 0 and y+j < self.h:
                    for i in (-1,0,1):
                        if x+i >= 0 and x+i < self.w and not (i == 0 and j == 0):
                            if classifications[y+j][x+i] == EdgeDetector.WEAK_EDGE:
                                classifications[y+j][x+i] = EdgeDetector.STRONG_EDGE
                                connect_weak_edges(x+i,y+j)
        
        for y in range(self.h):
            for x in range(self.w):
                if self.__matrix[y][x]['a'] < EdgeDetector.ALPHA_IGNORE_LIMIT:
                    classifications[y][x] = EdgeDetector.NO_EDGE
                else:
                    g = self.__matrix[y][x]['g']
                    if g < no_edge_threshold:
                        classifications[y][x] = EdgeDetector.NO_EDGE
                    elif g >= strong_edge_threshold:
                        classifications[y][x] = EdgeDetector.STRONG_EDGE
                        strongs.append((x,y))
                    else:
                        classifications[y][x] = EdgeDetector.WEAK_EDGE

        for strong_edge in strongs:
            connect_weak_edges(strong_edge[0],strong_edge[1])

        return classifications

    def detect_edges(self,no_edge_threshold:int,strong_edge_threshold:int) -> list:
        """
        Detects edges using the class's matrix and the given thresholds.

            Parameters:
                no_edge_threshold : int
                    The upper bound of what is considered a 'no edge'

                strong_edge_threshold : int 
                    The lower bound of what is considered a 'strong edge'

            Returns:
                edge_matrix : list
                    Matrix of the same width and height of the parent image, with boolean True if the cell belongs to an edge and boolean False otherwise.
        """

        if not no_edge_threshold is int:
            try:
                no_edge_threshold = int(no_edge_threshold)
            except:
                raise TypeError('no_edge_threshold should be type int, not %' % type(no_edge_threshold))
        if not strong_edge_threshold is int:
            try:
                strong_edge_threshold = int(strong_edge_threshold)
            except:
                raise TypeError('strong_edge_threshold should be type int, not %' % type(strong_edge_threshold))
        if no_edge_threshold < 0 or no_edge_threshold > 255:
            raise ValueError('no_edge_threshold should be 0 <= no_edge_threshold <= 255, not %' % no_edge_threshold)
        if strong_edge_threshold < 0 or strong_edge_threshold > 255:
            raise ValueError('strong_edge_threshold should be 0 <= strong_edge_threshold <= 255, not %' % strong_edge_threshold)
        if strong_edge_threshold < no_edge_threshold:
            raise ValueError('strong_edge_threshold should be larger than no_edge_threshold')

        return [ [ cell == EdgeDetector.STRONG_EDGE for cell in row ] for row in self.__double_threshold(no_edge_threshold,strong_edge_threshold) ]