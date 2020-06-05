from PIL import Image
import math

def grayscale(r,g,b):
    return ( (0.3 * r) + (0.59 * g) + (0.11 * b) )

class EdgeDetector:
    SOBEL_KX = [[-1,0,1],
                [-2,0,2],
                [-1,0,1]]

    SOBEL_KY = [[-1,-2,-1],
                [ 0, 0, 0],
                [ 1, 2, 1]]

    NO_EDGE     = 0
    WEAK_EDGE   = 1
    STRONG_EDGE = 2

    def __init__(self,img:Image,*,GAUSSIAN_KERNAL_SIGMA:float=1.4,GAUSSIAN_KERNAL_SIZE:int=5):

        self.sigma = GAUSSIAN_KERNAL_SIGMA
        self.ksize = GAUSSIAN_KERNAL_SIZE

        img = img.convert('RGBA')
        size = img.size
        self.w = size[0]
        self.h = size[1]

        self.matrix = [ [{'v': 0, 'a': 255} for _ in range(self.w)] for _ in range(self.h) ]

        for y in range(self.h):
            for x in range(self.w):
                px = img.getpixel((x,y))
                self.matrix[y][x]['v'] = min(255,round(grayscale(px[0],px[1],px[2])))
                self.matrix[y][x]['a'] = px[3]

        gaussian_matrix = self.__gaussian_filter_matrix()

        self.matrix = self.__convolute(gaussian_matrix)
        self.matrix = self.__sobel()
        self.__suppress()

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
                        v += kern*self.matrix[nY][nX]['v']
                        normalizer += kern
        
        if not normalize:
            normalizer = 1
        
        return {'v': math.floor(v/normalizer), 'a': self.matrix[y][x]['a']}

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
                isMax = isMax and (self.matrix[y+dy1][x+dx1]['g'] < g)
            if(x + dx2 >= 0 and x + dx2 < self.w and y + dy2 >= 0 and y + dy2 < self.h):
                isMax = isMax and (self.matrix[y+dy2][x+dx2]['g'] < g)

            if not isMax:
                self.matrix[y][x]['g'] = 0
        
        for y in range(self.h):
            for x in range(self.w):
                center = self.matrix[y][x]
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
                if self.matrix[y][x]['a'] < 128:
                    classifications[y][x] = EdgeDetector.NO_EDGE
                else:
                    g = self.matrix[y][x]['g']
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

    def detect_edges(self,lower,upper):
        return self.__double_threshold(lower,upper)