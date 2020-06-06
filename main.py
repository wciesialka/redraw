import argparse
from PIL import Image
import src.EdgeDetection as EdgeDetection
import pyautogui
import time

pyautogui.PAUSE = 0

USER_ABORTED = False

def squash(matrix,scale=2):
    squashed = []
    h = len(matrix)
    for y in range(0,h,scale):
        row = matrix[y]
        w = len(row)

        new_row = []

        for x in range(0,w,scale):
            total = 0
            true = 0
            for j in range(scale):
                for i in range(scale):
                    nX = x + i
                    nY = y + j
                    if nX >= 0 and nY >= 0 and nX < w and nY < h:
                        total += 1
                        if matrix[nY][nX]:
                            true += 1
            new_row.append(true >= total // 2)

        squashed.append(new_row)
    return squashed

def find_horizontal_lines(matrix):
    lines = []
    start = None
    for y,row in enumerate(matrix):
        w = len(row)
        for x,v in enumerate(row):
            if v:
                if start == None:
                    start = (x,y)
                if x == (w-1):
                    lines.append((start,(x,y)))
                    start = None                
            else:
                if start != None:
                    lines.append((start,(x,y)))
                    start = None
    return lines

def main(*,img:Image,weak:int,strong:int):

    MAX_SECONDS = 45
    STIME = time.time()

    START = (200,200)

    pyautogui.moveTo(START[0],START[1],duration=0)
    ed = EdgeDetection.EdgeDetector(img)
    
    if weak == None:
        weak = ed.q1
    if strong == None:
        strong = ed.q3
    
    edges = ed.detect_edges(weak,strong)
    squashed = squash(edges,2)



    # new = Image.new('RGB',(len(squashed),len(squashed[0])),(255,255,255))

    # for y,row in enumerate(squashed):
    #     for x,val in enumerate(row):
    #         if val:
    #             new.putpixel((x,y),(0,0,0))

    # new.show()

    # # drawing

    #lines = find_horizontal_lines(squashed)

    npx = 0
    for row in edges:
        for e in row:
            if e:
                npx += 1

    stamp = time.time()
    
    EDGE_DETECTION_DURATION = stamp - STIME
    TIME_REMAINING = MAX_SECONDS - EDGE_DETECTION_DURATION

    DURATION = TIME_REMAINING/npx

    for y,row in enumerate(edges):
        for x,val in enumerate(row):
            if val:
                pyautogui.click(START[0]+x,START[1]+y)
                time.sleep(DURATION)

    stampb = time.time()
    tdiff = (stampb - STIME)
    print("Took %d seconds" % tdiff)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Automatically redraw an Image using Python")
    parser.add_argument('-w','--weak',type=int,default=None,help="Upper bound for what is considered a NO EDGE during edge detection.")
    parser.add_argument('-s','--strong',type=int,default=None,help="Lower bound for what is considered a STRONG EDGE during edge detection.")
    parser.add_argument('image',type=argparse.FileType('rb'),help="Image to process.")
    args = parser.parse_args()
    img = Image.open(args.image)
    main(img=img,weak=args.weak,strong=args.strong)
    args.image.close()