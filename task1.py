import pickle
from matplotlib import pyplot as plt
import cv2
import numpy as np
import time

def get_x(angle):
    return np.array([np.math.cos(angle), np.math.sin(angle)])
    
def get_y(angle):
    return np.array([-np.math.sin(angle), np.math.cos(angle)])
    
def get_corners(rect):
    angle = rect[4]*np.math.pi/180
    vx = np.array([np.math.cos(angle), np.math.sin(angle)])
    vy = np.array([-np.math.sin(angle), np.math.cos(angle)])

    diff = np.array([[-0.5, -0.5],
                    [0.5, -0.5],
                    [0.5, 0.5],
                    [-0.5, 0.5]])

    
    M = np.array([vx*rect[2], vy*rect[3]])
    corners = np.array(rect[:2]) + np.matmul(diff, M)
    return corners

def get_projection(v, corner):
    return v[0]*corner[0] + v[1]*corner[1]

def is_collide(rect1, rect2):
    """check if both rectangle collides or not

    Args:
        rect1 (5 element tuple): (x,y, w, h, angle in degree)
        rect2 (5 element tuple): (x,y, w, h, angle in degree)
    
    Returns:
        bool : True if collides
    """
    vx1 = get_x(rect1[4]*np.math.pi/180)
    vy1 = get_y(rect1[4]*np.math.pi/180)
    vx2 = get_x(rect2[4]*np.math.pi/180)
    vy2 = get_y(rect2[4]*np.math.pi/180)

    corners1 = get_corners(rect1)
    corners2 = get_corners(rect2)

    for v in [vx1, vy1, vx2, vy2]:
        projections1 = [get_projection(v, corner) for corner in corners1]
        min_1 = min(projections1)
        max_1 = max(projections1)
        projections2 = [get_projection(v, corner) for corner in corners2]
        min_2 = min(projections2)
        max_2 = max(projections2)
        if min_1 >= max_2 or min_2 >= max_1:
            return False
    return True

def is_collide_cv2(rect1, rect2):
    rect1 = ((rect1[0],rect1[1]),(rect1[2], rect1[3]),rect1[4])
    rect2 = ((rect2[0],rect2[1]),(rect2[2], rect2[3]),rect2[4])
    return cv2.rotatedRectangleIntersection(tuple(rect1), tuple(rect2))
    
def test_get_corners():
    rect = [0,0, 10, 4, 90]
    corners = get_corners(rect)
    plt.plot(corners[:,0], corners[:,1]), plt.grid(True)
    plt.show()
    print(corners)


def test_collision():
    """Test collision function
    """
    rect1 = [0,0, 10, 4, 90]
    rect2 = [3, 5, 3 ,10, 10]
    start_time = time.time()
    for k in range(1000):
        status = is_collide(rect1, rect2)
    end_time = time.time()
    time1 = end_time- start_time
    # start_time = time.time()
    # for k in range(1000):
    #     status = is_collide_cv2(rect1, rect2)
    # end_time = time.time()
    time2 = end_time- start_time
    print(time1, time2)
    corners1 = get_corners(rect1)
    corners2 = get_corners(rect2)
    plt.plot(corners1[:,0], corners1[:,1],'b'), plt.grid(True)
    plt.title(status)
    plt.plot(corners2[:,0], corners2[:,1], 'r'), plt.show()
    
    
if __name__ == "__main__":
    test_collision()
