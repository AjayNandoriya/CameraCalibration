
import pickle
import json
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

def get_extrinisic_parameters(objectPoints, imagePoints, cameraMatrix):
    imagePoints=[d.astype(np.float32).reshape((-1,1,2)) for d in imagePoints]
    objectPoints=[objectPoints.astype(np.float32)]*len(imagePoints)
    retval, left_cameraMatrix, left_distCoeffs, left_rvecs, left_tvecs = cv2.calibrateCamera(objectPoints=objectPoints,
                imagePoints=imagePoints,
                imageSize=(2480, 2048),
                cameraMatrix=cameraMatrix,
                distCoeffs=tuple([0,0,0,0,0]))
    return left_rvecs, left_tvecs


def transform(t,x):
    # rvec = np.array([t[0], t[1], t[2]]).reshape((3,1))
    # R,_ = cv2.Rodrigues(rvec)
    # T = np.array([t[3], t[4], t[5]],dtype=np.float32).reshape((3,1))
    # T = np.concatenate(( R, T), axis=1).astype(np.float32)
    # T = np.vstack((T, np.zeros((1, T.shape[1]), dtype=np.float32)))
    # T[3,3] = 1
    T = t.reshape((4,4))
    # T[0,:3] = T[0,:3]/ np.linalg.norm(T[0,:3])
    # T[1,:3] = T[1,:3]/ np.linalg.norm(T[1,:3])
    # T[2,:3] = T[2,:3]/ np.linalg.norm(T[2,:3])
    # T[:3,0] = T[:3,0]/ np.linalg.norm(T[:3,0])
    # T[:3,1] = T[:3,1]/ np.linalg.norm(T[:3,1])
    # T[:3,2] = T[:3,2]/ np.linalg.norm(T[:3,2])
    
     
    y_pred = np.matmul(T, x)
    return y_pred

def transform_cost(t, x, y):
    y_pred = transform(t,x)
    return np.linalg.norm(y_pred - y)

def task2():
    # read data
    with open('calibration_data.pp', 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    # find camera exterinsic parameters w.e.t object coordinates
    cameraMatrix=data['leftCameraData']['cameraMatrix']
    objectPoints=[data['objectPoints'].astype(np.float32)]*10
    left_rvecs, left_tvecs = get_extrinisic_parameters(data['objectPoints'], data['leftCameraData']['imagePoints'], data['leftCameraData']['cameraMatrix'])
    right_rvecs, right_tvecs = get_extrinisic_parameters(data['objectPoints'], data['rightCameraData']['imagePoints'], data['rightCameraData']['cameraMatrix'])
    
    # find object coordinates in camera system
    tot_error = 0
    X_left = np.zeros((len(objectPoints),4,objectPoints[0].shape[0]), dtype=np.float32)
    X_right = np.zeros((len(objectPoints),4,objectPoints[0].shape[0]), dtype=np.float32)
    left_imagePoints=[d.astype(np.float32).reshape((-1,1,2)) for d in data['leftCameraData']['imagePoints']]
    right_imagePoints=[d.astype(np.float32).reshape((-1,1,2)) for d in data['rightCameraData']['imagePoints']]
    for i in range(len(objectPoints)):

        R,_ = cv2.Rodrigues(left_rvecs[i])
        T = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = left_tvecs[i][:,0]
        X = np.ones((4,left_imagePoints[i].shape[0]))
        X [:3,:] = objectPoints[0].T
        X_left[i,:,:] = np.matmul(T, X)

        R,_ = cv2.Rodrigues(right_rvecs[i])
        T = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = right_tvecs[i][:,0]
        X = np.ones((4,right_imagePoints[i].shape[0]))
        X [:3,:] = objectPoints[0].T
        X_right [i,:,:] = np.matmul(T, X)
        
        # Xt = np.matmul(T,X)
        # Xt_proj = Xt/Xt[2]
        # U = np.matmul(data['rightCameraData']['cameraMatrix'], Xt_proj)
        # imgpoints2 = U[:2,:].T.reshape((-1,1,2)).astype(np.float32)
        # # imgpoints2, _ = cv2.projectPoints(objectPoints[i], left_rvecs[i], left_tvecs[i], left_cameraMatrix, (0,0,0,0,0))
        # error = cv2.norm(left_imagePoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        # plt.subplot(2,5, i +1)
        # plt.plot(left_imagePoints[i][:,0,0],left_imagePoints[i][:,0,1], 'b*'),
        # plt.plot(imgpoints2[:,0,0],imgpoints2[:,0,1], 'r+'),
        # tot_error += error
    print("total error: ", tot_error/len(objectPoints))
    # plt.show()

    #  estiamte Transform matrix using least mean square
    N = X_right.shape[0]*X_right.shape[2]
    # x0 = np.zeros(6).reshape(-1)
    x0 = np.eye(4).reshape(-1)
    lower_bound = np.zeros(16)
    lower_bound[12:15] = - np.finfo(float).eps
    lower_bound[[0,1,2,4,5,6,8,9,10]] = -1 - np.finfo(float).eps
    lower_bound[[3,7,11]] = -np.inf
    lower_bound[15] = 1 - np.finfo(float).eps
    upper_bound = np.zeros(16)
    upper_bound[12:15] =  np.finfo(float).eps
    upper_bound[[0,1,2,4,5,6,8,9,10]] = 1 + np.finfo(float).eps
    upper_bound[[3,7,11]] = np.inf
    upper_bound[15] = 1 + np.finfo(float).eps
    bounds = (lower_bound, upper_bound)
    diff = X_right - X_left
    cost = np.linalg.norm(diff[:,:3,:])/np.sqrt(N)
    print("before optimize", cost)
    T_params = least_squares(fun=transform_cost,x0=x0,args=(X_right, X_left)) 
    y_pred = transform(T_params.x, X_right)
    diff = y_pred - X_left
    cost = np.linalg.norm(diff[:,:3,:])/np.sqrt(N)
    print("after optimize", cost)
    # T_params = least_squares(fun=transform_cost,x0=T_params.x,args=(X_right, X_left)) 
    # y_pred = transform(T_params.x, X_right)
    # diff = y_pred - X_left
    # cost = np.linalg.norm(diff)
    # print("after optimize", cost)
    

    src = np.swapaxes(X_right[:,:3,:],1,2).reshape(-1,3)
    dst = np.swapaxes(X_left[:,:3,:],1,2).reshape(-1,3)
    retval, out, inliers = cv2.estimateAffine3D(src,dst)

    rvec,_ = cv2.Rodrigues(out[:3,:3])

    # x0 = np.zeros(6)
    # x0[:3] = rvec[:,0]
    # x0[3:6] = out[:,3]
    x0 = np.eye(4)
    x0[:3,:] = out
    x0 = x0.reshape(-1)
    y_pred = transform(x0, X_right)
    diff = y_pred - X_left
    cost = np.linalg.norm(diff[:,:3,:])/np.sqrt(N)
    print("after cv2 estimation", cost)

    rvec = np.array([x0[0], x0[1], x0[2]]).reshape((3,1))
    R,_ = cv2.Rodrigues(rvec)
    T = np.array([x0[3], x0[4], x0[5]],dtype=np.float32).reshape((3,1))
    T = np.concatenate(( R, T), axis=1).astype(np.float32)
    T = np.vstack((T, np.zeros((1, T.shape[1]), dtype=np.float32)))
    T[3,3] = 1
    
    # print(out)
    # print(T)

    T_params = least_squares(fun=transform_cost,x0=x0,args=(X_right, X_left)) 
    y_pred = transform(T_params.x, X_right)
    diff = y_pred - X_left
    cost = np.linalg.norm(diff[:,:3,:])/np.sqrt(N)
    print("final optimize", cost)
    
    
if __name__ == "__main__":
    task2()
