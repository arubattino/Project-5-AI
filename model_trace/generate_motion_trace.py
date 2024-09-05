import pandas as pd
from tqdm import tqdm
import numpy as np
import cv2 as cv
import cv2

# Script to generate the line masks of all the videos and form a dataset with a CSV file.

def fingerprint(path, filename, dirr):
    '''
    path:     str: video address
    filename: str: name of the image to save
    dirr:     str: directory where you save the image
    
    '''
    cap = cv.VideoCapture(path)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create color
    color = np.array([0,255,0])

    # Take first frame and find corners in it
    for j in range(4):
        ret, old_frame = cap.read()
        old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    #-----------
    height, width, _ = old_frame.shape
    ny = height/10
    nx = width/15

    # Point selecction
    p0 = np.array([[[ny,nx]]])

    t=0
    for j in range(15):
        t+=1
        n=0
        y = ny*t
        for i in range(10):
            n+=1
            x = nx*n
            p0 = np.append(p0,[np.array([[y,x]])],axis= 0)

    p0 = np.float32(p0)

    # Black background
    imbk = cv2.imread('black.jpg') 
    imbk = cv2.resize(imbk, (width,height))

    count=0
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count+=1
        
    # Count frame n° 4
        if count > 4:
            imbk2 = imbk.copy()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
      
        # Calculate optical flow
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
      
        # Select good points
            if p1 is not None:
                good_new = p1[st==1]
                good_old = p0[st==1]

        # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                try:
                    mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color.tolist(), 1, lineType = 50) 
                    frame = cv.circle(frame, (int(a), int(b)), 5, color.tolist(), -1) 
                    imbk2 = cv.circle(imbk2, (int(a), int(b)), 5, color.tolist(), -1) 
                except:
                    pass

            img1 = cv.add(imbk2, mask)

        # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    cv2.imwrite(dirr + filename + '.jpg', img1)



train = pd.read_csv("finish_data_train_final.csv")


entrada = open('data_train_trace.csv', 'wt')
for j in tqdm(range(len(train)), desc='Loop'):
    try:
        path_video = str(train['dir'][j])
        movement = str(train['movement_label'][j])
        mv = str(train['movement_value'][j])
        entrada.write(f'{j}.jpg' + "," + movement + ',' + mv + "\n")
        fingerprint(path_video, f'{j}', './datatrain/') 
    except:
        print('Error', path_video, j)
        pass
entrada.close()