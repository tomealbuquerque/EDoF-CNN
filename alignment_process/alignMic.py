import cv2 
import numpy as np 
import pandas as pd

def alignMic(img, reference = 'reference.png', layout = 'referenceMicLocation.csv'):
    
    # Load microphone locations of reference image.
    micLoc = pd.read_csv(layout)
    micLocXY = [np.array(micLoc['X']), np.array(micLoc['Y'])]
    pts = np.float32(micLocXY).T.reshape(-1,1,2)

    # Open the image files. 
    img1_color = cv2.imread(reference) 

    # Convert to grayscale. 
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY) 
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    height, width = img2.shape 

    # Create ORB detector with 5000 features. 
    orb_detector = cv2.ORB_create(5000) 

    # Find keypoints and descriptors. 
    # The first arg is the image, second arg is the mask 
    # (which is not reqiured in this case). 
    kp1, d1 = orb_detector.detectAndCompute(img1, None) 
    kp2, d2 = orb_detector.detectAndCompute(img2, None) 

    # Match features between the two images. 
    # We create a Brute Force matcher with 
    # Hamming distance as measurement mode. 
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 

    # Match the two sets of descriptors. 
    matches = matcher.match(d1, d2) 

    # Sort matches on the basis of their Hamming distance. 
    matches.sort(key = lambda x: x.distance) 

    # Take the top 90 % matches forward. 
    matches = matches[:int(len(matches)*90)] 
    no_of_matches = len(matches) 

    # Define empty matrices of shape no_of_matches * 2. 
    p1 = np.zeros((no_of_matches, 2)) 
    p2 = np.zeros((no_of_matches, 2)) 

    for i in range(len(matches)): 
        p1[i, :] = kp1[matches[i].queryIdx].pt 
        p2[i, :] = kp2[matches[i].trainIdx].pt 

    # Find the homography matrix. 
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 

    # Use this matrix to transform the microphone locations wrt the reference image. 
    transformed_pts = cv2.perspectiveTransform(pts, homography, (width, height))

    # Save the output. 
    return transformed_pts


def getMedianFrame(vid):

    cap = cv2.VideoCapture(vid)
    ret, frame = cap.read()
    frameSet = np.zeros(np.insert(np.array(frame.shape),0,10))

    if cap.get(cv2.CAP_PROP_FRAME_COUNT) < 10000:
        print('Video have fewer than 10000 frames, returning first frame.')
        return frame

    for i in range(0,10):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/3 + i*1000)); # read frames from the middle of video
        ret, frame = cap.read()
        frameSet[i,:] = frame

    return np.median(frameSet, axis=0).astype('uint8')


if __name__ == "__main__":
    import argparse
    import os
    import glob
    from matplotlib import pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="video(es) to find microphones")
    parser.add_argument("reference", help="reference image to align to")
    parser.add_argument("layout", help="microphone layout for reference image")
    args = parser.parse_args()

    for vid in glob.glob(args.video):
        frame = getMedianFrame(vid)
        micLoc = alignMic(frame, args.reference, args.layout)

        np.save(rf'{os.path.dirname(vid)}/micLoc.npy', micLoc.squeeze())

        plt.figure(figsize=(30,20))
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.plot(micLoc.squeeze()[:,0],micLoc.squeeze()[:,1],'r*',markersize=12)

        plt.savefig(rf'{os.path.dirname(vid)}/micLoc.png')
        plt.close()