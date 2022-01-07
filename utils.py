import cv2 
import numpy as np

def edges(gray, threshold=0):
    src_gray = cv2.blur(gray, (3,3))    
    # Detect edges using Canny
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (255, 255, 255)
        cv2.drawContours(drawing, contours, i, color, 1, cv2.LINE_AA, hierarchy, 0)
    return cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)

def find_corner(gray):
    h, w = gray.shape
    # find Harris corners
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.07)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # discard first
    centroids = centroids[1:]
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    
    return corners.astype(int)

def extranct_coors(pred, gt):
    pred = pred.detach().cpu()[0].squeeze(0).numpy()
    gt = gt.detach().cpu()[0].squeeze(0).numpy()
    pred = (pred * 255).astype(np.uint8)
    src_edge = edges(pred)
    corners = find_corner(src_edge)

    src_corner = cv2.cvtColor(src_edge, cv2.COLOR_GRAY2BGR)
    src_edges = np.array(src_corner)

    for c in corners:
        cv2.circle(src_corner, tuple([x for x in c]), 5, (0,255,0), -1)
    
    return pred, gt, src_corner, src_edges