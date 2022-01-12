import cv2 
import numpy as np

def restore_corners(cor_map):
    print(np.max(cor_map))


def generate_distance_map(corners, w=1024, h=512):
    N = corners.shape[0]
    mapping = np.zeros((N, h, w), np.float32)
    for i, (x,y) in enumerate(corners):
        diagonal = np.linalg.norm(np.array([h, w]))
        cor = np.array([y,x])

        cor = np.tile(cor, (h, w)).reshape((h,w,2)) #512x1024x2
        
        indices = np.indices((h,w)) # 512x1024x2
        indices = np.transpose(indices, (1,2,0))
        diff = cor - indices

        distance = np.linalg.norm(diff, axis=-1)
        distance /= diagonal
        mapping[i] = distance

    mapping = np.min(mapping, axis=0)
    
    return 1.0 - mapping

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

if __name__ == '__main__':
    path = "G:/projects/HorizonNet/data/layoutnet_dataset/test/label_cor/camera_0000896878bd47b2a624ad180aac062e_conferenceRoom_3_frame_equirectangular_domain_.txt"
    corners = np.loadtxt(path).astype(np.int32)
    mapping = generate_distance_map(corners)

    
    tmp = cv2.resize(mapping, (512, 256))

    alpha = 0.9

    mask = tmp>alpha
    tmp[mask] = 1.0
    tmp[~mask] = 0.0
    cv2.imshow("gt", tmp)

    mapping = cv2.applyColorMap((mapping * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    for corner in corners:
        cv2.circle(mapping, corner, 5, (0,0,0), -1)

    pred = cv2.imread("G:/Downloads/media_images_validation_batch_0_21196_0.png", cv2.IMREAD_ANYDEPTH)
    pred = pred.astype(np.float32) / 255.0
    #pred = np.ones((256, 512)) *0.5
    
    mask = pred>alpha
    pred[mask] = 1.0
    pred[~mask] = 0.0

    intersection = pred * tmp
    union = np.clip((pred + tmp), 0,1)
    iou = np.sum(intersection) / np.sum(union)

    print(iou)

    cv2.imshow("pred", pred)
    cv2.imshow("intersection", intersection)
    cv2.imshow("union", union)
    
    #cv2.imshow("distances", mapping)
    cv2.waitKey(0)