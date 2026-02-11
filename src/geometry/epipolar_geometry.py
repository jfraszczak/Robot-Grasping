import cv2
import numpy as np


def draw_epilines(
    img1: np.ndarray,
    img2: np.ndarray,
    img_points1: np.ndarray,
    img_points2: np.ndarray,
    fundamental_matrix: np.ndarray
) -> None:
    img1_copy: np.ndarray = img1.copy()
    img2_copy: np.ndarray = img2.copy()

    height: int = min(np.shape(img1_copy)[0], np.shape(img2_copy)[0])
    width: int = min(np.shape(img1_copy)[1], np.shape(img2_copy)[1])
    img1_copy = img1_copy[:height, :width]
    img2_copy = img2_copy[:height, :width]

    combined_img: np.ndarray = np.hstack((img1_copy, img2_copy))
    w1: int = img1.shape[1]

    # Compute epilines in image 2 for points in image 1
    lines1_to_2 = cv2.computeCorrespondEpilines(img_points1.reshape(-1,1,2), 1, fundamental_matrix)
    lines1_to_2 = lines1_to_2.reshape(-1,3)
    
    # Compute epilines in image 1 for points in image 2
    lines2_to_1 = cv2.computeCorrespondEpilines(img_points2.reshape(-1,1,2), 2, fundamental_matrix)
    lines2_to_1 = lines2_to_1.reshape(-1,3)
    
    for i in range(len(img_points1)):
        color = tuple(np.random.randint(0,255,3).tolist())
        
        r = lines1_to_2[i]
        x0, y0 = 0, int(-r[2]/r[1])
        x1, y1 = img2.shape[1], int(-(r[2] + r[0]*img2.shape[1])/r[1])
        cv2.line(combined_img, (x0 + w1, y0), (x1 + w1, y1), color, 5)
        cv2.circle(combined_img, tuple(img_points1[i].astype(int)), 20, color, -1)
        
        r = lines2_to_1[i]
        x0, y0 = 0, int(-r[2]/r[1])
        x1, y1 = img1.shape[1], int(-(r[2] + r[0]*img1.shape[1])/r[1])
        cv2.line(combined_img, (x0, y0), (x1, y1), color, 5)
        cv2.circle(combined_img, tuple((img_points2[i] + np.array([w1,0])).astype(int)), 20, color, -1)
    
    cv2.imshow("Epilines", combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_corresponding_points(
    img_points1: np.ndarray,
    img_points2: np.ndarray,
    markers_ids1: np.ndarray,
    markers_ids2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    markers_ids1 = markers_ids1.flatten()
    markers_ids2 = markers_ids2.flatten()

    order1 = np.argsort(markers_ids1)
    order2 = np.argsort(markers_ids2)

    markers_ids1 = markers_ids1[order1]
    markers_ids2 = markers_ids2[order2]

    corners_per_marker: int = 4
    sorted_img_points1 = np.vstack([img_points1[i * corners_per_marker:(i + 1) * corners_per_marker] for i in order1])
    sorted_img_points2 = np.vstack([img_points2[i * corners_per_marker:(i + 1) * corners_per_marker] for i in order2])

    img_points1 = sorted_img_points1
    img_points2 = sorted_img_points2

    common_ids = set(list(markers_ids1)).intersection(set(list(markers_ids2)))
    mask1 = np.array([marker_id in common_ids for marker_id in markers_ids1])
    mask2 = np.array([marker_id in common_ids for marker_id in markers_ids2])

    mask1 = np.repeat(mask1, corners_per_marker)
    mask2 = np.repeat(mask2, corners_per_marker)

    img_points1 = img_points1[mask1]
    img_points2 = img_points2[mask2]

    img_points1 = img_points1.reshape(-1, 2).astype(np.float32)
    img_points2 = img_points2.reshape(-1, 2).astype(np.float32)

    return img_points1, img_points2
    

def fundamental_matrix_from_pose(
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray
) -> np.ndarray:
    translation = translation.reshape(3)
    tx = np.array([
        [0,     -translation[2],  translation[1]],
        [translation[2],   0,    -translation[0]],
        [-translation[1],  translation[0],  0]
    ])

    essential_matrix: np.ndarray = tx @ rotation
    fundamental_matrix: np.ndarray = np.linalg.inv(camera_matrix).T @ essential_matrix @ np.linalg.inv(camera_matrix)
    return fundamental_matrix
