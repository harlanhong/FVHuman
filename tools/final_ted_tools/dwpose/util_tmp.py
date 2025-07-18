import numpy as np
import cv2
import math
import matplotlib

eps = 0.01

def draw_pose_limb_mask(pose, H, W, body_score, face_score, hands_score):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    mask = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    mask = draw_connected_limbs(mask, candidate, subset, body_score)
    # mask = draw_handpose_mask(mask, hands, hands_score)
    # mask = draw_facepose_mask(mask, faces, face_score)

    return mask


def draw_connected_limbs(mask, candidate, subset, body_score, thickness=16):
    H, W, _ = mask.shape
    candidate = np.array(candidate)
    subset = np.array(subset)
    
    limbSeq = [
        [2, 3],
        [2, 6],
        [3, 4],
        [4, 5],
        [6, 7],
        [7, 8],
        [2, 9],
        [9, 10],
        [10, 11],
        [2, 12],
        [12, 13],
        [13, 14],
        [2, 1],
        [1, 15],
        [15, 17],
        [1, 16],
        [16, 18],
        # [3, 17],
        # [6, 18],
    ]

    for i, limb in enumerate(limbSeq):
        for n in range(len(subset)):
            index = subset[n][np.array(limb) - 1]
            if -1 in index:
                continue
            point1 = candidate[index[0].astype(int)]
            point2 = candidate[index[1].astype(int)]
            x1, y1, score1 = point1[0], point1[1], body_score[n][limb[0] - 1]
            x2, y2, score2 = point2[0], point2[1], body_score[n][limb[1] - 1]

            x1, y1 = int(x1 * W), int(y1 * H)
            x2, y2 = int(x2 * W), int(y2 * H)
            avg_score = (score1 + score2) / 2

            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(mask, (x1, y1), (x2, y2), (int(avg_score * 255), int(avg_score * 255), int(avg_score * 255)), thickness)
    
    return mask
    
def draw_pose_mask(pose, H, W, body_score, face_score, hands_score):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    mask = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    mask = draw_bodypose_mask(mask, candidate, subset, body_score)
    mask = draw_handpose_mask(mask, hands, hands_score)
    mask = draw_facepose_mask(mask, faces, face_score)

    return mask

def draw_bodypose_mask(mask, candidate, subset, body_score):
    H, W, _ = mask.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            score = body_score[n][i]  # Assuming body_score is an array of scores
            x = int(x * W)
            y = int(y * H)
            color = int(score * 255)
            cv2.circle(mask, (int(x), int(y)), 16, (color, color, color), thickness=-1)  # Set thickness to 1 instead of -1
    
    return mask

def draw_handpose_mask(mask, all_hand_peaks, hands_score):
    H, W, _ = mask.shape

    for peaks, score_list in zip(all_hand_peaks, hands_score):
        peaks = np.array(peaks)

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            score = score_list[i]  # Assuming hands_score is a list of score lists
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                color = int(score * 255)
                cv2.circle(mask, (x, y), 16, (color, color, color), thickness=-1)  # Set thickness to 1 instead of -1
    return mask

def draw_facepose_mask(mask, all_lmks, face_score):
    H, W, _ = mask.shape
    for lmks, score_list in zip(all_lmks, face_score):
        lmks = np.array(lmks)
        for i, lmk in enumerate(lmks):
            x, y = lmk
            score = score_list[i]  # Assuming face_score is a list of score lists
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                color = int(score * 255)
                cv2.circle(mask, (x, y), 16, (color, color, color), thickness=-1)  # Set thickness to 1 instead of -1
    return mask