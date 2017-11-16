import numpy as np
import cv2


def region_of_interest(img, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, 255)

    # returning the image only where mask pixels are nonzero
    return cv2.bitwise_and(img, mask)


def find_x(y, q, m):
    return int((y - q) / (m if m != 0 else 1))


Turn = 0
LeftQList = []
LeftMList = []
RightQList = []
RightMList = []
MaxMeanValues = 5


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    global Turn
    global LeftQList
    global LeftMList
    global RightQList
    global RightMList

    # Find the slopes and a sample point
    left_m_list = []
    right_m_list = []
    left_weights = []
    # Knowing y=mx+q we can find q given the sample point and the slope
    left_q_list = []
    right_q_list = []
    right_weights = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Left line has positive slope, Right line has negative.
            if x2 - x1 != 0:
                slope = ((y2 - y1) / (x2 - x1))
                if slope > 0:
                    left_m_list.append(slope)
                    left_q_list.append(y1 - (slope * x1))
                    left_weights.append(y2 - y1)
                else:
                    right_m_list.append(slope)
                    right_q_list.append(y1 - (slope * x1))
                    right_weights.append(y2 - y1)

    # Find the average left and right slope and quote
    left_m = np.average(left_m_list, weights=left_weights)
    left_q = np.average(left_q_list, weights=left_weights)
    right_m = np.average(right_m_list, weights=right_weights)
    right_q = np.average(right_q_list, weights=right_weights)

    # Calculate the current turn
    cur_pos = Turn % MaxMeanValues
    Turn = Turn + 1

    # Calculate a global average for the last (globalAverage) values
    if Turn <= MaxMeanValues:
        LeftQList.append(left_q)
        LeftMList.append(left_m)
        RightQList.append(right_q)
        RightMList.append(right_m)
    else:
        LeftQList[cur_pos] = left_q
        LeftMList[cur_pos] = left_m
        RightQList[cur_pos] = right_q
        RightMList[cur_pos] = right_m
    left_q = np.mean(LeftQList)
    left_m = np.mean(LeftMList)
    right_q = np.mean(RightQList)
    right_m = np.mean(RightMList)

    # Extend the line to the top and to the bottom
    bottom = img.shape[0]
    top = 320
    # Knowing all other parameters we can calculate the missing x for the bottom and the top
    x_bottom_left = find_x(bottom, left_q, left_m)
    x_top_left = find_x(top, left_q, left_m)
    x_bottom_right = find_x(bottom, right_q, right_m)
    x_top_right = find_x(top, right_q, right_m)
    # Draw both lines
    cv2.line(img, (x_bottom_left, bottom), (x_top_left, top), color, thickness)
    cv2.line(img, (x_bottom_right, bottom), (x_top_right, top), color, thickness)


cap = cv2.VideoCapture('test_videos/challenge.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('test_videos_output/challenge.mp4', fourcc, fps, (w, h), True)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur_gray, 50, 150)
        imshape = frame.shape
        vertices = np.array([[(0, imshape[0]), (430, 340), (540, 340), (imshape[1], imshape[0])]], dtype=np.int32)
        cv2.polylines(gray, vertices, True, (255, 255, 255), 5)
        masked_edges = region_of_interest(edges, vertices)
        lines = cv2.HoughLinesP(masked_edges, 0.5, np.pi / 180, 10, np.array([]), minLineLength=30, maxLineGap=30)
        line_img = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        draw_lines(line_img, lines)
        result = cv2.addWeighted(frame, 0.8, line_img, 1.0, 0.0)
        out.write(result)
    else:
        break

cap.release()
out.release()
