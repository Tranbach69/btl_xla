import cv2 as cv
import numpy as np

# Only for required region
def require_part(image, vertices):
    mask = np.zeros_like(image)
    match_mask_color = (255, 255, 255)
    # for filling the polygon
    cv.fillPoly(mask, vertices, match_mask_color)

    mask_img = cv.bitwise_and(image, mask)
    return mask_img

# For Draw The Line
def draw_lines(image, lines):
    image = np.copy(image)
    line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=15)
    image = cv.addWeighted(image, 0.8, line_image, 1, 0.0)
    return image
