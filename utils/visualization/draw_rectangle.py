from copy import copy

import cv2



def draw_rectangle(image, x1, y1, w, h, label, color):
    # For bounding box
    img = copy(image)
    text_color = (0, 0, 0)
    x2 = x1 + w
    y2 = y1 + h

    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # For the text background
    # Finds space required by the text so that we can put a background with that amount of width.
    (w, h), _ = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

    # Prints the text.
    img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    img = cv2.putText(img, str(label), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    # For printing text
    # img = cv2.putText(img, 'test', (x1, y1),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    return img
