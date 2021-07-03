import logging
from detectime.augmentations import get_max_bbox, result_crop

log = logging.getLogger(__name__)


def detect_hand(img,
                model_detector_faces,
                model_detector_hands):
    # DETECT HANDS
    result = model_detector_hands.detect([img], verbose=0)[0]
    all_hands, pred_score = result['rois'], result['scores']
    all_hands, pred_score = list(all_hands), list(pred_score)
    if all_hands:
        max_prob = max(pred_score)
        index_of_max_prob = pred_score.index(max_prob)
        y1, x1, y2, x2 = all_hands[index_of_max_prob]
        cropped_image = img[y1:y2, x1:x2]
        return cropped_image, 0
    else:
        # DETECT FACES
        detections = model_detector_faces.detect(img)
        all_faces = []
        for det in detections:
            x1, y1, x2, y2, s = det.tolist()
            w = x2 - x1
            h = y2 - y1
            bbox = [round(x1), round(y1), round(w), round(h)]
            all_faces.append(bbox)
        if len(all_faces) > 0:
            max_bbox = get_max_bbox(all_faces)
            x3, y3, x4, y4 = result_crop(img,
                                         max_bbox,
                                         crop_coefficient=0.75)
            cropped_image = img[y3:y4, x3:x4]
            return cropped_image, 1
        else:
            return img, 2


def modified_hand_detection(img,
                            model_detector_faces,
                            model_detector_hands,
                            crop_coefficient=1.5):
    # DETECTOR FACES
    detections = model_detector_faces.detect(img)
    all_faces = []
    for det in detections:
        x1, y1, x2, y2, s = det.tolist()
        w = x2 - x1
        h = y2 - y1
        bbox = [round(x1), round(y1), round(w), round(h)]
        all_faces.append(bbox)

    # DETECT HANDS
    result = model_detector_hands.detect([img], verbose=0)[0]
    all_hands = result['rois']

    # AREA
    area = []
    faces_and_hands = []

    # FIND MAX AREA OF HAND AND FACES
    for face in all_faces:
        for hand in all_hands:
            x3, y3, x4, y4 = result_crop(img=img,
                                         face=face,
                                         crop_coefficient=crop_coefficient)
            y1, x1, y2, x2 = hand
            left_x, left_y = max(x1, x3), max(y1, y3)
            right_x, right_y = min(x2, x4), min(y2, y4)
            width, height = right_x - left_x, right_y - left_y
            if width <= 0 or height <= 0:
                area.append(0)
            else:
                area.append(width * height)
            faces_and_hands.append((face, hand))

    if area:
        max_area = max(area)
        if max_area != 0:
            index_of_max_area = area.index(max_area)
            index_of_face_hand = faces_and_hands[index_of_max_area]
            final_face, final_hand = index_of_face_hand
            y1, x1, y2, x2 = final_hand
            # y1, x1, y2, x2 = result_crop(
            #     img,
            #     [x1, y1, x2 - x1, y2 - y1],
            #     crop_coefficient=0.75
            # )
            img_to_save = img[y1:y2, x1:x2]
            return img_to_save, 0

    return img, 2
