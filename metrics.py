
def non_max_suppression(bboxes, iou_threshold, threshold):
  # bboxes = [x1,x2,y1,y2,confidence]
  selected_boxes = []
  for i in range(len(bboxes)):
      discard = False
      for j in range(len(bboxes)):
          if i != j:
              iou = intersection_over_union(bboxes[i], bboxes[j])
              if iou > iou_threshold:
                  if bboxes[i][4] < bboxes[j][4]:
                      discard = True
                      break

      if not discard and bboxes[i][4] >= threshold:

          selected_boxes.append(bboxes[i])

  return selected_boxes

def intersection_over_union(predicted_bbox, gt_bbox) -> float:
    """
    :param: predicted_bbox - [x_min, y_min, x_max, y_max]
    :param: gt_bbox - [x_min, y_min, x_max, y_max]

    """
    intersection_bbox = np.array(
        [
            max(predicted_bbox[0], gt_bbox[0]),
            max(predicted_bbox[1], gt_bbox[1]),
            min(predicted_bbox[2], gt_bbox[2]),
            min(predicted_bbox[3], gt_bbox[3]),
        ]
    )

    intersection_area = max(intersection_bbox[2] - intersection_bbox[0], 0) * max(intersection_bbox[3] - intersection_bbox[1], 0)
    area_dt = (predicted_bbox[2] - predicted_bbox[0]) * (predicted_bbox[3] - predicted_bbox[1])
    area_gt = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])

    union_area = area_dt + area_gt - intersection_area

    iou = intersection_area / union_area

    return iou