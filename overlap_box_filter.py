def large_overlap(box1, box2, overlap_thresh=0.4):
    ymin1, xmin1, ymax1, xmax1 = box1
    ymin2, xmin2, ymax2, xmax2 = box2
    center1 = (xmin1 + xmax1) / 2
    center2 = (xmin2 + xmax2) / 2
    if abs(center1 - center2) < ((xmax2 - xmin2) + (xmax1 - xmin1)) / 4:
        # large overlap on horizon
        print 'horizon overlap'
        if ymax1 + ymin1 < ymax2 + ymin2:
            # 0 is lower, return 1 to set false flag
            return True, 0
        else:
            return True, 1
    if xmax1 < xmin2 or ymax1 < ymin2 or xmax2 < xmin1 or ymax2 < ymin1:
        # no overlap
        return False, None
    sx = sorted([xmin1, xmax1, xmin2, xmax2])
    sy = sorted([ymin2, ymin1, ymax1, ymax2])
    # get overlap area
    overlap_x = abs(sx[1] - sx[2])
    overlap_y = abs(sy[1] - sy[2])
    so = overlap_x * overlap_y

    # get overlap percentage
    s1 = abs(xmax1 - xmin1) * abs(ymax1 - ymin1)
    s2 = abs(xmax2 - xmin2) * abs(ymax2 - ymin2)
    o1 = so / float(s1)
    o2 = so / float(s2)

    # small overlap
    if o1 < overlap_thresh and o2 < overlap_thresh:
        return False, None
    if o1 > o2:
        # the box with more overlap has more active areas, reserve 0, return 1 to set false flag
        return True, 1
    else:
        # small
        return True, 0


def larger(box1, box2):
    x1, y1, x2, y2 = box1
    s1 = abs(x2-x1)*abs(y2-y1)
    x1, y1, x2, y2 = box2
    s2 = abs(x2-x1)*abs(y2-y1)
    return s1 > s2


def is_large_box(box1):
    ymin1, xmin1, ymax1, xmax1 = box1
    return (xmax1 - xmin1) * (ymax1 - ymin1) > 0.3


def overlap_box_filter(boxes):
    boxes_valid_flags = [True for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        if not boxes_valid_flags[i]:
            continue
        if is_large_box(box):
            boxes_valid_flags[i] = False
        for j in range(i+1, len(boxes)):
            if not boxes_valid_flags[j]:
                continue

            ret, smaller_overlap_item = large_overlap(box, boxes[j], 0.6)
            if ret:
                boxes_valid_flags[[i, j][smaller_overlap_item]] = False
    valid_boxes = list()
    for i, box_flag in enumerate(boxes_valid_flags):
        if box_flag:
            valid_boxes.append(boxes[i])
    return valid_boxes, boxes_valid_flags