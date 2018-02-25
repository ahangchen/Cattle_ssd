from overlap_box_filter import large_overlap


def is_new(box, last_boxes, sum_padding=-1.):
    print 'is new?'
    ymin1, xmin1, ymax1, xmax1 = box
    if sum_padding >= 0 and (xmin1 + sum_padding > 0.2):
        return False
    last_len = len(last_boxes)
    for i in range(last_len):
        last_box = last_boxes[last_len - i - 1]
        ymin2, xmin2, ymax2, xmax2 = last_box
        center1 = (xmin1 + xmax1) / 2
        center2 = (xmin2 + xmax2) / 2
        if abs(center1 - center2) < ((xmax2 - xmin2) + (xmax1 - xmin1)) / 4:
            print 'Old x [%f, %f, %f, %f] vs [%f, %f, %f, %f]' % (ymin1, xmin1, ymax1, xmax1, ymin2, xmin2, ymax2, xmax2)
            return False
        else:
            is_overlap, smaller_item = large_overlap(box, last_box, overlap_thresh=0.5)
            if is_overlap:
                print 'Old overlap [%f, %f, %f, %f] vs [%f, %f, %f, %f]' % (
                ymin1, xmin1, ymax1, xmax1, ymin2, xmin2, ymax2, xmax2)
                return False
    print 'New: %f, %f, %f, %f' % (ymin1, xmin1, ymax1, xmax1)
    log_str = ''
    for last_box in last_boxes:
        ymin2, xmin2, ymax2, xmax2 = last_box
        log_str += '[%f, %f, %f, %f], ' % (ymin2, xmin2, ymax2, xmax2)
    print log_str
    return True


def frame_padding(cur_boxes, last_boxes):
    avg_padding = 0
    valid_cnt = 0
    padding_boxes = []
    for box in cur_boxes:
        min_padding = 1.
        ymin1, xmin1, ymax1, xmax1 = box
        center1 = (xmin1 + xmax1) / 2
        for last_box in last_boxes:
            ymin2, xmin2, ymax2, xmax2 = last_box
            center2 = (xmin2 + xmax2) / 2
            padding = center1 - center2
            if min_padding > padding and padding > 0:
                min_padding = padding
        if min_padding > (xmax1 - xmin1) * 0.8:
            print 'invalid min padding %f with box width %f' % (min_padding, xmax1 - xmin1)
            continue
        avg_padding += min_padding
        valid_cnt += 1
        padding_boxes.append(box)
    if valid_cnt > 0:
        return padding_boxes, avg_padding / valid_cnt
    else:
        return padding_boxes, 0


def accum_frame_padding(old_padding, old_height, cur_boxes, last_boxes):
    if old_height == 0 and len(cur_boxes)> 0:
        old_height = avg_box_height(cur_boxes)
    if len(cur_boxes) == 0 or len(last_boxes) == 0:
        return old_padding, old_height
    else:
        padding_boxes, padding = frame_padding(cur_boxes, last_boxes)
        ret = len(padding_boxes) > 0
        if ret:
            confidence = 1 - abs(padding - old_padding) / (old_padding + 0.001)
            if confidence < 0:
                confidence = 0.3
            else:
                confidence = 0.5
            return (1 - confidence) * old_padding + confidence * padding, accum_box_height(old_height, padding_boxes, confidence)
        else:
            return old_padding, old_height


def avg_box_height(cur_boxes):
    sum_height = 0
    for cur_box in cur_boxes:
        ymin, xmin, ymax, xmax = cur_box
        cur_height = ymax - ymin
        sum_height += cur_height
    return sum_height / len(cur_boxes)


def accum_box_height(old_height, cur_boxes, confidence):
    avg_height = avg_box_height(cur_boxes)
    old_height = (1 - confidence) * old_height + confidence * avg_height
    return old_height


def frame_shift(padding, avg_height, avg_padding, last_boxes):
    # right shift cur frame
    shift_boxes = list()
    for box in last_boxes:
        cur_height = box[2] - box[0]
        # shift_boxes.append(
        #     [box[0], box[1] + padding, box[2], box[3] + padding]
        # )
        if avg_height == 0 or avg_padding == 0:
            shift_boxes.append(
                [box[0], box[1] + padding , box[2], box[3] + padding]
            )
        else:
            fix_padding = avg_padding * (1 - cur_height / avg_height)
            if fix_padding < 0:
                print 'cur_height: %f, avg_height: %f' % (cur_height, avg_height)
                fix_padding *= 2.
            ymin, xmin, ymax, xmax = box
            fix_time = xmin * avg_height / cur_height / avg_padding
            shift_boxes.append(
                [box[0], box[1] + padding + fix_time * fix_padding,
                 box[2], box[3] + padding + fix_time * fix_padding]
            )
            print 'fix_time: %f, fix_padding: %f, fix: %f' % (fix_time, fix_padding, fix_time * fix_padding)
    return shift_boxes