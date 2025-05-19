import cv2
import numpy as np


COLOR_MAP = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (102, 204, 153),
             (255, 255, 0), (106, 90, 205), (0, 255, 255), (255, 102, 0),
             (255, 153, 153), (153, 0, 51), (255, 182, 193), (220, 20, 60),
             (255, 240, 245), (255, 105, 180), (216, 191, 216), (128, 0, 128), 
             (128, 0, 0), (255, 204, 153), (0, 51, 0), (0, 51, 102), 
             (0, 128, 128), (51, 51, 153), (128, 128, 0), (255, 153, 204),
             (255, 0, 255), (153, 51, 0)]

# ==============================================================================
#                                                          BIRDS_EYE_POINT_CLOUD
# ==============================================================================
def birds_eye_point_cloud(points,
                          side_range=(-10, 10),
                          fwd_range=(-10,10),
                          res=0.1,
                          min_height = -2.73,
                          max_height = 1.27,
                          saveto=None):
    """ Creates an 2D birds eye view representation of the point cloud data.
        You can optionally save the image to specified filename.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        res:        (float) desired resolution in metres to use
                    Each output pixel will represent an square region res x res
                    in size.
        min_height:  (float)(default=-2.73)
                    Used to truncate height values to this minumum height
                    relative to the sensor (in metres).
                    The default is set to -2.73, which is 1 metre below a flat
                    road surface given the configuration in the kitti dataset.
        max_height: (float)(default=1.27)
                    Used to truncate height values to this maximum height
                    relative to the sensor (in metres).
                    The default is set to 1.27, which is 3m above a flat road
                    surface given the configuration in the kitti dataset.
        saveto:     (str or None)(default=None)
                    Filename to save the image as.
                    If None, then it just displays the image.
    """
    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    # r_lidar = points[:, 3]  # Reflectance

    # INDICES FILTER - of values within the desired rectangle
    # Note left side is positive y axis in LIDAR coordinates
    ff = np.logical_and((x_lidar > fwd_range[0]), (x_lidar < fwd_range[1]))
    ss = np.logical_and((y_lidar > -side_range[1]), (y_lidar < -side_range[0]))
    indices = np.argwhere(np.logical_and(ff,ss)).flatten()

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_lidar[indices]/res).astype(np.int32) # x axis is -y in LIDAR
    y_img = (-x_lidar[indices]/res).astype(np.int32)  # y axis is -x in LIDAR
                                                     # will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0]/res))
    y_img -= int(np.floor(fwd_range[0]/res))

    # # CLIP HEIGHT VALUES - to between min and max heights
    # pixel_values = np.clip(a = z_lidar[indices],
    #                        a_min=min_height,
    #                        a_max=max_height)

    # # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    # pixel_values  = scale_to_255(pixel_values, min=min_height, max=max_height)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((side_range[1] - side_range[0])/res)
    y_max = int((fwd_range[1] - fwd_range[0])/res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    # im[y_img, x_img] = pixel_values # -y because images start from top left
    im[y_img, x_img] = 255

    return im

def birds_eye_point_cloud_color(points, intensity, 
                          side_range=(-10, 10),
                          fwd_range=(-10,10),
                          res=0.1,
                          min_height = -2.73,
                          max_height = 1.27,
                          saveto=None):
    """ Creates an 2D birds eye view representation of the point cloud data.
        You can optionally save the image to specified filename.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        res:        (float) desired resolution in metres to use
                    Each output pixel will represent an square region res x res
                    in size.
        min_height:  (float)(default=-2.73)
                    Used to truncate height values to this minumum height
                    relative to the sensor (in metres).
                    The default is set to -2.73, which is 1 metre below a flat
                    road surface given the configuration in the kitti dataset.
        max_height: (float)(default=1.27)
                    Used to truncate height values to this maximum height
                    relative to the sensor (in metres).
                    The default is set to 1.27, which is 3m above a flat road
                    surface given the configuration in the kitti dataset.
        saveto:     (str or None)(default=None)
                    Filename to save the image as.
                    If None, then it just displays the image.
    """
    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    # r_lidar = points[:, 3]  # Reflectance

    # INDICES FILTER - of values within the desired rectangle
    # Note left side is positive y axis in LIDAR coordinates
    ff = np.logical_and((x_lidar > fwd_range[0]), (x_lidar < fwd_range[1]))
    ss = np.logical_and((y_lidar > -side_range[1]), (y_lidar < -side_range[0]))
    indices = np.argwhere(np.logical_and(ff,ss)).flatten()

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_lidar[indices]/res).astype(np.int32) # x axis is -y in LIDAR
    y_img = (-x_lidar[indices]/res).astype(np.int32)  # y axis is -x in LIDAR
                                                     # will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0]/res))
    y_img -= int(np.floor(fwd_range[0]/res))

    # # CLIP HEIGHT VALUES - to between min and max heights
    # pixel_values = np.clip(a = z_lidar[indices],
    #                        a_min=min_height,
    #                        a_max=max_height)

    # # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    # pixel_values  = scale_to_255(pixel_values, min=min_height, max=max_height)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((side_range[1] - side_range[0])/res)
    y_max = int((fwd_range[1] - fwd_range[0])/res)
    im = np.zeros([y_max, x_max, 3], dtype=np.uint8)
    # im[y_img, x_img] = pixel_values # -y because images start from top left
    # temp_color_map = COLOR_MAP.copy()
    for i, y_info in enumerate(y_img):
        # intensity_color = (np.int64(intensity[i]) % 10) %len(COLOR_MAP)
        if intensity[i] == 0:
            im[y_info, x_img[i]] = (255, 255, 255)
            continue
        intensity_color = (np.int64(intensity[i])) %len(COLOR_MAP)
        im[y_info, x_img[i]] = COLOR_MAP[intensity_color]
        # temp_color_map.pop(intensity_color)
        # im[y_info, x_img[i]] = [intensity[i], intensity[i], intensity[i]]

    return im

def rotation_matrix(yaw, pitch=0, roll=0):
    tx = roll
    ty = yaw
    tz = pitch

    Rx = np.array([[1,0,0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0,0,1]])


    return Ry.reshape([3,3])


def findVRange(img, id, undistort_color_img, is_car):
    # cv2.imwrite(str(id) + '_test_before.jpg', img)
    # 在二值图上寻找轮廓
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box = []
    max_contour = max(contours, key = cv2.contourArea)
    # 对每个轮廓点求最大外接矩形
    x, y, w, h = cv2.boundingRect(max_contour)
    # color_id = id % len(COLOR_MAP)
    temp_id = id
    if is_car:
        temp_id += 150
    cv2.rectangle(undistort_color_img, (x-10, y-10), (x+w+10, y+h+10), (0, 0, 255), 2)
    cv2.putText(undistort_color_img, str(temp_id), (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 3)


    for cont in contours:
        # 对每个轮廓点求最小外接矩形
        rect = cv2.minAreaRect(cont)
        # cv2.boxPoints可以将轮廓点转换为四个角点坐标
        box = cv2.boxPoints(rect)
        # 这一步不影响后面的画图，但是可以保证四个角点坐标为顺时针
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        # 在原图上画出预测的外接矩形
        box = box.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img, [box], True, 255, 3)


    min_v = 9999
    max_v = -9999
    for pt in box:
        if pt[0][1] > max_v:
            max_v = pt[0][1]
        if pt[0][1] < min_v:
            min_v = pt[0][1]

    v_thresh = int(round(min_v + 3 / 3 * (max_v - min_v)))
    # cv2.imwrite(os.path.join( '/home/zjlab/zcr/BEVHeight/results/bev_lidar/', str(id) + '_test.jpg'), img)

    return v_thresh

def lidar2img(img, undistort_img, coord_img_left, lidar, undistort_color_img, is_car = False):
    ids = np.unique(img) #去除重复元素，并从大到小排列
    ids = np.delete(ids, 0)
    id_flags = np.zeros((lidar.shape[0], 1))

    v_dict = {}

    for id in ids:
        img_id = img.copy()
        img_id[img_id != id] = 0
        img_id[img_id == id] = 255
        v_thresh = findVRange(img_id, id, undistort_color_img, is_car)
        # cv2.polylines(undistort_color_img, [box], True, 255, 3)
        v_dict[id] = v_thresh

    for i in range(coord_img_left.shape[0]):
        u = int(round(coord_img_left[i, 0], 0))
        v = int(round(coord_img_left[i, 1], 0))

        if u >= 0 and u < 1920 and v >= 0 and v < 1080:
            # cv2.circle(undistort_color_img, (u, v), 1, (255, 0, 0), -1)
            id = img[v, u]
            color_id = id % len(COLOR_MAP)
            if id > 0: # and v <= v_dict[id]:
            # if id > 0 and v <= v_dict[id]:
                # lidar[i][-1] = 10000 + id
                id_flags[i] = id
                # cv2.circle(undistort_color_img, (u, v), 3, COLOR_MAP[color_id], -1)
                cv2.circle(undistort_img, (u, v), 1, 255, 1)

    return ids, undistort_img, undistort_color_img, id_flags


def lidar2bev(obj_1_points, lidar2cam, offset, bev_lidar_path):
    box_rect = []
    box = []

    if not np.any(obj_1_points):
        return box_rect, box

    obj_1_points[:, -1] = 1
    obj_1_camera3d = np.matmul(lidar2cam, obj_1_points.T).T

    min_z = np.min(obj_1_camera3d[:, 2])

    obj_1_points = obj_1_points[obj_1_camera3d[:, 2] <= min_z + offset]

    img_raw = birds_eye_point_cloud(obj_1_points,
                                                     side_range=(-40, 40), fwd_range=(-30, 30),
                                                     res=0.05)
    img_copy = img_raw.copy()

    # 图像膨胀
    dilated_kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img_raw, dilated_kernel, 1)

    # 图像腐蚀
    erode_kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, erode_kernel, 1)

    # 在二值图上寻找轮廓
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = -9999
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > max_area:
            max_area = area
        else:
            continue
        # 对每个轮廓点求最小外接矩形
        rect = cv2.minAreaRect(cont)
        # cv2.boxPoints可以将轮廓点转换为四个角点坐标
        box = cv2.boxPoints(rect)
        # 这一步不影响后面的画图，但是可以保证四个角点坐标为顺时针
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        # 在原图上画出预测的外接矩形
        box = box.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img_copy, [box], True, 255, 1)

    for point in box:
        # y = (point[0][1] - 800) * 0.05 * -1
        # x = (point[0][0] - 400) * 0.05 * -1
        y = (point[0][0] - 800) * 0.05 * -1
        x = (point[0][1] - 600) * 0.05 * -1
        box_rect.append([x, y, 0, 1])
    box_rect = np.array(box_rect)

    # img_copy = cv2.rotate(img_copy, cv2.ROTATE_90_CLOCKWISE)
    # cv2.imwrite(bev_lidar_path, img_copy)

    return box_rect, box
