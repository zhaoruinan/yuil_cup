# 실행 방법
# cd ~/clean-pvnet
# source venv/bin/activate
# python yuil_pvnet_yolo.py


from lib.config import cfg, args
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import torch

def PVNet(): # PVNet Thread (6D pose estimation) 
    from lib.utils.pvnet import pvnet_pose_utils
    from lib.networks import make_network
    from lib.utils.net_utils import load_network

    global color_image
    global corner_2d_pred
    global pose_pred
    global mask
    global kpt_2d
    global yolo_mask

    torch.manual_seed(0)
    meta = {
        # 'K': np.array([[433.983, 0., 325.083], # D405
        #               [0., 432.934, 241.882],
        #               [0., 0., 1.]]),
        'K': np.array([[605.28, 0., 325.73],  # D435 카메라의 Intrinsic parameters
                      [0., 603.868, 236.881],
                      [0., 0., 1.]]),
        'kpt_3d': np.array([[3.31124291e-02, 3.00500095e-02, 1.39981493e-01], # cup 3D 모델의 3D keypoints 
                            [-3.00500002e-02, -3.31124291e-02, 1.39981493e-01],
                            [2.89015993e-02, -1.33980799e-03, 1.86264504e-09],
                            [-2.99086794e-02, 3.00500095e-02, 1.20750800e-01],
                            [2.96672098e-02, -2.98689492e-02, 1.18076198e-01],
                            [-3.06503791e-02, 2.87102209e-03, 1.64425392e-02],
                            [3.25382501e-03, -3.22385691e-02, 3.09616402e-02],
                            [7.84745812e-03, 3.15252990e-02, 3.17258015e-02],
                            [-5.00000000e-09, -5.00000000e-09, 7.01241509e-02]]),
        'corner_3d': np.array([[-4.575335e-02, -4.575335e-02, 1.862645e-09], # cup 3D 모델의 3D 코너 (6D pose 바운딩 박스 시각화를 위해 사용) 
                               [-4.575335e-02, -4.575335e-02, 1.402483e-01],
                               [-4.575335e-02, 4.575334e-02, 1.862645e-09],
                               [-4.575335e-02, 4.575334e-02, 1.402483e-01],
                               [4.575334e-02, -4.575335e-02, 1.862645e-09],
                               [4.575334e-02, -4.575335e-02, 1.402483e-01],
                               [4.575334e-02, 4.575334e-02, 1.862645e-09],
                               [4.575334e-02, 4.575334e-02, 1.402483e-01]]),
    }

    network = make_network(cfg).cuda()
    load_network(network, 'data/model/pvnet/cup', epoch=-1)
    network.eval()

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    K = np.array(meta['K'])
    kpt_3d = np.array(meta['kpt_3d'])
    corner_3d = np.array(meta['corner_3d'])

    while True:
        try:
            start = time.time()
            
            # 딥러닝 네트워크 입력을 위한 이미지 전처리
            demo_image = np.array(color_image).astype(np.float32)
            # demo_image = np.array(yolo_mask).astype(np.float32)
            inp = (((demo_image/255.)-mean)/std).transpose(2, 0, 1).astype(np.float32)
            inp = torch.Tensor(inp[None]).cuda()
            with torch.no_grad():
                output = network(inp)   # 딥러닝 네트워크 inference

            kpt_2d = output['kpt_2d'][0].detach().cpu().numpy() # 추론한 2D keypoints 

            mask = output['mask'][0].detach().cpu().numpy() # 추론한 segmentation mask

            mask = mask*255
            mask = mask.astype(np.uint8)

            pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K) # PnP Algorithm을 통한 6D pose 추정

            corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)  # 2D 바운딩 박스
            corner_2d_pred = corner_2d_pred.astype(np.int16)

            if np.all(np.logical_and(0 < kpt_2d[:, 0], kpt_2d[:, 0] <= 640)) and np.all(np.logical_and(0 < kpt_2d[:, 1], kpt_2d[:, 1] <= 480)):
                # print(kpt_2d)
                print(pose_pred)
                print("fps : ", 1/(time.time() - start))

        except:
            pass

        time.sleep(0.001)
def YOLO(   # YOLO Thread (Classification)
        weights='best.pt',  # model path or triton URL 
        source=0,  # file/dir/URL/glob/screen/0(webcam)
        data='coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):


    from pathlib import Path

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    from ultralytics.utils.plotting import Annotator, colors, save_one_box

    from models.common import DetectMultiBackend
    from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
    from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                               cv2,
                               increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
    from utils.torch_utils import select_device, smart_inference_mode

    global color_image
    global yolo_mask

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    while True:
        try:
            # 딥러닝 네트워크 입력을 위한 이미지 전처리
            image = color_image.copy()

            im = color_image.copy()
            im0 = color_image.copy()
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            pred = model(im, augment=augment, visualize=visualize)  # 딥러닝 네트워크 inference

            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            # print(pred)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1

                frame = 0
                # s += '%gx%g ' % im.shape[2:]  # print string
                s = '%gx%g ' % im.shape[2:]  # print string

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    yolo_mask = np.zeros_like(image)
                    
                    # FOV 내에 컵이 여러개 있을 때, 가장 왼쪽의 컵만 선택하는 마스크 생성
                    sl = det[torch.argmin(det[:, 0])]   # or torch.argmax(det[:, 0])

                    min_u = int(sl[0]) - 10
                    max_u = int(sl[2]) + 10
                    min_v = int(sl[1]) - 10
                    max_v = int(sl[3]) + 10
                    yolo_mask[min_v:max_v, min_u:max_u] = image[min_v:max_v, min_u:max_u]
        except:
            pass

        time.sleep(0.001)

def Camera():   # Camera Thread 
    ## License: Apache 2.0. See LICENSE file in root directory.
    ## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

    ###############################################
    ##      Open CV and Numpy integration        ##
    ###############################################

    import pyrealsense2.pyrealsense2 as rs
    import cv2

    global corner_2d_pred
    global pose_pred
    global mask
    global kpt_2d

    global color_image
    global yolo_mask

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)
    cv2.namedWindow('RGB', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('RGB', 1920, 1440)

    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
        
            aligned_frames = align.process(frames)
        
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            # depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            vis_image = color_image.copy()
            try: 
                if np.all(np.logical_and(0 < kpt_2d[:,0], kpt_2d[:,0] <= 640)) and np.all(np.logical_and(0 < kpt_2d[:,1], kpt_2d[:,1] <= 480)) and pose_pred[2,3] > 0.2 and pose_pred[2,3] < 0.8:
                    cv2.line(vis_image, tuple(corner_2d_pred[0]), tuple(corner_2d_pred[1]), (255, 0, 0), 3)
                    cv2.line(vis_image, tuple(corner_2d_pred[0]), tuple(corner_2d_pred[2]), (255, 0, 0), 3)
                    cv2.line(vis_image, tuple(corner_2d_pred[0]), tuple(corner_2d_pred[4]), (255, 0, 0), 3)
                    cv2.line(vis_image, tuple(corner_2d_pred[1]), tuple(corner_2d_pred[3]), (255, 0, 0), 3)
                    cv2.line(vis_image, tuple(corner_2d_pred[1]), tuple(corner_2d_pred[5]), (255, 0, 0), 3)
                    cv2.line(vis_image, tuple(corner_2d_pred[2]), tuple(corner_2d_pred[3]), (255, 0, 0), 3)
                    cv2.line(vis_image, tuple(corner_2d_pred[2]), tuple(corner_2d_pred[6]), (255, 0, 0), 3)
                    cv2.line(vis_image, tuple(corner_2d_pred[3]), tuple(corner_2d_pred[7]), (255, 0, 0), 3)
                    cv2.line(vis_image, tuple(corner_2d_pred[4]), tuple(corner_2d_pred[6]), (255, 0, 0), 3)
                    cv2.line(vis_image, tuple(corner_2d_pred[5]), tuple(corner_2d_pred[4]), (255, 0, 0), 3)
                    cv2.line(vis_image, tuple(corner_2d_pred[5]), tuple(corner_2d_pred[7]), (255, 0, 0), 3)
                    cv2.line(vis_image, tuple(corner_2d_pred[6]), tuple(corner_2d_pred[7]), (255, 0, 0), 3)
                else:
                    pass
            except:
                pass
            try:
                cv2.imshow('RGB', vis_image)
                cv2.imshow('MASK', mask)

                # cv2.imshow('YOLO', yolo_mask)
                cv2.waitKey(1)
            except:
                pass

    finally:

        # Stop streaming
        pipeline.stop()

def delay(robot):
    while True:
        if robot.get_robot_running_state() == None:
            pass
        else:
            time.sleep(0.1)
            break
    time.sleep(0.1)

def main(): # Robot 동작 코드 예시 (상황에 맞게 수정 필요)
    global pose_pred
    global mask
    global kpt_2d

    from yuil_lib import Yuil_robot
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

    robot = Yuil_robot("robot")

    speed = 50
    zero_pose = [0, 0, 0, 0, 0, 6 * np.pi/180.0]
    place_pose = [110 * np.pi/180.0, 0, 0, 0, 0, 6 * np.pi/180.0]
    Z = 0

    robot.go_home(speed=speed)
    delay(robot)
    robot.robot_movj(zero_pose, speed=speed, coord=0)
    delay(robot)

    robot.robot_stop()
    robot.robot_run('CINIT')
    delay(robot)

    time.sleep(7)
    T_0c = np.array([[0, 0, 1, 0.2],    # robot base to camera
                     [-1, 0, 0, 0],
                     [0, -1, 0, -0.2],
                     [0, 0, 0, 1]])

    robot.robot_stop()
    robot.gripper_open()
    delay(robot)

    count = 0
    while True:
        try:
            c_kpt = kpt_2d.copy()
            c_pose = pose_pred.copy()

            if np.all(np.logical_and(0 < c_kpt[:, 0], c_kpt[:, 0] <= 640)) and np.all(np.logical_and(0 < c_kpt[:, 1], c_kpt[:, 1] <= 480)) and c_pose[2, 3] > 0.2 and c_pose[2, 3] < 0.8:
                T_co = np.eye(4)

                # camera to object
                T_co[0, 3] = pose_pred[0, 3]
                T_co[1, 3] = pose_pred[1, 3]
                T_co[2, 3] = pose_pred[2, 3]

                T_0o = T_0c @ T_co  # robot base to object = robot base to camera * camera to object

                robot_pose = robot.robot_get_current_xyz_position()

                # 추정한 6D pose로 부터 로봇 동작 (필요에 따라 수정)
                robot_pose[0] = T_0o[0, 3] * 1000 - 180
                robot_pose[1] = T_0o[1, 3] * 1000 + 80*1
                robot_pose[2] = Z

                robot.robot_movec(robot_pose, speed, 1, speed, speed)
                delay(robot)

                robot.robot_stop()
                delay(robot)

                robot_pose[0] = robot_pose[0] + 90
                robot.robot_movec(robot_pose, speed, 1, speed, speed)

                delay(robot)

                robot.robot_stop()
                delay(robot)

                # time.sleep(0.001)

                robot.gripper_close()
                delay(robot)

                time.sleep(0.5)

                robot.go_home(speed=speed)
                robot.robot_stop()
                delay(robot)
                time.sleep(1)
                delay(robot)
                robot.robot_stop()
                time.sleep(0.001)
                robot.robot_movj(place_pose, speed=speed, coord=0)
                delay(robot)

                robot.robot_stop()

                robot_pose = robot.robot_get_current_xyz_position()

                robot_pose[0] = robot_pose[0] + (count % 2) * 120
                robot_pose[1] = robot_pose[1] - (count // 2) * 120


                robot.robot_movec(robot_pose, speed, 1, speed, speed)
                delay(robot)
                robot.robot_stop()


                time.sleep(1)
                robot_pose = robot.robot_get_current_xyz_position()
                robot_pose[2] = robot_pose[2] - 150

                delay(robot)
                robot.robot_stop()

                robot.robot_movec(robot_pose, speed, 1, speed, speed)
                delay(robot)
                robot.robot_stop()
                time.sleep(1)
                robot.gripper_open()
                delay(robot)

                time.sleep(1)

                robot.robot_movj(zero_pose, speed=speed, coord=0)
                delay(robot)

                count += 1
                if count == 4:
                    count = 0
        except:
            pass
        time.sleep(0.001)

    robot.disconnect_robot()


if __name__ == '__main__':
    import threading

    t1 = threading.Thread(target=Camera)
    t1.start()

    # t2 = threading.Thread(target=YOLO)
    # t2.start()

    t3 = threading.Thread(target=PVNet)
    t3.start()

    main()



