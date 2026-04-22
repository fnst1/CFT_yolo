import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QComboBox, QGroupBox,
                             QProgressBar, QTextEdit, QCheckBox, QSpinBox, QSlider)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np

from PyQt5.QtWidgets import QMessageBox

from utils.datasets import letterbox
from utils.general import LOGGER
from utils.plots import colors, plot_one_box

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: 未安装ultralytics库，无法进行真实检测。请运行: pip install ultralytics")


class VideoDetectionWorker(QThread):
    """视频检测工作线程"""
    frame_ready = pyqtSignal(object, object)  # 原始帧, 检测结果帧
    detection_info = pyqtSignal(list)  # 检测信息
    finished_signal = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, model_path, video_path, conf_threshold=0.5):
        super().__init__()
        self.model_path = model_path
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.running = True
        self.paused = False
        self.mutex = QMutex()

    def run(self):
        try:
            if not YOLO_AVAILABLE:
                raise ImportError("YOLO库未安装")

            # 加载模型
            model = YOLO(self.model_path)

            # 打开视频
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise Exception(f"无法打开视频文件: {self.video_path}")

            while self.running:
                self.mutex.lock()
                if self.paused:
                    self.mutex.unlock()
                    self.msleep(50)  # 避免CPU占用过高
                    continue
                self.mutex.unlock()

                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO推理
                results = model(frame, conf=self.conf_threshold, verbose=False)
                result = results[0]

                # 获取检测结果
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    cls_ids = result.boxes.cls.cpu().numpy()
                    names = result.names

                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = map(int, boxes[i])
                        conf = float(confs[i])
                        cls_id = int(cls_ids[i])
                        label = names[cls_id]
                        detections.append({
                            'box': [x1, y1, x2, y2],
                            'confidence': conf,
                            'label': label
                        })

                # 绘制检测结果
                annotated_frame = result.plot()

                # 发送信号
                self.frame_ready.emit(frame.copy(), annotated_frame.copy())
                self.detection_info.emit(detections)

                # 控制播放速度
                self.msleep(30)

            cap.release()
            self.finished_signal.emit()

        except Exception as e:
            self.error_occurred.emit(str(e))

    def pause_resume(self):
        self.mutex.lock()
        self.paused = not self.paused
        self.mutex.unlock()

    def stop(self):
        self.running = False


class CameraDetectionWorker(QThread):
    """摄像头检测工作线程"""
    frame_ready = pyqtSignal(object, object)
    detection_info = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_path, camera_index=0, conf_threshold=0.5):
        super().__init__()
        self.model_path = model_path
        self.camera_index = camera_index
        self.conf_threshold = conf_threshold
        self.running = True
        self.paused = False
        self.mutex = QMutex()

    def run(self):
        try:
            if not YOLO_AVAILABLE:
                raise ImportError("YOLO库未安装")

            # 加载模型
            model = YOLO(self.model_path)

            # 打开摄像头
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                raise Exception(f"无法打开摄像头: {self.camera_index}")

            while self.running:
                self.mutex.lock()
                if self.paused:
                    self.mutex.unlock()
                    self.msleep(50)
                    continue
                self.mutex.unlock()

                ret, frame = cap.read()
                if not ret:
                    break

                # YOLO推理
                results = model(frame, conf=self.conf_threshold, verbose=False)
                result = results[0]

                # 获取检测结果
                detections = []
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    cls_ids = result.boxes.cls.cpu().numpy()
                    names = result.names

                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = map(int, boxes[i])
                        conf = float(confs[i])
                        cls_id = int(cls_ids[i])
                        label = names[cls_id]
                        detections.append({
                            'box': [x1, y1, x2, y2],
                            'confidence': conf,
                            'label': label
                        })

                # 绘制检测结果
                annotated_frame = result.plot()

                # 发送信号
                self.frame_ready.emit(frame.copy(), annotated_frame.copy())
                self.detection_info.emit(detections)

                # 控制帧率
                self.msleep(30)

            cap.release()

        except Exception as e:
            self.error_occurred.emit(str(e))

    def pause_resume(self):
        self.mutex.lock()
        self.paused = not self.paused
        self.mutex.unlock()

    def stop(self):
        self.running = False


class DualStreamVideoDetectionWorker(QThread):
    """双模态视频检测工作线程 (可见光 + 热红外)"""
    # 注意：这里发送的是两个带检测框的帧
    frame_ready = pyqtSignal(object, object)  # 处理后的可见光帧, 处理后的热红外帧
    detection_info = pyqtSignal(list)  # 检测信息
    finished_signal = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, model_path, rgb_video_path, ir_video_path, conf_threshold=0.5):
        super().__init__()
        self.model_path = model_path
        self.rgb_video_path = rgb_video_path
        self.ir_video_path = ir_video_path
        self.conf_threshold = conf_threshold
        self.running = True
        self.paused = False
        self.mutex = QMutex()

    def run(self):
        try:
            # --- 1. 加载自定义双流模型 (与 DetectionWorker 中的逻辑一致) ---
            import torch
            from models.experimental import attempt_load
            from utils.general import check_img_size, scale_coords
            from ultralytics.utils.ops import non_max_suppression
            from utils.datasets import letterbox
            from utils.torch_utils import select_device
            from utils.plots import colors, plot_one_box

            device = select_device('')
            half = device.type != 'cpu'

            LOGGER.info(f"正在从 {self.model_path} 加载双流视频检测模型...")
            model = attempt_load(self.model_path, map_location=device)
            stride = int(model.stride.max())
            # 使用固定 640x640，无需 check_img_size（因为 auto=False）
            imgsz = 640  # 可保留用于日志，但不影响 letterbox
            names = model.module.names if hasattr(model, 'module') else model.names
            if half:
                model.half()
            model.eval()

            # --- 2. 打开两个视频文件 ---
            cap_rgb = cv2.VideoCapture(self.rgb_video_path)
            cap_ir = cv2.VideoCapture(self.ir_video_path)

            if not cap_rgb.isOpened():
                raise Exception(f"无法打开可见光视频文件: {self.rgb_video_path}")
            if not cap_ir.isOpened():
                raise Exception(f"无法打开热红外视频文件: {self.ir_video_path}")

            # --- 3. 主循环：同步读取并处理帧 ---
            while self.running:
                self.mutex.lock()
                if self.paused:
                    self.mutex.unlock()
                    self.msleep(50)  # 避免CPU占用过高
                    continue
                self.mutex.unlock()

                # 同时读取两路视频的下一帧
                ret_rgb, frame_rgb = cap_rgb.read()
                ret_ir, frame_ir_raw = cap_ir.read()

                # 如果任一视频流结束，则退出循环
                if not ret_rgb or not ret_ir:
                    break


                if len(frame_ir_raw.shape) == 2:
                    # 情况1: 原始帧是单通道灰度图 (shape: H, W)
                    # 直接转换为3通道BGR
                    frame_ir = cv2.cvtColor(frame_ir_raw, cv2.COLOR_GRAY2BGR)
                elif len(frame_ir_raw.shape) == 3:
                    if frame_ir_raw.shape[2] == 1:
                        # 情况2: 原始帧是(H, W, 1)的伪彩色图
                        # 先squeeze掉最后一个维度变成(H, W)，再转BGR
                        frame_ir = cv2.cvtColor(frame_ir_raw.squeeze(), cv2.COLOR_GRAY2BGR)
                    elif frame_ir_raw.shape[2] == 3:
                        # 情况3: 原始帧是3通道彩色图 (shape: H, W, 3)
                        # 先转为灰度，再转回3通道BGR，以保留单通道信息
                        gray = cv2.cvtColor(frame_ir_raw, cv2.COLOR_BGR2GRAY)
                        frame_ir = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    elif frame_ir_raw.shape[2] == 4:
                        # 情况4: 原始帧是4通道图 (如RGBA)
                        # 先转为3通道BGR，再按情况3处理
                        bgr = cv2.cvtColor(frame_ir_raw, cv2.COLOR_BGRA2BGR)
                        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                        frame_ir = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    else:
                        raise ValueError(f"不支持的热红外视频通道数: {frame_ir_raw.shape[2]}")
                else:
                    raise ValueError(f"无法解析的热红外视频帧形状: {frame_ir_raw.shape}")

                # 现在 frame_ir 已经被安全地初始化为3通道BGR格式
                # 可以继续进行后续的预处理...
                # --- 4. 预处理 (指定为 640x480) ---
                # 定义统一的目标尺寸 (height, width)
                target_h, target_w = 640, 640
                target_shape = (target_h, target_w)  # 注意：(H, W) 顺序

                # 对两个帧应用 letterbox，并强制输出为 target_shape
                rgb_letterboxed = letterbox(frame_rgb, target_shape, stride=stride, auto=False)[0]
                ir_letterboxed = letterbox(frame_ir, target_shape, stride=stride, auto=False)[0]

                # 转换颜色通道: BGR to RGB, 并调整维度顺序 HWC -> CHW
                rgb_letterboxed = rgb_letterboxed[:, :, ::-1].transpose(2, 0, 1)
                ir_letterboxed = ir_letterboxed[:, :, ::-1].transpose(2, 0, 1)

                # 关键步骤: 将 numpy array 转换为 torch tensor
                img_rgb = torch.from_numpy(np.ascontiguousarray(rgb_letterboxed)).to(device)
                img_ir = torch.from_numpy(np.ascontiguousarray(ir_letterboxed)).to(device)

                # 转换数据类型并归一化
                img_rgb = img_rgb.half() if half else img_rgb.float()
                img_ir = img_ir.half() if half else img_ir.float()
                img_rgb /= 255.0
                img_ir /= 255.0

                # 添加 batch 维度
                if img_rgb.ndimension() == 3:
                    img_rgb = img_rgb.unsqueeze(0)
                    img_ir = img_ir.unsqueeze(0)
                try:
                    # --- 推理 ---
                    with torch.no_grad():
                        pred = model(img_rgb, img_ir)[0]

                    # --- NMS ---
                    pred_nms = non_max_suppression(
                        pred,
                        conf_thres=self.conf_threshold,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False
                    )

                    # --- 准备绘制 ---
                    annotated_frame_rgb = frame_rgb.copy()
                    annotated_frame_ir = frame_ir.copy()
                    detections = []

                    det = pred_nms[0] if len(pred_nms) > 0 else torch.empty((0, 6), device=device)

                    if len(det):
                        # 缩放坐标
                        det[:, :4] = scale_coords(img_rgb.shape[2:], det[:, :4], frame_rgb.shape).round()
                        for *xyxy, conf, cls in det:
                            x1, y1, x2, y2 = map(int, xyxy)
                            label_name = names[int(cls)]
                            detections.append({
                                'box': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'label': label_name
                            })
                            plot_one_box(xyxy, annotated_frame_rgb, label=f'{label_name} {conf:.2f}',
                                         color=colors(int(cls), True), line_thickness=2)
                            plot_one_box(xyxy, annotated_frame_ir, label=f'{label_name} {conf:.2f}',
                                         color=colors(int(cls), True), line_thickness=2)

                    # --- 发送所有信号（统一在这里）---
                    self.frame_ready.emit(annotated_frame_rgb.copy(), annotated_frame_ir.copy())
                    self.detection_info.emit(detections)
                    # 如果你还需要 detection_result 信号，也可以在这里 emit
                    # self.detection_result.emit(frame_rgb.copy(), det.cpu().numpy())

                except Exception as e:
                    print(f"处理帧时出错: {e}")
                    # 出错时发送原始帧（无检测框）
                    self.frame_ready.emit(frame_rgb.copy(), frame_ir.copy())
                    self.detection_info.emit([])
                    # 可选：也 emit detection_result
                    # self.detection_result.emit(frame_rgb.copy(), np.array([]))

                # 注意：不再有 "continue"，也不再有 try 块外的 pred_nms 使用
                self.msleep(30)

            # --- 8. 清理资源 ---
            cap_rgb.release()
            cap_ir.release()
            self.finished_signal.emit()

        except Exception as e:
            import traceback
            self.error_occurred.emit(f"双模态视频检测出错: {str(e)}\n{traceback.format_exc()}")

    def pause_resume(self):
        """暂停/继续检测"""
        self.mutex.lock()
        self.paused = not self.paused
        self.mutex.unlock()

    def stop(self):
        """停止检测"""
        self.running = False

# --- 替换掉你代码中旧的 DetectionWorker 类 ---
class DetectionWorker(QThread):
    """图片检测工作线程（混合模式：单流用YOLO，双流用PyTorch）"""
    progress = pyqtSignal(int)
    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_path, image_sources, detection_mode, conf_threshold=0.5):
        super().__init__()
        self.model_path = model_path
        self.image_sources = image_sources
        self.detection_mode = detection_mode
        self.conf_threshold = conf_threshold

    def run(self):
        try:
            # === 双流模式：使用底层PyTorch ===
            if self.detection_mode == "both":
                import torch
                from models.experimental import attempt_load
                from utils.general import check_img_size, scale_coords
                from ultralytics.utils.ops import non_max_suppression
                from utils.datasets import letterbox
                from utils.torch_utils import select_device
                from utils.plots import colors, plot_one_box

                device = select_device('')
                half = device.type != 'cpu'

                LOGGER.info(f"正在从 {self.model_path} 加载双流模型...")
                model = attempt_load(self.model_path, map_location=device)
                stride = int(model.stride.max())
                imgsz = 640
                imgsz = check_img_size(imgsz, s=stride)
                names = model.module.names if hasattr(model, 'module') else model.names
                if half:
                    model.half()
                model.eval()

                total = len(self.image_sources)
                results = []
                for i, (rgb_path, ir_path) in enumerate(self.image_sources):
                    # 读取图像
                    im0_rgb = cv2.imread(rgb_path)
                    im0_ir_gray = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
                    if im0_rgb is None or im0_ir_gray is None:
                        continue
                    im0_ir = cv2.cvtColor(im0_ir_gray, cv2.COLOR_GRAY2BGR)

                    # 预处理
                    img_rgb = letterbox(im0_rgb, imgsz, stride=stride)[0]
                    img_ir = letterbox(im0_ir, imgsz, stride=stride)[0]
                    img_rgb = img_rgb.transpose((2, 0, 1))[::-1]
                    img_ir = img_ir.transpose((2, 0, 1))[::-1]
                    img_rgb = torch.from_numpy(np.ascontiguousarray(img_rgb)).to(device)
                    img_ir = torch.from_numpy(np.ascontiguousarray(img_ir)).to(device)
                    img_rgb = img_rgb.half() if half else img_rgb.float()
                    img_ir = img_ir.half() if half else img_ir.float()
                    img_rgb /= 255.0
                    img_ir /= 255.0
                    if img_rgb.ndimension() == 3:
                        img_rgb = img_rgb.unsqueeze(0)
                        img_ir = img_ir.unsqueeze(0)

                    # 推理
                    with torch.no_grad():
                        pred = model(img_rgb, img_ir)[0]

                    # 后处理
                    pred_nms = non_max_suppression(pred, conf_thres=self.conf_threshold, iou_thres=0.45, classes=None,
                                                   agnostic=False)
                    annotated_img_rgb = im0_rgb.copy()
                    annotated_img_ir = im0_ir.copy()
                    detections = []
                    det = pred_nms[0]
                    if det is not None and len(det):
                        det[:, :4] = scale_coords(img_rgb.shape[2:], det[:, :4], im0_rgb.shape).round()
                        for *xyxy, conf, cls in det:
                            x1, y1, x2, y2 = map(int, xyxy)
                            label_name = names[int(cls)]
                            detections.append({
                                'box': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'label': label_name
                            })
                            # 在两张图上都画框
                            plot_one_box(xyxy, annotated_img_rgb, label=f'{label_name} {conf:.2f}',
                                         color=colors(int(cls), True), line_thickness=2)
                            plot_one_box(xyxy, annotated_img_ir, label=f'{label_name} {conf:.2f}',
                                         color=colors(int(cls), True), line_thickness=2)

                    result_dict = {
                        'image_path': rgb_path,  # 可见光路径
                        'ir_path': ir_path,  # 新增：热红外路径
                        'detections': detections,
                        'processed_rgb': annotated_img_rgb,  # 新增：处理后的可见光图
                        'processed_ir': annotated_img_ir,  # 新增：处理后的热红外图
                        'original_shape': im0_rgb.shape,
                        'type': 'image'
                    }
                    results.append(result_dict)
                    self.progress.emit(int((i + 1) / total * 100))

                self.result_ready.emit(results)

            # === 单流模式：继续使用 ultralytics.YOLO ===
            else:
                if not YOLO_AVAILABLE:
                    raise ImportError("YOLO库未安装")

                model = YOLO(self.model_path)
                total = len(self.image_sources)
                results = []

                for i, img_path in enumerate(self.image_sources):
                    if not os.path.exists(img_path):
                        continue

                    # 对于 thermal_only 模式，YOLO 会自动将单通道图转为三通道
                    results_list = model(img_path, conf=self.conf_threshold, verbose=False)
                    result = results_list[0]

                    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
                    confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
                    cls_ids = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
                    names = result.names

                    original_img = cv2.imread(img_path)
                    if original_img is None:
                        continue

                    annotated_img = result.plot()

                    detections = []
                    for j in range(len(boxes)):
                        x1, y1, x2, y2 = map(int, boxes[j])
                        conf = float(confs[j])
                        cls_id = int(cls_ids[j])
                        label = names[cls_id]
                        detections.append({
                            'box': [x1, y1, x2, y2],
                            'confidence': conf,
                            'label': label
                        })

                    result_dict = {
                        'image_path': img_path,
                        'detections': detections,
                        'processed_image': annotated_img,
                        'original_shape': original_img.shape,
                        'type': 'image'
                    }
                    results.append(result_dict)
                    self.progress.emit(int((i + 1) / total * 100))

                self.result_ready.emit(results)

        except Exception as e:
            import traceback
            self.error_occurred.emit(f"错误: {str(e)}\n{traceback.format_exc()}")


# --- DetectionWorker 类结束 ---

# 注意: 你需要确保 `letterbox`, `plot_one_box`, `colors` 这些函数可以从你的项目 utils 中导入。
# 它们通常在 `utils.datasets` 和 `utils.plots` 中。


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv11实时视频检测系统")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 14px;
                margin: 4px 2px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QLabel {
                font-size: 12px;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            QProgressBar {
                border: 1px solid #aaa;
                border-radius: 5px;
                text-align: center;
                font-size: 12px;
            }
            QTextEdit {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 3px;
                font-family: Consolas, monospace;
            }
            QSlider {
                height: 20px;
            }
        """)

        # 初始化变量
        self.results = []
        self.current_result_index = 0
        self.current_worker = None
        self.is_video_mode = False
        self.is_camera_mode = False
        self.current_detection_mode = "visible_only"
        self.setup_ui()

    def on_mode_changed(self, text):
        mode_map = {
            "仅可见光": "visible_only",
            "仅热红外": "thermal_only",
            "可见光+热红外": "both"
        }
        self.current_detection_mode = mode_map[text]
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 标题
        title_label = QLabel("YOLOv11实时视频检测系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333; padding: 10px;")
        main_layout.addWidget(title_label)

        # 模型和数据选择区域
        selection_group = QGroupBox("模型和数据选择")
        selection_layout = QVBoxLayout(selection_group)

        # 模型选择
        model_layout = QHBoxLayout()
        self.model_label = QLabel("模型路径:")
        self.model_path_edit = QLabel("未选择模型")
        self.model_path_edit.setStyleSheet(
            "background-color: white; border: 1px solid #ccc; padding: 5px; border-radius: 3px;")
        self.model_button = QPushButton("选择模型")
        self.model_button.clicked.connect(self.select_model)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(self.model_button)
        selection_layout.addLayout(model_layout)

        # 文件选择
        file_layout = QHBoxLayout()
        self.file_label = QLabel("检测文件:")
        self.file_path_edit = QLabel("未选择文件")
        self.file_path_edit.setStyleSheet(
            "background-color: white; border: 1px solid #ccc; padding: 5px; border-radius: 3px;")
        self.file_button = QPushButton("选择文件")
        self.file_button.clicked.connect(self.select_files)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.file_button)
        selection_layout.addLayout(file_layout)

        main_layout.addWidget(selection_group)

        # 检测设置区域
        settings_group = QGroupBox("检测设置")
        settings_layout = QVBoxLayout(settings_group)

        # 检测模式选择
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("检测模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["仅可见光", "仅热红外", "可见光+热红外"])
        self.mode_combo.setCurrentIndex(0)
        mode_layout.addWidget(self.mode_combo)
        settings_layout.addLayout(mode_layout)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        # 检测类型选择
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("检测类型:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["图片检测", "视频检测", "摄像头检测"])
        self.type_combo.setCurrentIndex(0)
        self.type_combo.currentIndexChanged.connect(self.on_detection_type_changed)
        type_layout.addWidget(self.type_combo)
        settings_layout.addLayout(type_layout)

        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度阈值:"))
        self.conf_spinbox = QSpinBox()
        self.conf_spinbox.setRange(1, 100)
        self.conf_spinbox.setValue(50)
        self.conf_spinbox.setSuffix("%")
        conf_layout.addWidget(self.conf_spinbox)
        settings_layout.addLayout(conf_layout)

        main_layout.addWidget(settings_group)

        # 控制按钮区域
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("开始检测")
        self.start_button.clicked.connect(self.start_detection)
        self.pause_button = QPushButton("暂停/继续")
        self.pause_button.clicked.connect(self.pause_resume_detection)
        self.pause_button.setEnabled(False)
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        self.save_button = QPushButton("保存结果")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)

        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.save_button)
        main_layout.addLayout(control_layout)

        # 进度条（仅用于图片检测）
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 结果显示区域
        result_layout = QHBoxLayout()

        # 左侧原图/视频
        left_group = QGroupBox("原始画面（或可见光画面）")
        left_layout = QVBoxLayout(left_group)
        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(400, 300)
        self.original_image_label.setStyleSheet("border: 1px solid #ccc;")
        left_layout.addWidget(self.original_image_label)
        result_layout.addWidget(left_group)

        # 右侧检测结果
        right_group = QGroupBox("检测结果")
        right_layout = QVBoxLayout(right_group)
        self.result_image_label = QLabel()
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setMinimumSize(400, 300)
        self.result_image_label.setStyleSheet("border: 1px solid #ccc;")
        right_layout.addWidget(self.result_image_label)
        result_layout.addWidget(right_group)

        main_layout.addLayout(result_layout)

        # 图片检测的结果导航
        self.nav_widget = QWidget()
        nav_layout = QHBoxLayout(self.nav_widget)
        self.prev_button = QPushButton("上一张")
        self.prev_button.clicked.connect(self.show_prev_result)
        self.next_button = QPushButton("下一张")
        self.next_button.clicked.connect(self.show_next_result)
        self.index_label = QLabel("0/0")
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.index_label)
        nav_layout.addWidget(self.next_button)
        main_layout.addWidget(self.nav_widget)

        # 日志区域
        log_group = QGroupBox("日志信息")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group)

        # 初始化状态
        self.update_navigation()
        self.on_detection_type_changed(0)  # 初始化UI状态

        if not YOLO_AVAILABLE:
            self.log_text.append("警告: 未安装ultralytics库，无法进行真实检测")
            self.log_text.append("请运行: pip install ultralytics")

    def on_detection_type_changed(self, index):
        """检测类型改变时更新UI"""
        is_image_mode = (index == 0)

        # 显示/隐藏导航控件
        self.nav_widget.setVisible(is_image_mode)

        # 更新按钮文本
        if index == 0:
            self.file_button.setText("选择图片")
        elif index == 1:
            self.file_button.setText("选择视频")
        else:
            self.file_button.setText("使用摄像头")

    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pt *.onnx *.pb);;所有文件 (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            self.log_text.append(f"模型已选择: {os.path.basename(file_path)}")

    def select_files(self):
        current_type = self.type_combo.currentIndex()

        if current_type != 0:  # 非图片检测，走原逻辑
            if current_type == 1:  # 视频
                # --- 关键修改：判断是否为双模态视频 ---
                if self.current_detection_mode == "both":
                    # 让用户分别选择可见光和热红外视频
                    rgb_path, _ = QFileDialog.getOpenFileName(
                        self, "选择可见光视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv);;所有文件 (*)"
                    )
                    if not rgb_path:
                        return
                    ir_path, _ = QFileDialog.getOpenFileName(
                        self, "选择对应的热红外视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv);;所有文件 (*)"
                    )
                    if not ir_path:
                        return
                    # 存储为元组，与图片检测的 paired list 格式保持一致
                    self.selected_files = [(rgb_path, ir_path)]
                    self.file_path_edit.setText(f"已选择双模态视频对")
                else:
                    # 单模态视频逻辑不变
                    file_paths, _ = QFileDialog.getOpenFileNames(
                        self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv);;所有文件 (*)"
                    )
                    if file_paths:
                        self.selected_files = file_paths
                        self.file_path_edit.setText(f"已选择 {len(file_paths)} 个视频")

            else:  # 摄像头
                self.file_path_edit.setText("使用默认摄像头 (0)")
                self.selected_files = ["camera_0"]
                self.log_text.append("将使用默认摄像头进行检测")
            return

        # === 图片检测逻辑 ===
        current_mode_text = self.mode_combo.currentText()

        if current_mode_text == "可见光+热红外":
            # 弹出三种选项
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("选择多光谱匹配方式")
            msg_box.setText("请选择图像配对策略：")
            btn_folder = msg_box.addButton("📁 双文件夹：匹配同名图像", QMessageBox.ActionRole)
            btn_single = msg_box.addButton("🖼️ 手动选图：同目录找 _ir 配对", QMessageBox.ActionRole)
            btn_custom = msg_box.addButton("🔍 自由组合：选可见光 + 指定IR文件夹", QMessageBox.ActionRole)
            msg_box.addButton(QMessageBox.Cancel)
            msg_box.exec_()

            clicked_btn = msg_box.clickedButton()

            if clicked_btn == btn_folder:
                # ===== 方式A：双文件夹同名匹配 =====
                rgb_dir = QFileDialog.getExistingDirectory(self, "选择可见光图像文件夹（RGB）")
                if not rgb_dir: return
                ir_dir = QFileDialog.getExistingDirectory(self, "选择热红外图像文件夹（IR）")
                if not ir_dir: return

                image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

                def list_image_files(folder):
                    files = {}
                    for f in os.listdir(folder):
                        ext = os.path.splitext(f.lower())[1]
                        if ext in image_exts:
                            name_no_ext = os.path.splitext(f)[0]
                            files[name_no_ext] = os.path.join(folder, f)
                    return files

                rgb_files = list_image_files(rgb_dir)
                ir_files = list_image_files(ir_dir)
                common = set(rgb_files.keys()) & set(ir_files.keys())
                if not common:
                    self.log_text.append("❌ 双文件夹模式：未找到同名图像对！")
                    return
                paired = [(rgb_files[n], ir_files[n]) for n in sorted(common)]
                self.selected_files = paired
                self.file_path_edit.setText(f"已匹配 {len(paired)} 对（双文件夹模式）")
                self.log_text.append("✅ 双文件夹匹配完成")

            elif clicked_btn == btn_single:
                # ===== 方式B：同目录 _vis / _ir 配对 =====
                vis_paths, _ = QFileDialog.getOpenFileNames(
                    self, "选择可见光图片", "", "图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
                )
                if not vis_paths: return

                paired = []
                for vis in vis_paths:
                    base = os.path.splitext(vis)[0]
                    if base.endswith('_vis'):
                        ir_guess = base.replace('_vis', '_ir') + os.path.splitext(vis)[1]
                    else:
                        ir_guess = base + '_ir' + os.path.splitext(vis)[1]

                    if os.path.exists(ir_guess):
                        paired.append((vis, ir_guess))
                    else:
                        self.log_text.append(f"⚠️ 跳过 {os.path.basename(vis)}：未找到同目录红外图")

                if not paired:
                    self.log_text.append("❌ 手动配对模式：未找到任何有效对")
                    return
                self.selected_files = paired
                self.file_path_edit.setText(f"已加载 {len(paired)} 对（同目录配对模式）")
                self.log_text.append("✅ 同目录配对完成")

            elif clicked_btn == btn_custom:
                # ===== 方式C：自由组合 —— 手动选可见光 + 指定IR文件夹 =====
                vis_paths, _ = QFileDialog.getOpenFileNames(
                    self, "选择可见光图片（可跨目录）", "", "图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
                )
                if not vis_paths: return

                ir_dir = QFileDialog.getExistingDirectory(self, "选择红外图像文件夹（用于匹配）")
                if not ir_dir: return

                # 构建 IR 文件名索引（不含扩展名 -> 全路径）
                image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
                ir_index = {}
                for f in os.listdir(ir_dir):
                    ext = os.path.splitext(f.lower())[1]
                    if ext in image_exts:
                        name_no_ext = os.path.splitext(f)[0]
                        ir_index[name_no_ext] = os.path.join(ir_dir, f)

                # 尝试为每个可见光图匹配 IR 图
                paired = []
                for vis in vis_paths:
                    vis_name_no_ext = os.path.splitext(os.path.basename(vis))[0]
                    # 支持两种命名：原名 或 去掉 _vis
                    candidates = [vis_name_no_ext]
                    if vis_name_no_ext.endswith('_vis'):
                        candidates.append(vis_name_no_ext[:-4])  # 去掉 "_vis"

                    matched_ir = None
                    for cand in candidates:
                        if cand in ir_index:
                            matched_ir = ir_index[cand]
                            break

                    if matched_ir:
                        paired.append((vis, matched_ir))
                    else:
                        self.log_text.append(f"⚠️ 未在IR文件夹中找到 {vis_name_no_ext} 的红外图")

                if not paired:
                    self.log_text.append("❌ 自由组合模式：未匹配到任何图像对")
                    return

                self.selected_files = paired
                self.file_path_edit.setText(f"已匹配 {len(paired)} 对（自由组合模式）")
                self.log_text.append(f"✅ 从 {len(vis_paths)} 张可见光图中匹配成功 {len(paired)} 对")

            else:
                return  # 用户取消

        else:
            # 单模态逻辑不变
            label = "选择可见光图片" if current_mode_text == "仅可见光" else "选择热红外图片"
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, label, "", "图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
            )
            if file_paths:
                self.selected_files = file_paths
                self.file_path_edit.setText(f"已选择 {len(file_paths)} 张图片")
                self.log_text.append(f"已选择 {len(file_paths)} 张{current_mode_text}图片")

    def start_detection(self):
        if not YOLO_AVAILABLE:
            self.log_text.append("错误: 未安装ultralytics库，无法进行检测")
            return

        model_path = self.model_path_edit.text()
        if model_path == "未选择模型":
            self.log_text.append("请先选择模型文件")
            return

        current_type = self.type_combo.currentIndex()
        if current_type != 2 and (not hasattr(self, 'selected_files') or len(self.selected_files) == 0):
            self.log_text.append("请先选择检测文件")
            return

        # 获取检测参数
        conf_threshold = self.conf_spinbox.value() / 100.0

        if current_type == 0:  # 图片检测
            self.start_image_detection(model_path, conf_threshold)
        elif current_type == 1:  # 视频检测
            # --- 关键修改：根据模式选择不同的启动方法 ---
            if self.current_detection_mode == "both":
                self.start_dual_stream_video_detection(model_path, conf_threshold)
            else:
                self.start_video_detection(model_path, conf_threshold)
        else:  # 摄像头检测
            self.start_camera_detection(model_path, conf_threshold)

    def start_dual_stream_video_detection(self, model_path, conf_threshold):
        """启动双模态视频检测"""
        self.is_video_mode = True
        self.is_camera_mode = False

        # 从 selected_files 中取出视频对
        rgb_path, ir_path = self.selected_files[0]
        self.current_worker = DualStreamVideoDetectionWorker(model_path, rgb_path, ir_path, conf_threshold)
        self.current_worker.frame_ready.connect(self.on_dual_stream_video_frame_ready) # 注意：连接到新回调
        self.current_worker.detection_info.connect(self.on_detection_info)
        self.current_worker.finished_signal.connect(self.on_video_finished)
        self.current_worker.error_occurred.connect(self.on_detection_error)
        self.current_worker.start()

        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.log_text.append(f"开始双模态视频检测: RGB={os.path.basename(rgb_path)}, IR={os.path.basename(ir_path)}")

    def on_dual_stream_video_frame_ready(self, processed_rgb_frame, processed_ir_frame):
        """处理双模态视频帧"""
        # 左侧显示处理后的可见光帧
        self.display_frame(processed_rgb_frame, self.original_image_label)
        # 右侧显示处理后的热红外帧
        self.display_frame(processed_ir_frame, self.result_image_label)

    def start_image_detection(self, model_path, conf_threshold):
        """启动图片检测"""
        self.is_video_mode = False
        self.is_camera_mode = False

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.start_button.setEnabled(False)
        self.save_button.setEnabled(False)

        mode_text = self.mode_combo.currentText()
        detection_modes = {"仅可见光": "visible_only", "仅热红外": "thermal_only", "可见光+热红外": "both"}
        detection_mode = detection_modes[mode_text]
        self.current_worker = DetectionWorker(model_path, self.selected_files, detection_mode, conf_threshold)
        self.current_worker.progress.connect(self.progress_bar.setValue)
        self.current_worker.result_ready.connect(self.on_image_detection_complete)
        self.current_worker.error_occurred.connect(self.on_detection_error)
        self.current_worker.start()

        self.log_text.append("开始图片检测...")

    def start_video_detection(self, model_path, conf_threshold):
        """启动视频检测"""
        self.is_video_mode = True
        self.is_camera_mode = False

        video_path = self.selected_files[0]  # 只处理第一个视频
        self.current_worker = VideoDetectionWorker(model_path, video_path, conf_threshold)
        self.current_worker.frame_ready.connect(self.on_video_frame_ready)
        self.current_worker.detection_info.connect(self.on_detection_info)
        self.current_worker.finished_signal.connect(self.on_video_finished)
        self.current_worker.error_occurred.connect(self.on_detection_error)
        self.current_worker.start()

        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)  # 实时检测不支持批量保存
        self.progress_bar.setVisible(False)
        self.log_text.append(f"开始视频检测: {os.path.basename(video_path)}")

    def start_camera_detection(self, model_path, conf_threshold):
        """启动摄像头检测"""
        self.is_video_mode = False
        self.is_camera_mode = True

        camera_index = 0  # 默认摄像头
        self.current_worker = CameraDetectionWorker(model_path, camera_index, conf_threshold)
        self.current_worker.frame_ready.connect(self.on_video_frame_ready)
        self.current_worker.detection_info.connect(self.on_detection_info)
        self.current_worker.error_occurred.connect(self.on_detection_error)
        self.current_worker.start()

        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)  # 实时检测不支持批量保存
        self.progress_bar.setVisible(False)
        self.log_text.append("开始摄像头检测...")

    def on_video_frame_ready(self, original_frame, detected_frame):
        """处理视频帧"""
        # 显示原始帧
        self.display_frame(original_frame, self.original_image_label)
        # 显示检测结果帧
        self.display_frame(detected_frame, self.result_image_label)

    def on_detection_info(self, detections):
        """处理检测信息"""
        pass  # 可以在这里添加实时检测信息显示

    def display_frame(self, frame, label):
        """在标签中显示帧"""
        if frame is None or frame.size == 0:
            return

        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def pause_resume_detection(self):
        """暂停/继续检测"""
        if self.current_worker and hasattr(self.current_worker, 'pause_resume'):
            self.current_worker.pause_resume()

    def stop_detection(self):
        """停止检测"""
        if self.current_worker:
            if hasattr(self.current_worker, 'stop'):
                self.current_worker.stop()
            self.current_worker.quit()
            self.current_worker.wait()
            self.current_worker = None

        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(False)

        # 清空显示
        self.original_image_label.clear()
        self.result_image_label.clear()
        self.original_image_label.setText("原始画面")
        self.result_image_label.setText("检测结果")

    def on_image_detection_complete(self, results):
        """图片检测完成"""
        self.results = results
        self.current_result_index = 0
        self.progress_bar.setVisible(False)
        self.start_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.log_text.append(f"图片检测完成! 共处理 {len(results)} 张图片")
        self.update_navigation()
        self.show_current_result()

    def on_video_finished(self):
        """视频检测完成"""
        self.log_text.append("视频检测完成!")
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(False)

    def on_detection_error(self, error_msg):
        """检测出错"""
        self.log_text.append(f"检测出错: {error_msg}")
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)

    def show_current_result(self):
        if not self.results or self.current_result_index >= len(self.results):
            return

        result = self.results[self.current_result_index]

        # --- 关键修改：根据检测模式决定如何显示 ---
        detection_mode = self.current_detection_mode

        if detection_mode == "both":
            # 双模态：左-可见光，右-热红外
            # 显示左侧：可见光原图 -> 改为显示带框的可见光图
            processed_rgb = result['processed_rgb']
            height, width, channel = processed_rgb.shape
            bytes_per_line = 3 * width
            q_img_rgb = QImage(processed_rgb.data, width, height, bytes_per_line, QImage.Format_BGR888)
            pixmap_rgb = QPixmap.fromImage(q_img_rgb)
            self.original_image_label.setPixmap(pixmap_rgb.scaled(
                self.original_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

            # 显示右侧：热红外图（带框）
            processed_ir = result['processed_ir']
            height, width, channel = processed_ir.shape
            bytes_per_line = 3 * width
            q_img_ir = QImage(processed_ir.data, width, height, bytes_per_line, QImage.Format_BGR888)
            pixmap_ir = QPixmap.fromImage(q_img_ir)
            self.result_image_label.setPixmap(pixmap_ir.scaled(
                self.result_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

        else:
            # 单模态：保持原有逻辑
            img_path = result['image_path']
            # 显示原图
            pixmap = self.load_image_to_pixmap(img_path)
            self.original_image_label.setPixmap(pixmap.scaled(
                self.original_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

            # 显示检测结果
            processed_img = result['processed_image']
            height, width, channel = processed_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(processed_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)
            self.result_image_label.setPixmap(pixmap.scaled(
                self.result_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

        # 更新索引标签
        self.index_label.setText(f"{self.current_result_index + 1}/{len(self.results)}")

        # 记录检测结果
        detections = result['detections']
        self.log_text.append(f"第{self.current_result_index + 1}张图片检测到 {len(detections)} 个目标")
        for det in detections:
            self.log_text.append(f"  - {det['label']}: {det['confidence']:.2f}")

    def load_image_to_pixmap(self, img_path):
        if not os.path.exists(img_path):
            return QPixmap()

        img = cv2.imread(img_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channel = img_rgb.shape
            bytes_per_line = 3 * width
            q_img = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(q_img)
        return QPixmap()

    def update_navigation(self):
        has_results = len(self.results) > 0
        self.prev_button.setEnabled(has_results and self.current_result_index > 0)
        self.next_button.setEnabled(has_results and self.current_result_index < len(self.results) - 1)
        self.index_label.setText(f"{self.current_result_index + 1}/{len(self.results) if has_results else 0}")

    def show_prev_result(self):
        if self.current_result_index > 0:
            self.current_result_index -= 1
            self.show_current_result()
            self.update_navigation()

    def show_next_result(self):
        if self.current_result_index < len(self.results) - 1:
            self.current_result_index += 1
            self.show_current_result()
            self.update_navigation()

    def save_results(self):
        if not self.results:
            return

        save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not save_dir:
            return

        for i, result in enumerate(self.results):
            output_path = os.path.join(save_dir, f"detection_result_{i + 1}.jpg")
            processed_img = result['processed_image']
            cv2.imwrite(output_path, processed_img)
        self.log_text.append(f"结果已保存到: {save_dir}")

    def closeEvent(self, event):
        # 清理资源
        if self.current_worker:
            if hasattr(self.current_worker, 'stop'):
                self.current_worker.stop()
            self.current_worker.quit()
            self.current_worker.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())