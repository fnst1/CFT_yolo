# dual_stream_yolo.py
"""
基于您提供的代码，封装一个用于双光谱（RGB+IR）目标检测的 YOLO 模型。
"""

import torch

import torch.nn as nn
from pathlib import Path
from copy import deepcopy
import yaml
import math

# --- 关键：确保这些依赖项能被正确导入 ---
# 您需要根据您的项目结构调整这些导入路径。
# 这些模块通常来自 Ultralytics YOLO 或您自己的项目。
try:
    from models.common import *  # 包含 Conv, DWConv, Bottleneck 等基础模块
    from utils.general import LOGGER, make_divisible
    from utils.torch_utils import initialize_weights
    from ultralytics.nn.modules.block import C2f, SPPF, DFL
    from ultralytics.utils.tal import dist2bbox, make_anchors
except ImportError as e:
    raise ImportError(f"缺少必要的模块: {e}. 请确保您的 Python 路径包含了 YOLO 项目的根目录。")


class DualStreamDetect(nn.Module):
    """您自定义的检测头，完全保留您的逻辑。"""
    dynamic = False
    export = False
    end2end = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def _inference(self, x):
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

    def decode_bboxes(self, bboxes, anchors):
        return dist2bbox(bboxes, anchors, xywh=True, dim=1)


def parse_dual_stream_model(d, ch):
    """解析模型配置，支持 `from: -4` 的特殊标记。"""
    LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    nc, gd, gw = d['nc'], d['depth_multiple'], d['width_multiple']
    layers, save, c2 = [], [], ch[-1]

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n
        if m in [Conv, DWConv, Bottleneck, C2f, SPPF]:
            # --- 核心修改点：处理 f == -4 的情况 ---
            if f == -4:
                c1, c2 = 3, args[0]  # IR 输入通道为 3
            else:
                c1, c2 = ch[f], args[0]

            c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in [C2f]:
                args.insert(2, n)
                n = 1
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is DualStreamDetect:
            args.append([ch[x] for x in f])
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        m_.i, m_.f, m_.type = i, f, t
        LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{sum(x.numel() for x in m_.parameters()):10.0f}  {t:<45}{str(args):<30}')
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class DualStreamYOLOModel(nn.Module):
    """
    双流 YOLO 模型主类。
    使用方法:
        model = DualStreamYOLOModel('your_dual_stream_config.yaml')
        output = model(rgb_tensor, ir_tensor)
    """

    def __init__(self, cfg='yolov11n-dual.yaml', ch=3, nc=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"覆盖模型配置中的类别数 {self.yaml['nc']} 为 {nc}")
            self.yaml['nc'] = nc

        # 使用我们自定义的解析函数
        self.model, self.save = parse_dual_stream_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]
        self.inplace = self.yaml.get('inplace', True)

        # 初始化检测头
        m = self.model[-1]
        if isinstance(m, DualStreamDetect):
            m.inplace = self.inplace
            m.stride = torch.Tensor([8.0, 16.0, 32.0])
            self.stride = m.stride
            m.bias_init()

        initialize_weights(self)

    def forward(self, x_rgb, x_ir=None, augment=False, profile=False, visualize=False):
        """
        前向传播函数。

        Args:
            x_rgb (torch.Tensor): 可见光图像, 形状 [B, 3, H, W]
            x_ir (torch.Tensor): 红外图像, 形状 [B, 3, H, W]。如果为 None，则退化为单流模式。
        """
        if x_ir is None:
            return self._forward_single(x_rgb, profile, visualize)
        return self._forward_dual(x_rgb, x_ir, profile, visualize)

    def _forward_single(self, x, profile=False, visualize=False):
        """单流模式（兼容性）"""
        y = []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return x

    def _forward_dual(self, x_rgb, x_ir, profile=False, visualize=False):
        """双流模式：核心逻辑"""
        y = []
        for m in self.model:
            if m.f != -1:
                # 如果不是来自上一层 (-1)，则从保存的特征图中获取
                x = y[m.f] if isinstance(m.f, int) else [x_rgb if j == -1 else y[j] for j in m.f]
            else:
                x = x_rgb

            # --- 关键：路由到 IR 流 ---
            if m.f == -4:
                x = m(x_ir)  # 将红外图像送入此层
            else:
                x = m(x)  # 处理主（可见光）流或融合后的特征

            y.append(x if m.i in self.save else None)
        return x