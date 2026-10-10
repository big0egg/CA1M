# CA-1M 与 Cubify Anything

本仓库提供 Cubify Transformer（CuTR）的公开实现，以及配套的 CA-1M 数据集和衍生的 CA-VQA 标注。

本文档是原项目说明的中文版本，并补充当前阿里云 PAI DSW / PPU 环境的操作方案。软件名称、命令、字段名及 BibTeX 中的原始论文信息保留原样。

**当前目标：私人科研，先跑通官方 CA-1M 样例和 RGB-D 模型。** 原生 DSW 的 PPU 和基础 CUDA 运算已经通过检查；官方权重的完整推理仍待执行。请优先阅读[阿里云 DSW / PPU 适配操作](#dsw-ppu)。

## 最新消息

**2025 年 9 月：** 更新了 CA-VQA 的代码和链接。CA-1M 数据及其可视化已加入用于生成 CA-VQA 的**描述文本和属性**。

## 相关论文

**Apple**

[Cubify Anything：扩展室内三维物体检测的规模](https://arxiv.org/abs/2412.04458)

Justin Lazarow、David Griffiths、Gefen Kohavi、Francisco Crespo、Afshin Dehghan

**CVPR 2025**

![项目概览](teaser.jpg?raw=true "项目概览")

[MM-Spatial：探索多模态大语言模型的三维空间理解能力](https://arxiv.org/abs/2503.13111)

Erik Daxberger、Nina Wenzel、David Griffiths、Haiming Gang、Justin Lazarow、Gefen Kohavi、Kai Kang、Marcin Eichner、Yinfei Yang、Afshin Dehghan、Peter Grasch

**ICCV 2025**

![CA-VQA 项目概览](teaser_cavqa.png?raw=true "CA-VQA 项目概览")

## 仓库内容

1. CA-1M 数据及标注的下载链接，包括衍生的 CA-VQA 标注。
2. Cubify Anything 论文中发布的 CuTR 模型权重链接。
3. 用于读取所提供数据并运行 CuTR 的基础代码。
4. 通过 NeRF Capture 应用使用自有设备采集图像的基础支持。

CuTR 输出类别无关的物体三维包围盒，包括位置、尺寸、朝向和置信度。它不直接提供语义类别名称、分割掩码或完整场景网格。当前公开入口用于推理，未提供完整的训练、损失函数和微调流程。

## 通用安装说明

官方推荐 Python 3.10 和较新的 PyTorch 2.x 版本。`requirements.txt` 列出了项目依赖。普通 CPU 或 NVIDIA 环境应先安装与平台匹配的 PyTorch 和 torchvision，例如：

```bash
python -m pip install torch torchvision
```

然后在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

**当前 DSW / PPU 环境请使用下方专用步骤。** 镜像已包含能够使用 PPU 的 PyTorch，应保留这一组合，避免直接重装 PyTorch 或无约束地升级所有依赖。

<a id="dsw-ppu"></a>

## 阿里云 DSW / PPU 适配操作

### 适配结论与验证范围

以下信息来自 2026 年 10 月 10 日的检查和用户在 DSW 原生终端提供的输出。

| 项目 | 已确认的信息 |
| --- | --- |
| 项目目录 | `/mnt/workspace/ml-cubifyanything` |
| 系统与 Python | Ubuntu 24.04.2，x86_64；Python 3.12.3 |
| PyTorch / torchvision | 2.9.0 / 0.24.0；构建时 CUDA 版本为 13.0 |
| 加速卡 | 1 张 PPU-ZW810E，98304 MiB（96 GiB）设备内存 |
| 驱动与工具 | 驱动 2.0.0-2e1b14，PPU-SMI 1.28，HGGC 13.0 |
| 设备可用性 | `torch.cuda.is_available()` 为 `True`；设备矩阵乘法和矩阵求逆通过 |
| 内存与磁盘 | 约 110 GiB 系统内存；工作目录磁盘约 30 GB，剩余约 21 GB |
| 下载地址 | 两个模型和样例 `42898570` 均返回 HTTP 200，支持字节范围请求 |
| 默认环境缺少的依赖 | timm、rerun-sdk、webdataset、cyclonedds |

工具会话中借用已有虚拟环境的依赖，对 ViT-B 配置的 RGB、RGB-D 模型进行了 CPU 合成输入验证：预处理和模型前向通过，输出分数、中心和尺寸均为有限值。测试使用随机权重和较小的输入，不能说明预训练模型精度、完整分辨率性能或全部 PPU 算子的兼容性。

**判断：具备继续适配的条件。** 保留现有 Python 和厂商 PyTorch，建立专用虚拟环境补齐依赖，先运行一个官方场景，并将可视化保存为文件。完整模型的设备内存需求、速度和效果须实际测量。

原生 DSW 终端的结果才是判断实例设备与下载连通性的依据；工具会话可能受到设备和网络隔离限制。PPU 使用 `cuda` 兼容接口，但不应据此安装 NVIDIA 驱动或替换 SDK 运行库。

以下安装、下载和推理命令是待执行方案。本次文档更新仅修改 README，没有执行安装、下载或源码修改。目标用途为私人非商业科研，具体模型条款见 [LICENSE_MODEL](LICENSE_MODEL)。

### 第一步：复核原生终端环境

在 DSW 网页终端执行；更换镜像或重启实例后应重新检查。

```bash
cd /mnt/workspace/ml-cubifyanything
command -v python3 nvidia-smi ppu-smi
python3 --version
ppu-smi
df -h /mnt/workspace
free -h
```

检查解释器、框架与设备运算：

```bash
python3 -B - <<'PY'
import sys
import torch
print("Python 路径：", sys.executable)
print("PyTorch：", torch.__version__, torch.__file__)
print("构建时 CUDA 版本：", torch.version.cuda)
print("CUDA 可用：", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("当前终端无法使用加速卡，请先排查设备可见性和镜像环境。")
print("设备信息：", torch.cuda.get_device_properties(0))
with torch.inference_mode():
    x = torch.randn(32, 32, device="cuda")
    product = x @ x.T
    inverse = torch.linalg.inv(torch.eye(3, device="cuda"))
    torch.cuda.synchronize()
assert torch.isfinite(product).all().item()
assert torch.isfinite(inverse).all().item()
print("基础设备运算通过")
PY
```

### 第二步：建立专用环境，保留厂商 PyTorch

以下命令创建专用环境。`--system-site-packages` 使它能够继承镜像已有的 PyTorch 和 PPU 运行环境；新增依赖安装到虚拟环境。不要将其他项目的虚拟环境作为本项目正式环境。

```bash
cd /mnt/workspace/ml-cubifyanything
python3 -m venv --system-site-packages /mnt/workspace/venvs/cutr-ppu
source /mnt/workspace/venvs/cutr-ppu/bin/activate
python -m pip --version
```

先预览安装计划。固定已确认的 PyTorch、torchvision、NumPy 和 SciPy 版本，避免解析依赖时替换厂商框架。下方为候选组合，尚未在该专用环境中实际安装和联合验证。

```bash
python -m pip install --dry-run --only-binary=:all: \
  "torch==2.9.0" "torchvision==0.24.0" \
  "numpy==1.26.0" "scipy==1.11.3" \
  "timm==1.0.30" "rerun-sdk==0.22.1" \
  "webdataset==0.2.86" "cyclonedds==11.0.1" \
  "setuptools==80.10.2" tifffile Pillow
```

PyTorch 和 torchvision 应显示已满足要求。如果安装计划包含下载安装或替换这两个包，先停止并核实虚拟环境是否继承了系统包。`--dry-run` 不安装软件，但可能联网下载元数据或缓存文件。

候选版本的依据：

- timm 1.0.30 已用于之前的 CPU 合成输入验证。
- Rerun 0.22.1 保留源码使用的 `set_time_seconds` 等旧接口，并有适用于当前平台的预编译包；整个 Demo 尚未验证通过。[该版本接口](https://github.com/rerun-io/rerun/blob/0.22.1/rerun_py/rerun_sdk/rerun/__init__.py)
- WebDataset 保持仓库指定的 0.2.86。
- CycloneDDS 11.0.1 有 CPython 3.12 / Linux x86_64 预编译包。Demo 即使使用离线样例也会导入它，其项目接口兼容性需在下方导入检查中确认。[发行文件](https://pypi.org/project/cyclonedds/11.0.1/#files)
- `setup.py` 使用 `pkg_resources`，因此保留提供该接口的 setuptools 80.10.2。

确认安装计划后执行安装，不加入 `--upgrade` 或 `--force-reinstall`：

```bash
python -m pip install --only-binary=:all: \
  "torch==2.9.0" "torchvision==0.24.0" \
  "numpy==1.26.0" "scipy==1.11.3" \
  "timm==1.0.30" "rerun-sdk==0.22.1" \
  "webdataset==0.2.86" "cyclonedds==11.0.1" \
  "setuptools==80.10.2" tifffile Pillow

python -m pip install --no-deps --no-build-isolation -e .
```

镜像源缺少 wheel 时，先核实镜像源或尝试 PyPI 官方源，不直接移除版本限制或切换源码编译。可编辑安装会产生包元数据，但不修改项目 Python 源文件。

安装后检查项目模块和 PPU 可见性：

```bash
python -B - <<'PY'
import torch
import torchvision
import timm
import rerun as rr
import rerun.blueprint
import webdataset
import scipy
import tifffile
from PIL import Image
from cubifyanything.capture_stream import CaptureDataset
from cubifyanything.dataset import CubifyAnythingDataset
from cubifyanything.cubify_transformer import make_cubify_transformer
assert hasattr(rr, "set_time_seconds")
assert torch.cuda.is_available(), "虚拟环境中无法使用加速卡"
print("依赖和项目模块导入通过")
print("PyTorch：", torch.__version__, torch.__file__)
print("torchvision：", torchvision.__version__)
print("Rerun：", rr.__version__)
print("设备：", torch.cuda.get_device_name(0))
PY

python -m pip check
```

继承系统包后，`pip check` 可能报告其他应用原有的冲突，应先确定是否涉及本项目，不要为消除所有提示而升级整套镜像环境。

### 第三步：下载一个官方场景和 RGB-D 权重

先检查空间和下载地址。HEAD 请求只读取响应头。

```bash
df -h /mnt/workspace
for url in \
  https://ml-site.cdn-apple.com/models/cutr/cutr_rgbd.pth \
  https://ml-site.cdn-apple.com/datasets/ca1m/val/ca1m-val-42898570.tar
do
    printf '\n检查地址：%s\n' "$url"
    curl -I -L --connect-timeout 5 --max-time 15 "$url"
done
```

本次原生终端返回的文件长度如下，用于核对传输完整性，不是密码学校验值。官方更新文件后需重新读取响应头。

| 文件 | 字节数 | 约占用空间 |
| --- | --- | --- |
| `cutr_rgbd.pth` | 396866874 | 378.5 MiB |
| `cutr_rgb.pth` | 398103184 | 379.7 MiB |
| `ca1m-val-42898570.tar` | 826767360 | 788.5 MiB |

首次只下载 RGB-D 权重和一个场景，合计约 1.14 GiB。另需为环境、安装缓存和可视化记录预留空间。无需解压 tar，读取器可以直接处理归档。

```bash
cd /mnt/workspace/ml-cubifyanything
mkdir -p models data/val outputs

curl --fail --location --continue-at - --retry 3 \
  --output models/cutr_rgbd.pth \
  https://ml-site.cdn-apple.com/models/cutr/cutr_rgbd.pth

curl --fail --location --continue-at - --retry 3 \
  --output data/val/ca1m-val-42898570.tar \
  https://ml-site.cdn-apple.com/datasets/ca1m/val/ca1m-val-42898570.tar

stat -c '%n %s 字节' \
  models/cutr_rgbd.pth data/val/ca1m-val-42898570.tar
tar -tf data/val/ca1m-val-42898570.tar > /dev/null
```

下载失败、长度不符或 tar 检查报错时，先处理文件问题，再运行推理。此检查不能代替官方校验和。

### 第四步：无桌面推理并保存可视化

官方 Demo 会启动 Rerun 桌面窗口，当前未提供 `--headless` 或 `--save` 参数。下方命令在当前 Python 进程中包装记录创建函数，将结果保存为 `.rrd` 并跳过窗口启动；不改写 `tools/demo.py`。它会创建输出文件，完整运行仍待实际验证。

使用 `--device cuda` 调用已确认可用的 PPU 兼容接口：

```bash
cd /mnt/workspace/ml-cubifyanything
source /mnt/workspace/venvs/cutr-ppu/bin/activate

python -B - \
  /mnt/workspace/ml-cubifyanything/data/val/ca1m-val-42898570.tar \
  --model-path /mnt/workspace/ml-cubifyanything/models/cutr_rgbd.pth \
  --device cuda --every-nth-frame 300 --viz-on-gt-points <<'PY'
import runpy
import sys
import uuid
from pathlib import Path
import rerun as rr

output_dir = Path("/mnt/workspace/ml-cubifyanything/outputs")
output_dir.mkdir(parents=True, exist_ok=True)
recordings = []
original_new_recording = rr.new_recording

def create_saved_recording(*args, **kwargs):
    recording = original_new_recording(*args, **kwargs)
    output_path = output_dir / f"ca1m-{uuid.uuid4().hex}.rrd"
    recording.save(str(output_path))
    recordings.append(recording)
    print("可视化保存到：", output_path, flush=True)
    return recording

rr.new_recording = create_saved_recording
rr.spawn = lambda *args, **kwargs: None
sys.argv = ["tools/demo.py", *sys.argv[1:]]
try:
    runpy.run_path("tools/demo.py", run_name="__main__")
finally:
    for recording in recordings:
        recording.flush(blocking=True)
        recording.disconnect()
PY
```

`--every-nth-frame 300` 表示每 300 帧处理一帧，适合首次验证。它不限制总帧数、不减少下载量，读取器仍可能读取和解析被跳过的帧。通过后可改为 30、10 或不传该参数。

在另一个 DSW 终端中观察设备使用情况：

```bash
watch -n 2 ppu-smi
```

推理结束后检查记录文件：

```bash
ls -lh /mnt/workspace/ml-cubifyanything/outputs/*.rrd
```

包装命令依据 Rerun 0.22.1 的记录创建、保存和刷新接口编写。[记录接口](https://github.com/rerun-io/rerun/blob/0.22.1/rerun_py/rerun_sdk/rerun/recording_stream.py)、[保存接口](https://github.com/rerun-io/rerun/blob/0.22.1/rerun_py/rerun_sdk/rerun/sinks.py)

其他模式可复用同一包装命令，只调整 `python -B -` 后的参数：

- **只看数据：** 保留 tar 路径，使用 `--viz-only --every-nth-frame 300`，移除模型、设备和 `--viz-on-gt-points` 参数。
- **CPU 对照：** 将 `--device cuda` 改成 `--device cpu`，用于判断错误是否只出现在设备后端；运行速度须实际测量。
- **RGB 模型：** 另行下载 `cutr_rgb.pth` 并替换模型路径。官方样例读取器仍会加载 GT 数据用于读取或展示。

该包装命令只处理文件可视化，不新增 JSON 导出或服务接口。

### 第五步：查看结果与验收

从 DSW 下载 `.rrd`，在有桌面的本地电脑安装相同版本的查看器并打开。以下命令在**本地电脑**执行，替换为实际下载路径：

```bash
python -m pip install "rerun-sdk==0.22.1"
rerun /path/to/downloaded/ca1m-recording.rrd
```

首次验收标准：

1. 项目模块可导入，专用环境的 CUDA 接口仍能使用 PPU。
2. 官方权重加载成功，真实样例完成推理，没有未支持算子或设备内存错误。
3. `.rrd` 可打开，显示 RGB、深度、预测三维框和用于对照的 GT 点云。
4. 对照物体检查框的位置、尺寸与朝向。生成文件不等于精度合格。
5. 记录处理帧数、运行时间和设备内存使用情况。首次运行可能包含初始化或内核缓存开销，不能直接作为稳定吞吐量。

验证通过后保存实际依赖清单。它不能替代厂商 PyTorch 发行包和 SDK 镜像的记录。

```bash
python -m pip freeze > /mnt/workspace/ml-cubifyanything/outputs/cutr-ppu-freeze.txt
```

### 已知问题与排查

| 现象 | 原因或排查方向 | 当前操作 |
| --- | --- | --- |
| 工具会话报驱动未加载，原生终端可用 | 工具隔离可能隐藏设备 | 以原生终端设备运算结果为准 |
| CUDA 不可用 | 设备可见性、镜像或库路径问题 | 检查解释器、`ppu-smi` 和 SDK 环境，不自行重装驱动 |
| 模块缺失 | 未进入专用环境或安装失败 | 核对解释器路径，执行导入检查 |
| 缺少 `pkg_resources` | 安装脚本使用旧接口 | 保留 setuptools 80.10.2，核实构建隔离设置 |
| Rerun 缺少 `set_time_seconds` | SDK 接口版本不匹配 | 使用候选版本 0.22.1，或获得源码修改授权后迁移接口 |
| 桌面窗口无法启动 | DSW 无桌面显示 | 使用上方 `.rrd` 保存命令 |
| 相对路径转 URI 失败 | Demo 直接调用 `Path.as_uri()` | tar 路径使用绝对路径 |
| 手动传 `--score-thresh` 后报比较错误 | 参数未指定 `type=float`，输入成为字符串 | 首次使用默认阈值 0.25，不传此参数；修复需修改源码 |
| 空标注帧触发 `NameError` | `dataset.py` 引用未定义的 `empty_box` 和 `box_type` | 记录堆栈和触发帧，获得授权后修复；抽帧不保证绕开 |
| `torch.load` 反序列化报错 | 文件不完整或加载规则不兼容 | 核对文件及完整堆栈，不直接更改加载安全选项 |
| PPU 报未支持算子 | 基础 CUDA 运算通过不代表全部模型算子通过 | CPU 对照定位，记录算子后再评估 SDK 或局部实现适配 |

上述源码问题尚未修复，候选依赖组合和完整模型尚未在 PPU 上验证。后续需要修改 Python 源码时，应先列出具体变更并获得用户同意。

## CA-1M 与 ARKitScenes 的区别

本项目与 [ARKitScenes](https://machinelearning.apple.com/research/arkitscenes) 相关，二者总体使用相同的底层采集数据。CA-1M 的主要区别如下：

1. 每个场景都有详尽的类别无关三维框标注，坐标系为激光扫描仪坐标系。
2. 每次采集的每一帧都有逐帧三维框真值，按照论文中的渲染流程生成，因此这些标注独立于相机位姿。

此外，CA-1M 还提供：

1. 每一帧与激光扫描仪配准后的真值位姿。
2. 每一帧由激光扫描仪渲染得到的 512 × 384 真值深度。
3. 已调整为正立方向的图像。

**注意：** CA-1M 仅包含成功与激光扫描仪配准的采集数据，因此并非 ARKitScenes 中的所有采集数据都包含在 CA-1M 中。

## 下载 CA-1M 数据

### 数据许可

全部数据采用 [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) 许可发布。

所有数据链接位于 `data/train.txt` 和 `data/val.txt`，可使用 `curl` 下载。如果不需要提前下载完整数据集，可以将链接传给数据读取器，或将对应划分的 txt 文件传给 Demo，并使用 `--video-ids` 筛选视频。

传入 txt 文件时，下载的数据会缓存到 `data/[split]`。当前 DSW 的磁盘容量有限，首次只需下载一个场景。

### CA-VQA 下载链接

CA-VQA 验证集链接位于 `data/cavqa/val.txt`。各任务的训练集链接位于 `data/cavqa/train_[task].txt`，任务包括 `binary`、`cardinality`、`grounding2d`、`grounding3d`、`multichoice` 和 `regression`。

## CA-1M 数据格式

CA-1M 使用 WebDataset 格式发布，每次采集（一个视频）对应一个 tar 归档。`ca1m-[split]-XXXXXXX.tar` 包含采集编号 XXXXXXX 的全部数据。

训练集和验证集均以完整帧率发布。`CubifyAnythingDataset` 负责读取数据，具体解析逻辑见 [dataset.py](cubifyanything/dataset.py)。主要文件如下：

```text
[video_id]/[integer_timestamp].wide/image.png             # 主相机的 1024×768 RGB 图像。
[video_id]/[integer_timestamp].wide/depth.png             # 设备 LiDAR / ARKit 提供的 256×192 深度，UInt16，单位为毫米。
[video_id]/[integer_timestamp].wide/depth/confidence.tiff  # 256×192 深度置信度，取值为 [0, 1]，当前未使用。
[video_id]/[integer_timestamp].wide/instances.json        # 渲染得到的逐帧物体真值，包括三维框、描述等。
[video_id]/[integer_timestamp].wide/T_gravity.json         # 编码相机俯仰和横滚的旋转矩阵，假设可由 IMU 等来源获得。

[video_id]/[integer_timestamp].gt/RT.json                 # 激光扫描仪坐标系中的配准位姿，4×4 矩阵，按行存储为 JSON。
[video_id]/[integer_timestamp].gt/depth.png               # FARO 激光扫描仪配准得到的 512×384 深度，UInt16，单位为毫米。
```

图像已调整到正立方向，因此宽高可能互换。真值深度中的 0 表示未成功配准的点。读取器会将深度从毫米转换为米。

`[video_id]/world.gt/instances.json` 包含整个场景的三维标注，逐帧标注由这些场景标注生成。其中包含 `wall`、`floor`、`ceiling` 和 `door_frame` 等结构标签，可用于辅助渲染。

### CA-VQA 验证集格式

解压后的 CA-VQA 验证集可以通过 Hugging Face datasets 加载。将示例路径替换为实际解压目录：

```python
import datasets
cavqa_val = datasets.load_from_disk("/path/to/cavqa_val_dir")
```

返回字典中的每个条目对应一个任务的数据集，字段如下：

```text
id: str                             # 数据样本的唯一标识。
question: str                       # 问题。
answer: str                         # 真值答案。
reference_frame: dict               # 问题对应的参考帧。
support_frame_[1-4]: dict            # 前序支持帧，时间顺序为 [4, 3, 2, 1, reference]。

# 各帧包含以下字段。
image['path']: str                  # images 目录中的 RGB PNG 图像路径。
depth_map_ground_truth['path']: str # 真值深度图路径。
depth_map_arkit['path']: str         # ARKit 深度图路径。
depth_map_monocular['path']: str     # 单目估计深度图路径。
pose: List[List[float]]             # 相对于参考帧的位姿。
intrinsics: List[List[float]]       # 相机内参。
```

### CA-VQA 训练集格式

每个任务的 CA-VQA 训练集可以通过 TensorFlow Datasets（TFDS）加载：

```python
import tensorflow_datasets as tfds
builder = tfds.builder_from_directory("/path/to/cavqa_train_dir/task/1.0.0")
cavqa_task_train = builder.as_dataset(split="train")
```

将路径中的 `task` 替换为实际任务名。数据集字段如下：

```text
questions: Tensor[str]                # 与参考帧有关的问题。
answers: Tensor[str]                  # 真值答案。
images: Tensor[str]                   # RGB 图像，参考帧位于最后。
depth_maps_ground_truth: Tensor[str]  # 真值深度图，帧顺序与 images 一致。
depth_maps_arkit: Tensor[str]         # ARKit 深度图，帧顺序与 images 一致。
depth_maps_monocular: Tensor[str]     # 单目估计深度图，帧顺序与 images 一致。
poses: Tensor[float]                  # 相机位姿，帧顺序与 images 一致。
intrinsics: Tensor[float]             # 相机内参，帧顺序与 images 一致。
```

datasets 和 tensorflow_datasets 是 CA-VQA 数据读取的额外依赖，不是首次 CuTR 样例推理所必需的依赖。

## CA-1M 数据可视化

项目使用 [Rerun](https://rerun.io) 可视化数据。在有桌面的环境中，Demo 会自动启动查看器。如果只查看数据而不运行模型，使用 `--viz-only`。

推理时，可使用 `--viz-on-gt-points` 将预测框显示在由 FARO 真值深度生成的点云上，便于观察三维预测效果。

### 示例命令

以下直接调用 Demo 的命令适用于已配置依赖且可启动桌面查看器的环境。当前 DSW 请使用[无桌面保存命令](#dsw-ppu)。本地 tar 路径须为绝对路径。

```bash
python tools/demo.py /path/to/downloaded_data/ca1m-val-42898570.tar --viz-only
```

```bash
python tools/demo.py data/train.txt --viz-only --video-ids 45261548
```

## 抽帧处理

数据集帧率较高，可使用 `--every-nth-frame N` 每隔 N 帧处理一次。该参数减少模型处理的帧数，但不减少整个场景的下载量。

## 运行 CuTR 模型

所有模型均按 [LICENSE_MODEL](LICENSE_MODEL) 中的 Apple 机器学习研究模型条款发布。条款限定非商业科研用途，并排除产品开发和商业产品、服务使用。

1. [RGB-D 模型权重](https://ml-site.cdn-apple.com/models/cutr/cutr_rgbd.pth)
2. [RGB 模型权重](https://ml-site.cdn-apple.com/models/cutr/cutr_rgb.pth)

通过 `--model-path` 将权重路径传给 `demo.py`。程序会检查权重，判断模型是 RGB 还是 RGB-D，并相应调整是否向模型输入深度。

### RGB-D 模型

该模型需要 RGB 图像和具有实际距离尺度的深度图。训练时使用 ARKit 深度，也可以尝试其他提供实际距离尺度的深度来源，但效果需要验证。

以下为具有桌面查看器的环境示例；将权重路径替换为实际路径。

支持 Apple MPS 的设备：

```bash
python tools/demo.py data/val.txt --video-ids 42898570 --model-path /path/to/models/cutr_rgbd.pth --viz-on-gt-points --device mps
```

支持 CUDA 的设备：

```bash
python tools/demo.py data/val.txt --video-ids 42898570 --model-path /path/to/models/cutr_rgbd.pth --viz-on-gt-points --device cuda
```

使用 CPU：

```bash
python tools/demo.py data/val.txt --video-ids 42898570 --model-path /path/to/models/cutr_rgbd.pth --viz-on-gt-points --device cpu
```

### 仅 RGB 模型

该模型使用 RGB 图像，并尝试从图像中估计场景的实际尺度。它仍使用相机内参；“仅 RGB”表示不向模型输入深度，不表示无需相机信息，也不表示现有 Demo 可以直接读取任意 jpg 文件。

支持 Apple MPS 且可启动查看器的环境示例：

```bash
python tools/demo.py data/val.txt --video-ids 42898570 --model-path /path/to/models/cutr_rgb.pth --viz-on-gt-points --device mps
```

其他已验证可用的后端可以使用 `--device cuda` 或 `--device cpu`。当前 DSW 仍需使用下方包装命令保存可视化。

## 使用自有设备采集的数据

项目对通过 NeRF Capture 采集和发送 RGB / 深度数据提供基础支持。

1. 在设备上安装 [NeRF Capture](https://apps.apple.com/au/app/nerfcapture/id6446518379)。
2. 先启动 NeRF Capture，再运行 `demo.py`。如果无法连接或应用停止响应，可以强制退出后重新打开。
3. 使用常规推理参数，但将数据路径替换为 `stream`。
4. 在应用中点击发送按钮，将一帧发送给推理程序；结果会显示在 Rerun 查看器中。

等待期间，程序持续打印 `Still waiting`，表示仍在等待帧。配备 LiDAR 的设备可尝试 RGB-D 模型；没有 LiDAR 的设备使用 RGB 模型。

### RGB-D 流输入示例（MPS）

```bash
python tools/demo.py stream --model-path /path/to/models/cutr_rgbd.pth --device mps
```

### RGB 流输入示例（MPS）

```bash
python tools/demo.py stream --model-path /path/to/models/cutr_rgb.pth --device mps
```

流输入使用 DDS。手机与云端 DSW 通常不处于同一可发现网络，不能假设替换 `--device` 后就能直连；需要另外验证网络连通与 DDS 发现配置。当前首次样例验证不使用流输入。

接入自有离线 RGB / RGB-D 文件时，还需要适配数据格式、相机内参、图像方向和重力信息。RGB-D 的深度单位、图像对齐和分辨率比例也需校验。原有 CA-1M 读取器会读取标注和真值深度，不能直接将自有图片目录当作 CA-1M 输入。

## 论文引用

在研究中使用 CA-1M 或 CuTR 时，请引用以下文献。BibTeX 中的原始标题、作者和出版信息保留原样，便于准确引用。

```bibtex
@article{lazarow2024cubify,
  title={Cubify Anything: Scaling Indoor 3D Object Detection},
  author={Lazarow, Justin and Griffiths, David and Kohavi, Gefen and Crespo, Francisco and Dehghan, Afshin},
  journal={arXiv preprint arXiv:2412.04458},
  year={2024}
}
```

在研究中使用 CA-VQA 时，请引用以下文献：

```bibtex
@article{daxberger2025mmspatial,
  title={MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs},
  author={Daxberger, Erik and Wenzel, Nina and Griffiths, David and Gang, Haiming and Lazarow, Justin and Kohavi, Gefen and Kang, Kai and Eichner, Marcin and Yang, Yinfei and Dehghan, Afshin and Grasch, Peter},
  journal={arXiv preprint arXiv:2503.13111},
  year={2025}
}
```

## 许可

示例代码采用 [Apple 示例代码许可](LICENSE) 发布。

数据采用 [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) 许可发布。

模型采用 [Apple 机器学习研究模型许可](LICENSE_MODEL) 发布。中文说明用于理解项目，具体许可内容以相应原始许可文件为准。

## 致谢

本项目使用了多个开源项目的成果，贡献与致谢见 [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS)。
