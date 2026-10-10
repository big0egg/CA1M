# CA-1M 与 Cubify Anything

本仓库提供 Cubify Transformer（CuTR）的公开实现，以及配套的 CA-1M 数据集和衍生的 CA-VQA 标注。

本文档是原项目说明的中文版本，并补充当前阿里云 PAI DSW / PPU 环境的操作方案。软件名称、命令、字段名及 BibTeX 中的原始论文信息保留原样。

**当前目标：私人科研，运行官方 CA-1M 样例的 RGB 与 RGB-D 两种模型，批量导出带三维框的 PNG/JPG，分别保存结果用于比较。** 两种官方权重已通过真实样例的 CPU 批量图片导出验证；原生 DSW 的 PPU 基础运算通过，完整 PPU 推理仍待验证。请优先阅读[阿里云 DSW / PPU 适配操作](#dsw-ppu)。

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

硬件与基础环境信息来自 2026 年 10 月 10 日的检查和用户在 DSW 原生终端提供的输出；依赖与可编辑安装进展于 2026 年 10 月 11 日更新。

| 项目 | 已确认的信息 |
| --- | --- |
| 项目目录 | `/mnt/workspace/CA1M` |
| 系统与 Python | Ubuntu 24.04.2，x86_64；Python 3.12.3 |
| PyTorch / torchvision | 2.9.0 / 0.24.0；构建时 CUDA 版本为 13.0 |
| 加速卡 | 1 张 PPU-ZW810E，98304 MiB（96 GiB）设备内存 |
| 驱动与工具 | 驱动 2.0.0-2e1b14，PPU-SMI 1.28，HGGC 13.0 |
| PPU SDK | `/usr/local/PPU_SDK/release.yaml` 为 `2.1.0-a5f865`；与驱动的官方配套关系仍需核实 |
| 设备可用性 | `torch.cuda.is_available()` 为 `True`；设备矩阵乘法和矩阵求逆通过 |
| 内存与磁盘 | 约 110 GiB 系统内存；工作目录磁盘约 30 GB；下载两份权重和一个场景后，本次剩余约 11 GB |
| 下载地址 | 两个模型和样例 `42898570` 均返回 HTTP 200，支持字节范围请求 |
| 默认环境缺少的依赖 | timm、rerun-sdk、webdataset、cyclonedds |

2026 年 10 月 11 日在本项目专用环境中，对 ViT-B 配置的 RGB、RGB-D 模型重新进行了 CPU 合成输入验证：RGB 模型仅收到 image，RGB-D 模型收到 image 和 depth；预处理和模型前向均通过，各输出 100 个候选检测，分数、中心和尺寸均为有限值，权重键的模式识别也通过。测试使用随机权重、256 × 192 RGB 输入和 64 × 48 深度输入，不能说明预训练模型精度、完整分辨率性能或全部 PPU 算子的兼容性。

**判断：具备继续适配的条件。** 保留现有 Python 和厂商 PyTorch，建立专用虚拟环境补齐依赖，先在同一个官方场景分别验证两种模型，批量保存标记图片，并按需保存 Rerun 可视化记录。完整模型的设备内存需求、速度和效果须实际测量。

原生 DSW 终端的结果才是判断实例设备与下载连通性的依据；工具会话可能受到设备和网络隔离限制。PPU 使用 `cuda` 兼容接口，但不应据此安装 NVIDIA 驱动或替换 SDK 运行库。

### 本次落实情况

当前仓库位于 `/mnt/workspace/CA1M`。下表记录环境、图片导出和 Git 忽略配置的实际完成情况。

| 工作 | 实际状态 |
| --- | --- |
| 中文说明与操作路径 | 已更新本 README，统一使用当前仓库目录 |
| 项目专用环境 | 已创建 `/mnt/workspace/venvs/ca1m-ppu`，启用 `--system-site-packages` |
| 厂商框架继承 | 已确认专用环境使用系统 PyTorch 2.9.0、torchvision 0.24.0；未重装框架 |
| 项目新增依赖 | 专用环境已安装 timm 1.0.30、rerun-sdk 0.22.1、webdataset 0.2.86、cyclonedds 11.0.1；导入通过 |
| 项目可编辑安装 | 已使用显式 PEP 517、关闭构建隔离的命令安装 cubifyanything 0.0.1；Demo 帮助入口通过 |
| 模型与官方样例 | RGB、RGB-D 权重及官方场景 42898570 已在本地；字节数与之前的响应头一致 |
| RGB / RGB-D 模式适配 | 共用专用环境；官方权重在真实样例上分别通过 CPU 推理和批量导出，各保存 2 张 768 × 1024 PNG |
| 官方样例完整 PPU 推理 | RGB、RGB-D 均尚未执行完整验证，无可宣称的完整推理验收结果 |
| Python 源码 | 已按用户授权新增图片导出模块和 Demo 参数，并修复 `--score-thresh` 的浮点类型 |
| 验证 | 6 项投影、裁剪和图片保存测试通过；真实 RGB / RGB-D 图片已打开检查 |
| Git 忽略 | 权重、下载/解压的数据、图片和 `.rrd` 输出均已忽略；没有已跟踪的大文件 |

此前工具会话的联网检查返回 `Operation not permitted`，安装预览随后报告找不到 timm 版本；由于网络受限，该结果不能证明厂商源缺包或版本不兼容。之后用户在原生终端安装了依赖，并提供可编辑安装失败的日志。本次复核确认上述依赖已可导入，并在专用环境离线完成项目可编辑安装；随后新增了批量图片导出，两个模型都在 CPU 上用官方权重完成真实样例的抽帧导出（`--every-nth-frame 2 --max-frames 2`）。完整 PPU 推理仍需在具有设备权限的 DSW 原生终端验证。

目标用途为私人非商业科研，具体模型条款见 [LICENSE_MODEL](LICENSE_MODEL)。

### 现有镜像与官方 PPU-training 的选择

**当前决定：先保留现有镜像，完成项目依赖和真实样例验证。** 当前 Python 3.12、PyTorch 2.9、CUDA 13.0 的组合接近官方 PAI-PPU 2.1 主推环境；这说明没有必要仅因 Python 版本不同就换镜像，并不代表 CuTR 已通过官方 PPU 验证。

| 方案 | 官方镜像名称或版本 | 适用判断 |
| --- | --- | --- |
| 保留现有环境 | Python 3.12 / PyTorch 2.9 / CUDA 13.0 | 原生设备基础运算通过；继续验证项目，当前镜像完整标签尚未确认 |
| 与现有 2.0 驱动配套的候选 | `ppu-training:2.0.0-pytorch2.9.0-ppu-py312-cu129-ubuntu24.04` | 保持 PyTorch 2.9，切换为该系列配套的 SDK / CUDA |
| 接近上游 Python 3.10 的备选 | `ppu-training:2.0.0-pytorch2.4.0-ppu-py310-cu124-ubuntu22.04` | 仅在 Python 3.12 依赖或运行兼容问题无法解决时再评估 |
| PAI-PPU 2.1 的推荐组合 | `2.1.0-pytorch2.9.0-ppu-py312-cu130-ubuntu24.04` | 官方要求 2.1.0-ra1f23041713 驱动；不能只换容器就认定与当前驱动配套 |

PAI-PPU 2.0 发布说明要求驱动 `2.0.0-2e1b14`，与当前驱动一致；2.1 发布说明要求 `2.1.0-ra1f23041713`。当前已看到 SDK 2.1 与驱动 2.0 的跨版本组合，需要核实实例的完整镜像标签及平台支持关系，基础运算通过不能替代完整验证。[PAI-PPU 2.0 官方说明](https://help.aliyun.com/zh/pai/pai-ppu-v2-0-0-official-image-release-note)、[PAI-PPU 2.1 官方说明](https://help.aliyun.com/en/pai/pai-ppu-v2-1-0-official-image-release-note)

若后续出现仅在 PPU 上发生的算子、运行库或驱动错误，先保留完整堆栈并进行 CPU 对照，再决定更换镜像或咨询平台确认驱动配套。缺少普通 Python 包、Rerun 无桌面、仓库自身的读取错误，应分别处理，不能仅靠更换镜像判断问题已解决。

更换时在 DSW 官方镜像列表筛选 PPU 训练镜像，并核实区域和可用标签。PAI 文档中的 PPU 推理镜像用于 EAS / Model Gallery；DSW 运行本项目仍应选择训练镜像。换镜像后使用新的专用环境，并按新镜像的实际框架版本重写下方约束，不要继续照搬 `torch==2.9.0`、`torchvision==0.24.0` 等当前环境版本。

### 依赖源与冲突处理原则

当前镜像默认使用以下厂商依赖源；安装命令会继承它，无需修改全局 pip 配置：

```text
https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/ubuntu_cu130/simple/
```

阿里云还提供不含操作系统 / CUDA 路径的自动适配入口。它属于同一厂商服务，可在原生终端确认权限和连通性后作为排查选项；本次没有切换依赖源。[PPU PIP 服务说明](https://help.aliyun.com/zh/document_detail/2936857.html)

```text
https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/pypi_index/simple/
```

保留镜像提供的 torch、torchvision 和 SDK。专用环境继承系统包，因此它并非完全隔离；系统中的 vLLM、ONNX、OpenCV 等其他组件依赖冲突仍可能出现在 `pip check` 中。此前检查发现了 protobuf、NumPy、click、urllib3 等不一致项，需要结合项目实际导入和运行报错判断影响，不为了清空提示而升级全部组件。

### 第一步：复核原生终端环境

在 DSW 网页终端执行；更换镜像或重启实例后应重新检查。

```bash
cd /mnt/workspace/CA1M
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

本次已经创建下方专用环境，项目依赖和可编辑安装已完成。以下命令保留为首次建立环境的操作说明。`--system-site-packages` 使它能够继承镜像已有的 PyTorch 和 PPU 运行环境；新增依赖安装到虚拟环境。不要将其他项目的虚拟环境作为本项目正式环境。

```bash
cd /mnt/workspace/CA1M
if [ ! -x /mnt/workspace/venvs/ca1m-ppu/bin/python ]; then
    python3 -m venv --system-site-packages /mnt/workspace/venvs/ca1m-ppu
fi
cat /mnt/workspace/venvs/ca1m-ppu/pyvenv.cfg
source /mnt/workspace/venvs/ca1m-ppu/bin/activate
python -m pip --version
```

配置文件应包含 `include-system-site-packages = true`。如果现有环境没有继承系统包，应先确认其用途，再重建本项目专用环境。

首次安装时先预览安装计划。固定已确认的 PyTorch、torchvision、NumPy 和 SciPy 版本，避免解析依赖时替换厂商框架。下方组合已通过专用环境的依赖导入和 Demo 帮助入口检查，完整模型推理尚未验证。

```bash
python -m pip install --dry-run --only-binary=:all: \
  "torch==2.9.0" "torchvision==0.24.0" \
  "numpy==1.26.0" "scipy==1.11.3" \
  "timm==1.0.30" "rerun-sdk==0.22.1" \
  "webdataset==0.2.86" "cyclonedds==11.0.1" \
  "setuptools==80.10.2" "wheel==0.45.1" tifffile Pillow
```

PyTorch 和 torchvision 应显示已满足要求。如果安装计划包含下载安装或替换这两个包，先停止并核实虚拟环境是否继承了系统包。`--dry-run` 不安装软件，但可能联网下载元数据或缓存文件。

候选版本的依据：

- timm 1.0.30 已用于之前的 CPU 合成输入验证。
- Rerun 0.22.1 保留源码使用的 `set_time_seconds` 等旧接口，并有适用于当前平台的预编译包；整个 Demo 尚未验证通过。[该版本接口](https://github.com/rerun-io/rerun/blob/0.22.1/rerun_py/rerun_sdk/rerun/__init__.py)
- WebDataset 保持仓库指定的 0.2.86。
- CycloneDDS 11.0.1 有 CPython 3.12 / Linux x86_64 预编译包。Demo 即使使用离线样例也会导入它，项目的 CaptureDataset 导入已通过；DDS 流输入没有验证。[发行文件](https://pypi.org/project/cyclonedds/11.0.1/#files)
- `setup.py` 使用 `pkg_resources`，因此保留提供该接口的 setuptools 80.10.2；wheel 0.45.1 用于本地包构建。

确认安装计划后执行安装，不加入 `--upgrade` 或 `--force-reinstall`：

```bash
python -m pip install --only-binary=:all: \
  "torch==2.9.0" "torchvision==0.24.0" \
  "numpy==1.26.0" "scipy==1.11.3" \
  "timm==1.0.30" "rerun-sdk==0.22.1" \
  "webdataset==0.2.86" "cyclonedds==11.0.1" \
  "setuptools==80.10.2" "wheel==0.45.1" tifffile Pillow

python -m pip install --no-index --use-pep517 --no-deps --no-build-isolation -e .
```

#### 可编辑安装报 `No module named 'pkg_resources'`

当前 pip 24.0 对此仓库的原命令会进入 `setup.py develop`。setuptools 80.10.2 的 `develop` 又调用一次 `pip install -e . --use-pep517 --no-deps`，没有传递外层的 `--no-build-isolation`。内层启用了临时隔离构建，无法使用当前环境提供的 `pkg_resources`，在执行 `setup.py` 第 8 行时失败。外层的弃用警告不是此次失败的直接原因。

上方修正命令从入口显式选择 PEP 517，并关闭构建隔离，直接使用已安装的 setuptools / wheel；`--no-index` 防止此次项目安装联网，`--no-deps` 不重新解析运行依赖。该命令于 2026 年 10 月 11 日在本项目专用环境中执行成功，安装版本为 0.0.1，没有修改 `setup.py` 或其他项目源码。[pip 构建机制说明](https://pip.pypa.io/en/stable/reference/build-system/)

安装后可检查版本与帮助入口：

```bash
python -B -c 'import importlib.metadata as m; import cubifyanything; print(m.version("cubifyanything")); print(cubifyanything.__file__)'
python -B tools/demo.py --help
```

这一安装错误不涉及 PPU 算子、CUDA 或驱动，不需要为它更换镜像。仍应保留 `setuptools==80.10.2` 的约束；本次没有升级 pip，也没有重装 PyTorch。

若原生终端安装失败，保留完整输出，先核实厂商源的访问权限、连通性和 wheel 是否可用；必要时检查上方同一厂商的自动适配入口。不要全局切换为社区 PyPI 源并重装 torch / torchvision，也不要直接移除版本限制或切换源码编译。可编辑安装会产生包元数据，但不修改项目 Python 源文件。

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

### 第三步：下载一个官方场景和 RGB / RGB-D 两份权重

先检查空间和下载地址。HEAD 请求只读取响应头。

```bash
df -h /mnt/workspace
for url in \
  https://ml-site.cdn-apple.com/models/cutr/cutr_rgb.pth \
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

RGB、RGB-D 共用同一个官方场景，只需各下载一份权重；两份权重加一个场景合计约 1.51 GiB。另需为环境、安装缓存和两种模式的可视化记录预留空间。无需解压 tar，读取器可以直接处理归档。

```bash
cd /mnt/workspace/CA1M
mkdir -p models data/val outputs

curl --fail --location --continue-at - --retry 3 \
  --output models/cutr_rgb.pth \
  https://ml-site.cdn-apple.com/models/cutr/cutr_rgb.pth

curl --fail --location --continue-at - --retry 3 \
  --output models/cutr_rgbd.pth \
  https://ml-site.cdn-apple.com/models/cutr/cutr_rgbd.pth

curl --fail --location --continue-at - --retry 3 \
  --output data/val/ca1m-val-42898570.tar \
  https://ml-site.cdn-apple.com/datasets/ca1m/val/ca1m-val-42898570.tar

stat -c '%n %s 字节' \
  models/cutr_rgb.pth models/cutr_rgbd.pth \
  data/val/ca1m-val-42898570.tar
tar -tf data/val/ca1m-val-42898570.tar > /dev/null
```

下载失败、长度不符或 tar 检查报错时，先处理文件问题，再运行推理。此检查不能代替官方校验和。

### 第四步：分别运行 RGB 与 RGB-D，批量导出标记图片

#### 两种模式的输入与选择

两种模型使用相同的专用环境和 `--device cuda`，当前无需为 RGB 另换镜像或安装一套依赖。

| 项目 | RGB | RGB-D |
| --- | --- | --- |
| 权重 | `models/cutr_rgb.pth` | `models/cutr_rgbd.pth` |
| 送入模型的数据 | RGB 图像；当前流程使用相机内参和重力对齐信息 | RGB 图像和深度；当前流程使用相机内参和重力对齐信息 |
| 输入深度 | 不向模型输入深度 | 官方样例使用 ARKit 深度，保留实际距离尺度 |
| 标记图片目录 | `outputs/cutr_rgb/images/[video_id]/` | `outputs/cutr_rgbd/images/[video_id]/` |
| 可选 Rerun 记录目录 | `outputs/cutr_rgb/` | `outputs/cutr_rgbd/` |
| 深度面板 | 模型推理界面不创建深度面板 | 显示输入模型的深度 |
| GT 点云对照 | 可选，用于可视化对照 | 可选，用于可视化对照 |

Demo 检查权重中的 `backbone.0.patch_embed_depth.` 键来选择模型结构和预处理。模型类型由**实际权重**决定，改文件名不改变模式。`--no-depth` 不是 RGB / RGB-D 的模型切换开关；模型加载后读取器会按权重类型重新设置是否读取 ARKit 深度。因此 RGB 使用 `cutr_rgb.pth` 即可，RGB-D 使用 `cutr_rgbd.pth`，两种推理命令都无需额外传 `--no-depth`。

`--viz-on-gt-points` 只把预测显示在真值点云上，RGB 模型也可以使用；GT 深度不会因此成为 RGB 模型输入。即使不传此参数，现有 CA-1M 读取器仍读取 GT 深度、位姿和标注，因此这里的 RGB 命令不能直接用于缺少这些字段的任意图片目录。自有 RGB 数据还需要与图像对应的相机内参，接入现有 Demo 需要另外适配读取器；涉及源码的变更须先取得用户同意。

#### 批量导出 PNG/JPG：DSW 首选操作

标记图片将预测三维框的 8 个顶点投影到原始 RGB 图像，再绘制可见的 12 条边和置信度标签。两种模式都在 RGB 图像上绘制结果；RGB-D 的深度参与推理，不会额外导出深度图。

这是真实模型预测，不是 GT 标注。标签如 `#1 0.64` 表示当前帧的候选编号与置信度；模型没有语义类别名称，该编号也不表示跨帧跟踪 ID。投影使用原图相机内参和模型已变换到相机坐标的三维框，不重复施加重力变换；跨相机近裁剪面的边先在三维空间裁剪，投影后再裁剪到图像边界。完全在相机后方或画面外的框不绘制，没有可见检测时仍保存原图。

先进入专用环境：

```bash
cd /mnt/workspace/CA1M
source /mnt/workspace/venvs/ca1m-ppu/bin/activate
```

**RGB 批量图片导出：**

```bash
python -B tools/demo.py /mnt/workspace/CA1M/data/val/ca1m-val-42898570.tar \
  --model-path /mnt/workspace/CA1M/models/cutr_rgb.pth \
  --device cuda --every-nth-frame 300 --max-frames 5 \
  --image-output-dir /mnt/workspace/CA1M/outputs/cutr_rgb/images \
  --image-only
```

**RGB-D 批量图片导出：**

```bash
python -B tools/demo.py /mnt/workspace/CA1M/data/val/ca1m-val-42898570.tar \
  --model-path /mnt/workspace/CA1M/models/cutr_rgbd.pth \
  --device cuda --every-nth-frame 300 --max-frames 5 \
  --image-output-dir /mnt/workspace/CA1M/outputs/cutr_rgbd/images \
  --image-only
```

这两个命令直接导出图片，不启动 Rerun、不生成 `.rrd`，无需定义下方保存函数。首次各导出最多 5 帧；验证后去掉 `--max-frames 5` 即可处理全部抽样帧。同时去掉 `--every-nth-frame 300` 和 `--max-frames 5` 则逐帧处理整个场景，耗时和磁盘占用随导出帧数增加。

| 参数 | 含义 |
| --- | --- |
| `--image-output-dir PATH` | 图片根目录，按场景 ID 建立子目录 |
| `--image-only` | 只导出图片，跳过 Rerun 记录和窗口；必须同时指定输出目录 |
| `--image-format png` / `jpg` | 默认 PNG；JPG 使用质量 95，可加入 `--image-format jpg` |
| `--every-nth-frame N` | 每 N 帧处理一帧，N 为正整数 |
| `--max-frames N` | 最多处理 N 个抽样后的帧；不传则不限制，不改变下载量 |
| `--score-thresh 0.25` | 仅绘制达到置信度阈值的预测；默认 0.25，已修复浮点参数解析 |

图片文件名为 `frame_[处理序号]_[时间戳纳秒].png`，序号从 0 开始，表示本次处理的第几帧，不是原始视频帧号。同一场景、同一运行参数再次导出到相同目录会覆盖同名图片；比较不同阈值或后端时请指定不同目录。

本次已生成的 CPU 验证结果位于：

```text
outputs/cutr_rgb/images/42898570/frame_000000_752833256948125.png
outputs/cutr_rgb/images/42898570/frame_000001_752833456866708.png
outputs/cutr_rgbd/images/42898570/frame_000000_752833256948125.png
outputs/cutr_rgbd/images/42898570/frame_000001_752833456866708.png
```

可以直接在 DSW 文件浏览器打开或下载，无需 Rerun 查看器。RGB 使用 RGB 权重且不接收输入深度；RGB-D 使用 RGB-D 权重和 ARKit 深度。当前 CA-1M 读取器对 GT 数据的要求依然存在，图片导出功能不改变自有图片的接入条件。

#### 可选：同时保存图片与 Rerun 三维记录

如果需要交互查看三维场景，可使用下面的 `ca1m_demo` 函数保存 `.rrd`，并传入 `--image-output-dir` 同时导出图片。不要同时传 `--image-only`，否则只保存图片。

#### 在当前终端定义共用的 Rerun 保存函数

未使用 `--image-only` 时，Demo 会启动 Rerun 桌面窗口。下方函数在运行进程中包装记录创建接口，将结果保存为 `.rrd` 并跳过窗口启动。只导出图片时使用上方直接调用 Demo 的命令即可。

**先执行一次下方代码，再在同一个终端执行两个模型的命令。** 重新打开终端后需再次定义该函数。函数使用权重文件名区分输出目录；若重命名权重，目录名会相应变化，模型识别仍由 Demo 检查权重内容。

```bash
cd /mnt/workspace/CA1M
source /mnt/workspace/venvs/ca1m-ppu/bin/activate

ca1m_demo() {
    python -B - "$@" <<'PY'
import runpy
import sys
import uuid
from pathlib import Path
import rerun as rr

demo_args = sys.argv[1:]
if "--model-path" in demo_args:
    model_path = demo_args[demo_args.index("--model-path") + 1]
    run_name = Path(model_path).stem
else:
    run_name = "viz-only"

output_dir = Path("/mnt/workspace/CA1M/outputs") / run_name
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
sys.argv = ["tools/demo.py", *demo_args]
try:
    runpy.run_path("tools/demo.py", run_name="__main__")
finally:
    for recording in recordings:
        recording.flush(blocking=True)
        recording.disconnect()
PY
}
```

#### 可选：RGB 同时保存图片和 Rerun 记录

以下命令不向模型输入深度，也不请求 GT 点云展示。`.rrd` 保存到 `outputs/cutr_rgb/`，图片保存到其 `images/[video_id]/` 子目录。

```bash
ca1m_demo /mnt/workspace/CA1M/data/val/ca1m-val-42898570.tar \
  --model-path /mnt/workspace/CA1M/models/cutr_rgb.pth \
  --device cuda --every-nth-frame 300 --max-frames 5 \
  --image-output-dir /mnt/workspace/CA1M/outputs/cutr_rgb/images
```

需要与真值点云比较时，在该命令末尾加入 `--viz-on-gt-points`；仍是 RGB 模型推理。

#### 可选：RGB-D 同时保存图片和 Rerun 记录

以下命令使用 RGB 和 ARKit 深度推理，同时展示 GT 点云。`.rrd` 保存到 `outputs/cutr_rgbd/`，图片保存到其 `images/[video_id]/` 子目录。

```bash
ca1m_demo /mnt/workspace/CA1M/data/val/ca1m-val-42898570.tar \
  --model-path /mnt/workspace/CA1M/models/cutr_rgbd.pth \
  --device cuda --every-nth-frame 300 --max-frames 5 --viz-on-gt-points \
  --image-output-dir /mnt/workspace/CA1M/outputs/cutr_rgbd/images
```

建议首次按顺序运行两个模型。`--every-nth-frame 300` 表示每 300 帧处理一帧，它本身不限制总帧数、不减少下载量，读取器仍可能读取和解析被跳过的帧。两个模型使用相同场景和相同抽帧参数，便于对照同一时间戳；通过后可改为 30、10 或不传该参数。

在另一个 DSW 终端中观察设备使用情况：

```bash
watch -n 2 ppu-smi
```

推理结束后分别检查记录文件：

```bash
ls -lh /mnt/workspace/CA1M/outputs/cutr_rgb/*.rrd
ls -lh /mnt/workspace/CA1M/outputs/cutr_rgbd/*.rrd
```

包装函数依据 Rerun 0.22.1 的记录创建、保存和刷新接口编写。[记录接口](https://github.com/rerun-io/rerun/blob/0.22.1/rerun_py/rerun_sdk/rerun/recording_stream.py)、[保存接口](https://github.com/rerun-io/rerun/blob/0.22.1/rerun_py/rerun_sdk/rerun/sinks.py)

保存函数的独立记录检查通过，两个模型目录均生成非空 `.rrd`。另用官方 RGB-D 权重在 CPU 上处理真实样例的一帧，同时保存 `.rrd` 和 JPG，并验证了 GT 点云显示路径与 `--score-thresh 0.4` 参数。PPU 上的完整运行仍待验证。

#### 数据可视化与 CPU 对照

只看数据、不运行模型，输出到 `outputs/viz-only/`：

```bash
ca1m_demo /mnt/workspace/CA1M/data/val/ca1m-val-42898570.tar \
  --viz-only --every-nth-frame 300
```

RGB 和 RGB-D 都可以把各自命令中的 `--device cuda` 改成 `--device cpu`，用于定位 PPU 后端错误。文件仍保存在对应模型目录，每次生成唯一文件名；如需区分 CPU / PPU 结果，运行结束后按后端重命名文件并记录运行参数。

保存函数负责 `.rrd` 记录，Demo 的 `--image-output-dir` 负责 PNG/JPG 导出；此次没有新增 JSON 导出或服务接口。

### 第五步：查看结果与验收

PNG/JPG 可以直接在 DSW 文件浏览器或本地图片查看器打开。核对两个模式的场景 ID、时间戳和原图尺寸一致，检查三维框投影的位置与置信度标签。

只有选择同时保存 Rerun 记录时，才需从 `outputs/cutr_rgb/` 和 `outputs/cutr_rgbd/` 下载 `.rrd`，在有桌面的本地电脑安装相同版本的查看器并打开。以下命令在**本地电脑**执行，替换为实际下载路径：

```bash
python -m pip install "rerun-sdk==0.22.1"
rerun /path/to/downloaded/rgb/ca1m-recording.rrd
rerun /path/to/downloaded/rgbd/ca1m-recording.rrd
```

首次验收标准：

1. 项目模块可导入，专用环境的 CUDA 接口仍能使用 PPU。
2. RGB、RGB-D 两份官方权重分别加载成功，同一真实样例分别完成推理，没有未支持算子或设备内存错误。
3. 两种模式均生成可打开的 PNG/JPG，保留原图尺寸并绘制可见的预测三维框与置信度；抽样帧数符合 `--max-frames` 限制。可选的 `.rrd` 中 RGB-D 还显示输入深度；传入 `--viz-on-gt-points` 的运行应显示 GT 点云。
4. 对照相同时间戳的物体，分别检查两个模型的框位置、尺寸与朝向。RGB 估计实际尺度，误差需要用真实数据验证；生成文件不等于精度合格。
5. 分别记录模型模式、权重路径、抽帧间隔、处理帧数、运行时间和设备内存使用情况。首次运行可能包含初始化或内核缓存开销，不能直接作为稳定吞吐量。

验证通过后保存实际依赖清单。它不能替代厂商 PyTorch 发行包和 SDK 镜像的记录。

```bash
python -m pip freeze > /mnt/workspace/CA1M/outputs/ca1m-ppu-freeze.txt
```

### 已知问题与排查

| 现象 | 原因或排查方向 | 当前操作 |
| --- | --- | --- |
| 工具会话报驱动未加载，原生终端可用 | 工具隔离可能隐藏设备 | 以原生终端设备运算结果为准 |
| CUDA 不可用 | 设备可见性、镜像或库路径问题 | 检查解释器、`ppu-smi` 和 SDK 环境，不自行重装驱动 |
| 模块缺失 | 未进入专用环境或安装失败 | 核对解释器路径，执行导入检查 |
| 工具安装预览提示无可用版本，同时网络检查报 `Operation not permitted` | 工具网络隔离，尚不能判断源是否缺包 | 在 DSW 原生终端复核同一安装命令 |
| SDK / 驱动配套待核实 | 当前 SDK 2.1、驱动 2.0；官方发布说明有不同配套要求 | 记录镜像完整标签，结合完整推理结果确认平台支持关系 |
| 可编辑安装内层缺少 `pkg_resources` | `setup.py develop` 启动内层隔离构建，丢失外层设置 | 使用 `--use-pep517 --no-build-isolation` 直接构建，并保留 setuptools 80.10.2；本次已验证安装成功 |
| Rerun 缺少 `set_time_seconds` | SDK 接口版本不匹配 | 使用候选版本 0.22.1，或获得源码修改授权后迁移接口 |
| 桌面窗口无法启动 | DSW 无桌面显示 | 使用 `--image-only --image-output-dir PATH`，或可选的 `.rrd` 保存函数 |
| 相对路径转 URI 失败 | Demo 直接调用 `Path.as_uri()` | tar 路径使用绝对路径 |
| 传 `--no-depth` 却仍运行 RGB-D | 模型加载后按权重重新设置深度读取；该参数不改变模型结构 | RGB 使用 `cutr_rgb.pth`，RGB-D 使用 `cutr_rgbd.pth` |
| 旧版手动传 `--score-thresh` 后报比较错误 | 原参数未指定 `type=float` | 本次已修复；现在可传 `--score-thresh 0.4` 等浮点阈值 |
| 空标注帧触发 `NameError` | `dataset.py` 引用未定义的 `empty_box` 和 `box_type` | 记录堆栈和触发帧，获得授权后修复；抽帧不保证绕开 |
| `torch.load` 反序列化报错 | 文件不完整或加载规则不兼容 | 核对文件及完整堆栈，不直接更改加载安全选项 |
| PPU 报未支持算子 | 基础 CUDA 运算通过不代表全部模型算子通过 | CPU 对照定位，记录算子后再评估 SDK 或局部实现适配 |

本次已按授权新增批量图片导出，并修复阈值参数类型；空标注读取分支的问题尚未修改。6 项图片导出测试和两种官方权重的真实样例 CPU 抽帧导出通过，完整 PPU 推理仍待验证。后续其他源码变更仍须先取得用户同意。

复核导出模块的测试命令：

```bash
python -B -m unittest discover -s tests -v
```

### Git 管理：权重、数据集和生成结果保留在本地

已新增 `.gitignore`，忽略 `models/`、`weights/`、`checkpoints/`、下载/解压的数据目录、常见权重与压缩包扩展名，以及 `outputs/` 中的 PNG/JPG、`.rrd` 和其他生成结果。官方数据 URL 清单 `data/train.txt`、`data/val.txt`、`data/cavqa/*.txt` 与 `data/LICENSE_DATA` 保留在版本控制中。

本次检查确认两份权重和下载的场景都未被 Git 跟踪，忽略规则已生效；没有删除本地文件，也没有提交或推送到 GitHub。可以复核：

```bash
git check-ignore -v models/cutr_rgb.pth models/cutr_rgbd.pth \
  data/val/ca1m-val-42898570.tar \
  outputs/cutr_rgb/images/42898570/frame_000000_752833256948125.png
git ls-files -ci --exclude-standard
git status --short
```

本项目目前 `git ls-files -ci --exclude-standard` 没有输出，表示没有已跟踪但匹配忽略规则的文件。

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

该模型使用 RGB 图像，并尝试从图像中估计场景的实际尺度。它仍使用相机内参；“仅 RGB”表示不向模型输入深度，不表示无需相机信息，也不表示现有 Demo 可以直接读取任意 jpg 文件。官方样例还提供当前流程使用的重力对齐信息。

支持 Apple MPS 且可启动查看器的环境示例：

```bash
python tools/demo.py data/val.txt --video-ids 42898570 --model-path /path/to/models/cutr_rgb.pth --viz-on-gt-points --device mps
```

其他已验证可用的后端可以使用 `--device cuda` 或 `--device cpu`。当前 DSW 请使用[上方 RGB / RGB-D 保存函数与命令](#dsw-ppu)，分别保存两种模式的可视化。

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
