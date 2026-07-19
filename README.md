# asg-vlm

本仓库是在 cz 环境(离线 8×H200 + 在线 4090 + 共享卷,详见 [cz_runtime_env.md](cz_runtime_env.md))上 ms-swift SFT 实验的工作流。
本项目有三大要推进的事项:

1. cz 训练环境的最佳实践
2. 推进 PCB VLM 项目
3. 随着 PCB VLM 项目的推进,学习 MLOps、AI 实验工程管理

实验的可变输入分三类:数据、代码、环境。测试工程中，需要先用基础模型推理测试环境，再利用实验数据进行小规模微调完成工作流测试;实验日志按这三类记录,实验才可复现。

## 全局设置（只设置一次）

```bash
# 假设当前工作文件夹是共享文件夹,当前终端是 4090
git clone <本仓库> && cd asg_vlm
cp .env.example .env                 # 填 MODELSCOPE_API_TOKEN

# 缓存与解释器路径按 XDG 规范指到共享卷:先编辑 scripts/set_cache.sh 里的两个绝对路径,
# 再接入 bashrc(两台机器各做一次;bash 子进程 export 传不回父 shell,必须 source)
echo "source $PWD/scripts/set_cache.sh" >> ~/.bashrc
source ~/.bashrc
```
## 环境部署
```bash
# 4090
uv sync --group train

# 以下两个python模块在编译时要 import torch,因此只能在装完 torch 再单独装, 导致只能在 pyproject.toml 之外配置;
# 4090=sm_89、H200=sm_90,编译产物同时覆盖两种架构(fat binary,说明见下)
export TORCH_CUDA_ARCH_LIST="8.9;9.0"
uv pip install --no-build-isolation "deepspeed==0.16.*"
uv pip install --no-build-isolation "flash-attn==2.8.*"   # 轮子下载失败会转源码编译,MAX_JOBS=8 可限并发
```

`TORCH_CUDA_ARCH_LIST="8.9;9.0"` 实现的是同一份 venv 适配两种 GPU。CUDA 内核编译时必须指定目标架构,而一个 .so 文件可以并排容纳多套架构的机器码,这种产物叫 fat binary。设了这个变量,deepspeed 和 flash-attn 走源码编译时就把 sm_89、sm_90 两套内核都编进产物;运行时 CUDA 驱动按当前卡自动选用——4090 取 sm_89 版,H200 取 sm_90 版,一份二进制两机通用。不设则默认只编本机架构,产物拿到另一台机器上不可用。代价只是编译时间和文件体积翻倍。

flash-attn 版本说明:Qwen3.5-9B 经配置里的 `attn_impl: flash_attn` 走 transformers 的 flash_attention_2,对应 PyPI 的 `flash-attn` 2.x,这里版本固定为 2.8 系列。FA3 是 Hopper 专用,内核源码用了 sm_90 独有指令,不存在 sm_89 版本, 同一个FA3 .venv 无法做到h200/4090兼容.

环境测试,这一条在 4090 和 H200 上各跑一次,验 torch 带 CUDA、flash-attn 可导入、GPU 可见:

```bash
uv run --no-sync python -c "import torch, flash_attn; print(torch.__version__, torch.cuda.get_device_name(0))"
```

## 实验教程

教程按一次实验从头到尾的顺序展开:先准备数据和配置,再 smoke test、启动训练、观察实验数据曲线,训完追溯并做 checkpoint 版本控制。贯穿全程的示例是 PCB 两阶段训练:数据集 [`Mask2X/asg-pcb-cot-sft`](https://modelscope.cn/datasets/Mask2X/asg-pcb-cot-sft)(KiCad 原理图布局 DSL),先用 answer-only 数据学语法,再从其 checkpoint 续训 CoT 学推理,基座 Qwen/Qwen3.5-9B——两个阶段正好把两种模型来源(hub 基座、自产 checkpoint)都演示一遍。各步骤实践的出处见文末"延伸阅读"。

### 1. 定义实验

数据集和基座模型先在 4090 上下载(数据、代码、环境中的数据)。在完成全局设置后两者都会默认放进共享卷上的 modelscope 缓存区:

```bash
set -a; source .env; set +a   # modelscope 命令读 .env 里的 token
# 通式:--local_dir "$MODELSCOPE_CACHE/local/<名字-版本>"
# PCB 示例:
uv run --no-sync modelscope download --dataset Mask2X/asg-pcb-cot-sft --revision master
uv run --no-sync modelscope download --model Qwen/Qwen3.5-9B --revision master
```

MLOps 要点:

- 数据集的版本控制:"Data is immutable"(Cookiecutter Data Science 第一原则)。保证训练时的语料一致性;数据读取时必须要给出 revision。
- 模型和数据同理(Made With ML 的 versioning 一课把这条推广到所有实验输入:凡是影响结果的输入,都要能报出版本)。

然后在 `configs/` 新建一个 yaml,继承 `base.yaml` 后只写差异;字段即 ms-swift `TrainArguments` 同名参数,新增超参直接写,不用改代码。PCB 示例的三份配置就是这个形态:`stage1_answer_only.yaml`、`stage2_cot.yaml`、`lora_4090.yaml`,其中数据与模型路径用 `${oc.env:MODELSCOPE_CACHE}` 解析,没 source set_cache.sh 会在启动时报错而不是静默用错路径:

```yaml
defaults:
  - base
  - override hydra/job_logging: none
  - override hydra/hydra_logging: none
  - _self_

model: ${oc.env:MODELSCOPE_CACHE}/local/Qwen3.5-9B
dataset:
  - ${oc.env:MODELSCOPE_CACHE}/local/asg-pcb-cot-sft/data/answer_only_train.jsonl
val_dataset:
  - ${oc.env:MODELSCOPE_CACHE}/local/asg-pcb-cot-sft/data/answer_only_val.jsonl
```

配置(软编码)与硬编码之间存在模糊地带,处理原则按阶段分:前期快速验证不纠结,怎么快怎么来;进入寻超参阶段,要搜索的参数必须暴露成配置项;公开发布时做到什么程度,等有真实复现需求再定。一条硬规则贯穿始终:代码里禁止 `.get(xxx, default)` 式配置回退——默认值只允许出现在 base.yaml 这种看得见的地方,配置错误要在启动时暴露,不能被代码默默兜住(`report_to` 因此显式声明在 base.yaml)。

### 2. 训练前冒烟测试

冒烟测试的本质是用提前暴露错误换时间:清单做不全也不必做全,因为所有冒烟测试都只是为最终训练服务。如何权衡测试的数量，关键是把冒烟测试想全别遗漏，然后是根据代价和收益选择做哪些测试。一方面要主要想出测试方案，另外一方面 failure-driven ，也就是代码运行失败了，就要把测试给细化。

```bash
# 4090:单卡小配置,验环境与流程
export RUN_NAME=smoke-4090-$(date +%Y%m%d-%H%M%S)
uv run --no-sync torchrun --nproc_per_node 1 train/sft.py --config-name lora_4090 max_steps=3

# H200:正式配置,额外验 8 卡 NCCL、deepspeed、显存
export RUN_NAME=smoke-h200-$(date +%Y%m%d-%H%M%S)
uv run --no-sync torchrun --nproc_per_node 8 train/sft.py --config-name stage1_answer_only max_steps=3
```

通过判据:loss 是有限值(NaN 即数值问题)、无 OOM、`output/<run>/` 产出 checkpoint、MLflow 里能看到 run。


### 3. 启动与覆盖

正式实验前先 commit, MLflow 会自动把当次的 commit hash 记进 run 的 tags,这样才能追溯实验代码。然后启动,PCB 阶段一即标准形态:

```bash
export RUN_NAME=stage1-$(date +%Y%m%d-%H%M%S)
uv run --no-sync torchrun --nproc_per_node 8 train/sft.py --config-name stage1_answer_only
```

命令行可覆盖任意字段,优先级:命令行 > 实验配置 > base:

```bash
... --config-name stage1_answer_only learning_rate=2e-5
... --config-name stage1_answer_only 'dataset=[/abs/path/other_train.jsonl]'   # 列表要引号
```

模型第二次被引入的方式:PCB 阶段二从阶段一的产出续训,`model` 由命令行覆盖指向那个 checkpoint,谱系由此连起:

```bash
export RUN_NAME=stage2-$(date +%Y%m%d-%H%M%S)
uv run --no-sync torchrun --nproc_per_node 8 train/sft.py --config-name stage2_cot \
  model=output/stage1-xxxx/vN-xxxx/checkpoint-NNN
```

实验节奏遵循 Karpathy《A Recipe for Training Neural Networks》的增量复杂化:先用最简配置把端到端流程跑通,之后每次只改一个变量——一次改两个,指标变了不知道归因给谁。命令行覆盖正是为此准备的:单变量改动不必新建配置文件。随机种子默认固定(seed=42)并记录在 params,这是成本最低的可复现投资。

断点续训:`resume_from_checkpoint` 指向断点,`RUN_NAME` 复用原名,输出目录和曲线接在同一个 run 上。扫参:写 bash 循环逐个起(hydra 的 multirun 与 torchrun 冲突)。可用的环境变量还有 `CUDA_VISIBLE_DEVICES`(选卡)和 `--nproc_per_node`(卡数)。

### 4. 看曲线

实验追踪从第一个实验就在:MLflow 由启动器自动开启,不存在"先跑几个不记录的实验"的阶段。记录落在共享卷的 `mlruns/` 目录,训练机离线直接写入;看曲线在 4090 上起 UI,mac 经 ssh 端口转发打开:

```bash
uv run --no-sync mlflow ui --port 5000   # 4090,在仓库根目录执行
ssh -L 5000:localhost:5000 <4090>        # mac,然后浏览器开 http://localhost:5000
```

盯盘要点:eval loss 和 train loss 一起看,只降 train 不降 eval 是过拟合信号;grad_norm 突刺伴随 loss 跳变,先查数据再怀疑学习率。

### 5. 追溯 checkpoint

每个 run 的完整超参——包括 `model`(它指向的就是父 checkpoint)和数据路径——由 MLflowCallback 自动记为 params,git commit 记在 tags;UI 里勾选多个 run 点 Compare,参数差异和曲线对比一屏看完。ms-swift 还会在 output 目录里存一份 args.json 作为本地备份。想知道"这个 checkpoint 从哪来、用了什么数据和参数",打开它所在 run 的 params 即可,不靠记忆。

### 6. checkpoint 版本控制

产出的 checkpoint 上传 ModelScope 模型仓管理,metadata一般记三个参数就够(父模型、run 名、commit),上传前在 checkpoint 目录放一份 metadata 文件:

```markdown
---
base_model: Qwen/Qwen3.5-9B
base_model_revision: master
---
run: stage1-20260719-153000
commit: abc1234
```

```bash
set -a; source .env; set +a
uv run --no-sync modelscope upload <模型仓id> \
  output/<run>/vN-xxxx/checkpoint-NNN "<run>/checkpoint-NNN" --repo-type model
```

第三个参数是仓库内路径,以 run 名开头,不同 run 的同号 checkpoint 互不覆盖;目标模型仓先在 ModelScope 网页建一次。本地取回:`modelscope download --model <模型仓id> --local_dir <目录>`。

## 目录结构

```
.env       密钥与运行时开关(HF 离线等)
configs/   hydra 配置:base.yaml + 各实验差异
train/     sft.py 启动器
scripts/   set_cache.sh
output/    训练产出
mlruns/    MLflow 实验记录
```

数据集与基座模型不在仓库里,在共享卷的 `$MODELSCOPE_CACHE/local/` 下(XDG 目录派生,immutable),见教程第 1 步。

## 注意

以下是容易踩的坑:

- 训练命令一律 `uv run --no-sync`:`uv sync` 会清掉 pyproject 之外的 deepspeed 和 flash-attn。
- 新 shell 若没 source 过 set_cache.sh,uv 会把缓存和解释器装回本机 home,共享即失效;配置里的 `${oc.env:MODELSCOPE_CACHE}` 也会解析失败。接好 bashrc 后开新终端再干活。
- `RUN_NAME` 每次实验都要重新 export,复用旧值会把新实验写进旧 run 的目录。

## 延伸阅读

教程各处实践的出处与进阶材料,按建议阅读顺序排列:前四篇通读,之后的按场景取用,每条注明了什么时候读。

1. Karpathy《A Recipe for Training Neural Networks》(karpathy.github.io):增量复杂化、冒烟先行、固定种子的方法论来源,篇幅一小时,最先读。
2. Stas Bekman《Machine Learning Engineering Open Book》(GitHub: stas00/ml-engineering):BLOOM 训练工程师写的大规模训练实操——多卡调试、吞吐、稳定性,和 8×H200 full-tune 最对口,当工具书常备。
3. Patrick Mineault《The Good Research Code Handbook》(goodresearch.dev):研究代码从混乱到结构化的整理路径——目录结构、何时抽象、怎么测试,一个周末读完。
4. Google《Rules of Machine Learning》:第一条流水线要简单、基础设施先行的三阶段论,用来判断当前阶段该投入多少工程。
5. Michael Lones《How to avoid machine learning pitfalls》(arXiv,持续更新):从数据泄漏到评估作弊的研究者踩坑清单,设计实验和评估方案前过一遍。
6. Andrew Ng《Machine Learning Yearning》(免费电子书):实验迭代策略——先跑通、错误分析、何时加数据何时换模型,结果不如预期时读。
7. Josh Tobin《Troubleshooting Deep Neural Networks》(讲义/长文):把神经网络的沉默失败做成系统排查流程图,训练不对劲时查。
8. Cookiecutter Data Science:数据不可变原则与项目目录惯例,教程第 1 步的出处,浏览即可。
9. Made With ML(madewithml.com):课程体,从数据版本、实验追踪到部署的 MLOps 全流程,想系统补课时学。
10. 李沐《实用机器学习》(斯坦福 2021 秋季课,B 站有中文讲解):数据采集、标注到部署的工程视角课程,中文资源里最接近工程化教程的一个。
11. Papers with Code《ML Code Completeness Checklist》:代码发布与投稿前的完备性清单,开源或交接前对照一遍。
