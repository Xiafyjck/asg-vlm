#!/usr/bin/env python3
"""ms-swift SFT 启动器:hydra 组合配置,转 TrainArguments,调 sft_main。

经 train/*.sh 包装运行;直接调也行,任意字段可在命令行覆盖:
    torchrun --nproc_per_node 8 train/sft.py --config-name stage1_answer_only \
        learning_rate=2e-5 num_train_epochs=3
"""
import os
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


@hydra.main(version_base=None, config_path="../configs", config_name=None)
def main(dict_cfg: DictConfig) -> None:
    load_dotenv(ROOT / ".env")
    cfg = OmegaConf.to_container(dict_cfg, resolve=True)

    if not cfg.get("model"):
        raise SystemExit("未指定 model:请在配置里写,或用 model=<checkpoint 路径> 覆盖")

    # RUN_NAME 由启动命令 export,torchrun 各 rank 一致
    run_name = os.environ.get("RUN_NAME") or f"run-{time.strftime('%Y%m%d-%H%M%S')}"
    if not cfg.get("output_dir"):
        cfg["output_dir"] = str(ROOT / "output" / run_name)  # 运行时派生值,非配置默认

    # 相对路径统一到仓库根,不受 cwd 影响
    for key in ("dataset", "val_dataset"):
        if key in cfg:
            cfg[key] = [p if os.path.isabs(p) else str(ROOT / p) for p in cfg[key]]
    model_as_path = ROOT / str(cfg["model"])
    if model_as_path.exists():
        cfg["model"] = str(model_as_path)
    elif os.path.isabs(str(cfg["model"])) and not Path(cfg["model"]).exists():
        raise SystemExit(
            f"模型路径 {cfg['model']} 不存在:先在 4090 上用 modelscope download 下载(见 README 第 1 步)"
        )

    # MLflow 接线:file store 在共享卷,离线训练机直接写入。
    # report_to 在 base.yaml 显式声明,代码不做配置回退;
    # 环境变量 setdefault 属环境层接线,允许外部覆盖,不遮蔽配置。
    if "mlflow" in cfg["report_to"]:
        os.environ.setdefault("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri())
        os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "asg-vlm")
        if not cfg.get("run_name"):
            cfg["run_name"] = run_name  # 运行时派生值,非配置默认

    from swift.llm import TrainArguments, sft_main
    sft_main(TrainArguments(**cfg))


if __name__ == "__main__":
    main()
