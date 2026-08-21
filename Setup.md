# SmolVLA adaptation on LIBERO

Ноутбук (Experiments) запускает полный pipeline: подготовка LIBERO, seen fine-tuning, K=0 language control, target baselines, две исследовательские гипотезы, evaluation, Bonus B и выбор failure rollouts.

Исследуемые методы:

- H1: latent dynamics regularization.
- H2: video progress from TimeRewarder with RA-BC.


## Структура репозитория

```text
.
├── Experiments.ipynb
├── src/
│   ├── __init__.py
│   ├── settings.py
│   ├── utils.py
│   ├── data.py
│   ├── models.py
│   ├── train_baselines.py
│   ├── h1_dynamics.py
│   ├── h2_progress.py
│   ├── timerewarder_runner.py
│   ├── evaluation.py
│   ├── analysis.py
│   └── bonus.py
├── README.md
├── REPORT.docx
├── requirements.txt
└── .gitignore
```

`Experiments.ipynb` описывает и запускает эксперимент сверху вниз.

`src/settings.py` хранит dataset/model IDs, target instructions, K, seeds и пути runtime-каталогов.

`src/data.py` скачивает `nvidia/LIBERO_LeRobot_v3`, исправляет gripper convention, пересчитывает dataset statistics, создаёт first-K target subsets и выполняет replay smoke test.

`src/models.py` содержит только собственные model-level компоненты H1: `DynamicsConfig`, `LatentDynamicsHead`, latent pooling, dynamics loss и восстановление clean action chunk из SmolVLA flow velocity.

`src/train_baselines.py` запускает seen fine-tuning, обязательный native target fine-tune и matched LoRA-only control через официальный `lerobot-train`.

`src/h1_dynamics.py` содержит seen-only training dynamics prior и target LoRA training loop с latent-dynamics regularization.

`src/h2_progress.py` готовит target videos, обучает официальный TimeRewarder, переводит его reward в progress и запускает official RA-BC target training.

`src/timerewarder_runner.py` — небольшой runner для TimeRewarder inference в его отдельном Python environment.

`src/evaluation.py` запускает `lerobot-eval`, target checkpoint evaluation и paired K=0 wrong-instruction control.

`src/analysis.py` собирает rollout-level success, success rates с простым Wilson 95% interval, cost curve и список failed videos.

`src/bonus.py` оценивает rollout videos через TimeRewarder и Robometer-4B-LIBERO и считает ranking/proxy-pressure metrics.

`src/utils.py` содержит только общий subprocess runner с записью логов.

## Порядок эксперимента

Notebook выполняет:

```text
data
-> gripper replay
-> seen policy on full libero_90
-> K=0 correct/wrong instruction
-> native baseline
-> LoRA-only control
-> H1 latent dynamics
-> H2 TimeRewarder + RA-BC
-> main evaluation
-> Bonus B
-> failure candidates
```

Target budgets: `K = 5, 10, 25`.

Target-policy training seeds: `42, 123`.

Evaluation: 20 LIBERO episodes на каждую task/method/K/seed cell. K=0 использует один shared seen checkpoint.

## Runtime outputs

Runtime-каталоги не входят в git.

```text
data/
├── raw/                         downloaded NVIDIA suites
├── fixed/                       gripper-corrected libero_90/libero_goal
├── subsets/                     physical first-K target datasets
├── timerewarder/                videos prepared for TimeRewarder
└── subset_manifest.csv          exact source episodes used in each K

outputs/
├── seen/                        seen-policy checkpoint
├── heads/h1_dynamics/           H1 latent-dynamics prior
├── target/
│   ├── native/                  assignment baseline checkpoints
│   ├── lora/                    matched LoRA-only checkpoints
│   ├── h1_dynamics/             H1 policy checkpoints
│   └── h2_progress/             H2 RA-BC policy checkpoints
├── reward/
│   ├── timerewarder/            TimeRewarder checkpoints
│   └── progress/                progress parquet files for RA-BC
├── eval/                        raw LeRobot eval artifacts and rollout videos
├── bonus_b/                     raw reward-model inference artifacts
└── logs/                        training/evaluation logs

results/
├── replay_smoke.csv
├── language_control.csv
├── language_control_summary.csv
├── evaluation.csv               rollout-level simulator success
├── success_rates.csv            success + Wilson 95% CI by method/task/K/train seed
├── cost_curve.csv
├── cost_curve.png
├── bonus_b/
│   ├── scores.csv
│   ├── checkpoint_scores.csv
│   ├── ranking.csv
│   └── reward_pressure.csv
└── failure_candidates.csv
```


## Hardware

Финальный notebook рассчитан на один Linux-сервер класса:

```text
GPU: NVIDIA RTX PRO 6000 Blackwell, 96 GB
vCPU: 16+
free SSD: 200 GB recommended, 150 GB hard minimum
NVIDIA driver: >= 570.86
```

Batch sizes фиксированы и не меняются автоматически:

```text
seen policy:       32
target policies:    8
H1 dynamics prior: 32
TimeRewarder:      16
```

Основная среда проекта фиксирована на Python 3.12, LeRobot 0.6.1 и PyTorch 2.11.0 + CUDA 12.8. Notebook проверяет эти версии, CUDA, `ffmpeg`, минимум 80 GB VRAM и свободное место до начала обучения.

TimeRewarder запускается из отдельного окружения, которое `src/h2_progress.py` создаёт через `uv`: Python 3.9, PyTorch 2.7.1 + CUDA 12.8. Это Blackwell-compatible patch поверх исходного TimeRewarder environment, который был рассчитан на старый PyTorch/CUDA stack. Системные `libgl1/libegl1/libglib2.0-0` устанавливаются заранее, поэтому старый OpenCV/MMCV stack не зависит от desktop session.

Robometer остаётся в собственном `uv`-окружении его официального репозитория.

Для headless LIBERO notebook до импортов задаёт:

```text
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```


## Запуск

Требуются Linux, NVIDIA driver с поддержкой CUDA 12.8, Python 3.12 и persistent disk. Для Ubuntu/Debian перед Python environment системные зависимости:

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg libgl1 libegl1 libglib2.0-0
```

Создайте основную среду именно на Python 3.12 и установите зависимости до запуска Jupyter:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python -m jupyter notebook
```

Откройте `Experiments.ipynb` из корня репозитория и выполняйте ячейки сверху вниз. Setup-ячейки повторно проверяют requirements и фактические версии Python, LeRobot, PyTorch/CUDA, GPU, `ffmpeg` и свободный диск до начала обучения.

PyTorch ставится отдельной официальной CUDA 12.8 командой до `requirements.txt`; это исключает неоднозначность между CPU/PyPI и CUDA wheels. `requirements.txt` затем pin-ит LeRobot и совместимые dependency ranges:

```text
lerobot[training,evaluation,smolvla,libero,peft,sarm,notebook]==0.6.1
```

До полного запуска рекомендуется выполнить setup, data preparation и replay smoke. Все три target demonstrations должны replay-иться с `success=True`; если это не так, дальнейший training запускать нельзя.

Повторный запуск безопасен для законченных stages: функции возвращают уже существующие final checkpoints или raw eval artifacts. Незавершённый стандартный LeRobot training resume-ится из `checkpoints/last`. H1 сохраняет собственный lightweight resume state.
