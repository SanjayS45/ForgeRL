# Language-Conditioned RLHF for MetaWorld Robotics

A complete **Reinforcement Learning from Human Feedback (RLHF)** pipeline for training robotic manipulation policies using MetaWorld V3 environments. The system supports language-conditioned tasks, allowing natural language instructions to guide robot behavior.

## 🎯 Features

- **MetaWorld V3 Integration**: Properly configured Sawyer arm environments with correct Task/Env handling
- **Reward Model Training**: Neural reward models with sentence transformer encodings for language conditioning
- **PPO Policy Training**: Clean PPO implementation with reward model integration
- **React Dashboard**: Modern web UI for monitoring training and managing experiments
- **FastAPI Backend**: REST API for programmatic control of the training pipeline
- **Flexible Dataset Support**: Load any dataset of trajectory preference pairs

## 📁 Project Structure

```
language_rlhf_robotics/
├── backend/              # FastAPI backend
│   └── api.py           # REST API endpoints
├── checkpoints/         # Model checkpoints
├── envs/                # Environment wrappers
│   ├── metaworld_wrapper.py  # MetaWorld V3 wrapper
│   ├── language.py      # Sentence transformer encoding
│   └── instructions.py  # Task instructions
├── experiments/         # Datasets and experiment files
├── frontend/            # React dashboard
│   └── src/
│       └── App.jsx      # Main React application
├── logs/                # Training logs
├── policy/              # RL policy code
│   ├── networks.py      # Actor-Critic networks
│   └── ppo.py          # PPO algorithm
├── preferences/         # Preference labeling
│   └── synthetic.py     # Synthetic preference oracle
├── reward_model/        # Reward model
│   ├── model.py        # Neural reward models
│   └── trainer.py      # Training utilities
├── scripts/             # Utility scripts
│   ├── generate_trajectories.py  # Dataset generation
│   ├── preprocess_dataset.py     # Data preprocessing
│   └── test_env.py               # Environment testing
├── training/            # Training scripts
│   ├── train_reward_model.py     # Reward model training
│   └── train_policy.py           # Policy training
├── utils/               # Utilities
│   ├── data_utils.py    # Dataset loading/saving
│   ├── instruction_encoder.py   # Text encoding
│   └── logging_utils.py # Logging and checkpointing
└── requirements.txt     # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd language_rlhf_robotics

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Test Environment Setup

Verify MetaWorld is correctly installed:

```bash
python scripts/test_env.py --task reach-v3 --test-wrapper
```

### 3. Generate Training Dataset

Generate 1000+ preference pairs:

```bash
python scripts/generate_trajectories.py \
    --task reach-v3 \
    --num-pairs 1000 \
    --horizon 50 \
    --output experiments/trajs.pkl
```

### 4. Train Reward Model

```bash
python training/train_reward_model.py \
    --data experiments/trajs.pkl \
    --epochs 50 \
    --batch-size 32 \
    --checkpoint-dir checkpoints/reward_model
```

### 5. Train Policy with Learned Reward

```bash
python training/train_policy.py \
    --task reach-v3 \
    --reward-model checkpoints/reward_model/best_model.pt \
    --instruction "reach the target" \
    --total-steps 500000
```

## 🖥️ Web Dashboard

### Start Backend

```bash
cd language_rlhf_robotics
uvicorn backend.api:app --reload --port 8000
```

### Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Access the dashboard at `http://localhost:5173`

### Deploy to Netlify

1. Build the frontend:
   ```bash
   cd frontend
   npm run build
   ```

2. Deploy the `dist` folder to Netlify

3. Update `netlify.toml` with your backend URL

## 📦 Available Datasets

Pre-generated datasets for training:

| Dataset | Task | Pairs | Description |
|---------|------|-------|-------------|
| `reach_v3_large.pkl` | reach-v3 | 5,000 | Large reach task dataset |
| `push_v3_dataset.pkl` | push-v3 | 2,000 | Push task dataset |
| `pick_place_v3_dataset.pkl` | pick-place-v3 | 2,000 | Pick and place dataset |
| `door_open_v3_dataset.pkl` | door-open-v3 | 1,500 | Door opening dataset |
| `drawer_open_v3_dataset.pkl` | drawer-open-v3 | 1,500 | Drawer opening dataset |
| `multi_task_9k.pkl` | Multiple | 9,000 | Combined multi-task dataset |

### Generate Custom Dataset

```bash
python scripts/generate_large_dataset.py \
    --task reach-v3 \
    --num-pairs 5000 \
    --output experiments/my_dataset.pkl
```

### Combine Datasets

```bash
python scripts/combine_datasets.py \
    --inputs experiments/reach*.pkl experiments/push*.pkl \
    --output experiments/combined.pkl
```

## 📊 Dataset Format

Preference datasets use the format:
```python
[
    (instruction: str, trajectory_a: List[Tuple], trajectory_b: List[Tuple]),
    ...
]
```

Where each trajectory is:
```python
[(observation, action), (observation, action), ...]
```

Supported file formats: `.pkl`, `.pickle`, `.json`

## 🔧 Configuration

### Reward Model Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 50 | Training epochs |
| `batch_size` | 32 | Batch size |
| `learning_rate` | 1e-4 | Learning rate |
| `hidden_dim` | 256 | Hidden layer dimension |
| `loss_type` | cross_entropy | Loss function |

### Policy Training (PPO)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_steps` | 500000 | Total training steps |
| `steps_per_update` | 2048 | Steps between updates |
| `epochs_per_update` | 10 | PPO epochs per update |
| `gamma` | 0.99 | Discount factor |
| `clip_ratio` | 0.2 | PPO clip ratio |

## 🔌 API Reference

### Datasets

- `GET /api/datasets` - List available datasets
- `POST /api/datasets/upload` - Upload new dataset
- `DELETE /api/datasets/{name}` - Delete dataset

### Training

- `POST /api/reward-model/train` - Start reward model training
- `POST /api/policy/train` - Start policy training
- `GET /api/training/{job_id}` - Get training status
- `GET /api/training/{job_id}/history` - Get training history

### Models

- `GET /api/models` - List trained models
- `GET /api/models/{type}/{name}/download` - Download model

## 🐛 Common Issues

### MetaWorld Task vs Env Error

**Problem**: `'Task' object has no attribute 'reset'`

**Solution**: Use `ml1.train_classes[task_name]` to get the environment class, not `ml1.train_tasks[0]`. The `train_tasks` list contains Task configuration objects, not environments.

```python
# ❌ Wrong
env = ml1.train_tasks[0]

# ✅ Correct
env_cls = ml1.train_classes['reach-v3']
env = env_cls()
task = ml1.train_tasks[0]
env.set_task(task)
```

### Import Errors

Run all scripts from the project root:

```bash
cd language_rlhf_robotics
python scripts/generate_trajectories.py
```

### CUDA Out of Memory

Reduce batch size or use CPU:

```bash
python training/train_reward_model.py --batch-size 16 --device cpu
```

## 📝 Citation

If you use this codebase, please cite:

```bibtex
@software{language_rlhf_robotics,
  title = {Language-Conditioned RLHF for MetaWorld Robotics},
  year = {2024},
  url = {https://github.com/your-repo}
}
```

## 📄 License

MIT License - see LICENSE file for details.

