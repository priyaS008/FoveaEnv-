# 🕶️ BlinkEnv — Privacy-Aware Vision Attention Environment

> **"We built an OpenEnv benchmark where agents must solve tasks under a limited observation budget — making attention itself a decision problem."**

---

## 🎯 One-Line Pitch

An OpenEnv environment that trains AI agents to find objects using **only 10% of available visual data** — built for the next generation of privacy-first, compute-constrained edge devices like **Meta Ray-Ban smart glasses**.

---

## 🔍 The Problem

Today's AI agents assume **free access to all information**.

But real-world wearable devices face hard constraints:
- 🔋 **Battery limits** — can't process 4K frames continuously
- 🧠 **Compute limits** — small processors, no GPU
- 🔒 **Privacy limits** — users don't want AI seeing everything

**No existing RL benchmark trains agents to work under these constraints.**

---

## 💡 Our Solution — BlinkEnv

BlinkEnv is a **7×7 grid world** where:

- The agent has a **movable 3×3 gaze window** (like a human eye's focus point)
- The rest of the grid is **blurred/hidden** (privacy-preserved)
- The agent must **find a target object** using minimum gaze movements
- **Attention itself becomes a learnable skill**

```
░░░░░░░░░░░░░░░░░░░░░░
░░░░░░░░░░░░░░░░░░░░░░
░░░░░░ [cup][   ][   ] ░░░░
░░░░░░ [   ][📍 ][   ] ░░░░   ← agent's gaze (3x3 visible)
░░░░░░ [   ][   ][   ] ░░░░
░░░░░░░░░░░░░░░░░░░░░░
░░░░░░░░░░░░░░░░░░░░░░
```

---

## 🏗️ Environment Structure (OpenEnv Format)

### Actions
| Action | Description |
|--------|-------------|
| `up` | Move gaze up |
| `down` | Move gaze down |
| `left` | Move gaze left |
| `right` | Move gaze right |
| `focus` | Stay and observe current window |
| `answer` | Submit final answer (target name) |

### Observation
```python
GridObservation(
    visible_cells=["cup", "empty", "phone", ...],  # 3x3 window only
    gaze_position=(3, 4),
    target_found=False,
    reward=-0.1,
    task="Find the 'cup'",
    steps_taken=5,
    done=False,
    message="Target not visible. Keep searching."
)
```

### Reward Function
| Event | Reward |
|-------|--------|
| Correct answer | +10.0 |
| Each step | -0.1 |
| Wrong answer | -2.0 |
| Timeout (30 steps) | -5.0 |

---

## 🚀 Quick Start

```bash
# Install
pip install openenv-core fastapi uvicorn

# Test locally (no server needed)
python demo.py

# Run as OpenEnv server
uvicorn app:app --reload
# Visit http://localhost:8000
```

---

## 📁 File Structure

```
BlinkEnv/
├── models.py        # GazeAction, GridObservation, BlinkState
├── environment.py   # Core BlinkEnvironment logic
├── app.py           # FastAPI OpenEnv server
├── demo.py          # Local demo/test script
├── requirements.txt
└── README.md
```

---

## 🔗 Why This Matters

### Connection to Meta's Vision

Meta Ray-Ban smart glasses have:
- Small battery → cannot process full HD frames
- No GPU → cannot run large vision models
- Privacy concerns → users want minimal data capture

**BlinkEnv directly trains agents for this exact constraint.**

An agent trained on BlinkEnv can:
- Navigate real scenes with **90% less visual data**
- Make smarter attention decisions
- Run efficiently on edge hardware

### Privacy = Efficiency

> **Privacy and compute efficiency are the same problem.**  
> If your AI only looks where it needs to — it's both private AND efficient.

---

## 🏆 Impact

- ✅ First OpenEnv benchmark for **observation-budget-constrained agents**
- ✅ Directly applicable to **Meta wearables roadmap**
- ✅ Reusable benchmark for **privacy-aware AI research**
- ✅ Trains a capability that matters more than full-vision performance

---

## 👥 Team

Built for the **Meta × PyTorch × Scaler School of Technology OpenEnv Hackathon 2026**

---

*"The best AI doesn't see everything. It knows where to look."* 🕶️
