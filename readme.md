# 🚀 TD-Yoloflow 🚀  
**Combine Powerful YOLO Workflows With TouchDesigner**

![Python](https://img.shields.io/badge/python-3.12.9-blue.svg) ![CUDA](https://img.shields.io/badge/CUDA-11.8.89-green.svg)

---

## 🌟 Features
- ✅ Single **`.tox`** drop-in  
- ✅ Automatic repo clone + virtual-env setup  
- ✅ **OSC** control & **NDI / Shared-memory** data paths  
- ✅ Choice of **GUI** or **Headless** launch  

---

## 🛠️ Launch Modes
| Mode | Command | When to use it |
|------|---------|----------------|
| 🖥️ **Workflow GUI** | `python main.py` | Full pipeline: Capture → Review → Train → Test |
| 🚀 **Headless** | `python headless.py` | Production inference without a GUI |

---

## 🔄 Data & Control Pathways (v0.1)
| Direction | Transport |
|-----------|-----------|
| 🎥 **Video IN → TD-Yoloflow** | **NDI** *(shared-memory input coming soon)* |
| 🎛️ **Control TD ↔ TD-Yoloflow** | **OSC** *(default in 9000 / out 8860)* |
| 📸 **Frames & Detections → TD** | **OSC** + Shared-Memory TOP |

---

## 🎯 GUI Tabs – How to use them
| Tab | Typical workflow | Key tricks |
|-----|------------------|-----------|
| 📷 **Capture** | 1. Select NDI source → **Draw Box** → **Record**.<br>YOLO detects while the DINO tracker keeps the box centred; images and YOLO-format labels are saved to `data/`. | • Live class manager<br>• FPS-limited recorder<br>• Drag box during recording |
| 🔍 **Review** | 1. Choose a class folder → scrub thumbnails → double-click to edit boxes → **Save**. | • Batch delete with multi-select<br>• Auto-refresh every 2 s |
| 📈 **Train** | 1. Set epochs / image-size / augments → **Start Train**. Live loss & mAP curves stream in; checkpoints stored in `runs/train/`. | • Timestamped run names avoid collisions |
| 🚦 **Test** | 1. Load a `.pt` model → **Start** → watch real-time inference.<br>Adjust **Confidence** & **IoU** sliders. | • Annotated frames back to TD via shared memory<br>• Each detection sent as OSC bundle |

---

## 📌 Requirements
| Component | Version |
|-----------|---------|
| 🐍 **Python** | 3.12.9 (64-bit) |
| 🖥️ **CUDA Toolkit** | 11.8.89 |

> CPU-only works for testing, but a CUDA-capable GPU is strongly recommended.

---

## 🚧 Roadmap
- ☁️ Cheaper cloud training & live cloud inference  
- ⚡ Shared-memory video **input**  
- 🎯 Improved Capture-tab tracking (multi-object, re-ID)  
- 🧩 New tasks: oriented boxes, classification, segmentation  
- ✨ General quality-of-life improvements  

---

## 🚀 Quick-start
```bash
git clone https://github.com/yourname/td-yoloflow.git
cd td-yoloflow
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

# GUI mode
python main.py

# Headless mode
python headless.py
