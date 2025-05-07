# 🚀 TD-YOLOflow 🚀  
**Combine Powerful YOLO Workflows With TouchDesigner**

![Python](https://img.shields.io/badge/python-3.11.0-blue.svg) ![CUDA](https://img.shields.io/badge/CUDA-11.8.89-green.svg)

---

## 🌟 Features
- ✅ Single **`.tox`** drop-in for TouchDesigner  
- ✅ Automatic repository clone & virtual-env setup  
- ✅ Full YOLO detection workflow. Capture with auto tracking -> Review/ label data -> Train model (local and cloud options, nano to large model size, training analytics) -> Deploy model 

---

## 🔄 Data & Control Pathways (v0.1)
| Direction | Transport |
|-----------|-----------|
| 🎥 **Video IN → TD-Yoloflow** | **NDI** *(shared-memory input coming soon)* |
| 🎛️ **Control TD ↔ TD-Yoloflow** | **OSC** *(in 9000 / out 8860)* |
| 📸 **Frames & Detections → TD** | **OSC** + Shared-Memory TOP |

---

## 🎯 GUI Tabs – How to use them
| Tab | Typical workflow | Key tricks |
|-----|------------------|-----------|
| 📷 **Capture** | 1. Select NDI source → **Draw Box** → **Record**.<br>YOLO detects while the DINO tracker keeps the box centred; images & YOLO labels saved to `data/`. | • Live class manager<br>• FPS-limited recorder<br>• Drag box during recording |
| 🔍 **Review** | 1. Choose a class folder → scrub thumbnails → double-click to edit boxes → **Save**. | • Batch delete with multi-select<br>• Auto-refresh every 2 s |
| 📈 **Train** | 1. Set epochs / img-size / augments → **Start Train**.<br>Live loss & mAP curves stream in; checkpoints in `runs/train/`. | • Timestamped run names avoid collisions |
| 🚦 **Test** | 1. Load a `.pt` model → **Start** → watch real-time inference.<br>Adjust **Confidence** & **IoU** sliders. | • Annotated frames back to TD via shared memory<br>• Each detection sent as OSC bundle |

---

## 📌 Requirements
| Component | Version |
|-----------|---------|
| 🐍 **Python** | 3.11.0 (64-bit) |
| 🖥️ **CUDA Toolkit** | 11.8.89 |

> CPU-only runs for testing, but a CUDA-capable GPU is strongly recommended.

---

## 🚀 Quick-start (TouchDesigner-first)

1. **Drag `TD_YOLOflow.tox` into TouchDesigner** and let it cook.  
2. Select the new **TD_YOLOflow** COMP, open its **Parameters** pane, and set **Yoloflow Folder** – pick / create an empty folder where the repo will be cloned.  
3. Press **Clone Repo** ✛ – the component clones the latest TD-Yoloflow into that folder.  
4. When **Clone Status** shows ✅, click **Create Venv** – a Python 3.11 virtual-env is created and all requirements are installed.  
5. Finally hit **Launch YOLOflow** – the app opens in its own console window. GUI mode launches by default; toggle **Headless** for background service only.  

You can reopen the project later and jump straight to **Launch YOLOflow**; the COMP remembers the folder and virtual-env.

---
## 🚧 Roadmap
- ☁️ Cheaper cloud training & live cloud inference  
- ⚡ Shared-memory video **input**  
- 🎯 Improved Capture-tab tracking (multi-object, SAM for better accuracy and segmentation)  
- 🧩 New tasks: oriented boxes, classification, segmentation  
- ✨ General quality-of-life improvements  


---


## License

MIT License  

Copyright (c) 2025 Philip Liddell

Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the “Software”), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.





