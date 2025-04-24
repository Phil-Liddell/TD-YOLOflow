
<div align="center">

# 🚀 TD-Yoloflow 🚀

**Integrate Powerful YOLO Workflows Directly into TouchDesigner**

![Python](https://img.shields.io/badge/python-3.12.9-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8.89-green.svg)

</div>

---

## 🌟 Features

- ✅ One `.tox` file integration
- ✅ Automatic cloning and setup
- ✅ OSC & NDI/Shared-memory connectivity
- ✅ GUI and Headless launch options

---

## 🛠️ Launch Modes

| Mode              | Command                | Description                                         |
|-------------------|------------------------|-----------------------------------------------------|
| 🖥️ **Workflow GUI** | `python main.py`       | Capture → Review → Train → Test via Qt interface   |
| 🚀 **Headless**      | `python headless.py`   | Run inference only (ideal for production setups)   |

---

## 🔄 Data & Control Pathways (v0.1)

| Direction                      | Transport                                      |
|--------------------------------|------------------------------------------------|
| 🎥 **Video IN → TD-Yoloflow**  | NDI *(shared-memory coming soon)*              |
| 🎛️ **Control TD ↔ TD-Yoloflow**| OSC *(default: in 9000 / out 8860)*            |
| 📸 **Frames → TD**             | OSC + Shared-Memory TOP                        |

---

## 🎯 GUI Interface

| Tab           | Functionality                                        | Key Features                                                |
|---------------|------------------------------------------------------|-------------------------------------------------------------|
| 📷 **Capture**  | Record and track YOLO-labeled images                 | Class manager, FPS-limited, live tracking                   |
| 🔍 **Review**   | Scrub, edit, and manage datasets                     | Big previews, interactive bounding boxes, batch editing     |
| 📈 **Train**    | Launch and monitor YOLO model training               | Interactive configs, live metrics, cloud or local           |
| 🚦 **Test**     | Real-time inference and streaming results            | Adjustable confidence/IoU, frame annotations, OSC feedback  |

---

## 📌 Requirements

| Component     | Version              |
|---------------|----------------------|
| 🐍 Python     | 3.12.9 (64-bit)      |
| 🖥️ CUDA Toolkit | 11.8.89            |

⚠️ **Important:** While CPU-only usage is possible, GPU acceleration is highly recommended for optimal performance. Ensure your GPU drivers match CUDA Toolkit 11.8.

---

## 🚧 Project Roadmap

- ☁️ Cloud-based training & inference
- ⚡ Real-time shared-memory video inputs
- 🎯 Advanced tracking enhancements (multi-object, re-ID)
- 🧩 Expanded tasks: oriented boxes, classification, segmentation

---

## 🚀 Quick-start

```bash
git clone https://github.com/yourname/td-yoloflow.git
cd td-yoloflow
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# GUI Mode
python main.py

# Headless Mode
python headless.py
