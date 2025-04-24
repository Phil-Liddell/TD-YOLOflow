TD-Yoloflow
TD-Yoloflow drops into a project as a single .tox.
On first run it

clones this repo,

builds/updates a local virtual-env,

launches a Python application that talks to TouchDesigner over OSC and shared-memory/NDI**.**

Two launch flavours are provided:


Mode	Command inside the repo	Use-case
Workflow GUI	python main.py	Capture → Review → Train → Test via a Qt interface
Headless	python headless.py	Run inference only, no GUI (ideal for install machines)
Data / Control Path (v0.1)

Direction	Transport
Video IN → TD-Yoloflow	NDI (shared-memory input coming soon)
Control TD ↔ TD-Yoloflow	OSC (default in 9000 / out 8860)
Detections / Frames → TouchDesigner	OSC + Shared-Memory TOP
GUI Tabs

Tab	What you do with it	Highlights
Capture	Record images & YOLO labels from live NDI while a DINO tracker keeps the box on target.	Class manager, FPS-limited recorder, tracker follow.
Review	Scrub captured frames, tweak boxes, batch-clean the dataset.	Big preview, drag handles, delete images 
Train	Configure Ultralytics-YOLO, launch training, watch live metrics.	Collapsible parameter panes, timestamped run folders, local or future cheap-cloud back-ends.
Test	Run real-time inference on any model, stream results back to TD.	Confidence/IoU sliders, annotated frame sender, OSC burst per detection.

Requirements

Component	Version
Python	3.12.9 (64-bit)
CUDA Toolkit	11.8.89 
Note: TD-Yoloflow will still start on CPU-only machines, but training and real-time inference performance will be severely reduced.

Make sure your GPU drivers match CUDA 11.8 before installing the Python dependencies.


Roadmap
Cheaper cloud training & live cloud inference

Shared-memory video input for zero-latency grabs

Stronger Capture-tab tracking (multi-object, re-ID)

New tasks: oriented boxes, classification, segmentation

Manual Quick-start
bash
Copy
Edit
git clone https://github.com/yourname/td-yoloflow.git
cd td-yoloflow
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py                   # workflow GUI
# or
python headless.py               # headless mode
The TD-Yoloflow.tox handles these steps automatically inside TouchDesigner.

