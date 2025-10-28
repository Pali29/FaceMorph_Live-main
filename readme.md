# 🧠 Face-Morph Live

**Face-Morph Live** is a real-time face morphing and virtual camera application built in **Python** using **OpenCV**, **MediaPipe**, and **PyVirtualCam**.  
It captures live video from your webcam, detects facial landmarks, and seamlessly morphs your face with a target image — outputting the result to a virtual camera that can be used in apps like **Zoom**, **Discord**, or **OBS**.

---

## ✨ Features

- 🎥 Real-time webcam capture  
- 🧩 Face landmark detection using **MediaPipe Face Mesh**  
- 🌀 Smooth morphing between live face and target image  
- 📡 Output to a **virtual camera** (via PyVirtualCam)  
- ⚙️ Cross-platform: works on **Windows** and **Linux**  
- 🧵 Supports threading to improve performance (optional)

---

## 🏗️ Project Structure

```

FaceMorph_Live/
│
├── capture/
│   └── face_tracker.py          # Handles webcam capture and face tracking
│
├── morph/
│   ├── morph_core.py            # Core morphing logic
│   ├── utils.py                 # Helper functions (landmark extraction, etc.)
│   └── triangles.py             # Delaunay triangulation logic
│
├── assets/
│   └── faces/
│       └── source.jpeg          # Target image for morphing
│
├── misc/
│   └── linux_cam_start.py       # Helper for Linux virtual cam setup
│
├── trial.py                     # Main entry point
└── README.md                    # You are here

````

---

## ⚙️ Requirements

Before running, make sure you have the following installed:

### 🧩 Dependencies
```bash
pip install opencv-python mediapipe pyvirtualcam numpy
````

For Linux users:

```bash
sudo apt install v4l2loopback-dkms
```

---

## 🚀 Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/FaceMorph_Live.git
cd FaceMorph_Live
```

### 2. Add a target face image

Place your target image in:

```
assets/faces/source.jpeg
```

### 3. Start the virtual camera (Linux only)

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="FaceMorphCam" exclusive_caps=1
```

### 4. Run the live morphing app

```bash
python trial.py
```

You should see your morphed face stream to `/dev/video10`.
Select **FaceMorphCam** as your webcam source in Zoom, Discord, or OBS.

---

## 🧠 How It Works

1. **Capture:** Reads webcam frames via `cv2.VideoCapture`.
2. **Detect:** Uses **MediaPipe FaceMesh** to detect 468+ facial landmarks.
3. **Morph:** Triangulates both faces and warps each triangle smoothly.
4. **Stream:** Sends morphed frames to a **virtual camera** via `pyvirtualcam`.

---

## 🧩 Troubleshooting

| Issue                                                  | Cause                      | Fix                                                                   |
| ------------------------------------------------------ | -------------------------- | --------------------------------------------------------------------- |
| `Device /dev/video10 is already in use`                | Virtual cam already active | Run `sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback ...` |
| Morphing very slow                                     | Complex morphing per frame | Enable threaded morph or pre-scale images                             |
| Mediapipe “NORM_RECT without IMAGE_DIMENSIONS” warning | Safe to ignore             | None needed                                                           |
| Black screen on Discord                                | App caching old camera     | Restart Discord / app                                                 |

---

## 🧵 (Optional) Threaded Version

To reduce lag, morphing can be run on a **background thread**,
allowing the webcam feed to continue smoothly while morphing happens asynchronously.

---

## 🧑‍💻 Tech Stack

| Component      | Library                    |
| -------------- | -------------------------- |
| Face Detection | MediaPipe FaceMesh         |
| Morphing       | OpenCV (Affine & Delaunay) |
| Virtual Camera | PyVirtualCam               |
| UI / Display   | OpenCV HighGUI             |
| Language       | Python 3.10+               |

---

## 📸 Example Output

|             Input             |               Target              |               Morphed               |
| :---------------------------: | :-------------------------------: | :---------------------------------: |
| ![live](assets/demo/live.png) | ![target](assets/demo/target.png) | ![morphed](assets/demo/morphed.png) |

*(Add demo screenshots here once available)*

---

## 🧾 License

This project is released under the **MIT License**.
Feel free to use, modify, and distribute it with attribution.

---

## 💬 Credits

Created with ❤️ by [Pali](https://github.com/pali29)

---