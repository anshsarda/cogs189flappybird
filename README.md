# EEG Blink-Controlled Flappy Bird

This project implements a simple **EEG-based brain–computer interface (BCI)** that allows a user to control Flappy Bird using **eye blink signals recorded from an OpenBCI Cyton board**.

Motivated by real-world accessibility applications, this system is designed to enable hands-free interaction for individuals with limited or no hand mobility — allowing them to play games using only brain signals.

The full pipeline:
1. Collects labeled EEG blink data
2. Trains a classifier to detect intentional blink events
3. Streams live EEG data in real time
4. Translates classifier output into jump commands in the game

> Built on top of [FlapPyBird](https://github.com/sourabhv/FlapPyBird) by sourabhv, modified to integrate our BCI system. The blink data collection and labeling schema was adapted from Simon's original EEG recording code, which we modified to support our specific blink detection and labeling requirements.

---

## System Pipeline

The project has two distinct phases: an **offline training phase** and a **live game phase**. The training phase is run once ahead of time to build the model — it is not part of the real-time game loop.

**Offline Training (run once before playing):**
```
Participant Blink Session  (run_blink.py)
   ↓
Labeled EEG Dataset
   ↓
TRCA Classifier Training   (scripts/train_trca.py)
   ↓
model.joblib
```

**Live Game Pipeline (real-time during gameplay):**
```
OpenBCI Cyton Board
   ↓
BrainFlow Data Stream  (cyton_stream.py)
   ↓
EEG Preprocessing      (preprocessing.py)
   ↓
Blink Classifier       (classifier_interface.py + model.joblib)
   ↓
BCI Controller         (bci_controller.py)
   ↓
Flappy Bird Game       (flappy.py)
```

The game and EEG processing are intentionally **decoupled**, so the classifier can be swapped or retrained without modifying game logic.

---

## Project Structure

```
project/
│
├── flappy.py
│   Flappy Bird game engine (pygame)
│   Accepts jump commands from keyboard or BCI controller
│
├── bci_controller.py
│   Brain of the live interface — applies cooldown logic to
│   classifier output to prevent a single blink from
│   triggering multiple jumps
│
├── cyton_stream.py
│   Connects to the OpenBCI Cyton board via BrainFlow
│   Maintains hardware connection and streams raw EEG voltage
│
├── preprocessing.py
│   Filters and cleans the raw EEG signal
│   Isolates the high-amplitude, low-frequency features
│   characteristic of a deliberate eye blink
│
├── classifier_interface.py
│   Loads the trained model and exposes a simple yes/no
│   blink detection interface to the rest of the system
│
├── run_blink.py
│   Data collection script (offline, run before playing)
│   Prompts participant to blink at specific times to produce
│   a labeled EEG dataset. Adapted from Simon's original EEG
│   recording schema, modified for blink detection and labeling.
│
├── scripts/train_trca.py
│   Classifier training script (offline, run before playing)
│   Trains a TRCA-based model on the collected blink data
│   and outputs model.joblib for use during live gameplay
│
└── model.joblib
    Saved classifier weights used for live inference
```

---

## Environment Setup (Windows 11)

Create and activate a Python virtual environment:

```bash
pip install virtualenv
virtualenv pyenv --python=3.11.9
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
pyenv\Scripts\activate
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

Install **brainda** (used for EEG model utilities):

```bash
git clone https://github.com/TBC-TJU/brainda.git
cd brainda
pip install -r requirements.txt
pip install -e .
```

---

## Data Collection

Before collecting blink data, have the participant sit still to establish a **resting-state baseline**. This helps characterize background EEG and filter out noise unrelated to blinking (e.g., muscle tension, environmental interference).

Blink data is collected using a structured on-screen prompt ("Blink Now"):

```bash
python run_blink.py
```

- Each run collects **60 blink** and **60 non-blink** windows
- Windows are 1–2 seconds, centered on the "Blink Now" timestamps
- Participants should minimize jaw clenching and body movement, as EMG artifacts can interfere with blink detection
- Change the run number inside `run_blink.py` to save multiple runs

The recorded data is then used to train the blink detection classifier.

---

## Training the Classifier

After collecting data, train the TRCA-based classifier:

```bash
python scripts/train_trca.py
```

This script finds optimal spatial filters that amplify blink-related signal while suppressing background EEG noise. The trained model is saved as `model.joblib`.

> **Note on accuracy:** With 3 training runs (~60 blinks / 60 non-blinks per run), our model achieved ~56% accuracy — slightly above chance. Accuracy is limited by the small dataset size and high variance across runs. More training runs are strongly recommended.

---

## Running the Game

Start the game:

```bash
python flappy.py
```

Controls:
- **Spacebar** — manual jump
- **Intentional eye blink** — jump triggered by EEG classifier

During gameplay, the game loop repeatedly calls `should_jump()`. The BCI controller returns `True` when a blink is confidently detected, triggering the bird to flap.

---

## Real-Time Control Loop

```
Game Loop
   ↓
bci_controller.should_jump()
   ↓
Read EEG window          (cyton_stream.py)
   ↓
Preprocess signal        (preprocessing.py)
   ↓
Run classifier           (classifier_interface.py)
   ↓
Return True/False jump command
```

---

## Known Limitations

- **Class imbalance** — non-blink states occur far more frequently than blink states, causing the model to be overly conservative and require very deliberate blinks to trigger a jump
- **Processing latency** — preprocessing window sizes introduce delay between the physical blink and the in-game response, making controls feel sluggish at times
- **Small dataset** — only 3 training runs were completed; accuracy and variance improve significantly with more data

---

## Potential Improvements

- **Data augmentation / oversampling** of blink events to reduce classifier bias toward non-blink predictions
- **Asynchronous parallel processing** and smaller EEG windows to reduce end-to-end latency and improve real-time responsiveness
