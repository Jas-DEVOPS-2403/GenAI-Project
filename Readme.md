# RAG Agentic Fitness Coach

A real-time squat coaching system that runs entirely on your local machine. No cloud, no API keys, no internet connection required during use.

MediaPipe tracks your knee angle through a webcam at ~30fps. When you appear stuck or show a form fault, an agentic pipeline fires: FAISS retrieves relevant coaching cues from a local knowledge base, and a local LLM generates a short coaching cue that appears on screen in real time.

---

## How it works

The system runs two loops simultaneously:

**Fast loop (every frame)** — MediaPipe reads your knee angle and updates your rep count and movement phase (`up` / `down`). This runs at full camera speed and never blocks.

**Slow loop / agentic gate** — if you stay in the `down` phase below a threshold angle for ~1.5 seconds, the system classifies the fault, retrieves the 3 most relevant coaching cues from the FAISS index, and sends them as grounded context to the LLM. The LLM returns a single short cue which appears on screen.

Session memory prevents the coach from repeating the same cue if you keep triggering the same fault — it gives you time to self-correct.

```
Webcam
  └─ tracker.py          # MediaPipe — extracts knee angle
       └─ update_coach_logic.py   # Rep counting + anomaly gate
            └─ auditor.py         # Fault classification + snapshot
                 └─ rag_coach.py  # FAISS retrieval + Ollama LLM
                      └─ Screen overlay (COACH: ...)
```

---

## Requirements

- Python 3.10+
- A webcam
- [Ollama](https://ollama.com) installed and running

---

## Setup

**1. Install Ollama and pull the model**
```bash
ollama pull llama3.2:3b
```

**2. Start the Ollama server** (keep this terminal open)
```bash
ollama serve
```

**3. Install Python dependencies**
```bash
pip install faiss-cpu sentence-transformers ollama opencv-python
pip install mediapipe==0.10.14
```

> ⚠️ Pin mediapipe to `0.10.14` — newer versions have breaking API changes.

---

## Running

**Verify retrieval is working first (no webcam needed)**
```bash
python debug_rag.py
```
Check that the cues printed for each fault type make sense. If they look wrong, edit `FAULT_QUERIES` in `rag_coach.py` before proceeding.

**Run the full pipeline**
```bash
python test_tracker.py
```

Press `q` to quit.

---

## What you'll see

On screen:
- `Reps: N | Phase: up/down` — live rep counter and movement phase
- `COACH: <cue>` — coaching feedback when a fault is detected

In the console:
```
[RAGCoach] Loading sentence encoder...
[RAGCoach] Ready.
Agentic Tracker Running... Press 'q' to quit.
REPS: 1
--- VLM AUDIT TRIGGERED: Snapshot saved to audits/anomaly_1234567890.jpg ---
[Auditor] Triggered — fault=stuck phase=down angle=78
```

Snapshots of fault frames are saved to the `audits/` folder automatically.

---

## Fault types

The system classifies faults based on knee angle at the time the anomaly gate fires:

| Angle | Fault | Description |
|---|---|---|
| < 80° | `stuck` | Deep squat but can't drive up |
| 80° – 110° | `shallow_depth` | Not reaching parallel |
| ≥ 110° | `knee_valgus` | Potential up-phase form issue |

---

## Expanding the knowledge base

You can add coaching cues at runtime without rebuilding the FAISS index:

```python
coach = auditor._get_coach()
coach.add_to_knowledge_base([
    "Keep your chest tall throughout the movement.",
    "Drive your knees out as you descend.",
])
```

---

## Project structure

```
├── test_tracker.py        # Entry point — wires all modules, renders overlay
├── tracker.py             # PoseTracker — MediaPipe pose extraction
├── update_coach_logic.py  # State machine — rep counting + anomaly gate
├── auditor.py             # Agentic bridge — fault classification, calls RAGCoach
├── rag_coach.py           # RAGCoach — FAISS index, retrieval, Ollama LLM
├── debug_rag.py           # Standalone retrieval + LLM sanity check
├── audits/                # Saved snapshots of fault frames (auto-created)
├── CLAUDE.md              # Claude Code context file
└── README.md              # This file
```

---

## Current limitations

- Tracks the left leg only (MediaPipe landmarks 23, 25, 27)
- Fault classification uses angle only — true knee valgus detection requires 2D joint position analysis
- Knowledge base is hardcoded; no external dataset loaded by default
- Session memory resets on every run
- Coaching is visual only — no audio output

---

## Roadmap

- [ ] Right leg tracking + bilateral comparison
- [ ] Load coaching cues from an external dataset at startup
- [ ] VLM upgrade — `auditor.py` already saves snapshots and passes `image_path` through; swap in a vision model call
- [ ] Audio feedback via text-to-speech
- [ ] Support for additional exercises (deadlift, lunge, hip hinge)
- [ ] Persistent session memory across runs