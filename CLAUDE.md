# CLAUDE.md — RAG Agentic Fitness Coach

This file gives Claude Code the context needed to work on this project effectively.
Read it fully before making any changes.

---

## Project overview

A real-time squat coaching system that runs on a local webcam feed.
MediaPipe tracks knee angle at ~30fps. When the athlete appears stuck or shows a
form fault, an agentic gate fires a RAG pipeline: FAISS retrieves relevant coaching
cues from a local knowledge base, and a local LLM (llama3.2:3b via Ollama) generates
a short on-screen coaching cue. No cloud calls. No fine-tuning. Everything runs locally.

---

## Architecture

```
Webcam → tracker.py (MediaPipe)
              ↓ angle, knee_coords
         update_coach_logic.py (fast loop — rep counting + anomaly gate)
              ↓ is_anomaly = True
         auditor.py (agentic bridge — snapshot + fault classification)
              ↓ fault_type, phase, angle
         rag_coach.py (RAGCoach — FAISS retrieval + Ollama LLM)
              ↓ feedback string (or None if suppressed)
         test_tracker.py (overlay on screen)
```

**Two loops:**
- **Fast loop** — runs every frame, updates phase and rep count, increments `consecutive_stuck_frames`
- **Slow loop (agentic gate)** — fires only when `consecutive_stuck_frames > STUCK_LIMIT (45)`, roughly 1.5s of being stuck

---

## File responsibilities

| File | Role |
|---|---|
| `tracker.py` | `PoseTracker` — wraps MediaPipe, extracts hip/knee/ankle angle |
| `update_coach_logic.py` | Pure state machine — rep counting + anomaly detection, no I/O |
| `auditor.py` | Agentic bridge — classifies fault from angle, calls RAGCoach, saves snapshot |
| `rag_coach.py` | `RAGCoach` — FAISS index, sentence encoder, session memory, Ollama call |
| `test_tracker.py` | Entry point — wires all modules, renders OpenCV overlay |

---

## Key constants

```python
# update_coach_logic.py
THRESHOLD_DOWN = 90    # angle below this → phase transitions to "down"
THRESHOLD_UP   = 160   # angle above this → phase transitions to "up", rep counted
STUCK_LIMIT    = 45    # frames in "down" below 110° before anomaly fires (~1.5s at 30fps)

# auditor.py fault classification (angle-based)
angle < 80   → fault_type = "stuck"         # deep but can't drive up
angle < 110  → fault_type = "shallow_depth" # not hitting depth
angle >= 110 → fault_type = "knee_valgus"   # up-phase issue
```

---

## RAGCoach internals

- **Encoder:** `all-MiniLM-L6-v2` via `sentence-transformers` — loaded once at startup
- **Index:** `faiss.IndexFlatL2` — built from `KNOWLEDGE_BASE` strings in `rag_coach.py`
- **Retrieval:** `k=3` nearest cues per fault type, looked up via `FAULT_QUERIES` dict
- **LLM:** `ollama.generate(model="llama3.2:3b", prompt=...)` — expects ≤10 word output
- **Session memory:** `recent_faults` list (max 10). If the same fault appears 3 times in the last 3 entries, `get_feedback()` returns `None` — suppresses nagging

**To add cues at runtime (no rebuild needed):**
```python
coach.add_to_knowledge_base(["new cue 1", "new cue 2"])
```

---

## Local dependencies

All models run locally. No API keys needed.

```bash
# Python packages
pip install faiss-cpu sentence-transformers ollama opencv-python
pip install mediapipe==0.10.14   # pin this — newer versions have breaking changes

# Ollama
ollama serve          # must be running before test_tracker.py
ollama pull llama3.2:3b
```

---

## Running the project

```bash
# Step 1 — verify RAG retrieval (no webcam needed)
python debug_rag.py

# Step 2 — verify pose tracking only (no coach)
# Use the simple test_tracker.py (no auditor import)
python test_tracker.py

# Step 3 — full pipeline
python test_tracker.py   # full version with auditor import
```

**Expected console output (full pipeline):**
```
[RAGCoach] Loading sentence encoder...
[RAGCoach] Ready.
Agentic Tracker Running... Press 'q' to quit.
REPS: 1
--- VLM AUDIT TRIGGERED: Snapshot saved to audits/anomaly_<timestamp>.jpg ---
[Auditor] Triggered — fault=stuck phase=down angle=78
```

---

## How to debug

| Symptom | Likely cause | Fix |
|---|---|---|
| `[Auditor] Triggered` never appears | Not holding squat long enough, or MediaPipe not tracking | Hold squat for 2s; check angle prints to console |
| `COACH:` never appears on screen | LLM call failing silently | Check `ollama serve` is running; test `debug_rag.py` |
| Cues feel irrelevant to the fault | FAISS retrieving wrong chunks | Edit `FAULT_QUERIES` dict in `rag_coach.py`; rerun `debug_rag.py` |
| LLM output > 10 words | Prompt not tight enough | Tighten the prompt string in `get_feedback()` |
| Coach goes silent after a few reps | Session memory suppressing | Expected behaviour — same fault 3× in a row triggers suppression |
| `mediapipe` import errors | Wrong version installed | `pip install mediapipe==0.10.14` |

---

## What not to change without care

- **`STUCK_LIMIT`** — lowering this causes the agentic gate to fire too aggressively mid-rep; raising it delays feedback too long
- **`THRESHOLD_DOWN / THRESHOLD_UP`** — these define the squat rep window; changing them affects rep counting accuracy
- **`_should_skip()` in RAGCoach** — session memory logic; don't tighten without testing for over-suppression
- **The shared `_coach` singleton in `auditor.py`** — RAGCoach loads a sentence encoder on init; instantiating it per-call would cause severe lag

---

## Extending the project

**Add more exercises:**
- Add new fault keys to `FAULT_QUERIES` in `rag_coach.py`
- Add corresponding cue strings to `KNOWLEDGE_BASE`
- Update the fault classification block in `auditor.py` to handle new angle ranges or exercise types

**Add more coaching cues from a dataset:**
```python
# At startup in test_tracker.py, after RAGCoach is initialised:
coach = auditor._get_coach()
coach.add_to_knowledge_base(["cue from dataset", "another cue"])
```

**Upgrade to a VLM:**
- `auditor.py` already saves snapshots to `audits/anomaly_<timestamp>.jpg`
- The `image_path` parameter is passed through but unused beyond logging
- Swap `coach.get_feedback(...)` for a VLM call using the saved frame

**Switch LLM:**
- Change `model_name` in `RAGCoach.__init__()` — any model available in Ollama works
- Larger models (e.g. `llama3.1:8b`) give better output but add latency to the slow loop

---

## Current limitations

- Angle-based exercises track **both sides** and average them; single-side fallback if one side is occluded. Spatial/height exercises still use fixed landmark pairs.
- Fault classification is angle-only; true knee valgus requires 2D joint positions, not just angle
- `KNOWLEDGE_BASE` is hardcoded; no persistent dataset loaded yet
- Session memory resets on every run (not persisted to disk)
- No audio feedback — coaching is visual overlay only

---

## Project phases

| Phase | Goal | Status |
|---|---|---|
| **Phase 1** | Add all 23 Table 5 exercises — tracker, state machine, exercise-aware RAG | 🔄 In progress |
| **Phase 2** | Optimise RAG retrieval — better `KNOWLEDGE_BASE`, `FAULT_QUERIES` tuning, hit@k validation | ⏳ Pending |
| **Phase 3** | Benchmark — retrieval hit@k scores, LLM output quality metrics, regression tests | ⏳ Pending |
| **Phase 4** | Onboard fine-tuned model to replace/augment `llama3.2:3b` | ⏳ Pending |

**Parallel track:** Friend's repo uses Moondream 3 (VLM) instead of RAGCoach — separate experiment, different architecture.

**Phase 1 checklist:**
- [x] Squat pipeline working (fault detection, RAGCoach, session memory)
- [x] `shallow_depth` and `stuck` both fire correctly
- [ ] All 23 exercises in `tracker.py` (`get_exercise_config`)
- [ ] Generalised state machine in `update_coach_logic.py` (angle + spatial + height branches)
- [ ] Exercise-aware `FAULT_QUERIES` and `KNOWLEDGE_BASE` in `rag_coach.py`
- [ ] `exercise_name` passed through `auditor.py` → `RAGCoach.get_feedback()`
- [ ] Full `test_tracker.py` with `EXERCISE_MAP` keyboard switching
