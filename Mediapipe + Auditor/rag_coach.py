import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Knowledge base — covers all 23 Table 5 exercises
# To expand at runtime: coach.add_to_knowledge_base(["new cue"])
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    # ── SQUAT (original 15 cues) ────────────────────────────────────────────
    "Knee valgus (knees caving in): Push your knees out over your toes as you descend.",
    "Knees collapsing inward: Think about spreading the floor apart with your feet.",
    "Shallow squat depth: Push your hips back and down, aim for thighs parallel to the floor.",
    "Not hitting depth: Sit into it, do not just bend your knees.",
    "Rounded back: Keep your chest up and spine neutral the entire rep.",
    "Forward lean: Weight in your heels, chest tall, brace your core.",
    "Stuck at the bottom: Drive through your heels and squeeze your glutes to power out.",
    "Pausing too long at the bottom: Stay tight, do not relax at the bottom position.",
    "Good depth on that rep, keep that consistency going.",
    "Strong rep, stay controlled and keep breathing.",
    "You are doing well, focus on keeping form solid.",
    "Brace your core before every rep like you are about to take a punch.",
    "Control the descent, do not just drop down.",
    "Squeeze your glutes at the top for full hip extension.",
    "Keep your gaze forward, looking down can cause your back to round.",

    # ── LUNGE / WALKING LUNGE / LUNGE JUMPS ────────────────────────────────
    "Lunge knee valgus: Drive your front knee outward, track it over your second toe.",
    "Lunge depth: Drop your back knee close to the floor for full range of motion.",
    "Lunge forward lean: Keep your torso upright, shoulders stacked over hips.",
    "Lunge stuck: Push through your front heel to drive back up, stay tight.",
    "Walking lunge: Take a long stride, keep your front shin vertical.",
    "Lunge jump: Land softly with bent knees, absorb the impact before the next jump.",

    # ── HIP HINGE / GOOD MORNINGS ───────────────────────────────────────────
    "Good morning rounded back: Maintain a neutral spine, hinge from the hips not the waist.",
    "Good morning depth: Push hips back until you feel a strong hamstring stretch.",
    "Hip hinge: Imagine pushing a wall behind you with your glutes as you hinge.",
    "Hip hinge stuck: Drive hips forward and squeeze glutes hard at the top.",

    # ── PUSH-UPS / MOVING PLANK ─────────────────────────────────────────────
    "Push-up elbow flare: Keep elbows at 45 degrees to your torso, not flared wide.",
    "Push-up sagging hips: Squeeze your abs and glutes — body should be one straight line.",
    "Push-up depth: Lower your chest until it nearly touches the floor every rep.",
    "Push-up stuck: Push the floor away explosively, think about spreading your hands apart.",
    "Moving plank: Keep hips level as you walk your hands, do not rock side to side.",
    "Plank taps: Minimise hip rotation when lifting each hand for a tap.",

    # ── HIGH KNEES / BUTT KICKERS ───────────────────────────────────────────
    "High knees: Drive each knee up to hip height, stay on the balls of your feet.",
    "High knees posture: Keep your torso tall with a slight forward lean from the ankle.",
    "High knees arms: Pump your arms to match each knee drive, stay rhythmic.",
    "Butt kickers: Actively contract your hamstring to drive the heel up to your glute.",
    "Butt kickers posture: Keep a quick cadence, land lightly and stay upright.",

    # ── JUMPING JACKS / AIR JUMP ROPE ───────────────────────────────────────
    "Jumping jacks arms: Reach arms fully overhead on every rep, do not cut it short.",
    "Jumping jacks sync: Arms and legs should move together, stay coordinated.",
    "Jumping jacks low drive: Jump wider, push your feet apart with each rep.",
    "Air jump rope: Keep wrists turning in small circles, stay light on your feet.",
    "Air jump rope rhythm: Find a consistent bounce, do not stop between turns.",

    # ── SQUAT JUMPS / SQUAT KICKS ───────────────────────────────────────────
    "Squat jump depth: Hit parallel before you explode up, do not shallow-dip.",
    "Squat jump landing: Land softly with bent knees, absorb the impact through your legs.",
    "Squat jump knee valgus on landing: Push knees out as you absorb the landing.",
    "Squat kick: Full squat down first, then kick out powerfully at the top.",

    # ── MOUNTAIN CLIMBERS / FLOOR TOUCHES / QUICK FEET ─────────────────────
    "Mountain climbers: Drive each knee toward your chest, keep hips level and low.",
    "Mountain climbers speed: Pick up the pace, alternate legs as fast as you can control.",
    "Floor touches: Hinge at the hips to reach down, keep a flat back as you lower.",
    "Quick feet: Stay on the balls of your feet, keep your steps light and rapid.",
    "Quick feet posture: Stay low, slight squat position, keep your core braced.",

    # ── STANDING KICKS / BOXING SQUAT PUNCHES ──────────────────────────────
    "Standing kicks: Drive the kick from the hip, snap the leg back under control.",
    "Standing kicks balance: Squeeze your standing glute and fix your gaze on one point.",
    "Boxing squat punches: Full squat depth first, then punch with full arm extension.",

    # ── COOL-DOWN STRETCHES ─────────────────────────────────────────────────
    "Deltoid stretch: Pull your arm across your chest until you feel the shoulder stretch.",
    "Quad stretch: Pull your heel to your glute, stand tall and squeeze the standing glute.",
    "Shoulder gators: Keep the movement slow and controlled, feel the full range.",
    "Toe touchers: Hinge from the hips with a soft knee bend, reach toward the floor.",

    # ── GENERAL POSITIVES / UNIVERSAL CUES ─────────────────────────────────
    "Great rep, maintain that range of motion every single time.",
    "Breathing is key — exhale on the effort, inhale on the return.",
    "Keep your core engaged throughout the entire movement.",
    "Slow down the lowering phase to build more control.",
    "Good symmetry, keep both sides working equally.",
    "Full range of motion beats partial reps every time.",
    "Stay consistent with your tempo, do not rush.",
    "Focus on the muscle you are training, make each rep intentional.",
]

# ---------------------------------------------------------------------------
# Fault queries — nested by exercise, with a _generic fallback
# Keys must match exercise_name values used in test_tracker.py (snake_case)
# ---------------------------------------------------------------------------
FAULT_QUERIES = {
    "squats": {
        "knee_valgus":   "knees caving inward squat correction cue",
        "shallow_depth": "squat not deep enough parallel correction",
        "rounded_back":  "back rounding forward lean posture fix",
        "stuck":         "stuck at bottom squat cannot stand up correction",
        "good_form":     "positive reinforcement squat good rep encouragement",
    },
    "walking_lunges": {
        "knee_valgus":   "front knee caving inward lunge correction",
        "shallow_depth": "lunge not deep enough back knee floor",
        "forward_lean":  "excessive forward lean lunge torso upright",
        "stuck":         "stuck at bottom lunge cannot rise correction",
        "good_form":     "positive reinforcement lunge good rep",
    },
    "lunge_jumps": {
        "knee_valgus":   "front knee caving lunge jump landing correction",
        "shallow_depth": "lunge jump not deep enough range of motion",
        "stuck":         "stuck lunge jump cannot explode up correction",
        "good_form":     "positive reinforcement lunge jump explosive rep",
    },
    "good_mornings": {
        "rounded_back":  "back rounding good morning neutral spine correction",
        "shallow_depth": "good morning not hinging enough hamstring stretch",
        "stuck":         "stuck at bottom good morning correction drive up",
        "good_form":     "positive reinforcement good morning hip hinge",
    },
    "push-ups": {
        "elbow_flare":   "elbows flaring out push up tricep path correction",
        "shallow_depth": "push up chest not reaching floor range of motion",
        "sagging_hips":  "hips sagging push up core engagement plank position",
        "stuck":         "stuck at bottom push up pressing strength correction",
        "good_form":     "positive reinforcement push up full rep encouragement",
    },
    "plank_taps": {
        "sagging_hips":  "hips sagging plank taps core engagement",
        "shallow_depth": "plank tap not reaching full extension",
        "good_form":     "positive reinforcement plank taps strong core",
    },
    "moving_plank": {
        "sagging_hips":  "hips sagging moving plank core engagement",
        "rotation":      "hip rotation moving plank keep hips level",
        "good_form":     "positive reinforcement moving plank strong hold",
    },
    "high_knees": {
        "low_drive":     "knees not driving high enough high knees correction",
        "forward_lean":  "leaning backward high knees torso upright cue",
        "good_form":     "positive reinforcement high knees good drive rhythm",
    },
    "butt_kickers": {
        "low_drive":     "heels not reaching glutes butt kickers correction",
        "forward_lean":  "torso leaning too far forward butt kickers posture",
        "good_form":     "positive reinforcement butt kickers good heel drive",
    },
    "jumping_jacks": {
        "shallow_depth": "arms not reaching overhead jumping jack full extension",
        "low_drive":     "feet not jumping wide enough jumping jack correction",
        "timing":        "arms and legs out of sync jumping jack coordination",
        "good_form":     "positive reinforcement jumping jack good rep",
    },
    "air_jump_rope": {
        "timing":        "air jump rope rhythm lost coordination correction",
        "low_drive":     "not bouncing enough air jump rope light feet",
        "good_form":     "positive reinforcement air jump rope good rhythm",
    },
    "squat_jumps": {
        "shallow_depth": "squat jump not hitting parallel before jumping",
        "landing":       "heavy landing squat jump soft knees absorb impact",
        "knee_valgus":   "knees caving on landing squat jump correction",
        "good_form":     "positive reinforcement squat jump explosive rep",
    },
    "squat_kicks": {
        "shallow_depth": "squat kick not squatting deep enough before kick",
        "stuck":         "stuck squat kick cannot complete movement",
        "good_form":     "positive reinforcement squat kick good rep",
    },
    "mountain_climbers": {
        "low_drive":     "knees not driving to chest mountain climbers correction",
        "sagging_hips":  "hips rising mountain climbers keep body flat",
        "good_form":     "positive reinforcement mountain climbers good pace",
    },
    "floor_touches": {
        "rounded_back":  "back rounding floor touches hip hinge correction",
        "shallow_depth": "floor touches not reaching far enough range of motion",
        "good_form":     "positive reinforcement floor touches good reach",
    },
    "quick_feet": {
        "low_drive":     "feet not lifting quick feet stay light correction",
        "good_form":     "positive reinforcement quick feet good cadence",
    },
    "standing_kicks": {
        "low_drive":     "kick not reaching high enough standing kick correction",
        "balance_loss":  "losing balance standing kick squeeze glute correction",
        "good_form":     "positive reinforcement standing kick good height",
    },
    "boxing_squat_punches": {
        "shallow_depth": "squat not deep enough boxing squat punches correction",
        "stuck":         "stuck at bottom boxing squat punches correction",
        "good_form":     "positive reinforcement boxing squat punches good rep",
    },
    "puddle_jumps": {
        "shallow_depth": "puddle jump not jumping far enough lateral distance",
        "landing":       "heavy landing puddle jump soft knees absorb",
        "good_form":     "positive reinforcement puddle jump good lateral power",
    },
    "deltoid_stretch": {
        "shallow_depth": "deltoid stretch not pulling arm far enough across",
        "good_form":     "positive reinforcement deltoid stretch good hold",
    },
    "quad_stretch": {
        "shallow_depth": "quad stretch heel not reaching glute pull closer",
        "balance_loss":  "losing balance quad stretch squeeze standing glute",
        "good_form":     "positive reinforcement quad stretch good hold",
    },
    "shoulder_gators": {
        "shallow_depth": "shoulder gators not reaching full arm opening",
        "good_form":     "positive reinforcement shoulder gators good range",
    },
    "toe_touchers": {
        "shallow_depth": "toe touchers not reaching far enough hip hinge depth",
        "rounded_back":  "back rounding toe touchers hinge from hips correction",
        "good_form":     "positive reinforcement toe touchers good reach",
    },
    # Fallback for any exercise not explicitly listed above
    "_generic": {
        "shallow_depth": "incomplete range of motion exercise correction cue",
        "rounded_back":  "back rounding neutral spine posture correction",
        "sagging_hips":  "hips dropping core engagement correction",
        "low_drive":     "not reaching target height or distance correction",
        "asymmetry":     "left right asymmetry exercise correction cue",
        "stuck":         "stuck unable to complete movement correction",
        "landing":       "landing too hard soft knees absorb impact",
        "good_form":     "positive reinforcement good form encouragement",
    },
}


class RAGCoach:
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.llm_model = model_name
        print("[RAGCoach] Loading sentence encoder...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.recent_faults: list = []  # session memory — tracks last N fault types
        self._build_index()
        print("[RAGCoach] Ready.")

    def _build_index(self):
        embeddings = self.encoder.encode(KNOWLEDGE_BASE).astype("float32")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def add_to_knowledge_base(self, new_entries: list):
        """
        Expand the knowledge base at runtime without rebuilding.
        e.g. coach.add_to_knowledge_base(["cue 1", "cue 2"])
        """
        KNOWLEDGE_BASE.extend(new_entries)
        embeddings = self.encoder.encode(new_entries).astype("float32")
        self.index.add(embeddings)

    def _retrieve(self, fault_type: str, exercise_name: str = "squats", k: int = 3) -> list:
        exercise_key = exercise_name.lower().replace(" ", "_").replace("-", "_")
        exercise_faults = FAULT_QUERIES.get(exercise_key, FAULT_QUERIES["_generic"])
        query = exercise_faults.get(
            fault_type,
            FAULT_QUERIES["_generic"].get(fault_type, "good exercise form coaching cue")
        )
        vec = self.encoder.encode([query]).astype("float32")
        _, indices = self.index.search(vec, k)
        return [KNOWLEDGE_BASE[i] for i in indices[0]]

    def _should_skip(self, fault_type: str) -> bool:
        """Session memory: suppress if same fault coached 3 times in a row."""
        return self.recent_faults[-3:].count(fault_type) >= 3

    def _update_memory(self, fault_type: str):
        self.recent_faults.append(fault_type)
        if len(self.recent_faults) > 10:
            self.recent_faults.pop(0)

    def get_feedback(self, fault_type: str, exercise_name: str = "squats",
                     phase: str = "down", angle: float = 100.0) -> str | None:
        """
        Core agentic step:
          1. Check session memory — skip if already nagging about same fault
          2. Retrieve exercise-specific coaching cues (RAG)
          3. Call LLM with grounded, exercise-aware prompt
          4. Update session memory
        Returns a short feedback string, or None if suppressed.
        """
        if self._should_skip(fault_type):
            return None  # give the athlete time to self-correct

        retrieved_cues = self._retrieve(fault_type, exercise_name)
        context = "\n".join(f"- {cue}" for cue in retrieved_cues)

        prompt = (
            f"You are a real-time fitness coach. The athlete is doing {exercise_name}.\n"
            f"Current state: phase={phase}, joint_angle={int(angle)} degrees, detected_issue={fault_type}\n\n"
            f"Relevant coaching knowledge:\n{context}\n\n"
            f"Give ONE short coaching cue. Maximum 10 words. Be direct and encouraging. "
            f"Output only the cue, nothing else."
        )

        response = ollama.generate(model=self.llm_model, prompt=prompt)
        feedback = response["response"].strip().split("\n")[0]

        self._update_memory(fault_type)
        return feedback
