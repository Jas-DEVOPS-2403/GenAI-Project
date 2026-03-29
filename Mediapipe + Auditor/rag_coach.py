import ollama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Knowledge base — hardcoded for now, add dataset chunks here later
# to expand: embed extra strings and call index.add() before use
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    # Knee valgus
    "Knee valgus (knees caving in): Push your knees out over your toes as you descend.",
    "Knees collapsing inward: Think about spreading the floor apart with your feet.",
    # Depth
    "Shallow squat depth: Push your hips back and down, aim for thighs parallel to the floor.",
    "Not hitting depth: Sit into it, do not just bend your knees.",
    # Back / posture
    "Rounded back: Keep your chest up and spine neutral the entire rep.",
    "Forward lean: Weight in your heels, chest tall, brace your core.",
    # Stuck / grinding
    "Stuck at the bottom: Drive through your heels and squeeze your glutes to power out.",
    "Pausing too long at the bottom: Stay tight, do not relax at the bottom position.",
    # Positive reinforcement
    "Good depth on that rep, keep that consistency going.",
    "Strong rep, stay controlled and keep breathing.",
    "You are doing well, focus on keeping form solid.",
    # General cues
    "Brace your core before every rep like you are about to take a punch.",
    "Control the descent, do not just drop down.",
    "Squeeze your glutes at the top for full hip extension.",
    "Keep your gaze forward, looking down can cause your back to round.",
]

FAULT_QUERIES = {
    "knee_valgus":   "knees caving inward squat correction cue",
    "shallow_depth": "squat not deep enough parallel correction",
    "rounded_back":  "back rounding forward lean posture fix",
    "stuck":         "stuck at bottom squat cannot stand up correction",
    "good_form":     "positive reinforcement squat good rep encouragement",
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
        Expand the knowledge base at runtime.
        Call this to add dataset chunks without rebuilding from scratch.
        e.g. coach.add_to_knowledge_base(["cue 1", "cue 2"])
        """
        KNOWLEDGE_BASE.extend(new_entries)
        embeddings = self.encoder.encode(new_entries).astype("float32")
        self.index.add(embeddings)

    def _retrieve(self, fault_type: str, k: int = 3) -> list:
        query = FAULT_QUERIES.get(fault_type, FAULT_QUERIES["stuck"])
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

    def get_feedback(self, fault_type: str, phase: str, angle: float) -> str | None:
        """
        Core agentic step:
          1. Check session memory — skip if already nagging about same fault
          2. Retrieve relevant coaching cues (RAG)
          3. Call LLM with grounded prompt
          4. Update session memory
        Returns a short feedback string, or None if suppressed.
        """
        if self._should_skip(fault_type):
            return None  # give the athlete time to self-correct

        retrieved_cues = self._retrieve(fault_type)
        context = "\n".join(f"- {cue}" for cue in retrieved_cues)

        prompt = (
            f"You are a real-time fitness coach. The athlete is doing squats.\n"
            f"Current state: phase={phase}, knee_angle={int(angle)} degrees, detected_issue={fault_type}\n\n"
            f"Relevant coaching knowledge:\n{context}\n\n"
            f"Give ONE short coaching cue. Maximum 10 words. Be direct and encouraging. "
            f"Output only the cue, nothing else."
        )

        response = ollama.generate(model=self.llm_model, prompt=prompt)
        feedback = response["response"].strip().split("\n")[0]

        self._update_memory(fault_type)
        return feedback
