import os
os.environ["OLLAMA_HOST"] = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

import json
import time
import itertools
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from tqdm import tqdm
from src.llm_conversation.ai_agent import AIAgent
from src.llm_conversation.conversation_manager import ConversationManager

# Schema de salida mínimo compatible con get_response()
class BaselineResponse(BaseModel):
    response: str

INCLUDED_CATEGORIES = {
    "causal_judgment",
    "moral_permissibility",
    "simple_ethical_questions",
}

# PARÁMETROS DE EXPERIMENTACIÓN
TEMPERATURES = [0.1, 0.7]
CONTEXT_SIZES = [4096]
MODELS = ["llama3.2:3b", "mistral-small3.2:24b"]
NUM_RUNS = 3

SOLO_AGENT_PROMPT = (
    "You are a careful and impartial reasoner. "
    "Analyze the following dilemma step by step, weigh the different "
    "perspectives, and provide your final answer clearly. "
    "Task: Resolve the following dilemma."
)


class BaselineRunner:
    def __init__(self, base_dir="debates", results_dir="results_baseline"):
        self.current_dir = Path(__file__).parent.absolute()
        self.base_dir = self.current_dir / base_dir
        self.results_dir = self.current_dir / results_dir
        self.results_dir.mkdir(exist_ok=True)

    def run_all(self):
        if not self.base_dir.exists():
            print(f"ERROR: La carpeta '{self.base_dir}' no existe.")
            return

        json_files = list(self.base_dir.glob("*.json"))
        if not json_files:
            print("AVISO: No se encontraron archivos .json en la carpeta debates.")
            return

        valid_files = [f for f in json_files if f.stem in INCLUDED_CATEGORIES]

        for json_file in tqdm(valid_files, desc="Categorías", unit="cat"):
            category = json_file.stem
            tqdm.write(f"\n>>> BASELINE PARA CATEGORÍA: {category.upper()}")

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                tqdm.write(f"  [!] Error leyendo {json_file.name}: {e}")
                continue

            self._run_matrix(category, data)

    def _run_matrix(self, category, questions):
        configurations = list(itertools.product(
            TEMPERATURES, CONTEXT_SIZES, MODELS, range(NUM_RUNS)
        ))

        for temp, ctx_size, model, run_id in tqdm(
            configurations, desc=f"  Configs [{category}]", unit="cfg",
            leave=False
        ):
            scenario_name = (
                f"{category}_baseline_temp{temp}_ctx{ctx_size}"
                f"_{model.replace(':', '-')}_run{run_id}"
            )

            scenario_output_dir = self.results_dir / scenario_name
            scenario_output_dir.mkdir(exist_ok=True)

            for idx, item in enumerate(tqdm(
                questions, desc=f"    Dilemas", unit="deb", leave=False
            )):
                output_file_path = scenario_output_dir / f"question_{idx}.json"
                if output_file_path.exists():
                    continue
                self._execute_single_query(
                    item, category, temp, ctx_size, model, run_id,
                    scenario_name, idx, scenario_output_dir
                )

    def _execute_single_query(self, item, category, temp, ctx_size, model,
                              run_id, scenario_name, q_idx, output_dir):
        question = item.get('input', '')
    
        start_ts = time.time()
        start_iso = datetime.now().isoformat(timespec="seconds")
    
        try:
            agent = AIAgent(
                name="Solo_Agent",
                model=model,
                system_prompt=SOLO_AGENT_PROMPT,
                temperature=temp,
                ctx_size=ctx_size
            )
    
            agent.add_message(
                name="user",
                role="user",
                content=question
            )
    
            response_chunks = []
            for chunk in agent.get_response(output_format=BaselineResponse):
                response_chunks.append(chunk)
    
            raw_response = "".join(response_chunks)
    
            try:
                parsed = BaselineResponse.model_validate_json(raw_response)
                response = parsed.response
            except Exception:
                response = raw_response
    
            elapsed_total = round(time.time() - start_ts, 3)
    
            result_payload = {
                "metadata": {
                    "scenario": scenario_name,
                    "category": category,
                    "num_agents": 1,
                    "temperature": temp,
                    "ctx_size": ctx_size,
                    "model": model,
                    "turn_policy": "none",
                    "run_id": run_id,
                    "experiment_type": "baseline_solo",
                    "ollama_host": os.environ.get("OLLAMA_HOST"),
                    "target_scores": item.get('target_scores', {}),
                    "timing": {
                        "started_at": start_iso,
                        "elapsed_seconds_total": elapsed_total,
                        "n_turns": 1,
                        "avg_seconds_per_turn": elapsed_total,
                        "per_turn_seconds": [elapsed_total]
                    },
                    "dynamics_raw": {
                        "speaker_counts": {"Solo_Agent": 1},
                        "turn_lengths_chars": [len(response)]
                    }
                },
                "input": question,
                "debate_transcript": [f"[Solo_Agent]: {response}"]
            }
    
            with open(output_dir / f"question_{q_idx}.json", "w",
                      encoding='utf-8') as out_f:
                json.dump(result_payload, out_f, indent=4)
    
            tqdm.write(
                f"     [OK] q_{q_idx} guardada — "
                f"{elapsed_total}s ({len(response)} chars)"
            )
    
        except Exception as e:
            elapsed_total = round(time.time() - start_ts, 3)
            tqdm.write(f"     [!] Error en q_{q_idx} tras {elapsed_total}s: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print(f"Servidor Ollama: {os.environ.get('OLLAMA_HOST')}")
    print(f"Modelos: {MODELS}\n")
    runner = BaselineRunner()
    runner.run_all()
