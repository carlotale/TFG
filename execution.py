import os

os.environ["OLLAMA_HOST"] = "https://ollama2.gsi.upm.es"

import json
import time
import itertools
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from src.llm_conversation.ai_agent import AIAgent
from src.llm_conversation.conversation_manager import ConversationManager
import sys

# CONFIGURACIÓN DE AGENTES POR TEMÁTICA
ROLE_POOLS = {
    "causal_judgment": [
        {"role": "Causal Scientist", "prompt": "Focus on physical cause-effect chains and counterfactual logic."},
        {"role": "Ethical Philosopher", "prompt": "Analyze moral responsibility and intentionality behind actions."},
        {"role": "Ordinary Citizen", "prompt": "Use common sense and everyday intuition to attribute blame."},
        {"role": "Legal Expert", "prompt": "Focus on negligence, duty of care, and legal liability."}
    ],
    "moral_permissibility": [
        {"role": "Utilitarian", "prompt": "Focus on the greatest good for the greatest number. Minimize total harm."},
        {"role": "Deontologist", "prompt": "Follow strict moral rules. Some actions are wrong regardless of consequences."},
        {"role": "Human Rights Advocate", "prompt": "Prioritize individual inalienable rights and bodily autonomy."},
        {"role": "Pragmatist", "prompt": "Focus on practical outcomes and social stability."}
    ],
    "simple_ethical_questions": [
        {"role": "Ethicist", "prompt": "Apply ethical theories to determine right and wrong."},
        {"role": "Cultural Relativist", "prompt": "Consider how cultural norms influence moral judgments."},
        {"role": "Legal Scholar", "prompt": "Focus on legality and social contract principles."},
        {"role": "Empath", "prompt": "Consider the emotional impact on all parties involved."}
    ]
}

# PARÁMETROS DE EXPERIMENTACIÓN
NUM_AGENTS_TO_TEST = [2, 3, 4]
TEMPERATURES = [0.1, 0.7]
CONTEXT_SIZES = [4096]
MODELS = ["llama3.2:3b", "mistral-small3.2:24b"]  
TURN_POLICIES = ["round_robin", "random", "moderator"]
NUM_RUNS = 3

class ExperimentOrchestrator:
    def __init__(self, base_dir="debates", results_dir="results_tfg"):
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

        # Filtrar las categorías incluidas en el experimento final.
        valid_files = [f for f in json_files if f.stem in ROLE_POOLS]

        # Barra de progreso a nivel de categoría.
        for json_file in tqdm(valid_files, desc="Categorías", unit="cat"):
            category = json_file.stem
            tqdm.write(f"\n>>> CATEGORÍA: {category.upper()}")

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                tqdm.write(f"  [!] Error leyendo {json_file.name}: {e}")
                continue

            self._run_matrix(category, data)

    def _run_matrix(self, category, questions):
        # Producto cartesiano de todas las combinaciones de configuración.
        configurations = list(itertools.product(
            NUM_AGENTS_TO_TEST, TEMPERATURES, CONTEXT_SIZES,
            MODELS, TURN_POLICIES, range(NUM_RUNS)
        ))

        # Barra de progreso a nivel de configuración dentro de la categoría.
        for n_agents, temp, ctx_size, model, turn_policy, run_id in tqdm(
            configurations, desc=f"  Configs [{category}]", unit="cfg",
            leave=False
        ):
            selected_roles = ROLE_POOLS[category][:n_agents]

            scenario_name = (
                f"{category}_agents{n_agents}_temp{temp}_ctx{ctx_size}"
                f"_{model.replace(':', '-')}_turn-{turn_policy}_run{run_id}"
            )

            scenario_output_dir = self.results_dir / scenario_name
            scenario_output_dir.mkdir(exist_ok=True)

            # Barra de progreso a nivel de dilema, mostrando el tiempo medio por debate en tiempo real.
            for idx, item in enumerate(tqdm(
                questions, desc=f"    Dilemas", unit="deb", leave=False
            )):
                output_file_path = scenario_output_dir / f"question_{idx}.json"
                if output_file_path.exists():
                    continue

                self._execute_single_debate(
                    item, selected_roles, temp, ctx_size, model,
                    turn_policy, run_id, scenario_name, idx, scenario_output_dir
                )

    def _execute_single_debate(self, item, roles, temp, ctx_size, model,
                               turn_policy, run_id, scenario_name, q_idx,
                               output_dir):
        question = item.get('input', '')

        # Marcado del tiempo de inicio 
        start_ts = time.time()
        start_iso = datetime.now().isoformat(timespec="seconds")

        try:
            agents = []
            for r in roles:
                full_prompt = (
                    f"Role: {r['role']}. {r['prompt']} "
                    "Task: Resolve the following dilemma."
                )
                agent = AIAgent(
                    name=r['role'].replace(" ", "_"),
                    model=model,
                    system_prompt=full_prompt,
                    temperature=temp,
                    ctx_size=ctx_size
                )
                agents.append(agent)

            manager = ConversationManager(
                agents=agents,
                initial_message=question,
                allow_termination=True,
                turn_order=turn_policy
            )

            conv_iterator = manager.run_conversation()
            conversation_history = []
            speaker_counts = {}
            turn_lengths = []
            turn_timestamps = []   # tiempo acumulado al final de cada turno

            for turn_idx, (agent_name, message_stream) in enumerate(conv_iterator):
                content = ""
                for chunk in message_stream:
                    content = chunk

                conversation_history.append(f"[{agent_name}]: {content}")
                speaker_counts[agent_name] = speaker_counts.get(agent_name, 0) + 1
                turn_lengths.append(len(content))
                turn_timestamps.append(round(time.time() - start_ts, 3))

            # Cálculo de métricas de tiempo
            elapsed_total = round(time.time() - start_ts, 3)
            n_turns = len(conversation_history)
            avg_turn_time = round(elapsed_total / n_turns, 3) if n_turns else 0

            # Tiempo individual por turno
            per_turn_seconds = []
            prev = 0.0
            for ts in turn_timestamps:
                per_turn_seconds.append(round(ts - prev, 3))
                prev = ts

            result_payload = {
                "metadata": {
                    "scenario": scenario_name,
                    "category": scenario_name.split('_')[0],
                    "num_agents": len(agents),
                    "temperature": temp,
                    "ctx_size": ctx_size,
                    "model": model,
                    "turn_policy": turn_policy,
                    "run_id": run_id,
                    "experiment_type": "multi_agent",
                    "ollama_host": os.environ.get("OLLAMA_HOST"),
                    "target_scores": item.get('target_scores', {}),
                    "timing": {
                        "started_at": start_iso,
                        "elapsed_seconds_total": elapsed_total,
                        "n_turns": n_turns,
                        "avg_seconds_per_turn": avg_turn_time,
                        "per_turn_seconds": per_turn_seconds
                    },
                    "dynamics_raw": {
                        "speaker_counts": speaker_counts,
                        "turn_lengths_chars": turn_lengths
                    }
                },
                "input": question,
                "debate_transcript": conversation_history
            }

            with open(output_dir / f"question_{q_idx}.json", "w",
                      encoding='utf-8') as out_f:
                json.dump(result_payload, out_f, indent=4)

            tqdm.write(
                f"     [OK] q_{q_idx} guardada — "
                f"{n_turns} turnos en {elapsed_total}s "
                f"({avg_turn_time}s/turno)"
            )

        except Exception as e:
            elapsed_total = round(time.time() - start_ts, 3)
            tqdm.write(f"     [!] Error en q_{q_idx} tras {elapsed_total}s: {e}")
            import traceback
            print("\n" + "!"*30)
            traceback.print_exc()
            print("!"*30 + "\n")


if __name__ == "__main__":
    print(f"Servidor Ollama: {os.environ.get('OLLAMA_HOST')}")
    print(f"Modelos: {MODELS}\n")
    orchestrator = ExperimentOrchestrator()
    orchestrator.run_all()