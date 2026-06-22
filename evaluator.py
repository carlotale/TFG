import os
os.environ["OLLAMA_HOST"] = "https://ollama2.gsi.upm.es"

import json
import re
import time
import pandas as pd
import numpy as np
from pathlib import Path
from ollama import Client

# Métricas NLP:
#   pip install bert-score nltk rouge-score
#   python -c "import nltk; nltk.download('punkt')"
from bert_score import score as bert_score_fn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# CONFIGURACIÓN
JUDGE_MODEL    = "command-r:35b"
JUDGE_CTX_SIZE = 4096
JUDGE_TIMEOUT  = 600        
JUDGE_RETRIES  = 3         

MAS_RESULTS_DIR      = "results_tfg"
BASELINE_RESULTS_DIR = "results_baseline"
OUTPUT_CSV           = "tfg_analisis_command.csv"

EXPERIMENT_CATEGORIES = {
    "causal_judgment",
    "moral_permissibility",
    "simple_ethical_questions",
}

_BLEU_SMOOTH = SmoothingFunction().method1
_ROUGE_SCORER = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
)

# Cliente Ollama global con timeout extendido
OLLAMA_CLIENT = Client(host="https://ollama2.gsi.upm.es", timeout=JUDGE_TIMEOUT)


# FUNCIONES AUXILIARES
def category_from_scenario(scenario_name: str) -> str:
    """Extrae la categoría del nombre del escenario."""
    for cat in EXPERIMENT_CATEGORIES:
        if scenario_name.startswith(cat):
            return cat
    return scenario_name.split('_')[0]


def call_judge(prompt: str) -> str:
    """Llama al juez con reintentos automáticos en caso de timeout.
    Devuelve la respuesta cruda del modelo, o cadena vacía si todos los intentos fallan."""
    system_msg = (
        "You are a data extraction bot. "
        "You only output the exact option chosen, nothing else."
    )

    for attempt in range(1, JUDGE_RETRIES + 1):
        try:
            response = OLLAMA_CLIENT.chat(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": prompt}
                ],
                options={
                    "num_ctx":     JUDGE_CTX_SIZE,
                    "temperature": 0.0,
                },
                stream=False
            )
            return response["message"]["content"]

        except Exception as e:
            err_msg = str(e).lower()
            is_timeout = "timed out" in err_msg or "timeout" in err_msg

            if is_timeout and attempt < JUDGE_RETRIES:
                wait = 5 * attempt   # 5s, 10s, 15s entre reintentos
                print(f"     [!] Judge timeout (intento {attempt}/{JUDGE_RETRIES}), "
                      f"reintentando en {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"     [!] Judge error definitivo: {e}")
                return ""

    return ""


def extract_choice(judge_response: str, options: list) -> str:
    """Extrae la opción elegida por el juez en su respuesta textual."""
    if not judge_response:
        return "No_Consensus"

    # Si vino como JSON, intentar parsear
    try:
        parsed = json.loads(judge_response)
        if isinstance(parsed, dict) and "verdict" in parsed:
            judge_response = parsed["verdict"]
    except Exception:
        pass

    sorted_options = sorted(options, key=len, reverse=True)
    for option in sorted_options:
        pattern = rf"\b{re.escape(option)}\b"
        if re.search(pattern, judge_response, re.IGNORECASE):
            return option
    return "No_Consensus"


def get_consensus_text(transcript_lines: list, n_last: int = 2) -> str:
    """Últimos N turnos del debate."""
    if not transcript_lines:
        return ""
    return " ".join(transcript_lines[-n_last:])


def compute_nlp_metrics(candidate: str, reference: str) -> dict:
    """BERTScore F1, BLEU y ROUGE-L entre consenso y respuesta correcta."""
    metrics = {"bertscore_f1": None, "bleu": None, "rouge_l": None}

    if not candidate or not reference:
        return metrics

    try:
        _, _, F1 = bert_score_fn(
            [candidate], [reference], lang="en",
            verbose=False, rescale_with_baseline=False
        )
        metrics["bertscore_f1"] = round(F1.item(), 4)
    except Exception as e:
        print(f"     [!] BERTScore error: {e}")

    try:
        bleu = sentence_bleu(
            [reference.lower().split()],
            candidate.lower().split(),
            smoothing_function=_BLEU_SMOOTH
        )
        metrics["bleu"] = round(bleu, 4)
    except Exception as e:
        print(f"     [!] BLEU error: {e}")

    try:
        scores = _ROUGE_SCORER.score(reference, candidate)
        metrics["rouge_l"] = round(scores['rougeL'].fmeasure, 4)
    except Exception as e:
        print(f"     [!] ROUGE error: {e}")

    return metrics


def gini_coefficient(values: list) -> float:
    """Coeficiente de Gini para medir desigualdad de turnos.
    0 = todos hablan por igual; cercano a 1 = un agente acapara el debate."""
    if not values or sum(values) == 0:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_v))
    return round((2 * cumsum) / (n * sum(sorted_v)) - (n + 1) / n, 4)


def compute_conversation_dynamics(metadata: dict, transcript_lines: list):
    """Dinámica conversacional de los datos capturados en ejecución."""
    if metadata.get("experiment_type") != "multi_agent":
        return None

    dyn_raw = metadata.get("dynamics_raw", {})
    speaker_counts = dyn_raw.get("speaker_counts", {})
    turn_lengths = dyn_raw.get("turn_lengths_chars", [])
    counts = list(speaker_counts.values())
    n_agents_expected = metadata.get("num_agents", 0)

    return {
        "turns_taken":               len(transcript_lines),
        "unique_speakers":           len(speaker_counts),
        "silent_agents":             max(0, n_agents_expected - len(speaker_counts)),
        "speaker_dominance_gini":    gini_coefficient(counts),
        "avg_turn_length_chars":     round(np.mean(turn_lengths),    2) if turn_lengths else 0,
        "median_turn_length_chars":  round(np.median(turn_lengths),  2) if turn_lengths else 0,
        "max_speaker_share": (
            round(max(counts) / sum(counts), 4)
            if counts and sum(counts) > 0 else 0
        )
    }


# CLASE EVALUATOR

class Evaluator:
    def __init__(self):
        self.current_dir   = Path(__file__).parent.absolute()
        self.mas_dir       = self.current_dir / MAS_RESULTS_DIR
        self.baseline_dir  = self.current_dir / BASELINE_RESULTS_DIR
        self.output_csv    = self.current_dir / OUTPUT_CSV

    def run_evaluation(self):
        print("INICIANDO EVALUACIÓN SISTEMÁTICA")
        print(f"Juez: {JUDGE_MODEL}  (Alibaba — familia distinta a debatientes)")
        print(f"Timeout juez: {JUDGE_TIMEOUT}s  |  Reintentos: {JUDGE_RETRIES}")
        print(f"Servidor: {os.environ.get('OLLAMA_HOST')}\n")

        all_metrics = []

        if self.mas_dir.exists():
            print(f"→ Procesando experimentos multi-agente: {self.mas_dir}")
            self._process_directory(self.mas_dir, all_metrics)
        else:
            print(f"[AVISO] No se encuentra {self.mas_dir}")

        if self.baseline_dir.exists():
            print(f"\n→ Procesando experimentos baseline: {self.baseline_dir}")
            self._process_directory(self.baseline_dir, all_metrics)
        else:
            print(f"[AVISO] No se encuentra {self.baseline_dir}")

        if not all_metrics:
            print("\n[!] No se encontraron datos para evaluar.")
            return

        df = pd.DataFrame(all_metrics)
        df.to_csv(self.output_csv, index=False)
        print(f"\n[OK] Datos exportados a {self.output_csv}")

        self._print_full_statistics(df)

    def _process_directory(self, results_dir: Path, all_metrics: list):
        scenario_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
        for scenario_dir in scenario_dirs:
            for json_file in scenario_dir.glob("*.json"):
                try:
                    self._evaluate_file(json_file, all_metrics)
                except Exception as e:
                    print(f"[!] Error evaluando {json_file.name}: {e}")

    def _evaluate_file(self, json_file: Path, all_metrics: list):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        target_scores = meta.get("target_scores", {})
        options = list(target_scores.keys())
        correct_option = next(
            (k for k, v in target_scores.items() if v == 1), None
        )
        transcript_lines = data.get("debate_transcript", [])
        exp_type = meta.get("experiment_type", "multi_agent")

        # Saltar si ya está evaluado correctamente (excluyendo No_Consensus por timeout, para darle otra oportunidad si la primera falló)
        already = (
            "evaluation_commandr" in data
            and "nlp_metrics" in data["evaluation_commandr"]
            and data["evaluation_commandr"].get("verdict") not in (None, "No_Consensus", "")
        )

        if already:
            ev = data["evaluation_commandr"]
            final_choice = ev.get("verdict")
            is_correct   = ev.get("is_correct")
            nlp          = ev.get("nlp_metrics", {})
            dynamics     = ev.get("dynamics", None)
            timing       = meta.get("timing", {})
            print(f"[SKIP] {json_file.name} ya evaluado")
        else:
            print(
                f"Evaluando: {json_file.name} | tipo: {exp_type} "
                f"| modelo: {meta.get('model')}"
            )

            transcript_text = "\n".join(transcript_lines)

            # Capa 1: LLM-as-a-Judge
            judge_prompt = (
                "You are an impartial and strict judge. "
                "Read the following debate from AI agent(s).\n"
                f"Read the debate and identify which option the agents converged on as their final answer.\n"
                f"among these exact options: {options}.\n"
                "If multiple agents debated, identify the majority consensus.\n"
                "RESPOND STRICTLY WITH THE EXACT TEXT OF THE CHOSEN OPTION AND NOTHING ELSE.\n"
                "Important: consider ALL options equally. Do not default to any particular option.\n"
                f"TEXT:\n{transcript_text}"
            )
            

            raw_judge_response = call_judge(judge_prompt)
            final_choice = extract_choice(raw_judge_response, options)
            is_correct   = 1 if final_choice == correct_option else 0

            # Capa 2: Métricas NLP
            consensus_text = get_consensus_text(transcript_lines, n_last=2)
            reference_text = correct_option if correct_option else ""
            nlp = compute_nlp_metrics(consensus_text, reference_text)

            # Dinámicas conversacionales (solo MAS)
            dynamics = compute_conversation_dynamics(meta, transcript_lines)
            timing   = meta.get("timing", {})

            data["evaluation_commandr"] = {
                "raw_judge_output": raw_judge_response,
                "verdict":          final_choice,
                "correct_option":   correct_option,
                "is_correct":       is_correct,
                "judge_model":      JUDGE_MODEL,
                "nlp_metrics":      nlp,
                "dynamics":         dynamics
            }

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            print(
                f"  -> Veredicto: {final_choice} | OK: {bool(is_correct)} "
                f"| BERT: {nlp.get('bertscore_f1')} "
                f"| BLEU: {nlp.get('bleu')} "
                f"| ROUGE: {nlp.get('rouge_l')}"
            )

        # Recopilar fila para CSV
        scenario_name = meta.get("scenario", "")
        row = {
            "Experiment_Type":  exp_type,
            "Category":         category_from_scenario(scenario_name),
            "Num_Agents":       meta.get("num_agents", 1),
            "Temperature":      meta.get("temperature"),
            "Ctx_Size":         meta.get("ctx_size"),
            "Model":            meta.get("model"),
            "Turn_Policy":      meta.get("turn_policy", "n/a"),
            "Run_ID":           meta.get("run_id", 0),
            "Turns_Taken":      len(transcript_lines),
            "Correct_Option":   correct_option,
            "Agents_Verdict":   final_choice,
            "Is_Correct":       is_correct,
            "BERTScore_F1":     nlp.get("bertscore_f1"),
            "BLEU":             nlp.get("bleu"),
            "ROUGE_L":          nlp.get("rouge_l"),
            "Elapsed_Seconds":  timing.get("elapsed_seconds_total"),
            "Avg_Turn_Seconds": timing.get("avg_seconds_per_turn"),
            "N_Turns":          timing.get("n_turns"),
            "File": str(json_file.relative_to(self.current_dir))
        }

        if dynamics:
            row.update({
                "Speaker_Gini":      dynamics.get("speaker_dominance_gini"),
                "Silent_Agents":     dynamics.get("silent_agents"),
                "Avg_Turn_Length":   dynamics.get("avg_turn_length_chars"),
                "Max_Speaker_Share": dynamics.get("max_speaker_share")
            })

        all_metrics.append(row)

    # ESTADÍSTICAS

    def _print_full_statistics(self, df: pd.DataFrame):
        print("\n" + "="*60)
        print("RESUMEN ESTADÍSTICO")
        print("="*60)

        # 1. MAS vs Baseline
        print("\n[1] PRECISIÓN GLOBAL: MULTI-AGENTE vs BASELINE")
        comp = df.groupby("Experiment_Type")["Is_Correct"].agg(["mean", "std", "count"])
        comp["mean"] = (comp["mean"] * 100).round(2)
        comp["std"]  = (comp["std"]  * 100).round(2)
        comp.columns = ["Accuracy_%", "StdDev_%", "N_Samples"]
        print(comp.to_string())

        # 2. Varianza entre runs
        print("\n[2] VARIANZA ENTRE RUNS")
        config_keys = ["Experiment_Type", "Category", "Model",
                       "Temperature", "Turn_Policy", "Num_Agents"]
        run_var = df.groupby(config_keys)["Is_Correct"].agg(["mean", "std"]).reset_index()
        run_var["mean"] = (run_var["mean"] * 100).round(2)
        run_var["std"]  = (run_var["std"]  * 100).round(2)
        print(f"   Desviación típica media entre runs: {run_var['std'].mean():.2f}%")

        # 3. Por modelo
        print("\n[3] PRECISIÓN POR MODELO")
        model_stats = df.groupby(["Experiment_Type", "Model"])["Is_Correct"].agg(["mean", "std", "count"])
        model_stats["mean"] = (model_stats["mean"] * 100).round(2)
        model_stats["std"]  = (model_stats["std"]  * 100).round(2)
        print(model_stats.to_string())

        mas_df = df[df["Experiment_Type"] == "multi_agent"]
        if not mas_df.empty:
            print("\n[4] PRECISIÓN POR POLÍTICA DE TURNOS (solo MAS)")
            print((mas_df.groupby("Turn_Policy")["Is_Correct"].mean() * 100).round(2).to_string())

            print("\n[5] PRECISIÓN POR NÚMERO DE AGENTES")
            print((mas_df.groupby("Num_Agents")["Is_Correct"].mean() * 100).round(2).to_string())

        print("\n[6] PRECISIÓN POR TEMPERATURA")
        print((df.groupby(["Experiment_Type", "Temperature"])["Is_Correct"].mean() * 100).round(2).to_string())

        print("\n[7] PRECISIÓN POR TEMÁTICA")
        print((df.groupby(["Experiment_Type", "Category"])["Is_Correct"].mean() * 100).round(2).to_string())

        nlp_cols = ["BERTScore_F1", "BLEU", "ROUGE_L"]
        print("\n[8] MÉTRICAS NLP MEDIAS POR TIPO DE EXPERIMENTO")
        print(df.groupby("Experiment_Type")[nlp_cols].mean().round(4).to_string())

        print("\n[9] CORRELACIÓN MÉTRICAS NLP ↔ ACIERTO DEL JUEZ")
        corr = df[nlp_cols + ["Is_Correct"]].corr()["Is_Correct"].drop("Is_Correct")
        print(corr.round(4).to_string())

        if not mas_df.empty and "Speaker_Gini" in mas_df.columns:
            print("\n[10] DINÁMICAS CONVERSACIONALES POR POLÍTICA DE TURNOS")
            dyn_cols = ["Speaker_Gini", "Avg_Turn_Length", "Max_Speaker_Share", "Turns_Taken"]
            existing = [c for c in dyn_cols if c in mas_df.columns]
            print(mas_df.groupby("Turn_Policy")[existing].mean().round(3).to_string())

            print("\n[11] AGENTES SILENCIADOS POR POLÍTICA DE TURNOS")
            if "Silent_Agents" in mas_df.columns:
                print(mas_df.groupby("Turn_Policy")["Silent_Agents"].mean().round(3).to_string())

            print("\n[12] CORRELACIÓN: BALANCE DE TURNOS ↔ ACIERTO")
            corr_dyn = mas_df[["Speaker_Gini", "Max_Speaker_Share",
                                "Turns_Taken", "Is_Correct"]].corr()["Is_Correct"]
            print(corr_dyn.drop("Is_Correct").round(4).to_string())

        timing_cols = ["Elapsed_Seconds", "Avg_Turn_Seconds", "N_Turns"]
        existing_timing = [c for c in timing_cols if c in df.columns]
        if existing_timing:
            print("\n[13] ANÁLISIS DE TIEMPOS POR MODELO")
            print(df.groupby(["Experiment_Type", "Model"])[existing_timing].mean().round(2).to_string())

            if not mas_df.empty:
                print("\n[14] TIEMPOS POR POLÍTICA DE TURNOS (solo MAS)")
                print(mas_df.groupby("Turn_Policy")[existing_timing].mean().round(2).to_string())


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run_evaluation()