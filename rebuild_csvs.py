import os, json, pandas as pd, numpy as np
from pathlib import Path

current_dir  = Path("/home/jovyan/work/TFG")
mas_dir      = current_dir / "results_tfg"
baseline_dir = current_dir / "results_baseline"

EXPERIMENT_CATEGORIES = {"causal_judgment", "moral_permissibility", "simple_ethical_questions"}

def category_from_scenario(scenario_name):
    for cat in EXPERIMENT_CATEGORIES:
        if scenario_name.startswith(cat):
            return cat
    return scenario_name.split('_')[0]

def build_rows(results_dir, rows_granite, rows_commandr):
    for scenario_dir in sorted(results_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        for json_file in sorted(scenario_dir.glob("*.json")):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            meta           = data.get("metadata", {})
            transcript     = data.get("debate_transcript", [])
            target_scores  = meta.get("target_scores", {})
            correct_option = next((k for k, v in target_scores.items() if v == 1), None)
            timing         = meta.get("timing", {})
            scenario_name  = meta.get("scenario", "")
            exp_type       = meta.get("experiment_type", "multi_agent")
            file_id        = str(json_file.relative_to(current_dir))

            base_row = {
                "Experiment_Type":  exp_type,
                "Category":         category_from_scenario(scenario_name),
                "Num_Agents":       meta.get("num_agents", 1),
                "Temperature":      meta.get("temperature"),
                "Ctx_Size":         meta.get("ctx_size"),
                "Model":            meta.get("model"),
                "Turn_Policy":      meta.get("turn_policy", "n/a"),
                "Run_ID":           meta.get("run_id", 0),
                "Turns_Taken":      len(transcript),
                "Correct_Option":   correct_option,
                "Elapsed_Seconds":  timing.get("elapsed_seconds_total"),
                "Avg_Turn_Seconds": timing.get("avg_seconds_per_turn"),
                "N_Turns":          timing.get("n_turns"),
                "File":             file_id,
            }

            # Granite
            ev_g = data.get("evaluation", {})
            nlp_g = ev_g.get("nlp_metrics", {})
            dyn_g = ev_g.get("dynamics") or {}
            row_g = {**base_row,
                "Agents_Verdict": ev_g.get("verdict"),
                "Is_Correct":     ev_g.get("is_correct"),
                "BERTScore_F1":   nlp_g.get("bertscore_f1"),
                "BLEU":           nlp_g.get("bleu"),
                "ROUGE_L":        nlp_g.get("rouge_l"),
            }
            if dyn_g:
                row_g.update({
                    "Speaker_Gini":      dyn_g.get("speaker_dominance_gini"),
                    "Silent_Agents":     dyn_g.get("silent_agents"),
                    "Avg_Turn_Length":   dyn_g.get("avg_turn_length_chars"),
                    "Max_Speaker_Share": dyn_g.get("max_speaker_share"),
                })
            rows_granite.append(row_g)

            # Command-R
            ev_c = data.get("evaluation_commandr", {})
            if ev_c:
                nlp_c = ev_c.get("nlp_metrics", {})
                dyn_c = ev_c.get("dynamics") or {}
                row_c = {**base_row,
                    "Agents_Verdict": ev_c.get("verdict"),
                    "Is_Correct":     ev_c.get("is_correct"),
                    "BERTScore_F1":   nlp_c.get("bertscore_f1"),
                    "BLEU":           nlp_c.get("bleu"),
                    "ROUGE_L":        nlp_c.get("rouge_l"),
                }
                if dyn_c:
                    row_c.update({
                        "Speaker_Gini":      dyn_c.get("speaker_dominance_gini"),
                        "Silent_Agents":     dyn_c.get("silent_agents"),
                        "Avg_Turn_Length":   dyn_c.get("avg_turn_length_chars"),
                        "Max_Speaker_Share": dyn_c.get("max_speaker_share"),
                    })
                rows_commandr.append(row_c)

rows_granite, rows_commandr = [], []
build_rows(mas_dir,      rows_granite, rows_commandr)
build_rows(baseline_dir, rows_granite, rows_commandr)

df_g = pd.DataFrame(rows_granite)
df_c = pd.DataFrame(rows_commandr)

print(f"Granite:  {len(df_g)} filas | únicos: {df_g['File'].nunique()}")
print(f"CommandR: {len(df_c)} filas | únicos: {df_c['File'].nunique()}")
print("Categorías Granite:",  df_g['Category'].value_counts().to_dict())
print("Categorías CommandR:", df_c['Category'].value_counts().to_dict())

df_g.to_csv(current_dir / "tfg_analisis.csv",         index=False)
df_c.to_csv(current_dir / "tfg_analisis_command.csv",  index=False)
print("CSVs regenerados correctamente.")
