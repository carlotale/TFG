"""
Smoke test: ejecuta una sola configuración mínima para validar que el
servidor de Ollama del GSI responde correctamente y que la latencia de los
modelos elegidos es manejable antes de lanzar el experimento completo.

Genera UN debate con cada modelo y mide el tiempo total. Si todo va bien,
puedes lanzar `execution.py` con confianza. Si hay errores o tarda demasiado,
ajusta los parámetros antes del barrido masivo.
"""

import os

os.environ["OLLAMA_HOST"] = "https://ollama2.gsi.upm.es/"

import json
import time
from pathlib import Path
from src.llm_conversation.ai_agent import AIAgent
from src.llm_conversation.conversation_manager import ConversationManager

# Modelos a validar
MODELS_TO_TEST = ["llama3.3:latest", "phi4:14b"]

# Dilema de prueba (ejemplo simple de juicio causal)
TEST_QUESTION = (
    "A man pushed a heavy boulder down a hill. The boulder struck a tree, "
    "which fell and damaged a fence. Did the man cause the fence to be damaged? "
    "Answer with Yes or No and briefly explain."
)

# Roles de prueba (los mismos que se usarán en producción)
TEST_ROLES = [
    {"role": "Causal Scientist",
     "prompt": "Focus on physical cause-effect chains and counterfactual logic."},
    {"role": "Legal Expert",
     "prompt": "Focus on negligence, duty of care, and legal liability."}
]


def run_smoke_test_for_model(model):
    print(f"\n{'═' * 60}")
    print(f"  TESTANDO: {model}")
    print(f"{'═' * 60}")

    start = time.time()

    try:
        agents = []
        for r in TEST_ROLES:
            full_prompt = (
                f"Role: {r['role']}. {r['prompt']} "
                "Task: Resolve the following dilemma."
            )
            agents.append(AIAgent(
                name=r['role'].replace(" ", "_"),
                model=model,
                system_prompt=full_prompt,
                temperature=0.7,
                ctx_size=2048
            ))

        manager = ConversationManager(
            agents=agents,
            initial_message=TEST_QUESTION,
            allow_termination=True,
            turn_order="round_robin"
        )

        conv_iterator = manager.run_conversation()
        n_turns = 0
        total_chars = 0

        for turn_idx, (agent_name, message_stream) in enumerate(conv_iterator):
            content = ""
            for chunk in message_stream:
                content = chunk
            n_turns += 1
            total_chars += len(content)
            print(f"  [{agent_name}] respondió ({len(content)} chars)")

        elapsed = time.time() - start

        print(f"\n  ✓ Modelo OK")
        print(f"  ✓ Turnos generados: {n_turns}")
        print(f"  ✓ Caracteres totales: {total_chars}")
        print(f"  ✓ Tiempo total: {elapsed:.1f} s")
        print(f"  ✓ Promedio por turno: {elapsed / n_turns:.1f} s")

        return {
            "model": model,
            "status": "ok",
            "turns": n_turns,
            "elapsed_s": round(elapsed, 1),
            "avg_per_turn_s": round(elapsed / n_turns, 1)
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ✗ ERROR tras {elapsed:.1f}s: {e}")
        return {
            "model": model,
            "status": "error",
            "error": str(e),
            "elapsed_s": round(elapsed, 1)
        }


def estimate_full_experiment(results):
    """Estima cuánto tardará el experimento completo basándose en el tiempo
    de los smoke tests."""
    print(f"\n{'═' * 60}")
    print("  ESTIMACIÓN DEL EXPERIMENTO COMPLETO")
    print(f"{'═' * 60}")

    # Configuración real del experimento (debe coincidir con execution.py)
    n_agents = 3                    # 2, 3, 4
    n_temps = 2                     # 0.1, 0.7
    n_ctx = 2                       # 2048, 4096
    n_models = 2                    # llama3.3, phi4
    n_turns_pol = 3                 # round_robin, random, moderator
    n_runs = 3
    avg_dilemmas_per_category = 10  # estimación
    n_categories = 7

    total_scenarios = (
        n_agents * n_temps * n_ctx * n_models * n_turns_pol * n_runs
    )
    total_debates = total_scenarios * n_categories * avg_dilemmas_per_category

    avg_turn_time = sum(
        r.get("avg_per_turn_s", 0) for r in results if r["status"] == "ok"
    ) / max(1, sum(1 for r in results if r["status"] == "ok"))

    # Promedio de turnos por debate (estimación: ~6 turnos)
    avg_turns_per_debate = 6
    total_seconds = total_debates * avg_turns_per_debate * avg_turn_time
    total_hours = total_seconds / 3600

    print(f"  Escenarios totales:       {total_scenarios}")
    print(f"  Debates totales (aprox):  {total_debates}")
    print(f"  Tiempo estimado:          {total_hours:.1f} horas "
          f"({total_hours / 24:.1f} días)")
    print()
    if total_hours < 24:
        print("  → Viable en una sesión de cómputo")
    elif total_hours < 72:
        print("  → Viable si se ejecuta en background durante varios días")
    else:
        print("  ⚠️  Considera reducir matriz: menos categorías, menos runs,")
        print("     o solo 2 turn_policies en lugar de 3")


if __name__ == "__main__":
    print(f"Servidor: {os.environ.get('OLLAMA_HOST')}")
    print(f"Modelos a testar: {MODELS_TO_TEST}\n")

    results = []
    for model in MODELS_TO_TEST:
        results.append(run_smoke_test_for_model(model))

    print(f"\n{'═' * 60}")
    print("  RESUMEN")
    print(f"{'═' * 60}")
    for r in results:
        status_icon = "✓" if r["status"] == "ok" else "✗"
        print(f"  {status_icon} {r['model']}: {r['status']} "
              f"({r.get('elapsed_s', 'N/A')}s)")

    if all(r["status"] == "ok" for r in results):
        estimate_full_experiment(results)
        print("\n[OK] Todo correcto. Puedes lanzar execution.py y baseline.py")
    else:
        print("\n[!] Hay errores. Revisa la conexión al servidor o los modelos.")
