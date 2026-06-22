import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

granite  = pd.read_csv("tfg_analisis.csv")
commandr = pd.read_csv("tfg_analisis_command.csv")

merged = pd.merge(
    granite[['File', 'Is_Correct', 'Agents_Verdict', 'Correct_Option',
             'Experiment_Type', 'Category', 'Model', 'Temperature', 'Turn_Policy', 'Num_Agents']].rename(columns={
        'Is_Correct':    'Granite_Is_Correct',
        'Agents_Verdict':'Granite_Verdict'
    }),
    commandr[['File', 'Is_Correct', 'Agents_Verdict']].rename(columns={
        'Is_Correct':    'CommandR_Is_Correct',
        'Agents_Verdict':'CommandR_Verdict'
    }),
    on='File'
)

print(f"Total ficheros cruzados: {len(merged)}")

print("\n" + "="*60)
print("INTER-JUDGE AGREEMENT: Granite vs Command-R")
print("="*60)

valid = merged[['Granite_Is_Correct', 'CommandR_Is_Correct']].dropna()
kappa = cohen_kappa_score(valid['Granite_Is_Correct'].astype(int),
                          valid['CommandR_Is_Correct'].astype(int))
pct_agree = (merged['Granite_Verdict'] == merged['CommandR_Verdict']).mean() * 100

print(f"\n  Cohen's Kappa:          {kappa:.4f}")
print(f"  % acuerdo en veredicto: {pct_agree:.2f}%")
print(f"  N evaluaciones:         {len(valid)}")

print("\n[1] KAPPA POR TIPO DE EXPERIMENTO")
for exp_type, group in merged.groupby('Experiment_Type'):
    v = group[['Granite_Is_Correct', 'CommandR_Is_Correct']].dropna()
    k = cohen_kappa_score(v['Granite_Is_Correct'].astype(int),
                          v['CommandR_Is_Correct'].astype(int))
    ag = (group['Granite_Verdict'] == group['CommandR_Verdict']).mean() * 100
    print(f"  {exp_type}: Kappa={k:.4f}  Acuerdo={ag:.2f}%  N={len(v)}")

print("\n[2] KAPPA POR CATEGORÍA")
for cat, group in merged.groupby('Category'):
    v = group[['Granite_Is_Correct', 'CommandR_Is_Correct']].dropna()
    k = cohen_kappa_score(v['Granite_Is_Correct'].astype(int),
                          v['CommandR_Is_Correct'].astype(int))
    ag = (group['Granite_Verdict'] == group['CommandR_Verdict']).mean() * 100
    print(f"  {cat}: Kappa={k:.4f}  Acuerdo={ag:.2f}%  N={len(v)}")

print("\n[3] KAPPA POR MODELO DEBATIENTE")
for model, group in merged.groupby('Model'):
    v = group[['Granite_Is_Correct', 'CommandR_Is_Correct']].dropna()
    k = cohen_kappa_score(v['Granite_Is_Correct'].astype(int),
                          v['CommandR_Is_Correct'].astype(int))
    ag = (group['Granite_Verdict'] == group['CommandR_Verdict']).mean() * 100
    print(f"  {model}: Kappa={k:.4f}  Acuerdo={ag:.2f}%  N={len(v)}")

print("\n[4] ANÁLISIS DE DESACUERDOS")
disagree = merged[merged['Granite_Verdict'] != merged['CommandR_Verdict']]
print(f"  Total desacuerdos: {len(disagree)} ({len(disagree)/len(merged)*100:.2f}%)")
print(f"  Granite correcto, CommandR incorrecto: {((disagree['Granite_Is_Correct']==1) & (disagree['CommandR_Is_Correct']==0)).sum()}")
print(f"  CommandR correcto, Granite incorrecto: {((disagree['Granite_Is_Correct']==0) & (disagree['CommandR_Is_Correct']==1)).sum()}")
print(f"  Ambos incorrectos pero distinta opción: {((disagree['Granite_Is_Correct']==0) & (disagree['CommandR_Is_Correct']==0)).sum()}")
print("\n  Desacuerdos por categoría:")
print(disagree.groupby('Category').size().to_string())
print("\n  Desacuerdos por tipo de experimento:")
print(disagree.groupby('Experiment_Type').size().to_string())

print("\n[5] ACCURACY COMPARADA: GRANITE vs COMMAND-R")
for exp_type, group in merged.groupby('Experiment_Type'):
    g_acc = group['Granite_Is_Correct'].mean() * 100
    c_acc = group['CommandR_Is_Correct'].mean() * 100
    print(f"  {exp_type}: Granite={g_acc:.2f}%  CommandR={c_acc:.2f}%  Diff={c_acc-g_acc:+.2f}%")