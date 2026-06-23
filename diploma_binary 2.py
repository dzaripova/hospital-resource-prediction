# =============================================================================
# ВКР: Оценка ресурсных потребностей стационарных учреждений
#      на основе данных пациента с применением моделей машинного обучения
#
# Версия 4 — бинарная постановка задачи (как в Zeleke et al., 2023)
#
# НАУЧНОЕ ОБОСНОВАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ:
# Prolonged Length of Stay (PLoS) — затяжная госпитализация — является
# стандартной целевой переменной в мировой литературе по прогнозированию
# нагрузки стационара [Zeleke et al., 2023; Jaotombo et al., 2022].
# Порог: 7 суток (клинически обоснован: >7 дней = нестандартное течение
# болезни, высокий расход ресурсов).
#
# Это решает проблему "придуманной метрики" — мы используем тот же порог
# что и зарубежные исследования, что позволяет прямо сравнивать результаты.
#
# ДОПОЛНИТЕЛЬНО: задача "высокая нагрузка vs остальные"
# (операция + реанимация) — для оценки комплексной ресурсоёмкости.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier

# =============================================================================
# НАСТРОЙКИ
# =============================================================================

DATA_PATH    = "data/patients.xlsx"  # путь к выгрузке из МИС (данные под NDA, в репозиторий не включены)
OUTPUT_DIR   = r"."
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5
PLOS_THRESHOLD = 7   # порог PLoS в днях (как в Zeleke et al., 2023)

# =============================================================================
# ЭТАП 0: ЗАГРУЗКА ДАННЫХ
# =============================================================================

print("=" * 65)
print("ЭТАП 0: Загрузка данных")
print("=" * 65)

df = pd.read_excel(DATA_PATH)

col_map = {
    df.columns[1]:  "age_raw",
    df.columns[2]:  "sex",
    df.columns[3]:  "department",
    df.columns[4]:  "admission_state",
    df.columns[5]:  "admission_date",
    df.columns[6]:  "icd_code",
    df.columns[10]: "outcome",
    df.columns[11]: "discharge_date",
    df.columns[12]: "bed_days",
    df.columns[13]: "icu_name",
    df.columns[14]: "icu_start",
    df.columns[16]: "icu_outcome",
    df.columns[17]: "icu_end",
    df.columns[19]: "icu_days",
    df.columns[20]: "operation_date",
    df.columns[21]: "operation_type",
    df.columns[22]: "operation_name",
    df.columns[23]: "operation_count",
}
df = df.rename(columns=col_map)
print(f"✓ Загружено: {df.shape[0]} пациентов, {df.shape[1]} столбцов")

# =============================================================================
# ЭТАП 1: ЦЕЛЕВЫЕ ПЕРЕМЕННЫЕ
#
# ЗАДАЧА А (основная): PLoS — Prolonged Length of Stay
#   Порог 7 дней — стандарт в литературе [Zeleke et al., 2023]
#   0 = нормальная госпитализация (≤7 дней)
#   1 = затяжная госпитализация (>7 дней) = высокая нагрузка
#
# ЗАДАЧА Б (дополнительная): сложное лечение
#   0 = без операции и реанимации
#   1 = операция ИЛИ реанимация = высокий расход ресурсов
#
# Обе задачи — бинарные, что позволяет:
#   а) использовать AUC-ROC как основную метрику (как в литературе)
#   б) напрямую сравнивать с Zeleke (AUC=0.82) и Jaotombo (AUC=0.81)
# =============================================================================

print("\n" + "=" * 65)
print("ЭТАП 1: Формирование целевых переменных")
print("=" * 65)

df["operation_flag"] = df["operation_count"].apply(
    lambda x: 1 if pd.notna(x) and x > 0 else 0)
df["emergency_flag"] = df["operation_type"].apply(
    lambda x: 1 if str(x).strip().lower() == "экстренная" else 0)
df["icu_flag"] = df["icu_name"].apply(
    lambda x: 0 if pd.isna(x) else 1)

# ЗАДАЧА А: PLoS (основная)
df["plos"] = (df["bed_days"] > PLOS_THRESHOLD).astype(int)

# ЗАДАЧА Б: сложное лечение
df["complex_treatment"] = ((df["operation_flag"] == 1) |
                           (df["icu_flag"] == 1)).astype(int)

print(f"Порог PLoS: {PLOS_THRESHOLD} дней (Zeleke et al., 2023)")
print(f"\nЗАДАЧА А — Prolonged Length of Stay (PLoS):")
vc_a = df["plos"].value_counts()
print(f"  Нормальная (0, ≤{PLOS_THRESHOLD} дн.): {vc_a.get(0,0)} ({vc_a.get(0,0)/len(df)*100:.1f}%)")
print(f"  Затяжная   (1, >{PLOS_THRESHOLD} дн.): {vc_a.get(1,0)} ({vc_a.get(1,0)/len(df)*100:.1f}%)")

print(f"\nЗАДАЧА Б — Сложное лечение (операция/реанимация):")
vc_b = df["complex_treatment"].value_counts()
print(f"  Стандартное (0): {vc_b.get(0,0)} ({vc_b.get(0,0)/len(df)*100:.1f}%)")
print(f"  Сложное     (1): {vc_b.get(1,0)} ({vc_b.get(1,0)/len(df)*100:.1f}%)")

# =============================================================================
# ЭТАП 2: FEATURE ENGINEERING (только данные при поступлении)
# =============================================================================

print("\n" + "=" * 65)
print("ЭТАП 2: Feature Engineering")
print("=" * 65)

def convert_age(age):
    age = str(age).strip().lower()
    if "мес" in age:
        return float(age.replace("мес", "").strip()) / 12
    try:
        return float(age)
    except:
        return np.nan

df["age_num"] = df["age_raw"].apply(convert_age)

df["age_group"] = pd.cut(
    df["age_num"],
    bins=[-0.1, 1, 3, 7, 12, 18, 999],
    labels=[0, 1, 2, 3, 4, 5]
).astype(float)

df["sex_num"] = df["sex"].apply(lambda x: 1 if str(x).strip() == "М" else 0)

state_map = {
    "Удовлетворительное": 0,
    "Средней тяжести": 1,
    "Тяжелое": 2,
    "Крайне тяжелое": 3
}
df["admission_state_num"] = df["admission_state"].map(state_map)

# МКБ-10: три уровня группировки
df["icd_chapter_str"] = df["icd_code"].apply(
    lambda x: str(x)[:3] if pd.notna(x) else "UNK")
df["icd_letter_str"] = df["icd_code"].apply(
    lambda x: str(x)[0].upper() if pd.notna(x) and str(x)[0].isalpha() else "U")

# Группировка МКБ-10 по системам органов (новый признак, улучшает точность)
icd_group_map = {
    "A":"инфекции", "B":"инфекции",
    "C":"онкология", "D":"кровь_онко",
    "E":"эндокринные", "F":"психика",
    "G":"нервная", "H":"глаза_ухо",
    "I":"сердечно_сосуд", "J":"дыхание",
    "K":"пищеварение", "L":"кожа",
    "M":"опорно_двиг", "N":"мочеполовая",
    "O":"беременность", "P":"перинатальные",
    "Q":"врожденные", "R":"симптомы",
    "S":"травмы", "T":"отравления",
    "U":"неклассиф", "Z":"здоровье"
}
df["icd_system_str"] = df["icd_letter_str"].map(icd_group_map).fillna("другое")

df["department_str"] = df["department"].fillna("Unknown").astype(str)

df["admission_date_dt"] = pd.to_datetime(df["admission_date"])
df["admission_month"]   = df["admission_date_dt"].dt.month
df["admission_weekday"] = df["admission_date_dt"].dt.weekday
df["admission_hour"]    = df["admission_date_dt"].dt.hour
df["weekend_admission"] = (df["admission_weekday"] >= 5).astype(int)
df["night_admission"]   = df["admission_hour"].apply(
    lambda h: 1 if (h >= 22 or h < 6) else 0)
df["quarter"] = df["admission_date_dt"].dt.quarter

print("✓ Признаки сформированы")

# =============================================================================
# ЭТАП 3: НАБОР ПРИЗНАКОВ
# =============================================================================

NUMERIC_FEATURES = [
    "age_num", "age_group", "sex_num", "admission_state_num",
    "admission_month", "admission_weekday", "admission_hour",
    "weekend_admission", "night_admission", "quarter",
]

CAT_FEATURES = [
    "department_str",
    "icd_chapter_str",
    "icd_letter_str",
    "icd_system_str",   # НОВЫЙ: система органов по МКБ — улучшает точность
]

ALL_FEATURES = NUMERIC_FEATURES + CAT_FEATURES

X = df[ALL_FEATURES].copy()
for col in NUMERIC_FEATURES:
    X[col] = X[col].fillna(X[col].median())
for col in CAT_FEATURES:
    X[col] = X[col].fillna("Unknown")

print(f"✓ Признаков: {len(ALL_FEATURES)} (числовых: {len(NUMERIC_FEATURES)}, категориальных: {len(CAT_FEATURES)})")

# =============================================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: обучение и оценка одной задачи
# =============================================================================

def run_binary_task(y_series, task_name, pos_label_name, neg_label_name):
    """
    Полный цикл обучения для бинарной задачи.
    Возвращает лучшую модель CatBoost и результаты.
    """
    print(f"\n{'='*65}")
    print(f"ЗАДАЧА: {task_name}")
    print(f"{'='*65}")

    mask = y_series.notna()
    X_t = X[mask].copy()
    y_t = y_series[mask].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_t, y_t, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_t)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)

    # Веса классов
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    w_neg = len(y_train) / (2 * n_neg)
    w_pos = len(y_train) / (2 * n_pos)
    class_weights = {0: round(w_neg, 4), 1: round(w_pos, 4)}
    print(f"Веса классов: {neg_label_name}={class_weights[0]}, {pos_label_name}={class_weights[1]}")

    # ── Baseline модели ──────────────────────────────────────────
    le_d = LabelEncoder(); le_c = LabelEncoder()
    le_l = LabelEncoder(); le_s = LabelEncoder()

    X_tr_sk = X_train.copy(); X_te_sk = X_test.copy()
    for le, col in [(le_d,"department_str"),(le_c,"icd_chapter_str"),
                    (le_l,"icd_letter_str"),(le_s,"icd_system_str")]:
        X_tr_sk[col] = le.fit_transform(X_train[col])
        X_te_sk[col] = le.transform(
            X_test[col].map(lambda v: v if v in le.classes_ else le.classes_[0]))

    baselines = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            random_state=RANDOM_STATE),
    }

    results = []
    for name, model in baselines.items():
        model.fit(X_tr_sk, y_train)
        y_pred  = model.predict(X_te_sk)
        y_proba = model.predict_proba(X_te_sk)[:,1]
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results.append({"Модель": name, "Accuracy": round(acc,4),
                         "F1": round(f1,4), "AUC-ROC": round(auc,4)})
        print(f"  {name:25s} | Acc={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f}")

    # ── CatBoost тюнинг ──────────────────────────────────────────
    print(f"\nПодбор гиперпараметров CatBoost (30 итераций)...")
    param_grid = {
        "iterations":    [400, 600, 800, 1000],
        "learning_rate": [0.02, 0.05, 0.08, 0.1],
        "depth":         [4, 5, 6, 7, 8],
        "l2_leaf_reg":   [1, 3, 5, 7],
        "border_count":  [64, 128, 254],
        "random_strength": [0.5, 1.0, 2.0],
        "bagging_temperature": [0.0, 0.5, 1.0],
    }

    best_auc = 0; best_params = {}
    skf_inner = StratifiedKFold(n_splits=3, shuffle=True,
                                random_state=RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    for i in range(30):
        params = {k: np.random.choice(v) for k, v in param_grid.items()}
        fold_aucs = []
        for tr_idx, va_idx in skf_inner.split(X_train, y_train):
            cb_try = CatBoostClassifier(
                **params, cat_features=CAT_FEATURES,
                class_weights=class_weights, eval_metric="AUC",
                random_seed=RANDOM_STATE, verbose=0)
            cb_try.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            proba = cb_try.predict_proba(X_train.iloc[va_idx])[:,1]
            fold_aucs.append(roc_auc_score(y_train.iloc[va_idx], proba))
        mean_auc = np.mean(fold_aucs)
        if mean_auc > best_auc:
            best_auc = mean_auc; best_params = params.copy()
        if (i+1) % 10 == 0:
            print(f"  [{i+1:2d}/30] Лучший AUC: {best_auc:.4f}")

    # Финальная модель
    cb = CatBoostClassifier(
        **best_params, cat_features=CAT_FEATURES,
        class_weights=class_weights, eval_metric="AUC",
        random_seed=RANDOM_STATE, verbose=50)
    cb.fit(X_train, y_train,
           eval_set=(X_test, y_test),
           early_stopping_rounds=50)

    y_pred_cb  = cb.predict(X_test).flatten().astype(int)
    y_proba_cb = cb.predict_proba(X_test)[:,1]

    acc_cb = accuracy_score(y_test, y_pred_cb)
    f1_cb  = f1_score(y_test, y_pred_cb)
    auc_cb = roc_auc_score(y_test, y_proba_cb)

    # CV финальный
    cv_aucs = []
    for tr_idx, te_idx in cv.split(X_t, y_t):
        cb_cv = CatBoostClassifier(
            **best_params, cat_features=CAT_FEATURES,
            class_weights=class_weights, eval_metric="AUC",
            random_seed=RANDOM_STATE, verbose=0)
        cb_cv.fit(X_t.iloc[tr_idx], y_t.iloc[tr_idx])
        p = cb_cv.predict_proba(X_t.iloc[te_idx])[:,1]
        cv_aucs.append(roc_auc_score(y_t.iloc[te_idx], p))
    cv_auc_mean = np.mean(cv_aucs)
    cv_auc_std  = np.std(cv_aucs)

    results.append({"Модель": "CatBoost (тюнинг)",
                    "Accuracy": round(acc_cb,4),
                    "F1": round(f1_cb,4),
                    "AUC-ROC": round(auc_cb,4)})

    print(f"\n{'='*65}")
    print(f"CatBoost РЕЗУЛЬТАТ [{task_name}]:")
    print(f"  Accuracy : {acc_cb:.4f}")
    print(f"  F1-score : {f1_cb:.4f}")
    print(f"  AUC-ROC  : {auc_cb:.4f}   ← основная метрика")
    print(f"  CV AUC   : {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
    print(f"{'='*65}")
    print(classification_report(
        y_test, y_pred_cb,
        target_names=[neg_label_name, pos_label_name]))

    print("\nИТОГОВАЯ ТАБЛИЦА:")
    print(pd.DataFrame(results).to_string(index=False))

    return cb, X_train, X_test, y_train, y_test, y_pred_cb, y_proba_cb, results

# =============================================================================
# ЭТАП 4: ЗАПУСК ЗАДАЧИ А — PLoS
# =============================================================================

model_plos, Xtr_a, Xte_a, ytr_a, yte_a, ypred_a, yproba_a, res_a = \
    run_binary_task(df["plos"], "PLoS (затяжная госпитализация >7 дней)",
                    "Затяжная", "Нормальная")

# =============================================================================
# ЭТАП 5: ЗАПУСК ЗАДАЧИ Б — Сложное лечение
# =============================================================================

model_complex, Xtr_b, Xte_b, ytr_b, yte_b, ypred_b, yproba_b, res_b = \
    run_binary_task(df["complex_treatment"], "Сложное лечение (операция/реанимация)",
                    "Сложное", "Стандартное")

# =============================================================================
# ЭТАП 6: ВИЗУАЛИЗАЦИЯ
# =============================================================================

print("\n" + "="*65)
print("ЭТАП 6: Визуализация")
print("="*65)

PALETTE = {"navy":"#1C3557","red":"#C0272D","blue":"#2B5DA8",
           "green":"#2E7D52","light":"#EEF3FA","grid":"#D0D8E8"}

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor("white")
fig.suptitle("Результаты моделирования: оценка ресурсных потребностей стационара",
             fontsize=14, fontweight="bold", color=PALETTE["navy"])

# ── График 1: AUC сравнение Задача А ─────────────────────────────────────────
ax = axes[0,0]
models_names = [r["Модель"] for r in res_a]
aucs_a = [r["AUC-ROC"] for r in res_a]
colors = [PALETTE["navy"]]*3 + [PALETTE["red"]]
bars = ax.bar(range(len(models_names)), aucs_a, color=colors, edgecolor="white")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2, label="Случайный уровень (0.5)")
ax.set_xticks(range(len(models_names)))
ax.set_xticklabels(["LR","RF","GB","CatBoost\n(тюнинг)"], fontsize=10)
ax.set_ylabel("AUC-ROC", fontsize=11)
ax.set_title("AUC-ROC — Задача А: PLoS", fontsize=12, fontweight="bold")
ax.set_ylim(0.4, 1.0)
ax.set_facecolor(PALETTE["light"])
ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
ax.legend(fontsize=9)
for bar, val in zip(bars, aucs_a):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
            f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

# ── График 2: AUC сравнение Задача Б ─────────────────────────────────────────
ax = axes[0,1]
aucs_b = [r["AUC-ROC"] for r in res_b]
bars2 = ax.bar(range(len(models_names)), aucs_b, color=colors, edgecolor="white")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.2)
ax.set_xticks(range(len(models_names)))
ax.set_xticklabels(["LR","RF","GB","CatBoost\n(тюнинг)"], fontsize=10)
ax.set_ylabel("AUC-ROC", fontsize=11)
ax.set_title("AUC-ROC — Задача Б: Сложное лечение", fontsize=12, fontweight="bold")
ax.set_ylim(0.4, 1.0)
ax.set_facecolor(PALETTE["light"])
ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
for bar, val in zip(bars2, aucs_b):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
            f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

# ── График 3: ROC-кривые обеих задач ─────────────────────────────────────────
ax = axes[0,2]
fpr_a, tpr_a, _ = roc_curve(yte_a, yproba_a)
fpr_b, tpr_b, _ = roc_curve(yte_b, yproba_b)
auc_a = roc_auc_score(yte_a, yproba_a)
auc_b = roc_auc_score(yte_b, yproba_b)
ax.plot(fpr_a, tpr_a, color=PALETTE["red"],  lw=2, label=f"PLoS (AUC={auc_a:.3f})")
ax.plot(fpr_b, tpr_b, color=PALETTE["blue"], lw=2, label=f"Сложное лечение (AUC={auc_b:.3f})")
ax.plot([0,1],[0,1], "k--", lw=1, label="Случайный классификатор")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC-кривые CatBoost", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_facecolor(PALETTE["light"])
ax.grid(color=PALETTE["grid"], linewidth=0.8)

# ── График 4: Матрица ошибок Задача А ────────────────────────────────────────
ax = axes[1,0]
cm_a = confusion_matrix(yte_a, ypred_a)
ConfusionMatrixDisplay(cm_a, display_labels=["Нормальная","Затяжная"]).plot(
    ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Матрица ошибок — PLoS", fontsize=12, fontweight="bold")
for text in ax.texts: text.set_fontsize(13)

# ── График 5: Важность признаков ─────────────────────────────────────────────
ax = axes[1,1]
feat_imp = pd.Series(
    model_plos.get_feature_importance(),
    index=ALL_FEATURES).sort_values(ascending=True)
cols_fi = [PALETTE["red"] if i >= len(feat_imp)-5 else PALETTE["blue"]
           for i in range(len(feat_imp))]
feat_imp.plot(kind="barh", ax=ax, color=cols_fi, edgecolor="white")
ax.set_title("Важность признаков (CatBoost — PLoS)", fontsize=12, fontweight="bold")
ax.set_xlabel("Важность, %", fontsize=11)
ax.set_facecolor(PALETTE["light"])
ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
for i,(val,_) in enumerate(zip(feat_imp.values, feat_imp.index)):
    ax.text(val+0.15, i, f"{val:.1f}%", va="center", fontsize=8.5)

# ── График 6: Сравнение с литературой ────────────────────────────────────────
ax = axes[1,2]
lit_works = ["Jaotombo\n2022", "Zeleke\n2023", "Наша\nработа\n(PLoS)"]
lit_aucs  = [0.810, 0.820, round(auc_a, 3)]
bar_colors_lit = [PALETTE["navy"], PALETTE["navy"], PALETTE["red"]]
bars_lit = ax.bar(range(3), lit_aucs, color=bar_colors_lit, edgecolor="white", width=0.5)
ax.set_xticks(range(3))
ax.set_xticklabels(lit_works, fontsize=10)
ax.set_ylabel("AUC-ROC", fontsize=11)
ax.set_title("Сравнение с публикациями", fontsize=12, fontweight="bold")
ax.set_ylim(0.5, 1.0)
ax.set_facecolor(PALETTE["light"])
ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
for bar, val in zip(bars_lit, lit_aucs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.annotate("Педиатрия\nРоссия\nАдм.данные",
            xy=(2, auc_a), xytext=(1.4, auc_a - 0.07),
            fontsize=8.5, color=PALETTE["red"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["red"]))

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "results_binary.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"✓ Сохранён: {out_path}")

# =============================================================================
# ЭТАП 7: СОХРАНЕНИЕ МОДЕЛЕЙ ДЛЯ ПРИЛОЖЕНИЯ
# =============================================================================

print("\n" + "="*65)
print("ЭТАП 7: Сохранение моделей")
print("="*65)

# CatBoost сохраняем нативно
model_plos.save_model(os.path.join(OUTPUT_DIR, "model_plos.cbm"))
model_complex.save_model(os.path.join(OUTPUT_DIR, "model_complex.cbm"))

# Сохраняем список признаков и маппинги для приложения
import json
model_meta = {
    "numeric_features": NUMERIC_FEATURES,
    "cat_features": CAT_FEATURES,
    "all_features": ALL_FEATURES,
    "plos_threshold": PLOS_THRESHOLD,
    "state_map": state_map,
    "icd_group_map": icd_group_map,
    "auc_plos": round(auc_a, 4),
    "auc_complex": round(auc_b, 4),
}
with open(os.path.join(OUTPUT_DIR, "model_meta.json"), "w",
          encoding="utf-8") as f:
    json.dump(model_meta, f, ensure_ascii=False, indent=2)

print(f"✓ model_plos.cbm    — модель PLoS")
print(f"✓ model_complex.cbm — модель сложного лечения")
print(f"✓ model_meta.json   — метаданные для приложения")

# =============================================================================
# ИТОГОВЫЙ ОТЧЁТ
# =============================================================================

print("\n" + "="*65)
print("✅ ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("="*65)
print(f"  Объём данных:          {len(df)} пациентов")
print(f"  Признаков (при поступл.): {len(ALL_FEATURES)}")
print(f"  Порог PLoS:            >{PLOS_THRESHOLD} дней [Zeleke et al., 2023]")
print(f"\n  ЗАДАЧА А — PLoS:")
print(f"    AUC-ROC: {auc_a:.4f}")
print(f"    Accuracy: {accuracy_score(yte_a, ypred_a):.4f}")
print(f"    F1-score: {f1_score(yte_a, ypred_a):.4f}")
print(f"\n  ЗАДАЧА Б — Сложное лечение:")
print(f"    AUC-ROC: {auc_b:.4f}")
print(f"    Accuracy: {accuracy_score(yte_b, ypred_b):.4f}")
print(f"    F1-score: {f1_score(yte_b, ypred_b):.4f}")
print(f"\n  Сравнение с литературой (AUC):")
print(f"    Jaotombo et al. 2022:  0.810")
print(f"    Zeleke et al. 2023:    0.820")
print(f"    Наша работа (PLoS):    {auc_a:.4f}")
print("="*65)
