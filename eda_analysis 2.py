# =============================================================================
# ВКР: Оценка ресурсных потребностей стационара
# РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)
#
# Что делает скрипт:
#   1. Загружает исходный Excel-файл с данными
#   2. Выводит в консоль все цифры для главы 2 диплома
#   3. Сохраняет 4 графика (Рисунки 1–4) в PNG
#   4. Сохраняет таблицу топ-10 диагнозов в CSV
#
# Запуск:
#   python eda_analysis.py
#
# Все результаты сохраняются в подпапку ./eda_output/
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# НАСТРОЙКИ
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "data/patients.xlsx"  # путь к выгрузке из МИС (данные под NDA, в репозиторий не включены)
OUTPUT_DIR = r"./eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Цветовая палитра (синие/тёмные тона, согласованная с остальными графиками ВКР)
COLOR_MAIN = "#1C3557"
COLOR_ACCENT = "#2B5DA8"
COLOR_HIGHLIGHT = "#C0272D"
COLOR_GRID = "#E2E7EF"
COLORS_BARS = ["#1C3557", "#2B5DA8", "#3A7BC8", "#5B95D8", "#7FAEE6",
               "#A5C8F0", "#C0272D", "#7A1518", "#1E6B40", "#3FAF72"]

# ─────────────────────────────────────────────────────────────────────────────
# СПРАВОЧНИКИ
# ─────────────────────────────────────────────────────────────────────────────
ICD_CLASS_NAMES = {
    'A': 'Инфекционные и паразитарные',
    'B': 'Инфекционные и паразитарные',
    'C': 'Новообразования',
    'D': 'Болезни крови и кроветворных органов',
    'E': 'Болезни эндокринной системы',
    'F': 'Психические расстройства',
    'G': 'Болезни нервной системы',
    'H': 'Болезни глаза и уха',
    'I': 'Болезни системы кровообращения',
    'J': 'Болезни органов дыхания',
    'K': 'Болезни органов пищеварения',
    'L': 'Болезни кожи и подкожной клетчатки',
    'M': 'Болезни костно-мышечной системы',
    'N': 'Болезни мочеполовой системы',
    'O': 'Беременность и роды',
    'P': 'Перинатальные состояния',
    'Q': 'Врождённые аномалии',
    'R': 'Симптомы и отклонения от нормы',
    'S': 'Травмы',
    'T': 'Травмы и отравления',
    'U': 'Неклассифицированные',
    'Z': 'Факторы, влияющие на здоровье',
}

# ─────────────────────────────────────────────────────────────────────────────
# ЗАГРУЗКА ДАННЫХ
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("ЗАГРУЗКА ДАННЫХ")
print("=" * 70)

df = pd.read_excel(DATA_PATH)
print(f"✓ Загружено: {len(df)} пациентов, {df.shape[1]} столбцов")
print(f"✓ Имена столбцов:")
for i, col in enumerate(df.columns):
    print(f"   [{i}] {col}")
print()

# Стандартизация имён столбцов (как в diploma_binary.py)
col_map = {
    df.columns[1]:  "age_raw",
    df.columns[2]:  "sex",
    df.columns[3]:  "department",
    df.columns[4]:  "admission_state",
    df.columns[5]:  "admission_date",
    df.columns[6]:  "icd_code",
    df.columns[7]:  "icd_name",
    df.columns[12]: "bed_days",
    df.columns[13]: "icu_name",
    df.columns[23]: "operation_count",
}
df = df.rename(columns=col_map)

# ─────────────────────────────────────────────────────────────────────────────
# ПОДГОТОВКА ПРИЗНАКОВ ДЛЯ EDA
# ─────────────────────────────────────────────────────────────────────────────

# Возраст: переводим "6 мес" в годы
def parse_age(s):
    s = str(s).strip().lower()
    if "мес" in s:
        try: return float(s.replace("мес","").strip()) / 12
        except: return np.nan
    try: return float(s)
    except: return np.nan

df["age_num"] = df["age_raw"].apply(parse_age)

# Возрастная группа
def age_group_label(age):
    if pd.isna(age): return "не определён"
    if age < 1: return "0–1 год"
    elif age < 3: return "1–3 года"
    elif age < 7: return "3–7 лет"
    elif age < 12: return "7–12 лет"
    elif age < 18: return "12–18 лет"
    else: return "18+"

df["age_group_label"] = df["age_num"].apply(age_group_label)

# Класс МКБ-10 (первая буква)
df["icd_class"] = df["icd_code"].astype(str).str[0]

# Целевые переменные
df["plos"] = (df["bed_days"] > 7).astype(int)
df["operation_flag"] = (df["operation_count"].fillna(0).astype(float) > 0).astype(int)
df["icu_flag"] = df["icu_name"].notna().astype(int)
df["complex_treatment"] = ((df["operation_flag"] == 1) | (df["icu_flag"] == 1)).astype(int)

total = len(df)

# =============================================================================
# БЛОК 1: РАСПРЕДЕЛЕНИЕ ПО ТЯЖЕСТИ СОСТОЯНИЯ
# =============================================================================
print("=" * 70)
print("БЛОК 1: РАСПРЕДЕЛЕНИЕ ПО ТЯЖЕСТИ СОСТОЯНИЯ ПРИ ПОСТУПЛЕНИИ")
print("=" * 70)

severity_counts = df["admission_state"].value_counts()
print(f"{'Состояние':<30} {'Количество':>12} {'Доля, %':>10}")
print("-" * 56)
for state, cnt in severity_counts.items():
    pct = cnt / total * 100
    print(f"{str(state)[:30]:<30} {cnt:>12} {pct:>9.2f}%")
print()

# График: Рисунок 1
fig, ax = plt.subplots(figsize=(9, 5))
states_order = ["Удовлетворительное", "Средней тяжести", "Тяжелое", "Крайне тяжелое"]
counts_ordered = [severity_counts.get(s, 0) for s in states_order]
bars = ax.bar(states_order, counts_ordered,
              color=[COLOR_ACCENT, COLOR_MAIN, "#7A1518", COLOR_HIGHLIGHT])

for bar, cnt in zip(bars, counts_ordered):
    pct = cnt / total * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
            f"{cnt}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)

ax.set_ylabel("Количество пациентов, чел.", fontsize=12)
ax.set_title("Рисунок 1 — Распределение пациентов по тяжести состояния\nпри поступлении в стационар",
             fontsize=12, pad=15)
ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
ax.set_ylim(0, max(counts_ordered) * 1.18)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_severity.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"✓ Сохранён: {OUTPUT_DIR}/fig1_severity.png\n")

# =============================================================================
# БЛОК 2: ВОЗРАСТНОЕ РАСПРЕДЕЛЕНИЕ
# =============================================================================
print("=" * 70)
print("БЛОК 2: ВОЗРАСТНОЕ РАСПРЕДЕЛЕНИЕ ПАЦИЕНТОВ")
print("=" * 70)

age_order = ["0–1 год", "1–3 года", "3–7 лет", "7–12 лет", "12–18 лет"]
age_counts = df["age_group_label"].value_counts()
print(f"{'Возрастная группа':<20} {'Количество':>12} {'Доля, %':>10}")
print("-" * 46)
for grp in age_order:
    cnt = age_counts.get(grp, 0)
    pct = cnt / total * 100
    print(f"{grp:<20} {cnt:>12} {pct:>9.2f}%")
print()

# График: Рисунок 2
fig, ax = plt.subplots(figsize=(9, 5))
counts_age = [age_counts.get(grp, 0) for grp in age_order]
bars = ax.bar(age_order, counts_age, color=COLOR_MAIN, edgecolor=COLOR_ACCENT, linewidth=1.2)

for bar, cnt in zip(bars, counts_age):
    pct = cnt / total * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
            f"{cnt}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)

ax.set_ylabel("Количество пациентов, чел.", fontsize=12)
ax.set_xlabel("Возрастная группа", fontsize=12)
ax.set_title("Рисунок 2 — Возрастное распределение пациентов,\nгоспитализированных в стационар",
             fontsize=12, pad=15)
ax.grid(axis="y", color=COLOR_GRID, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
ax.set_ylim(0, max(counts_age) * 1.18)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_age.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"✓ Сохранён: {OUTPUT_DIR}/fig2_age.png\n")

# =============================================================================
# БЛОК 3: РАСПРЕДЕЛЕНИЕ ПО ПОЛУ
# =============================================================================
print("=" * 70)
print("БЛОК 3: РАСПРЕДЕЛЕНИЕ ПО ПОЛУ")
print("=" * 70)

sex_counts = df["sex"].value_counts()
print(f"{'Пол':<20} {'Количество':>12} {'Доля, %':>10}")
print("-" * 46)
for sex, cnt in sex_counts.items():
    pct = cnt / total * 100
    print(f"{str(sex):<20} {cnt:>12} {pct:>9.2f}%")
print()

# График: Рисунок 3 (круговая)
fig, ax = plt.subplots(figsize=(7, 6))
labels = list(sex_counts.index)
sizes = list(sex_counts.values)
percentages = [s/total*100 for s in sizes]
colors_pie = [COLOR_ACCENT, COLOR_HIGHLIGHT]
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=[f"{lbl}\n({cnt} чел., {pct:.1f}%)" for lbl, cnt, pct in zip(labels, sizes, percentages)],
    colors=colors_pie,
    startangle=90,
    autopct='',
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
    textprops=dict(fontsize=11),
)

ax.set_title("Рисунок 3 — Распределение пациентов по полу",
             fontsize=12, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_sex.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"✓ Сохранён: {OUTPUT_DIR}/fig3_sex.png\n")

# =============================================================================
# БЛОК 4: РАСПРЕДЕЛЕНИЕ ПО ОТДЕЛЕНИЯМ
# =============================================================================
print("=" * 70)
print("БЛОК 4: ТОП-10 ОТДЕЛЕНИЙ ПО ЧИСЛУ ПАЦИЕНТОВ")
print("=" * 70)

dept_counts = df["department"].value_counts().head(10)
print(f"{'№':<3} {'Отделение':<45} {'Количество':>12} {'Доля, %':>10}")
print("-" * 74)
for i, (dept, cnt) in enumerate(dept_counts.items(), 1):
    pct = cnt / total * 100
    print(f"{i:<3} {str(dept)[:45]:<45} {cnt:>12} {pct:>9.2f}%")
print()

# График: Рисунок 4 (горизонтальные бары)
fig, ax = plt.subplots(figsize=(10, 6))
dept_labels = [str(d)[:35] + ("..." if len(str(d)) > 35 else "")
               for d in dept_counts.index]
y_pos = np.arange(len(dept_labels))

bars = ax.barh(y_pos, dept_counts.values, color=COLOR_MAIN, edgecolor=COLOR_ACCENT)
ax.set_yticks(y_pos)
ax.set_yticklabels(dept_labels, fontsize=10)
ax.invert_yaxis()

for bar, cnt in zip(bars, dept_counts.values):
    pct = cnt / total * 100
    ax.text(bar.get_width() + total*0.005, bar.get_y() + bar.get_height()/2,
            f"{cnt} ({pct:.1f}%)", va="center", fontsize=9)

ax.set_xlabel("Количество пациентов, чел.", fontsize=12)
ax.set_title("Рисунок 4 — Распределение пациентов по отделениям\n(топ-10 по числу госпитализаций)",
             fontsize=12, pad=15)
ax.grid(axis="x", color=COLOR_GRID, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
ax.set_xlim(0, max(dept_counts.values) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_departments.png"), dpi=200, bbox_inches="tight")
plt.close()
print(f"✓ Сохранён: {OUTPUT_DIR}/fig4_departments.png\n")

# =============================================================================
# БЛОК 5: РАСПРЕДЕЛЕНИЕ ПО КЛАССАМ МКБ-10
# =============================================================================
print("=" * 70)
print("БЛОК 5: РАСПРЕДЕЛЕНИЕ ПО КЛАССАМ МКБ-10")
print("=" * 70)

class_counts = df["icd_class"].value_counts()
print(f"{'Класс':<8} {'Название':<45} {'Кол-во':>8} {'Доля, %':>10}")
print("-" * 75)
for cls, cnt in class_counts.head(15).items():
    pct = cnt / total * 100
    name = ICD_CLASS_NAMES.get(cls, "?")[:45]
    print(f"{cls:<8} {name:<45} {cnt:>8} {pct:>9.2f}%")
print()

# Сохраняем как CSV
class_df = pd.DataFrame({
    "Класс МКБ-10": class_counts.index,
    "Название": [ICD_CLASS_NAMES.get(c, "?") for c in class_counts.index],
    "Количество": class_counts.values,
    "Доля, %": [round(c/total*100, 2) for c in class_counts.values],
})
class_df.to_csv(os.path.join(OUTPUT_DIR, "table_icd_classes.csv"),
                index=False, encoding="utf-8-sig", sep=";")
print(f"✓ Сохранена таблица: {OUTPUT_DIR}/table_icd_classes.csv\n")

# =============================================================================
# БЛОК 6: ТОП-10 КОНКРЕТНЫХ ДИАГНОЗОВ
# =============================================================================
print("=" * 70)
print("БЛОК 6: ТОП-10 КОНКРЕТНЫХ ДИАГНОЗОВ ПО МКБ-10 (Таблица 1)")
print("=" * 70)

top10_codes = df["icd_code"].value_counts().head(10)
rows = []
for i, (code, cnt) in enumerate(top10_codes.items(), 1):
    name = df[df["icd_code"] == code]["icd_name"].iloc[0]
    pct = cnt / total * 100
    rows.append({"№": i, "Код МКБ-10": code, "Наименование диагноза": name,
                 "Количество случаев": cnt, "Доля, %": round(pct, 2)})

top10_df = pd.DataFrame(rows)
print(f"{'№':<3} {'Код':<8} {'Наименование':<55} {'Кол-во':>8} {'Доля':>8}")
print("-" * 86)
for r in rows:
    print(f"{r['№']:<3} {r['Код МКБ-10']:<8} {str(r['Наименование диагноза'])[:55]:<55} "
          f"{r['Количество случаев']:>8} {r['Доля, %']:>7.2f}%")
print()

top10_df.to_csv(os.path.join(OUTPUT_DIR, "table1_top10_icd.csv"),
                index=False, encoding="utf-8-sig", sep=";")
print(f"✓ Сохранена таблица: {OUTPUT_DIR}/table1_top10_icd.csv\n")

# =============================================================================
# БЛОК 7: РАСПРЕДЕЛЕНИЕ ЦЕЛЕВЫХ ПЕРЕМЕННЫХ
# =============================================================================
print("=" * 70)
print("БЛОК 7: РАСПРЕДЕЛЕНИЕ ЦЕЛЕВЫХ ПЕРЕМЕННЫХ")
print("=" * 70)

print(f"\nЗАДАЧА А — PLoS (затяжная госпитализация > 7 дней):")
for v in [0, 1]:
    cnt = (df["plos"] == v).sum()
    pct = cnt / total * 100
    label = "Нормальная (≤ 7 дн.)" if v == 0 else "Затяжная (> 7 дн.)"
    print(f"  {label:<25} {cnt:>5} ({pct:.2f} %)")

print(f"\nЗАДАЧА Б — Сложное лечение (операция или ОРИТ):")
for v in [0, 1]:
    cnt = (df["complex_treatment"] == v).sum()
    pct = cnt / total * 100
    label = "Стандартное" if v == 0 else "Сложное (опер./ОРИТ)"
    print(f"  {label:<25} {cnt:>5} ({pct:.2f} %)")

print(f"\n  В т.ч. с операцией:  {df['operation_flag'].sum()} "
      f"({df['operation_flag'].sum()/total*100:.2f} %)")
print(f"  В т.ч. в ОРИТ:       {df['icu_flag'].sum()} "
      f"({df['icu_flag'].sum()/total*100:.2f} %)")
print()

# =============================================================================
# БЛОК 8: ИТОГОВАЯ СВОДКА
# =============================================================================
print("=" * 70)
print("ИТОГОВАЯ СВОДКА EDA")
print("=" * 70)
print(f"Общее число пациентов:        {total}")
print(f"Уникальных кодов МКБ-10:      {df['icd_code'].nunique()}")
print(f"Уникальных отделений:         {df['department'].nunique()}")
print(f"Возрастной диапазон:          от {df['age_num'].min():.2f} до {df['age_num'].max():.0f} лет")
print(f"Медианный возраст:            {df['age_num'].median():.1f} лет")
print(f"Медианная длит. госпитализации: {df['bed_days'].median():.0f} дней")
print(f"Средняя длит. госпитализации:   {df['bed_days'].mean():.1f} дней")
print()
print("=" * 70)
print(f"✅ ВСЕ РЕЗУЛЬТАТЫ EDA СОХРАНЕНЫ В ПАПКУ: {OUTPUT_DIR}/")
print("=" * 70)
print(f"  fig1_severity.png        — Рисунок 1 (тяжесть состояния)")
print(f"  fig2_age.png             — Рисунок 2 (возраст)")
print(f"  fig3_sex.png             — Рисунок 3 (пол)")
print(f"  fig4_departments.png     — Рисунок 4 (отделения)")
print(f"  table1_top10_icd.csv     — Таблица 1 (топ-10 МКБ-10)")
print(f"  table_icd_classes.csv    — Распределение по классам МКБ-10")
print("=" * 70)
