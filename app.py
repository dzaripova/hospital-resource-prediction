"""
╔══════════════════════════════════════════════════════════════════╗
║  ПРИЛОЖЕНИЕ: Оценка ресурсных потребностей стационара            ║
║  Streamlit web-app — дипломная работа Зариповой Д.И., 2026       ║
╚══════════════════════════════════════════════════════════════════╝

КАК ЗАПУСТИТЬ:

  1. Откройте Anaconda Prompt (НЕ обычный cmd!)
  2. Перейдите в папку с приложением:
     cd path/to/project
  3. Запустите:
     streamlit run app.py
  4. Браузер откроется сам на http://localhost:8501
     Если не открылся — откройте этот адрес вручную
  5. Чтобы остановить — Ctrl+C в окне Anaconda Prompt

ЧТО ДОЛЖНО БЫТЬ В ОДНОЙ ПАПКЕ С app.py:
  ✓ model_plos.cbm        (модель прогноза затяжной госпитализации)
  ✓ model_complex.cbm     (модель прогноза операции/реанимации)
  ✓ model_meta.json       (метаданные моделей)

ВАЖНО:
  - Не закрывайте окно Anaconda Prompt пока работает приложение!
  - Файлы моделей создаются автоматически после запуска diploma_binary.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from catboost import CatBoostClassifier


# ═══════════════════════════════════════════════════════════════════
# НАСТРОЙКИ СТРАНИЦЫ
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Оценка ресурсных потребностей стационара",
    page_icon="🏥",
    layout="wide",
)


# ═══════════════════════════════════════════════════════════════════
# СТИЛИ
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.stApp { background: #F4F6FA; }

.main-header {
    background: linear-gradient(120deg, #1C3557 0%, #2B5DA8 100%);
    border-radius: 14px;
    padding: 26px 34px;
    margin-bottom: 24px;
    color: white;
    box-shadow: 0 4px 16px rgba(28, 53, 87, 0.15);
}
.main-header h1 {
    color: white;
    font-size: 1.6rem;
    margin: 0;
    font-weight: 700;
}
.main-header p {
    color: #AAC4E8;
    margin: 6px 0 0 0;
    font-size: 0.88rem;
}

.section-header {
    color: #1C3557;
    font-weight: 700;
    border-left: 4px solid #2B5DA8;
    padding: 4px 0 4px 12px;
    margin: 24px 0 14px 0;
    font-size: 1.05rem;
}

.result-card {
    border-radius: 12px;
    padding: 22px;
    color: white;
    text-align: center;
    margin: 8px 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
}
.result-danger { background: linear-gradient(135deg, #C0272D, #9B1B1F); }
.result-success { background: linear-gradient(135deg, #1E6B40, #2E7D52); }

.result-label { font-size: 1.5rem; font-weight: 700; margin: 0; }
.result-sub { font-size: 0.85rem; opacity: 0.88; margin-top: 6px; }

.prob-display {
    text-align: center;
    margin: 12px 0 6px 0;
}
.prob-big { font-size: 2.4rem; font-weight: 700; margin: 0; }
.prob-big.danger { color: #C0272D; }
.prob-big.success { color: #1E6B40; }
.prob-label { font-size: 0.8rem; color: #6B7A8D; margin-top: 2px; }

.rec-box, .warning-box {
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.9rem;
    line-height: 1.55;
}
.rec-box {
    background: #EDFAF3;
    border-left: 4px solid #1E6B40;
    color: #145030;
}
.warning-box {
    background: #FFF0F0;
    border-left: 4px solid #C0272D;
    color: #7A1518;
}
.rec-box b, .warning-box b {
    display: block;
    margin-bottom: 4px;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    opacity: 0.75;
}

.patient-pill {
    display: inline-block;
    background: white;
    border: 1px solid #D8E1EE;
    border-radius: 8px;
    padding: 8px 14px;
    margin: 3px;
    font-size: 0.85rem;
    color: #2C3E50;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.patient-pill b { color: #1C3557; }

table.summary-table {
    width: 100%;
    border-collapse: collapse;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    font-size: 0.92rem;
}
table.summary-table th {
    background: #1C3557;
    color: white;
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
table.summary-table td {
    padding: 12px 16px;
    background: white;
    border-bottom: 1px solid #E8EDF4;
    color: #2C3E50;
}
table.summary-table tr:last-child td { border-bottom: none; }

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge.danger { background: #FEE2E2; color: #991B1B; }
.badge.success { background: #D1FAE5; color: #065F46; }

div[data-testid="stMetric"] {
    background: white;
    border-radius: 10px;
    padding: 12px 16px;
    border-left: 3px solid #2B5DA8;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# ЗАГРУЗКА МОДЕЛЕЙ
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    """Загружает обе модели CatBoost и метаданные из папки с app.py"""
    model_dir = os.path.dirname(os.path.abspath(__file__))

    m_plos = CatBoostClassifier()
    m_plos.load_model(os.path.join(model_dir, "model_plos.cbm"))

    m_complex = CatBoostClassifier()
    m_complex.load_model(os.path.join(model_dir, "model_complex.cbm"))

    with open(os.path.join(model_dir, "model_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    return m_plos, m_complex, meta


try:
    model_plos, model_complex, meta = load_models()
    models_ok = True
    load_err = ""
except Exception as e:
    models_ok = False
    load_err = str(e)


# ═══════════════════════════════════════════════════════════════════
# ЗАГОЛОВОК
# ═══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="main-header">
    <h1>🏥 Оценка ресурсных потребностей стационара</h1>
    <p>Система поддержки принятия решений на основе CatBoost &nbsp;|&nbsp;
       Зарипова Д.И., КФУ, 2026 &nbsp;|&nbsp;
       ВКР: 12.03.04 Биотехнические системы и технологии</p>
</div>
""", unsafe_allow_html=True)


if not models_ok:
    st.error(f"""
    ⚠️ **Модели не найдены.**

    В папке с `app.py` должны лежать файлы:
    - `model_plos.cbm`
    - `model_complex.cbm`
    - `model_meta.json`

    Они создаются автоматически после запуска `diploma_binary.py`.

    Ошибка: `{load_err}`
    """)
    st.stop()


# ═══════════════════════════════════════════════════════════════════
# БОКОВАЯ ПАНЕЛЬ
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📊 О системе")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("AUC PLoS", meta.get("auc_plos", "—"))
    with col2:
        st.metric("AUC Сложное", meta.get("auc_complex", "—"))

    st.markdown("---")
    st.markdown(f"""
    **Задача А — PLoS**
    Прогноз затяжной госпитализации
    (порог >{meta.get('plos_threshold', 7)} дней,
    по Zeleke et al., 2023)

    **Задача Б — Сложное лечение**
    Прогноз операции или реанимации
    """)

    st.markdown("---")
    st.markdown(f"""
    **Признаков:** {len(meta.get('all_features', []))}
    **Обучено на:** 5 868 пациентов
    **Алгоритм:** CatBoost + ранняя остановка
    """)

    st.markdown("---")
    st.caption("Сравнение AUC-ROC с литературой:")
    st.markdown(f"""
    - Zeleke 2023: `0.820`
    - Jaotombo 2022: `0.810`
    - **Эта работа**: `{meta.get('auc_plos', '—')}`
    """)


# ═══════════════════════════════════════════════════════════════════
# ФОРМА ВВОДА
# ═══════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Данные пациента при поступлении</div>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Демографические данные**")
    age_input = st.text_input(
        "Возраст",
        value="8",
        placeholder="8  или  6 мес",
        help="Возраст в годах. Для детей до 1 года — в месяцах: «6 мес»"
    )
    sex = st.selectbox("Пол", ["Мужской", "Женский"])
    admission_state = st.selectbox(
        "Тяжесть состояния при поступлении",
        ["Удовлетворительное", "Средней тяжести", "Тяжелое", "Крайне тяжелое"]
    )

with col2:
    st.markdown("**🩺 Клинические данные**")
    department = st.text_input(
        "Отделение",
        value="Аллергологическое отделение",
        help="Полное название отделения"
    )
    icd_code = st.text_input(
        "Код МКБ-10",
        value="J45.0",
        placeholder="Например: J45.0, C91.0, Q35.1",
        help="Основной диагноз по МКБ-10"
    ).strip().upper()

with col3:
    st.markdown("**📅 Дата и время поступления**")
    admission_date = st.date_input("Дата поступления")
    admission_hour = st.slider(
        "Час поступления", 0, 23, 10, format="%d:00"
    )

    weekday = admission_date.weekday()
    is_night = admission_hour >= 22 or admission_hour < 6
    is_weekend = weekday >= 5

    if is_weekend:
        st.info(f"🟡 Выходной день, {admission_hour}:00")
    elif is_night:
        st.info(f"🌙 Ночное поступление, {admission_hour}:00")
    else:
        st.info(f"🟢 Будний день, {admission_hour}:00")


st.markdown("---")
predict_btn = st.button(
    "🔍 Рассчитать ресурсные потребности",
    type="primary",
    use_container_width=True
)


# ═══════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════
def parse_age(s):
    """'8' → 8.0; '6 мес' → 0.5"""
    s = str(s).strip().lower()
    if "мес" in s:
        try:
            return float(s.replace("мес", "").strip()) / 12
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def age_to_group(age):
    """Возраст в годах → группа 0–5"""
    if age < 1: return 0
    elif age < 3: return 1
    elif age < 7: return 2
    elif age < 12: return 3
    elif age < 18: return 4
    else: return 5


def build_patient_df(age_raw, sex_str, dept, icd, state_str, date_obj, hour):
    """Собирает DataFrame с признаками пациента в формате модели"""
    age = parse_age(age_raw)
    if age is None:
        return None, "Неверный формат возраста. Введите число (например: 8) или '6 мес'."

    icd = str(icd).strip().upper()
    icd_chapter = icd[:3] if len(icd) >= 3 else "UNK"
    icd_letter = icd[0] if (icd and icd[0].isalpha()) else "U"
    icd_system = meta.get("icd_group_map", {}).get(icd_letter, "другое")

    state_map = meta.get("state_map", {})
    state_num = state_map.get(state_str, 1)

    wd = date_obj.weekday()
    month = date_obj.month
    quarter = (month - 1) // 3 + 1

    row = {
        "age_num": age,
        "age_group": float(age_to_group(age)),
        "sex_num": 1 if sex_str == "Мужской" else 0,
        "admission_state_num": float(state_num),
        "admission_month": float(month),
        "admission_weekday": float(wd),
        "admission_hour": float(hour),
        "weekend_admission": float(wd >= 5),
        "night_admission": float(hour >= 22 or hour < 6),
        "quarter": float(quarter),
        "department_str": str(dept),
        "icd_chapter_str": icd_chapter,
        "icd_letter_str": icd_letter,
        "icd_system_str": icd_system,
    }
    return pd.DataFrame([row]), None


# ═══════════════════════════════════════════════════════════════════
# ОБРАБОТКА КЛИКА «РАССЧИТАТЬ»
# ═══════════════════════════════════════════════════════════════════
if predict_btn:
    pat_df, err = build_patient_df(
        age_input, sex, department, icd_code,
        admission_state, admission_date, admission_hour
    )

    if err:
        st.error(f"❌ Ошибка ввода: {err}")
        st.stop()

    # ── Предсказания ────────────────────────────────────────────────
    pred_plos = int(model_plos.predict(pat_df).flatten()[0])
    proba_plos = float(model_plos.predict_proba(pat_df)[0][1])
    pred_complex = int(model_complex.predict(pat_df).flatten()[0])
    proba_complex = float(model_complex.predict_proba(pat_df)[0][1])

    # ── Карточка пациента ──────────────────────────────────────────
    st.markdown('<div class="section-header">Результаты оценки</div>',
                unsafe_allow_html=True)

    icd_sys_display = meta.get("icd_group_map", {}).get(
        icd_code[0].upper() if icd_code else "U", "—"
    )

    st.markdown(f"""
    <div style="margin: 14px 0;">
        <span class="patient-pill">👤 <b>{age_input}</b> · {sex}</span>
        <span class="patient-pill">🏥 <b>{department[:30]}{"..." if len(department)>30 else ""}</b></span>
        <span class="patient-pill">🩺 <b>{icd_code}</b> ({icd_sys_display})</span>
        <span class="patient-pill">⚕️ {admission_state}</span>
        <span class="patient-pill">📅 {'Выходной' if admission_date.weekday()>=5 else 'Будний'}, {admission_hour}:00</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Две карточки результатов ──────────────────────────────────
    res_col1, res_col2 = st.columns(2)

    # Задача А: PLoS
    with res_col1:
        st.markdown("#### Задача А: Длительность госпитализации")

        if pred_plos == 1:
            st.markdown(f"""
            <div class="result-card result-danger">
                <p class="result-label">⚠️ ЗАТЯЖНАЯ</p>
                <p class="result-sub">Прогноз: более {meta.get('plos_threshold', 7)} дней</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-success">
                <p class="result-label">✅ НОРМАЛЬНАЯ</p>
                <p class="result-sub">Прогноз: ≤{meta.get('plos_threshold', 7)} дней</p>
            </div>
            """, unsafe_allow_html=True)

        cls_a = "danger" if pred_plos == 1 else "success"
        st.markdown(f"""
        <div class="prob-display">
            <p class="prob-big {cls_a}">{proba_plos*100:.1f}%</p>
            <p class="prob-label">вероятность затяжной госпитализации</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(proba_plos)

        if pred_plos == 1:
            st.markdown("""
            <div class="warning-box">
                <b>Рекомендация</b>
                Зарезервировать койку на длительный срок.
                Уведомить профильных специалистов о потребности в ресурсах.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="rec-box">
                <b>Рекомендация</b>
                Стандартное планирование коечного фонда.
                Ожидаемое пребывание в пределах нормы.
            </div>
            """, unsafe_allow_html=True)

    # Задача Б: Сложное лечение
    with res_col2:
        st.markdown("#### Задача Б: Операция / Реанимация")

        if pred_complex == 1:
            st.markdown("""
            <div class="result-card result-danger">
                <p class="result-label">⚠️ ВЫСОКАЯ НАГРУЗКА</p>
                <p class="result-sub">Вероятна операция или реанимация</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card result-success">
                <p class="result-label">✅ СТАНДАРТНОЕ</p>
                <p class="result-sub">Операция/ОРИТ маловероятна</p>
            </div>
            """, unsafe_allow_html=True)

        cls_b = "danger" if pred_complex == 1 else "success"
        st.markdown(f"""
        <div class="prob-display">
            <p class="prob-big {cls_b}">{proba_complex*100:.1f}%</p>
            <p class="prob-label">вероятность операции или реанимации</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(proba_complex)

        if pred_complex == 1:
            st.markdown("""
            <div class="warning-box">
                <b>Рекомендация</b>
                Уведомить хирургическую бригаду и ОРИТ.
                Зарезервировать операционный ресурс и место в реанимации.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="rec-box">
                <b>Рекомендация</b>
                Стандартный протокол лечения.
                Оперативное вмешательство маловероятно.
            </div>
            """, unsafe_allow_html=True)

    # ── Сводная таблица (HTML — БЕЗ pyarrow!) ─────────────────────
    st.markdown('<div class="section-header">Сводка по пациенту</div>',
                unsafe_allow_html=True)

    def make_badge(pred, label_yes, label_no):
        cls = "danger" if pred == 1 else "success"
        text = label_yes if pred == 1 else label_no
        return f'<span class="badge {cls}">{text}</span>'

    conf_a = max(proba_plos, 1 - proba_plos) * 100
    conf_b = max(proba_complex, 1 - proba_complex) * 100

    summary_html = f"""
    <table class="summary-table">
        <thead>
            <tr>
                <th>Показатель</th>
                <th>Вероятность</th>
                <th>Прогноз</th>
                <th>Уверенность</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Затяжная госпитализация (&gt;{meta.get('plos_threshold', 7)} дн.)</td>
                <td><b>{proba_plos*100:.1f}%</b></td>
                <td>{make_badge(pred_plos, "⚠ Затяжная", "✓ Нормальная")}</td>
                <td>{conf_a:.1f}%</td>
            </tr>
            <tr>
                <td>Операция или реанимация</td>
                <td><b>{proba_complex*100:.1f}%</b></td>
                <td>{make_badge(pred_complex, "⚠ Высокая нагрузка", "✓ Стандартное")}</td>
                <td>{conf_b:.1f}%</td>
            </tr>
        </tbody>
    </table>
    """
    st.markdown(summary_html, unsafe_allow_html=True)

    # ── Общая оценка ──────────────────────────────────────────────
    st.markdown("---")
    if pred_plos == 1 and pred_complex == 1:
        st.error("🔴 **Высокий приоритет планирования.** Оба показателя указывают на ресурсоёмкое лечение. Рекомендуется приоритетное резервирование коечного фонда, операционной и ОРИТ.")
    elif pred_plos == 0 and pred_complex == 0:
        st.success("🟢 **Стандартная госпитализация.** Оба показателя в норме. Специального резервирования ресурсов не требуется.")
    else:
        st.warning("🟡 **Смешанный прогноз.** Один из показателей повышен. Рекомендуется наблюдение и частичное резервирование ресурсов.")


# ═══════════════════════════════════════════════════════════════════
# ФУТЕР
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("ℹ️ О методологии и ограничениях"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        **Алгоритм:** CatBoost (Яндекс, 2017)
        - Нативная работа с категориальными признаками
        - Упорядоченное целевое кодирование — без утечки данных
        - Ранняя остановка — защита от переобучения

        **Целевая переменная Задачи А:**
        Prolonged Length of Stay (PLoS), стандарт в литературе.
        Порог {meta.get('plos_threshold', 7)} дней по Zeleke et al., 2023.

        **Признаки:** только данные при поступлении —
        без лабораторных показателей и данных о ходе лечения.
        """)
    with c2:
        st.markdown(f"""
        **Качество моделей (AUC-ROC):**

        | Задача | AUC |
        |--------|-----|
        | PLoS | {meta.get('auc_plos', '—')} |
        | Сложное лечение | {meta.get('auc_complex', '—')} |

        **⚠️ Ограничения:**
        - Поддержка административных решений,
          не замена клинического суждения врача
        - Только административные данные
        - Обучена на данных одного стационара
        """)

st.markdown("""
<p style="text-align:center;color:#9BA8B5;font-size:0.78rem;margin-top:16px;">
Зарипова Д.И. · ВКР 12.03.04 Биотехнические системы и технологии · КФУ, Институт физики · 2026
</p>
""", unsafe_allow_html=True)
