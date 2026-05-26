# -*- coding: utf-8 -*-
"""
生成式 AI 製造設備異常診斷系統 - Streamlit 正式版

部署說明：
1. 將本檔案放在 GitHub 專案根目錄，建議命名為 app.py。
2. 同一層放置：資料集.xlsx
3. 同一層放置 requirements.txt，內容至少包含：
   streamlit
   pandas
   openpyxl
   plotly
   numpy
   Pillow
4. Streamlit Cloud 的 main file path 請填：app.py

注意：
本檔案是正式 Python / Streamlit 程式，不包含 Colab 專用語法：
- 沒有 !pip install
- 沒有 google.colab
- 沒有 files.upload()
- 沒有 files.download()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from PIL import Image
except Exception:
    Image = None


# =========================================================
# Streamlit 基本設定
# =========================================================

st.set_page_config(
    page_title="生成式 AI 製造設備異常診斷系統",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXCEL_FILENAME = "資料集.xlsx"


# =========================================================
# 通用工具
# =========================================================

def clean_text(value, default: str = "未提供") -> str:
    """將 NaN / None / 空字串轉成友善文字。"""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "nat"}:
        return default
    return text


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def find_file(filename: str) -> Optional[str]:
    """在常見部署路徑中尋找檔案。"""
    candidates = [
        filename,
        f"./{filename}",
        f"/mount/src/{filename}",
        f"/mount/src/genai_final/{filename}",
        f"/mount/src/generative-ai/{filename}",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # 從目前 repo 往下找，避免資料檔被放到子資料夾。
    for root, _, files in os.walk("."):
        if filename in files:
            return os.path.join(root, filename)

    return None


def ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """確保 DataFrame 有指定欄位，缺少則補空值。"""
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df


# =========================================================
# SOP 可讀化輔助
# =========================================================

def classify_sop_stage(step_title, step_content) -> str:
    text = f"{clean_text(step_title, '')} {clean_text(step_content, '')}"

    if any(k in text for k in ["確認是否", "判斷是否", "確認異常", "持續", "觀察", "是否仍", "根節點"]):
        return "異常確認"
    if any(k in text for k in ["安全", "停機", "隔離", "防護", "危險"]):
        return "安全處置"
    if any(k in text for k in ["檢查", "比對", "量測", "查看", "監控", "確認", "判斷"]):
        return "原因排查"
    if any(k in text for k in ["調整", "更換", "清除", "修正", "處理", "復歸", "退爐", "降速"]):
        return "處置修正"
    if any(k in text for k in ["復機", "試軋", "恢復", "結案", "OK", "再確認"]):
        return "復機確認"
    if any(k in text for k in ["紀錄", "回報", "上傳", "改善", "修訂", "回填"]):
        return "紀錄回饋"

    return "原因排查"


def translate_check_type(check_type) -> str:
    value = clean_text(check_type, "").lower()
    if value == "auto":
        return "系統自動判斷"
    if value == "manual":
        return "人工確認"
    if value == "hybrid":
        return "系統輔助＋人工確認"
    return "未標示，需人工確認"


def build_evidence_required(row: pd.Series) -> str:
    evidence: List[str] = []

    if clean_text(row.get("monitor_name"), ""):
        evidence.append("系統參數截圖 / 感測數值")

    image_needed = clean_text(row.get("image_needed"), "").lower()
    if image_needed in {"true", "1", "yes", "y", "需要"}:
        evidence.append("現場照片")

    if clean_text(row.get("safety_note"), ""):
        evidence.append("安全確認紀錄")

    if not evidence:
        evidence.append("人工確認備註")

    return "、".join(evidence)


def build_next_action_hint(row: pd.Series) -> str:
    text = f"{clean_text(row.get('step_title'), '')} {clean_text(row.get('step_content'), '')}"

    if any(k in text for k in ["加熱", "溫度", "在爐", "鋼種"]):
        return "若加熱條件異常，請比對鋼種標準，必要時調整加熱參數或通知製程工程師。"
    if any(k in text for k in ["軸承", "傳動", "馬達", "電流", "Roll", "軋輥", "Guide", "導輪"]):
        return "若設備機構或電流異常，請通知設備工程師檢查磨耗、鬆動、卡滯或破損。"
    if any(k in text for k in ["訊號", "PLC", "sensor", "感測", "通訊", "loss"]):
        return "若訊號異常，請確認 PLC、通訊狀態、感測器連線與資料是否缺漏。"
    if any(k in text for k in ["水量", "噴嘴", "冷卻", "流量", "壓力"]):
        return "若水量、噴嘴或壓力異常，請確認水量設定、阻塞狀態、角度與壓力來源。"

    return "若此步驟判定異常，請依 SOP 進入下一步，並留下處置紀錄。"


# =========================================================
# 資料讀取與資料表建立
# =========================================================

@st.cache_data(show_spinner=False)
def load_excel_tables() -> Tuple[Dict[str, pd.DataFrame], Optional[str]]:
    excel_path = find_file(EXCEL_FILENAME)

    if not excel_path:
        return {}, f"找不到 {EXCEL_FILENAME}。請確認 GitHub 專案根目錄有上傳資料集.xlsx。"

    try:
        xls = pd.ExcelFile(excel_path)
        tables = {sheet: pd.read_excel(excel_path, sheet_name=sheet) for sheet in xls.sheet_names}
        return tables, None
    except Exception as exc:
        return {}, f"讀取 Excel 失敗：{exc}"


@st.cache_data(show_spinner=False)
def build_views(tables: Dict[str, pd.DataFrame]):
    required = [
        "05_cate_detail",
        "06_equipment",
        "07_state",
        "09_monitor_function",
        "11_sop_main",
        "12_sop_step",
        "13_abnormal_event",
        "14_event_step_check",
        "15_sensor_snapshot",
    ]

    missing = [name for name in required if name not in tables]
    if missing:
        return None, None, None, None, f"Excel 缺少必要工作表：{missing}"

    cate_detail = tables["05_cate_detail"].copy()
    equipment = tables["06_equipment"].copy()
    state = tables["07_state"].copy()
    monitor_function = tables["09_monitor_function"].copy()
    sop_main = tables["11_sop_main"].copy()
    sop_step = tables["12_sop_step"].copy()
    abnormal_event = tables["13_abnormal_event"].copy()
    event_step_check = tables["14_event_step_check"].copy()
    sensor_snapshot = tables["15_sensor_snapshot"].copy()

    # ---------------------------------------------------------
    # 1. 異常事件可讀表
    # ---------------------------------------------------------
    event = abnormal_event.copy()

    if {"equipment_id"}.issubset(event.columns) and {"equipment_id"}.issubset(equipment.columns):
        eq_cols = [c for c in ["equipment_id", "equipment_code", "equipment_name"] if c in equipment.columns]
        event = event.merge(equipment[eq_cols].drop_duplicates(), on="equipment_id", how="left")

    if {"state_id"}.issubset(event.columns) and {"state_id"}.issubset(state.columns):
        st_cols = [c for c in ["state_id", "state_name", "default_severity"] if c in state.columns]
        event = event.merge(state[st_cols].drop_duplicates(), on="state_id", how="left")

    if {"cate_detail_id"}.issubset(event.columns) and {"cate_detail_id"}.issubset(cate_detail.columns):
        if "name" in cate_detail.columns:
            event = event.merge(
                cate_detail[["cate_detail_id", "name"]].rename(columns={"name": "cate_detail_name"}).drop_duplicates(),
                on="cate_detail_id",
                how="left",
            )

    if {"sop_id"}.issubset(event.columns) and {"sop_id"}.issubset(sop_main.columns):
        sop_cols = [c for c in ["sop_id", "sop_name", "sop_desc", "owner_role"] if c in sop_main.columns]
        event = event.merge(sop_main[sop_cols].drop_duplicates(), on="sop_id", how="left")

    event = ensure_columns(
        event,
        [
            "event_id",
            "occurred_at",
            "equipment_code",
            "equipment_name",
            "state_name",
            "severity",
            "downtime_min",
            "root_cause_category",
            "cause_summary",
            "action_summary",
            "close_result",
            "sop_id",
            "sop_name",
        ],
    )

    event["downtime_min"] = safe_numeric(event["downtime_min"])
    event["occurred_at"] = pd.to_datetime(event["occurred_at"], errors="coerce")

    for col in ["equipment_code", "equipment_name", "state_name", "severity", "root_cause_category"]:
        event[col] = event[col].astype(str).replace("nan", pd.NA)

    # ---------------------------------------------------------
    # 2. SOP 可讀表
    # ---------------------------------------------------------
    sop = sop_step.copy()

    if {"sop_id"}.issubset(sop.columns) and {"sop_id"}.issubset(sop_main.columns):
        sop_cols = [
            c
            for c in ["sop_id", "equipment_id", "state_id", "sop_name", "sop_desc", "version", "status", "owner_role"]
            if c in sop_main.columns
        ]
        sop = sop.merge(sop_main[sop_cols].drop_duplicates(), on="sop_id", how="left")

    if {"equipment_id"}.issubset(sop.columns) and {"equipment_id"}.issubset(equipment.columns):
        eq_cols = [c for c in ["equipment_id", "equipment_code", "equipment_name", "line_area"] if c in equipment.columns]
        sop = sop.merge(equipment[eq_cols].drop_duplicates(), on="equipment_id", how="left")

    if {"state_id"}.issubset(sop.columns) and {"state_id"}.issubset(state.columns):
        st_cols = [c for c in ["state_id", "state_name", "default_severity"] if c in state.columns]
        sop = sop.merge(state[st_cols].drop_duplicates(), on="state_id", how="left")

    if {"monitor_id"}.issubset(sop.columns) and {"monitor_id"}.issubset(monitor_function.columns):
        mon_cols = [c for c in ["monitor_id", "monitor_name", "subgroup", "data_source_system"] if c in monitor_function.columns]
        sop = sop.merge(monitor_function[mon_cols].drop_duplicates(), on="monitor_id", how="left")

    sop = ensure_columns(
        sop,
        [
            "sop_id",
            "sop_name",
            "sop_desc",
            "equipment_code",
            "equipment_name",
            "state_name",
            "step_id",
            "parent_step_id",
            "sort_order",
            "branch_label",
            "step_title",
            "step_content",
            "check_type",
            "monitor_id",
            "monitor_name",
            "standard_text",
            "image_needed",
            "safety_note",
            "owner_role",
        ],
    )

    sop["sort_order"] = safe_numeric(sop["sort_order"])
    sop["sop_stage"] = sop.apply(lambda row: classify_sop_stage(row.get("step_title"), row.get("step_content")), axis=1)
    sop["check_method_label"] = sop["check_type"].apply(translate_check_type)
    sop["evidence_required"] = sop.apply(build_evidence_required, axis=1)
    sop["next_action_hint"] = sop.apply(build_next_action_hint, axis=1)

    for col in ["equipment_code", "state_name", "sop_id", "step_id"]:
        sop[col] = sop[col].astype(str).replace("nan", pd.NA)

    # ---------------------------------------------------------
    # 3. 感測器快照可讀表
    # ---------------------------------------------------------
    sensor = sensor_snapshot.copy()

    if {"monitor_id"}.issubset(sensor.columns) and {"monitor_id"}.issubset(monitor_function.columns):
        mon_cols = [c for c in ["monitor_id", "monitor_name", "subgroup", "data_source_system"] if c in monitor_function.columns]
        sensor = sensor.merge(monitor_function[mon_cols].drop_duplicates(), on="monitor_id", how="left")

    if {"event_id"}.issubset(sensor.columns) and {"event_id"}.issubset(event.columns):
        event_cols = [c for c in ["event_id", "occurred_at", "equipment_code", "equipment_name", "state_name", "severity", "root_cause_category"] if c in event.columns]
        sensor = sensor.merge(event[event_cols].drop_duplicates(), on="event_id", how="left")

    sensor = ensure_columns(
        sensor,
        [
            "snapshot_id",
            "event_id",
            "occurred_at",
            "captured_at",
            "equipment_code",
            "state_name",
            "monitor_id",
            "monitor_name",
            "parameter_name",
            "actual_value",
            "unit",
            "spec_lower",
            "spec_upper",
            "judgement",
            "source_system",
        ],
    )

    for col in ["actual_value", "spec_lower", "spec_upper"]:
        sensor[col] = safe_numeric(sensor[col])

    sensor["captured_at"] = pd.to_datetime(sensor["captured_at"], errors="coerce")

    # ---------------------------------------------------------
    # 4. SOP 檢查紀錄
    # ---------------------------------------------------------
    check = event_step_check.copy()

    return event, sop, sensor, check, None


# =========================================================
# 查詢與分析函式
# =========================================================

def get_equipment_options(event_df: pd.DataFrame) -> List[str]:
    values = event_df["equipment_code"].dropna().astype(str).unique().tolist()
    return sorted([v for v in values if v.lower() not in {"nan", "none", ""}])


def get_state_options(event_df: pd.DataFrame, equipment_code: Optional[str] = None) -> List[str]:
    df = event_df.copy()
    if equipment_code and equipment_code != "全部":
        df = df[df["equipment_code"].astype(str) == str(equipment_code)]
    values = df["state_name"].dropna().astype(str).unique().tolist()
    return sorted([v for v in values if v.lower() not in {"nan", "none", ""}])


def get_matched_events(event_df: pd.DataFrame, equipment_code: Optional[str] = None, state_name: Optional[str] = None) -> pd.DataFrame:
    df = event_df.copy()

    if equipment_code and equipment_code != "全部":
        df = df[df["equipment_code"].astype(str) == str(equipment_code)]

    if state_name and state_name != "全部":
        df = df[df["state_name"].astype(str) == str(state_name)]

    return df.copy()


def get_sop_by_equipment_state(sop_df: pd.DataFrame, event_df: pd.DataFrame, equipment_code: Optional[str] = None, state_name: Optional[str] = None) -> pd.DataFrame:
    events = get_matched_events(event_df, equipment_code, state_name)

    if events.empty:
        return pd.DataFrame()

    sop_ids = events["sop_id"].dropna().astype(str).unique().tolist()
    result = sop_df[sop_df["sop_id"].astype(str).isin(sop_ids)].copy()

    # 若事件找不到 SOP ID，退而用設備與異常名稱查 SOP。
    if result.empty:
        result = sop_df.copy()
        if equipment_code and equipment_code != "全部":
            result = result[result["equipment_code"].astype(str) == str(equipment_code)]
        if state_name and state_name != "全部":
            result = result[result["state_name"].astype(str) == str(state_name)]

    if not result.empty and "sort_order" in result.columns:
        result = result.sort_values(["sop_id", "sort_order"], na_position="last")

    return result


def get_recommended_cases(event_df: pd.DataFrame, equipment_code: str, state_name: str, top_n: int = 5) -> pd.DataFrame:
    events = get_matched_events(event_df, equipment_code, state_name)

    if events.empty:
        return pd.DataFrame()

    events = events.copy()
    score = pd.Series(0.0, index=events.index)

    score += (events["equipment_code"].astype(str) == str(equipment_code)).astype(int) * 3
    score += (events["state_name"].astype(str) == str(state_name)).astype(int) * 3

    action_text = events["action_summary"].fillna("").astype(str)
    score += action_text.str.len().clip(upper=100) / 100
    score += action_text.str.contains("復機|改善|恢復|完成|排除|正常|更換|調整|校正", regex=True, na=False).astype(int) * 2

    downtime = pd.to_numeric(events["downtime_min"], errors="coerce")
    events["_downtime_for_sort"] = downtime
    events["_recommend_score"] = score.round(2)

    return events.sort_values(["_recommend_score", "_downtime_for_sort"], ascending=[False, True]).head(top_n)


def build_ai_diagnosis_text(event_df: pd.DataFrame, sop_df: pd.DataFrame, sensor_df: pd.DataFrame, equipment_code: str, state_name: str) -> str:
    events = get_matched_events(event_df, equipment_code, state_name)
    sop = get_sop_by_equipment_state(sop_df, event_df, equipment_code, state_name)
    cases = get_recommended_cases(event_df, equipment_code, state_name, top_n=3)

    lines: List[str] = []
    lines.append("【AI 異常診斷建議】")
    lines.append("")
    lines.append("一、異常概況")
    lines.append(f"設備：{equipment_code}")
    lines.append(f"異常狀況：{state_name}")

    if events.empty:
        lines.append("目前找不到相似歷史事件，建議先依現場 SOP 與工程師經驗處理。")
        return "\n".join(lines)

    avg_down = events["downtime_min"].mean()
    lines.append(f"相似歷史事件：{len(events)} 筆")
    if pd.notna(avg_down):
        lines.append(f"平均停機時間：約 {avg_down:.1f} 分鐘")
    lines.append("")

    lines.append("二、可能原因排序")
    cause_counts = events["root_cause_category"].dropna().value_counts().head(3)
    if cause_counts.empty:
        lines.append("目前原因分類不足，需現場補充判斷。")
    else:
        for i, (cause, count) in enumerate(cause_counts.items(), start=1):
            lines.append(f"{i}. {cause}：歷史相似事件中出現 {count} 次")
    lines.append("")

    # Sensor summary
    event_ids = events["event_id"].dropna().astype(str).tolist()
    sensor = sensor_df[sensor_df["event_id"].astype(str).isin(event_ids)].copy()
    abnormal_sensor = sensor[
        sensor["judgement"].astype(str).str.contains("異常|NG|Fail|fail|超出|不正常", regex=True, na=False)
    ]

    lines.append("三、感測器觀察")
    if abnormal_sensor.empty:
        lines.append("目前未整理出明確異常感測器項目，仍需確認資料是否完整。")
    else:
        sensor_counts = abnormal_sensor["monitor_name"].dropna().value_counts().head(5)
        for monitor, count in sensor_counts.items():
            lines.append(f"- {monitor}：相似事件中曾出現 {count} 次異常判斷")
    lines.append("")

    lines.append("四、建議處置順序")
    lines.append("1. 先確認異常是否仍持續發生。")
    lines.append("2. 查詢對應 SOP，依異常確認、原因排查、處置修正、復機確認順序處理。")
    lines.append("3. 檢查 sensor snapshot 是否有異常、缺漏或接近上下限的項目。")
    lines.append("4. 比對相似歷史案例，參考過去有效處置方式。")
    lines.append("5. 完成處置後回填原因、處置與復機結果。")
    lines.append("")

    lines.append("五、對應 SOP")
    if sop.empty:
        lines.append("目前找不到明確對應 SOP。")
    else:
        for sop_id, group in sop.groupby("sop_id"):
            first = group.iloc[0]
            lines.append(f"- {sop_id}｜{clean_text(first.get('sop_name'))}")
    lines.append("")

    lines.append("六、相似歷史案例")
    if cases.empty:
        lines.append("目前沒有足夠相似案例。")
    else:
        for _, row in cases.iterrows():
            lines.append(
                f"- {clean_text(row.get('event_id'))}｜停機 {clean_text(row.get('downtime_min'))} 分鐘｜"
                f"原因：{clean_text(row.get('root_cause_category'))}｜處置：{clean_text(row.get('action_summary'))}"
            )
    lines.append("")
    lines.append("七、提醒")
    lines.append("本頁為 AI 輔助整理，不取代工程師、班長與安全人員的現場判斷。")

    return "\n".join(lines)


def build_predictive_warning(sensor_df: pd.DataFrame, equipment_code: str) -> pd.DataFrame:
    df = sensor_df.copy()

    if equipment_code and equipment_code != "全部":
        df = df[df["equipment_code"].astype(str) == str(equipment_code)]

    if df.empty:
        return pd.DataFrame()

    def risk_score(row: pd.Series) -> int:
        score = 0
        judgement = clean_text(row.get("judgement"), "").lower()
        actual = row.get("actual_value")
        lower = row.get("spec_lower")
        upper = row.get("spec_upper")

        if any(k.lower() in judgement for k in ["異常", "ng", "fail", "超出", "不正常"]):
            score += 70

        if pd.notna(actual) and pd.notna(upper) and upper != 0:
            if actual > upper:
                score += 80
            elif actual > upper * 0.9:
                score += 35

        if pd.notna(actual) and pd.notna(lower) and lower != 0:
            if actual < lower:
                score += 80
            elif actual < lower * 1.1:
                score += 35

        return int(score)

    df["_risk_score"] = df.apply(risk_score, axis=1)

    risk = (
        df.groupby(["equipment_code", "monitor_name", "parameter_name"], dropna=False)
        .agg(
            risk_score=("_risk_score", "max"),
            abnormal_count=("judgement", lambda x: x.astype(str).str.contains("異常|NG|Fail|fail|超出", regex=True, na=False).sum()),
            avg_value=("actual_value", "mean"),
            max_value=("actual_value", "max"),
            min_value=("actual_value", "min"),
            spec_lower=("spec_lower", "first"),
            spec_upper=("spec_upper", "first"),
        )
        .reset_index()
    )

    def level(score: float) -> str:
        if score >= 80:
            return "高"
        if score >= 35:
            return "中"
        return "低"

    risk["risk_level"] = risk["risk_score"].apply(level)
    risk = risk.sort_values(["risk_score", "abnormal_count"], ascending=False)

    return risk


# =========================================================
# 頁面共用元件
# =========================================================

def select_equipment_state(event_df: pd.DataFrame, key_prefix: str = "main") -> Tuple[str, str]:
    equipment_options = get_equipment_options(event_df)

    if not equipment_options:
        st.error("資料中找不到 equipment_code，請確認資料集格式。")
        st.stop()

    equipment_code = st.selectbox("選擇設備", equipment_options, key=f"{key_prefix}_equipment")

    state_options = get_state_options(event_df, equipment_code)

    if not state_options:
        st.warning("此設備目前沒有可選異常狀況。")
        state_options = ["全部"]

    state_name = st.selectbox("選擇異常狀況", state_options, key=f"{key_prefix}_state")

    return equipment_code, state_name


def render_sidebar(event_df: pd.DataFrame) -> str:
    st.sidebar.title("功能頁面")
    st.sidebar.caption("建議依照展示流程，由上往下介紹。")

    groups = {
        "系統與資料概況": [
            "系統總覽",
            "異常分析看板",
        ],
        "異常處理流程": [
            "異常事件查詢",
            "SOP 處理流程查詢",
            "AI 異常診斷建議",
            "現場照片佐證紀錄",
        ],
        "預警與改善": [
            "主動預警 Agent",
            "SOP 改善建議",
        ],
        "知識圖譜展示": [
            "設備原因查詢",
        ],
    }

    if "page" not in st.session_state:
        st.session_state["page"] = "系統總覽"

    for group_name, pages in groups.items():
        st.sidebar.markdown(f"### {group_name}")
        for page in pages:
            label = f"• {page}"
            if st.sidebar.button(label, key=f"nav_{page}", use_container_width=True):
                st.session_state["page"] = page

    st.sidebar.divider()
    st.sidebar.markdown("### 資料概況")
    st.sidebar.metric("異常事件數", len(event_df))
    st.sidebar.metric("涉及設備數", event_df["equipment_code"].nunique())

    return st.session_state["page"]


# =========================================================
# 各頁面
# =========================================================

def page_overview(event_df: pd.DataFrame, sop_df: pd.DataFrame, sensor_df: pd.DataFrame) -> None:
    st.title("🛠️ 生成式 AI 製造設備異常診斷系統")
    st.write("本系統將異常事件、SOP、感測器快照與歷史處置紀錄整合，用於輔助現場異常查詢、AI 診斷、預警與 SOP 改善。")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("異常事件數", len(event_df))
    c2.metric("涉及設備數", event_df["equipment_code"].nunique())
    c3.metric("SOP 步驟數", len(sop_df))
    c4.metric("感測器快照數", len(sensor_df))

    st.markdown("### 展示流程")
    st.info(
        "先看整體異常狀況 → 查單一異常事件 → 查 SOP → 看 AI 診斷 → 補充現場照片佐證 → "
        "進一步做主動預警與 SOP 改善。"
    )

    st.markdown("### 系統特色")
    st.markdown(
        """
- **資料可讀化**：將原始代碼轉為設備名稱、異常名稱、SOP 名稱與處置摘要。
- **SOP 查詢**：依設備與異常狀況快速找到對應 SOP。
- **AI 診斷建議**：根據歷史案例、原因分類與 SOP 輔助產生診斷建議。
- **照片佐證紀錄**：照片不直接判斷原因，只作為現場處置與人工確認佐證。
- **主動預警 Agent**：根據 sensor snapshot 找出可能風險項目。
- **SOP 改善建議**：從停機時間、人工確認點與重複原因找改善重點。
        """
    )


def page_dashboard(event_df: pd.DataFrame) -> None:
    st.title("異常分析看板")
    st.write("用歷史異常事件快速掌握哪些設備、異常狀況與原因最常出現。")

    avg_down = event_df["downtime_min"].mean()
    max_down = event_df["downtime_min"].max()

    c1, c2, c3 = st.columns(3)
    c1.metric("總異常事件", len(event_df))
    c2.metric("平均停機時間", f"{avg_down:.1f} 分鐘" if pd.notna(avg_down) else "無資料")
    c3.metric("最大停機時間", f"{max_down:.0f} 分鐘" if pd.notna(max_down) else "無資料")

    col1, col2 = st.columns(2)

    with col1:
        top_equipment = event_df["equipment_code"].value_counts().head(10).reset_index()
        top_equipment.columns = ["equipment_code", "count"]
        fig = px.bar(top_equipment, x="equipment_code", y="count", title="Top 10 異常設備")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_state = event_df["state_name"].value_counts().head(10).reset_index()
        top_state.columns = ["state_name", "count"]
        fig = px.bar(top_state, x="state_name", y="count", title="Top 10 異常狀況")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        sev = event_df["severity"].value_counts().reset_index()
        sev.columns = ["severity", "count"]
        fig = px.pie(sev, names="severity", values="count", title="嚴重度分布")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        cause = event_df["root_cause_category"].value_counts().head(10).reset_index()
        cause.columns = ["root_cause_category", "count"]
        fig = px.bar(cause, x="root_cause_category", y="count", title="原因分類 Pareto")
        st.plotly_chart(fig, use_container_width=True)

    if "occurred_at" in event_df.columns:
        trend = event_df.dropna(subset=["occurred_at"]).copy()
        if not trend.empty:
            trend["month"] = trend["occurred_at"].dt.to_period("M").astype(str)
            monthly = trend.groupby("month").size().reset_index(name="count")
            fig = px.line(monthly, x="month", y="count", markers=True, title="每月異常件數趨勢")
            st.plotly_chart(fig, use_container_width=True)


def page_event_query(event_df: pd.DataFrame) -> None:
    st.title("異常事件查詢")
    equipment_code, state_name = select_equipment_state(event_df, "event_query")

    filtered = get_matched_events(event_df, equipment_code, state_name)
    st.write(f"找到 {len(filtered)} 筆事件。")

    show_cols = [
        "event_id",
        "occurred_at",
        "equipment_code",
        "state_name",
        "severity",
        "downtime_min",
        "root_cause_category",
        "cause_summary",
        "action_summary",
        "sop_name",
    ]
    show_cols = [c for c in show_cols if c in filtered.columns]
    st.dataframe(filtered[show_cols], use_container_width=True)

    csv = filtered[show_cols].to_csv(index=False, encoding="utf-8-sig")
    st.download_button("下載查詢結果 CSV", csv, file_name="異常事件查詢.csv", mime="text/csv")


def page_sop_query(event_df: pd.DataFrame, sop_df: pd.DataFrame) -> None:
    st.title("SOP 處理流程查詢")
    equipment_code, state_name = select_equipment_state(event_df, "sop_query")

    sop = get_sop_by_equipment_state(sop_df, event_df, equipment_code, state_name)

    if sop.empty:
        st.warning("目前找不到對應 SOP。")
        return

    st.success(f"找到 {sop['sop_id'].nunique()} 份可能對應 SOP。")

    for sop_id, group in sop.groupby("sop_id"):
        first = group.iloc[0]
        with st.expander(f"{sop_id}｜{clean_text(first.get('sop_name'))}", expanded=True):
            st.write(f"**SOP 說明：** {clean_text(first.get('sop_desc'))}")
            st.write(f"**負責角色：** {clean_text(first.get('owner_role'))}")

            for stage, stage_df in group.groupby("sop_stage", sort=False):
                st.markdown(f"#### {stage}")
                stage_df = stage_df.sort_values("sort_order", na_position="last")

                for _, step in stage_df.iterrows():
                    if clean_text(step.get("step_title"), "") == "根節點":
                        continue

                    with st.container(border=True):
                        order = clean_text(step.get("sort_order"), "")
                        title = clean_text(step.get("step_title"))
                        st.markdown(f"**{order}. {title}**")
                        st.write(clean_text(step.get("step_content")))

                        c1, c2 = st.columns(2)
                        c1.write(f"判斷方式：{clean_text(step.get('check_method_label'))}")
                        c1.write(f"監控項目：{clean_text(step.get('monitor_name'), '無，需人工確認')}")
                        c2.write(f"判斷標準：{clean_text(step.get('standard_text'))}")
                        c2.write(f"需留存佐證：{clean_text(step.get('evidence_required'))}")
                        st.caption(f"異常時建議：{clean_text(step.get('next_action_hint'))}")


def page_ai_diagnosis(event_df: pd.DataFrame, sop_df: pd.DataFrame, sensor_df: pd.DataFrame) -> None:
    st.title("AI 異常診斷建議")
    st.write("系統會依設備、異常狀況、歷史案例、SOP 與 sensor snapshot 產生輔助診斷。")

    equipment_code, state_name = select_equipment_state(event_df, "ai_diag")
    report = build_ai_diagnosis_text(event_df, sop_df, sensor_df, equipment_code, state_name)

    st.text_area("診斷報告", report, height=540)

    st.download_button(
        "下載 AI 診斷 TXT",
        report,
        file_name=f"{equipment_code}_{state_name}_AI診斷建議.txt",
        mime="text/plain",
    )

    st.markdown("### 相似歷史事件推薦")
    cases = get_recommended_cases(event_df, equipment_code, state_name, top_n=5)
    if cases.empty:
        st.info("目前沒有相似案例。")
    else:
        cols = [
            "event_id",
            "occurred_at",
            "equipment_code",
            "state_name",
            "downtime_min",
            "root_cause_category",
            "cause_summary",
            "action_summary",
        ]
        cols = [c for c in cols if c in cases.columns]
        st.dataframe(cases[cols], use_container_width=True)


def analyze_photo(uploaded_file):
    if uploaded_file is None:
        return None

    if Image is None:
        return {
            "filename": uploaded_file.name,
            "quality": "無法分析",
            "notes": ["目前環境未安裝 Pillow，無法分析照片品質。"],
        }

    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert("RGB")
        arr = np.asarray(img).astype("float32")
        gray = arr.mean(axis=2)

        width, height = img.size
        brightness = float(gray.mean())
        contrast = float(gray.std())
        gy, gx = np.gradient(gray)
        sharpness = float((gx ** 2 + gy ** 2).mean())

        notes: List[str] = []
        score = 0

        if width >= 600 and height >= 400:
            score += 1
            notes.append("解析度足夠，可作為現場佐證。")
        else:
            notes.append("解析度偏低，建議補拍近照或提高解析度。")

        if 55 <= brightness <= 210:
            score += 1
            notes.append("亮度大致正常。")
        elif brightness < 55:
            notes.append("照片偏暗，可能看不清楚異常部位。")
        else:
            notes.append("照片偏亮，可能過曝。")

        if contrast >= 22:
            score += 1
            notes.append("對比尚可。")
        else:
            notes.append("對比偏低，建議補拍光線較清楚的照片。")

        if sharpness >= 18:
            score += 1
            notes.append("清晰度尚可。")
        else:
            notes.append("照片可能模糊，建議補拍。")

        quality = "高" if score >= 4 else ("中" if score >= 2 else "低")

        uploaded_file.seek(0)
        return {
            "filename": uploaded_file.name,
            "width": width,
            "height": height,
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "quality": quality,
            "notes": notes,
        }

    except Exception as exc:
        return {
            "filename": uploaded_file.name,
            "quality": "無法分析",
            "notes": [f"圖片讀取失敗：{exc}"],
        }


def page_photo_record(event_df: pd.DataFrame) -> None:
    st.title("現場照片佐證紀錄")
    st.write("此頁不直接判斷照片中的異常原因，只用於上傳照片、檢查照片品質、整理現場補充紀錄。")

    equipment_code, state_name = select_equipment_state(event_df, "photo")

    purpose = st.selectbox(
        "照片用途",
        ["異常部位近照", "設備全景", "處置前紀錄", "處置後紀錄", "安全確認", "其他"],
    )

    uploaded_file = st.file_uploader("上傳現場照片", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="已上傳照片", use_container_width=True)

    note = st.text_area("現場補充說明", placeholder="例如：異常位置、是否有異音、是否漏油、是否已停機、現場處置方式等。")
    analysis = analyze_photo(uploaded_file)

    record_lines: List[str] = []
    record_lines.append("【現場照片佐證紀錄】")
    record_lines.append(f"設備：{equipment_code}")
    record_lines.append(f"異常狀況：{state_name}")
    record_lines.append(f"照片用途：{purpose}")

    if analysis is None:
        record_lines.append("照片狀態：尚未上傳")
    else:
        record_lines.append(f"照片檔名：{analysis.get('filename')}")
        record_lines.append(f"照片品質等級：{analysis.get('quality')}")
        if analysis.get("width"):
            record_lines.append(f"尺寸：{analysis.get('width')} x {analysis.get('height')}")
        record_lines.append("品質提醒：")
        for item in analysis.get("notes", []):
            record_lines.append(f"- {item}")

    record_lines.append("")
    record_lines.append("現場補充說明：")
    record_lines.append(note if note else "未填寫")
    record_lines.append("")
    record_lines.append("提醒：照片僅作為人工確認與後續追蹤佐證，不直接取代工程師判斷。")

    record_text = "\n".join(record_lines)
    st.text_area("照片佐證紀錄", record_text, height=360)

    st.download_button(
        "下載照片佐證紀錄 TXT",
        record_text,
        file_name=f"{equipment_code}_{state_name}_照片佐證紀錄.txt",
        mime="text/plain",
    )


def page_warning(event_df: pd.DataFrame, sensor_df: pd.DataFrame) -> None:
    st.title("主動預警 Agent")
    st.write("本頁用 sensor snapshot 做簡化風險排序，找出可能需要提前巡檢的設備或監控項目。")

    equipment_options = ["全部"] + get_equipment_options(event_df)
    equipment_code = st.selectbox("選擇設備", equipment_options, key="warning_equipment")
    risk = build_predictive_warning(sensor_df, equipment_code)

    if risk.empty:
        st.warning("目前沒有可用的 sensor snapshot 資料。")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("高風險項目", int((risk["risk_level"] == "高").sum()))
    c2.metric("中風險項目", int((risk["risk_level"] == "中").sum()))
    c3.metric("低風險項目", int((risk["risk_level"] == "低").sum()))

    st.dataframe(risk.head(30), use_container_width=True)

    top = risk.iloc[0]
    st.info(
        f"目前最高風險項目為：{clean_text(top.get('equipment_code'))} / "
        f"{clean_text(top.get('monitor_name'))} / {clean_text(top.get('parameter_name'))}。"
        f"風險等級為 {clean_text(top.get('risk_level'))}。"
        "建議提前安排巡檢，確認設備狀態、感測器可信度與近期保養紀錄。"
    )


def page_sop_improvement(event_df: pd.DataFrame, sop_df: pd.DataFrame) -> None:
    st.title("SOP 改善建議")
    st.write("從歷史異常事件與 SOP 步驟資料中，找出最值得改善的設備、異常與流程節點。")

    tab1, tab2, tab3, tab4 = st.tabs(["停機時間優先改善", "人工確認節點", "重複發生原因", "報告可用結論"])

    with tab1:
        group = (
            event_df.groupby(["equipment_code", "state_name"], dropna=False)
            .agg(
                event_count=("event_id", "count"),
                avg_downtime_min=("downtime_min", "mean"),
                max_downtime_min=("downtime_min", "max"),
            )
            .reset_index()
            .sort_values(["avg_downtime_min", "event_count"], ascending=[False, False])
        )
        st.dataframe(group.head(20), use_container_width=True)

    with tab2:
        manual = sop_df.copy()
        manual = manual[
            manual["check_method_label"].astype(str).str.contains("人工", na=False)
            | manual["evidence_required"].astype(str).str.contains("照片|人工|備註", na=False)
        ]
        cols = [
            "equipment_code",
            "state_name",
            "sop_id",
            "sop_name",
            "step_title",
            "check_method_label",
            "evidence_required",
        ]
        cols = [c for c in cols if c in manual.columns]
        st.dataframe(manual[cols].head(50), use_container_width=True)

    with tab3:
        repeated = (
            event_df.groupby(["equipment_code", "state_name", "root_cause_category"], dropna=False)
            .size()
            .reset_index(name="event_count")
            .sort_values("event_count", ascending=False)
        )
        st.dataframe(repeated.head(30), use_container_width=True)

    with tab4:
        st.markdown(
            """
本系統不只是查詢 SOP，而是利用歷史異常事件與 SOP 步驟資料，找出最值得改善的設備、異常與流程節點。

例如：
- 平均停機時間較高的設備，可作為優先改善對象。
- 最常需要人工確認的 SOP 節點，代表目前尚無法完全自動化判斷。
- 重複發生的原因分類，可作為後續修訂 SOP、補充監控項目或調整保養策略的依據。
            """
        )


def page_equipment_reason(event_df: pd.DataFrame) -> None:
    st.title("設備原因查詢")
    st.write("選擇設備後，直接查看該設備常見異常、原因分類、歷史原因摘要與處置方式。")

    equipment_code = st.selectbox("選擇設備", get_equipment_options(event_df), key="reason_equipment")
    df = get_matched_events(event_df, equipment_code, None)

    if df.empty:
        st.warning("此設備目前沒有歷史事件資料。")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("事件數", len(df))
    c2.metric("異常類型數", df["state_name"].nunique())
    avg_down = df["downtime_min"].mean()
    c3.metric("平均停機時間", f"{avg_down:.1f} 分鐘" if pd.notna(avg_down) else "無資料")

    tab1, tab2, tab3, tab4 = st.tabs(["常見異常", "原因分類", "歷史原因摘要", "處置方式"])

    with tab1:
        table = df["state_name"].value_counts().reset_index()
        table.columns = ["state_name", "count"]
        st.dataframe(table, use_container_width=True)

    with tab2:
        table = df["root_cause_category"].value_counts().reset_index()
        table.columns = ["root_cause_category", "count"]
        st.dataframe(table, use_container_width=True)

    with tab3:
        cols = ["event_id", "occurred_at", "state_name", "downtime_min", "cause_summary"]
        st.dataframe(df[[c for c in cols if c in df.columns]].head(50), use_container_width=True)

    with tab4:
        cols = ["event_id", "state_name", "root_cause_category", "action_summary", "close_result"]
        st.dataframe(df[[c for c in cols if c in df.columns]].head(50), use_container_width=True)


# =========================================================
# 主程式
# =========================================================

def main() -> None:
    tables, load_error = load_excel_tables()

    if load_error:
        st.error(load_error)
        st.stop()

    event_df, sop_df, sensor_df, check_df, build_error = build_views(tables)

    if build_error:
        st.error(build_error)
        st.stop()

    page = render_sidebar(event_df)

    if page == "系統總覽":
        page_overview(event_df, sop_df, sensor_df)
    elif page == "異常分析看板":
        page_dashboard(event_df)
    elif page == "異常事件查詢":
        page_event_query(event_df)
    elif page == "SOP 處理流程查詢":
        page_sop_query(event_df, sop_df)
    elif page == "AI 異常診斷建議":
        page_ai_diagnosis(event_df, sop_df, sensor_df)
    elif page == "現場照片佐證紀錄":
        page_photo_record(event_df)
    elif page == "主動預警 Agent":
        page_warning(event_df, sensor_df)
    elif page == "SOP 改善建議":
        page_sop_improvement(event_df, sop_df)
    elif page == "設備原因查詢":
        page_equipment_reason(event_df)
    else:
        page_overview(event_df, sop_df, sensor_df)


if __name__ == "__main__":
    main()
