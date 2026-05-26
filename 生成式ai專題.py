import os
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import html
import math
from datetime import datetime

# =========================================================
# Streamlit 基本設定
# =========================================================

st.set_page_config(
    page_title="生成式 AI 製造設備異常診斷系統",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 欄位設定
# =========================================================

EVENT_COLS = [
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
    "sop_id",
    "sop_name"
]

SOP_COLS = [
    "sop_id",
    "sop_name",
    "sop_desc",
    "equipment_code",
    "equipment_name",
    "state_name",
    "step_id",
    "parent_step_id",
    "owner_role",
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
    "sop_stage",
    "check_method_label",
    "evidence_required",
    "next_action_hint"
]

SENSOR_COLS = [
    "snapshot_id",
    "event_id",
    "captured_at",
    "monitor_id",
    "monitor_name",
    "parameter_name",
    "actual_value",
    "unit",
    "spec_lower",
    "spec_upper",
    "judgement",
    "source_system",
    "source_note"
]

HUMAN_ASSIST_COLS = [
    "source_table",
    "equipment_code",
    "state_keyword",
    "assist_type",
    "check_item",
    "human_action",
    "related_monitor_id",
    "evidence_required",
    "escalation_rule",
    "source_note"
]


# =========================================================
# 安全讀檔函式
# =========================================================

def safe_read_csv(path, usecols=None):
    """
    安全讀取 CSV。
    如果指定欄位不存在，會自動忽略不存在欄位。
    """
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        if usecols is None:
            return pd.read_csv(path)

        return pd.read_csv(
            path,
            usecols=lambda col: col in usecols
        )
    except Exception:
        return pd.read_csv(path)


# =========================================================
# 讀取資料
# =========================================================

@st.cache_data(show_spinner=False)
def load_data():
    """
    讀取 Streamlit 需要的資料。

    重要：
    - Colab 版本會先產生 output/*.csv，所以原本 app 只讀 output。
    - Streamlit Cloud / GitHub 上如果沒有 output 資料夾，就會變成 0 筆，後面也會 KeyError。
    - 這版會先讀 output；若 output 不存在或主要資料為空，會自動從「Dataset/資料集.xlsx」重新建立資料表。
    """

    def _find_file(filename):
        candidates = [
            filename,
            f"./{filename}",
            f"/mount/src/{filename}",
            f"/mount/src/genai_final/{filename}",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        for root, _, files in os.walk("."):
            if filename in files:
                return os.path.join(root, filename)
        return None

    def _ensure_columns(df, cols):
        for col in cols:
            if col not in df.columns:
                df[col] = pd.NA
        return df

    def _classify_stage(title, content):
        text = f"{title} {content}"
        if any(k in text for k in ["根節點", "確認是否", "判斷是否", "確認異常", "持續", "觀察", "是否仍"]):
            return "1. 異常確認"
        if any(k in text for k in ["安全", "停機", "隔離", "防護", "通知", "危險"]):
            return "2. 安全處置"
        if any(k in text for k in ["檢查", "確認", "比對", "量測", "查看", "判斷", "監控"]):
            return "3. 原因排查"
        if any(k in text for k in ["調整", "更換", "清除", "修正", "處理", "復歸", "退爐", "降速"]):
            return "4. 處置修正"
        if any(k in text for k in ["復機", "試軋", "恢復", "結案", "再確認", "OK"]):
            return "5. 復機確認"
        if any(k in text for k in ["紀錄", "回報", "上傳", "改善", "修訂", "回填"]):
            return "6. 紀錄回饋"
        return "3. 原因排查"

    def _translate_check_type(value):
        value = str(value).lower().strip()
        if value == "auto":
            return "系統自動判斷"
        if value == "manual":
            return "人工確認"
        if value == "hybrid":
            return "系統輔助 + 人工確認"
        return "未標示，需人工確認"

    def _evidence(row):
        evidence = []
        if pd.notna(row.get("monitor_name")) and str(row.get("monitor_name")).strip() not in ["", "nan", "None"]:
            evidence.append("系統參數截圖 / 感測數值")
        if str(row.get("image_needed", "")).lower() in ["true", "1", "yes", "y", "需要"]:
            evidence.append("現場照片")
        if pd.notna(row.get("safety_note")) and str(row.get("safety_note")).strip() not in ["", "nan", "None"]:
            evidence.append("安全確認紀錄")
        if not evidence:
            evidence.append("人工確認備註")
        return "、".join(evidence)

    def _next_hint(row):
        text = f"{row.get('step_title', '')} {row.get('step_content', '')}"
        if any(k in text for k in ["加熱", "溫度", "在爐", "鋼種"]):
            return "若加熱條件異常，請比對鋼種標準，必要時調整加熱參數或通知製程工程師。"
        if any(k in text for k in ["軸承", "傳動", "馬達", "電流", "Guide", "Roll", "軋輥"]):
            return "若設備機構或電流異常，請通知設備工程師檢查磨耗、鬆動、卡滯或破損。"
        if any(k in text for k in ["訊號", "PLC", "sensor", "感測", "通訊", "loss"]):
            return "若訊號異常，請確認 PLC、通訊狀態、感測器連線與資料是否缺漏。"
        if any(k in text for k in ["水量", "噴嘴", "冷卻", "流量", "壓力"]):
            return "若水量、噴嘴或壓力異常，請確認水量設定、阻塞狀態、角度與壓力來源。"
        return "若此步驟判定異常，請依 SOP 進入下一步，並留下處置紀錄。"

    # ---------------------------------------------------------
    # 1) 先讀 output CSV。這是 Colab 版本原本的路徑。
    # ---------------------------------------------------------
    clean_event_view = safe_read_csv("output/clean_event_view.csv", usecols=EVENT_COLS)

    clean_sop_view = safe_read_csv("output/sop_detail_view.csv", usecols=SOP_COLS)
    if clean_sop_view.empty:
        clean_sop_view = safe_read_csv("output/clean_sop_view.csv", usecols=SOP_COLS)

    clean_sensor_view = safe_read_csv("output/clean_sensor_view.csv", usecols=SENSOR_COLS)
    human_assist_view = safe_read_csv("output/human_assist_view.csv", usecols=HUMAN_ASSIST_COLS)

    graph_nodes = safe_read_csv("output/graph_nodes.csv") if os.path.exists("output/graph_nodes.csv") else None
    graph_edges = safe_read_csv("output/graph_edges.csv") if os.path.exists("output/graph_edges.csv") else None

    # ---------------------------------------------------------
    # 2) 若 output 不存在或主要資料為空，自動改從 Excel 建立。
    # ---------------------------------------------------------
    if clean_event_view.empty or "equipment_code" not in clean_event_view.columns:
        excel_path = _find_file("Dataset/資料集.xlsx")
        if excel_path is None:
            st.error("找不到 Dataset/資料集.xlsx。請確認 GitHub 的 Dataset 資料夾內有資料集.xlsx，或上傳 output 資料夾。")
            st.stop()

        try:
            xls = pd.ExcelFile(excel_path)
            tables = {sheet: pd.read_excel(excel_path, sheet_name=sheet) for sheet in xls.sheet_names}
        except Exception as exc:
            st.error(f"讀取 Dataset/資料集.xlsx 失敗：{exc}")
            st.stop()

        required = ["05_cate_detail", "06_equipment", "07_state", "09_monitor_function", "11_sop_main", "12_sop_step", "13_abnormal_event", "14_event_step_check", "15_sensor_snapshot"]
        missing = [s for s in required if s not in tables]
        if missing:
            st.error(f"Dataset/資料集.xlsx 缺少必要工作表：{missing}")
            st.stop()

        cate_detail = tables["05_cate_detail"].copy()
        equipment = tables["06_equipment"].copy()
        state = tables["07_state"].copy()
        monitor_function = tables["09_monitor_function"].copy()
        sop_main = tables["11_sop_main"].copy()
        sop_step = tables["12_sop_step"].copy()
        abnormal_event = tables["13_abnormal_event"].copy()
        sensor_snapshot = tables["15_sensor_snapshot"].copy()

        # clean_event_view
        clean_event_view = abnormal_event.copy()
        if "equipment_id" in clean_event_view.columns and "equipment_id" in equipment.columns:
            eq_cols = [c for c in ["equipment_id", "equipment_code", "equipment_name"] if c in equipment.columns]
            clean_event_view = clean_event_view.merge(equipment[eq_cols].drop_duplicates(), on="equipment_id", how="left")
        if "state_id" in clean_event_view.columns and "state_id" in state.columns:
            st_cols = [c for c in ["state_id", "state_name", "default_severity"] if c in state.columns]
            clean_event_view = clean_event_view.merge(state[st_cols].drop_duplicates(), on="state_id", how="left")
        if "cate_detail_id" in clean_event_view.columns and "cate_detail_id" in cate_detail.columns and "name" in cate_detail.columns:
            clean_event_view = clean_event_view.merge(
                cate_detail[["cate_detail_id", "name"]].rename(columns={"name": "cate_detail_name"}).drop_duplicates(),
                on="cate_detail_id", how="left"
            )
        if "sop_id" in clean_event_view.columns and "sop_id" in sop_main.columns:
            sop_cols = [c for c in ["sop_id", "sop_name", "sop_desc", "owner_role"] if c in sop_main.columns]
            clean_event_view = clean_event_view.merge(sop_main[sop_cols].drop_duplicates(), on="sop_id", how="left")
        clean_event_view = _ensure_columns(clean_event_view, EVENT_COLS + ["close_result", "equipment_id", "state_id", "root_cause_category"])

        # clean_sop_view
        clean_sop_view = sop_step.copy()
        if "sop_id" in clean_sop_view.columns and "sop_id" in sop_main.columns:
            sop_cols = [c for c in ["sop_id", "equipment_id", "state_id", "sop_name", "sop_desc", "version", "status", "owner_role"] if c in sop_main.columns]
            clean_sop_view = clean_sop_view.merge(sop_main[sop_cols].drop_duplicates(), on="sop_id", how="left")
        if "equipment_id" in clean_sop_view.columns and "equipment_id" in equipment.columns:
            eq_cols = [c for c in ["equipment_id", "equipment_code", "equipment_name", "line_area"] if c in equipment.columns]
            clean_sop_view = clean_sop_view.merge(equipment[eq_cols].drop_duplicates(), on="equipment_id", how="left")
        if "state_id" in clean_sop_view.columns and "state_id" in state.columns:
            st_cols = [c for c in ["state_id", "state_name", "default_severity"] if c in state.columns]
            clean_sop_view = clean_sop_view.merge(state[st_cols].drop_duplicates(), on="state_id", how="left")
        if "monitor_id" in clean_sop_view.columns and "monitor_id" in monitor_function.columns:
            mon_cols = [c for c in ["monitor_id", "monitor_name", "subgroup", "data_source_system", "description"] if c in monitor_function.columns]
            clean_sop_view = clean_sop_view.merge(monitor_function[mon_cols].drop_duplicates(), on="monitor_id", how="left")
        clean_sop_view = _ensure_columns(clean_sop_view, SOP_COLS)
        clean_sop_view["sop_stage"] = clean_sop_view.apply(lambda r: _classify_stage(r.get("step_title", ""), r.get("step_content", "")), axis=1)
        clean_sop_view["check_method_label"] = clean_sop_view["check_type"].apply(_translate_check_type)
        clean_sop_view["evidence_required"] = clean_sop_view.apply(_evidence, axis=1)
        clean_sop_view["next_action_hint"] = clean_sop_view.apply(_next_hint, axis=1)

        # clean_sensor_view
        clean_sensor_view = sensor_snapshot.copy()
        if "monitor_id" in clean_sensor_view.columns and "monitor_id" in monitor_function.columns:
            mon_cols = [c for c in ["monitor_id", "monitor_name", "subgroup", "data_source_system", "description"] if c in monitor_function.columns]
            clean_sensor_view = clean_sensor_view.merge(monitor_function[mon_cols].drop_duplicates(), on="monitor_id", how="left")
        if "event_id" in clean_sensor_view.columns and "event_id" in clean_event_view.columns:
            event_cols = [c for c in ["event_id", "occurred_at", "equipment_code", "equipment_name", "state_name", "severity", "root_cause_category"] if c in clean_event_view.columns]
            clean_sensor_view = clean_sensor_view.merge(clean_event_view[event_cols].drop_duplicates(), on="event_id", how="left")
        clean_sensor_view = _ensure_columns(clean_sensor_view, SENSOR_COLS)

        # human_assist_view：若沒有原本 output，就從 SOP 步驟建立簡化版人工輔助資料。
        records = []
        for _, row in clean_sop_view.iterrows():
            title = str(row.get("step_title", "")).strip()
            if title in ["", "根節點", "nan", "None"]:
                continue
            check_type = str(row.get("check_type", "")).lower()
            manual_related = (
                check_type in ["manual", "hybrid"]
                or pd.isna(row.get("monitor_name"))
                or str(row.get("image_needed", "")).lower() in ["true", "1", "yes", "y", "需要"]
                or pd.notna(row.get("safety_note"))
            )
            if not manual_related:
                continue
            records.append({
                "source_table": "12_sop_step",
                "equipment_code": row.get("equipment_code"),
                "state_keyword": row.get("state_name"),
                "assist_type": row.get("sop_stage", "SOP人工確認"),
                "check_item": row.get("step_title"),
                "human_action": row.get("step_content"),
                "related_monitor_id": row.get("monitor_id"),
                "evidence_required": row.get("evidence_required"),
                "escalation_rule": row.get("next_action_hint"),
                "source_note": f"來自 {row.get('sop_id', '')}｜{row.get('sop_name', '')}"
            })
        human_assist_view = pd.DataFrame(records)
        human_assist_view = _ensure_columns(human_assist_view, HUMAN_ASSIST_COLS)

        # graph_nodes / graph_edges：建立穩定可展示的簡化圖譜資料。
        node_records = []
        edge_records = []
        def add_node(node_id, node_type, label):
            node_records.append({"node_id": node_id, "node_type": node_type, "label": label})
        def add_edge(src, tgt, rel, name):
            edge_records.append({"source": src, "target": tgt, "relation": rel, "relation_name": name})
        for _, row in clean_event_view.iterrows():
            eq = row.get("equipment_code")
            stn = row.get("state_name")
            sopid = row.get("sop_id")
            eventid = row.get("event_id")
            cause = row.get("root_cause_category")
            if pd.notna(eq): add_node(f"Equipment:{eq}", "Equipment", str(eq))
            if pd.notna(stn): add_node(f"State:{stn}", "State", str(stn))
            if pd.notna(eventid): add_node(f"Event:{eventid}", "Event", str(eventid))
            if pd.notna(sopid): add_node(f"SOP:{sopid}", "SOP", str(row.get("sop_name", sopid)))
            if pd.notna(cause): add_node(f"Cause:{cause}", "Cause", str(cause))
            if pd.notna(eq) and pd.notna(stn): add_edge(f"Equipment:{eq}", f"State:{stn}", "HAS_STATE", "設備可能發生此異常")
            if pd.notna(eventid) and pd.notna(eq): add_edge(f"Event:{eventid}", f"Equipment:{eq}", "OCCURRED_ON", "事件發生於此設備")
            if pd.notna(eventid) and pd.notna(stn): add_edge(f"Event:{eventid}", f"State:{stn}", "HAS_ABNORMAL_STATE", "事件屬於此異常狀況")
            if pd.notna(eventid) and pd.notna(sopid): add_edge(f"Event:{eventid}", f"SOP:{sopid}", "USED_SOP", "事件使用此 SOP")
            if pd.notna(eventid) and pd.notna(cause): add_edge(f"Event:{eventid}", f"Cause:{cause}", "HAS_CAUSE", "事件原因分類")
        for _, row in clean_sop_view.iterrows():
            sopid = row.get("sop_id")
            stepid = row.get("step_id")
            if pd.notna(sopid) and pd.notna(stepid):
                add_node(f"SOPStep:{stepid}", "SOPStep", str(row.get("step_title", stepid)))
                add_edge(f"SOP:{sopid}", f"SOPStep:{stepid}", "HAS_STEP", "SOP 包含此步驟")
        graph_nodes = pd.DataFrame(node_records).drop_duplicates() if node_records else pd.DataFrame(columns=["node_id", "node_type", "label"])
        graph_edges = pd.DataFrame(edge_records).drop_duplicates() if edge_records else pd.DataFrame(columns=["source", "target", "relation", "relation_name"])

    # ---------------------------------------------------------
    # 3) 共同清理型別與保底欄位，避免 KeyError。
    # ---------------------------------------------------------
    clean_event_view = _ensure_columns(clean_event_view, EVENT_COLS + ["close_result", "root_cause_category"])
    clean_sop_view = _ensure_columns(clean_sop_view, SOP_COLS)
    clean_sensor_view = _ensure_columns(clean_sensor_view, SENSOR_COLS)
    human_assist_view = _ensure_columns(human_assist_view, HUMAN_ASSIST_COLS)

    for col in ["equipment_code", "state_name", "severity", "root_cause_category"]:
        clean_event_view[col] = clean_event_view[col].astype(str).replace("nan", pd.NA)

    for col in ["equipment_code", "state_name", "sop_id", "step_id", "parent_step_id"]:
        clean_sop_view[col] = clean_sop_view[col].astype(str).replace("nan", pd.NA)

    for col in ["equipment_code", "state_keyword", "check_item", "human_action"]:
        human_assist_view[col] = human_assist_view[col].astype(str).replace("nan", pd.NA)

    clean_event_view["downtime_min"] = pd.to_numeric(clean_event_view["downtime_min"], errors="coerce")
    clean_event_view["occurred_at"] = pd.to_datetime(clean_event_view["occurred_at"], errors="coerce")
    clean_sensor_view["captured_at"] = pd.to_datetime(clean_sensor_view["captured_at"], errors="coerce")

    for col in ["actual_value", "spec_lower", "spec_upper"]:
        clean_sensor_view[col] = pd.to_numeric(clean_sensor_view[col], errors="coerce")

    for col in ["monitor_name", "parameter_name", "judgement", "unit"]:
        clean_sensor_view[col] = clean_sensor_view[col].astype(str).replace("nan", pd.NA)

    clean_sop_view["sort_order"] = pd.to_numeric(clean_sop_view["sort_order"], errors="coerce")

    return clean_event_view, clean_sop_view, clean_sensor_view, graph_nodes, graph_edges, human_assist_view


with st.spinner("載入製造異常資料中..."):
    clean_event_view, clean_sop_view, clean_sensor_view, graph_nodes, graph_edges, human_assist_view = load_data()


# =========================================================
# 預先彙總 Dashboard 統計
# =========================================================

@st.cache_data(show_spinner=False)
def build_dashboard_summary(event_df):
    summary = {}

    summary["total_events"] = len(event_df)

    if "equipment_code" in event_df.columns:
        summary["equipment_count"] = event_df["equipment_code"].nunique()

        top_equipment = (
            event_df["equipment_code"]
            .dropna()
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_equipment.columns = ["equipment_code", "count"]
    else:
        summary["equipment_count"] = 0
        top_equipment = pd.DataFrame(columns=["equipment_code", "count"])

    if "state_name" in event_df.columns:
        top_state = (
            event_df["state_name"]
            .dropna()
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_state.columns = ["state_name", "count"]
    else:
        top_state = pd.DataFrame(columns=["state_name", "count"])

    if "severity" in event_df.columns:
        severity_df = (
            event_df["severity"]
            .dropna()
            .value_counts()
            .reset_index()
        )
        severity_df.columns = ["severity", "count"]
    else:
        severity_df = pd.DataFrame(columns=["severity", "count"])

    if "root_cause_category" in event_df.columns:
        cause_df = (
            event_df["root_cause_category"]
            .dropna()
            .value_counts()
            .reset_index()
        )
        cause_df.columns = ["root_cause_category", "count"]
    else:
        cause_df = pd.DataFrame(columns=["root_cause_category", "count"])

    if "downtime_min" in event_df.columns:
        summary["avg_downtime"] = event_df["downtime_min"].mean()
        summary["max_downtime"] = event_df["downtime_min"].max()
    else:
        summary["avg_downtime"] = None
        summary["max_downtime"] = None

    if "occurred_at" in event_df.columns:
        trend_df = event_df.dropna(subset=["occurred_at"]).copy()

        if len(trend_df) > 0:
            trend_df["month"] = trend_df["occurred_at"].dt.to_period("M").astype(str)

            monthly_count = (
                trend_df
                .groupby("month")
                .size()
                .reset_index(name="count")
                .sort_values("month")
            )
        else:
            monthly_count = pd.DataFrame(columns=["month", "count"])
    else:
        monthly_count = pd.DataFrame(columns=["month", "count"])

    return summary, top_equipment, top_state, severity_df, cause_df, monthly_count


dashboard_summary, top_equipment_df, top_state_df, severity_df, cause_df, monthly_count_df = build_dashboard_summary(clean_event_view)


# =========================================================
# 加強版：相似案例推薦與 SOP 改善分析
# =========================================================

def get_recommended_cases(equipment_code, state_name, top_n=5):
    """
    依同設備、同異常狀況、處置紀錄與停機時間排序，推薦最值得參考的歷史案例。
    """
    events = get_matched_events(equipment_code, state_name).copy()

    if events.empty:
        return pd.DataFrame()

    score = pd.Series(0, index=events.index, dtype="float")

    if "equipment_code" in events.columns:
        score += (events["equipment_code"].astype(str) == str(equipment_code)).astype(int) * 3

    if "state_name" in events.columns:
        score += (events["state_name"].astype(str) == str(state_name)).astype(int) * 3

    if "action_summary" in events.columns:
        action_text = events["action_summary"].fillna("").astype(str)
        score += action_text.str.len().clip(upper=100) / 100

        success_keywords = "復機|改善|恢復|完成|排除|正常|更換|調整|校正"
        score += action_text.str.contains(success_keywords, regex=True, na=False).astype(int) * 2

    if "downtime_min" in events.columns:
        downtime = pd.to_numeric(events["downtime_min"], errors="coerce")
        if downtime.notna().any():
            # 停機時間較短代表該案例可能較有效，作為小幅加分。
            max_dt = downtime.max()
            min_dt = downtime.min()
            if pd.notna(max_dt) and pd.notna(min_dt) and max_dt != min_dt:
                score += (1 - (downtime - min_dt) / (max_dt - min_dt)).fillna(0)
            events["_downtime_for_sort"] = downtime
        else:
            events["_downtime_for_sort"] = pd.NA
    else:
        events["_downtime_for_sort"] = pd.NA

    events["_recommend_score"] = score.round(2)

    sort_cols = ["_recommend_score"]
    ascending = [False]
    if "_downtime_for_sort" in events.columns:
        sort_cols.append("_downtime_for_sort")
        ascending.append(True)

    return events.sort_values(sort_cols, ascending=ascending).head(top_n)


def build_formal_ai_report(equipment_code, state_name, result):
    """
    將多 Agent 結果整理成老師與現場人員較容易閱讀的正式診斷報告格式。
    """
    diagnosis = result.get("diagnosis", {})
    sop = result.get("sop", {})
    human = result.get("human", {})
    parts = result.get("parts", {})
    notification = result.get("notification", {})

    possible_causes = diagnosis.get("possible_causes", [])
    matched_event_count = diagnosis.get("matched_event_count", 0)
    avg_downtime = diagnosis.get("avg_downtime", None)

    if possible_causes:
        cause_lines = []
        for idx, item in enumerate(possible_causes, start=1):
            cause_lines.append(f"{idx}. {item.get('cause', '未分類原因')}：{item.get('evidence', '歷史事件曾出現')}")
    else:
        cause_lines = ["1. 目前資料不足，尚無法判定明確主因，需先補齊現場觀察與感測器紀錄。"]

    if avg_downtime is not None and pd.notna(avg_downtime):
        downtime_text = f"相似事件 {matched_event_count} 筆，平均停機約 {avg_downtime:.1f} 分鐘。"
    else:
        downtime_text = f"相似事件 {matched_event_count} 筆，停機時間資料不足。"

    if human.get("assist_df") is not None and not human["assist_df"].empty:
        manual_df = human["assist_df"].head(4)
        manual_lines = []
        for idx, row in manual_df.iterrows():
            manual_lines.append(f"- {row.get('check_item', '依現場狀況確認')}；建議佐證：{row.get('evidence_required', '現場紀錄或照片')}")
    else:
        manual_lines = [
            "- 確認現場異常是否仍持續，並留下照片或班長確認紀錄。",
            "- 確認感測器是否有 loss、資料缺漏或異常跳動。",
            "- 確認 SOP 是否符合實際作業方式，若不符需回饋修訂。"
        ]

    if parts.get("status") == "ok" and parts.get("parts"):
        parts_lines = [f"- {item.get('part', '待確認備件')}：{item.get('reason', '依歷史處置或 SOP 關鍵字推估')}" for item in parts["parts"][:4]]
    else:
        parts_lines = ["- 目前沒有明確備件線索，需由維修單位依現場狀況確認。"]

    notify_units = notification.get("notify_units", [])
    notify_text = "、".join(notify_units) if notify_units else "設備維修單位、製程工程單位"

    lines = []
    lines.append("【正式 AI 異常診斷報告】")
    lines.append("")
    lines.append("一、異常概況")
    lines.append(f"- 設備：{equipment_code}")
    lines.append(f"- 異常狀況：{state_name}")
    lines.append(f"- 歷史資料基礎：{downtime_text}")
    lines.append("")
    lines.append("二、可能原因排序")
    lines.extend(cause_lines)
    lines.append("")
    lines.append("三、建議處置順序")
    lines.append("1. 先確認異常是否仍持續發生，避免處理已恢復但未紀錄的事件。")
    lines.append("2. 依對應 SOP 檢查關鍵節點，優先檢查可由感測器判斷的項目。")
    lines.append("3. 對 AI 無法自動判斷的節點，要求現場人員補上照片、量測值或備註。")
    lines.append("4. 比對下方相似案例，優先參考過去曾成功復機且停機時間較短的處置方式。")
    lines.append("5. 完成處置後回填原因、動作與復機結果，作為後續 SOP 改善依據。")
    lines.append("")
    lines.append("四、需要人工確認的項目")
    lines.extend(manual_lines)
    lines.append("")
    lines.append("五、備件或物料確認方向")
    lines.extend(parts_lines)
    lines.append("")
    lines.append("六、建議通報對象")
    lines.append(f"- {notify_text}")
    lines.append("")
    lines.append("七、使用限制")
    lines.append("- 本報告為生成式 AI 輔助整理，不取代工程師、安全人員與現場主管的判斷。")
    lines.append("- 若感測器資料缺漏、現場環境不明或 SOP 與實況不一致，應以人工確認結果為準。")

    return "\n".join(lines)


def build_sop_improvement_tables(event_df, sop_df):
    """
    建立 SOP 改善頁面需要的三種分析表：
    1. 停機時間較高的設備與異常組合
    2. 最常需要人工確認的 SOP 步驟
    3. 最常出現感測器 / 規則異常的設備異常組合
    """
    if event_df.empty:
        high_downtime = pd.DataFrame()
        repeated_abnormal = pd.DataFrame()
    else:
        high_downtime = event_df.copy()
        if "downtime_min" in high_downtime.columns:
            high_downtime["downtime_min"] = pd.to_numeric(high_downtime["downtime_min"], errors="coerce")
            group_cols = [c for c in ["equipment_code", "state_name"] if c in high_downtime.columns]
            if group_cols:
                high_downtime = (
                    high_downtime
                    .groupby(group_cols, dropna=False)
                    .agg(
                        event_count=("event_id", "count") if "event_id" in high_downtime.columns else ("downtime_min", "size"),
                        avg_downtime_min=("downtime_min", "mean"),
                        max_downtime_min=("downtime_min", "max")
                    )
                    .reset_index()
                    .sort_values(["avg_downtime_min", "event_count"], ascending=[False, False])
                    .head(15)
                )
            else:
                high_downtime = pd.DataFrame()
        else:
            high_downtime = pd.DataFrame()

        if "root_cause_category" in event_df.columns:
            group_cols = [c for c in ["equipment_code", "state_name", "root_cause_category"] if c in event_df.columns]
            if group_cols:
                repeated_abnormal = (
                    event_df
                    .groupby(group_cols, dropna=False)
                    .size()
                    .reset_index(name="event_count")
                    .sort_values("event_count", ascending=False)
                    .head(15)
                )
            else:
                repeated_abnormal = pd.DataFrame()
        else:
            repeated_abnormal = pd.DataFrame()

    if sop_df.empty:
        manual_steps = pd.DataFrame()
    else:
        manual_df = sop_df.copy()
        check_cols = [c for c in ["check_type", "image_needed", "evidence_required", "owner_role"] if c in manual_df.columns]

        def _manual_score(row):
            text = " ".join([str(row.get(c, "")) for c in check_cols])
            score = 0
            if any(k in text for k in ["人工", "目視", "照片", "拍照", "確認", "現場", "手動"]):
                score += 1
            if str(row.get("image_needed", "")).lower() in ["y", "yes", "true", "1", "需要"]:
                score += 1
            if pd.notna(row.get("evidence_required", pd.NA)) and str(row.get("evidence_required", "")).strip() not in ["", "nan", "None"]:
                score += 1
            return score

        manual_df["_manual_score"] = manual_df.apply(_manual_score, axis=1)
        manual_steps = manual_df[manual_df["_manual_score"] > 0].copy()

        display_cols = [
            "equipment_code", "state_name", "sop_id", "sop_name", "step_id",
            "step_title", "check_type", "owner_role", "evidence_required", "_manual_score"
        ]
        existing_cols = [c for c in display_cols if c in manual_steps.columns]
        manual_steps = (
            manual_steps[existing_cols]
            .sort_values("_manual_score", ascending=False)
            .head(20)
        )

    return high_downtime, manual_steps, repeated_abnormal



# =========================================================
# 共用查詢函式
# =========================================================

def get_matched_events(equipment_code=None, state_name=None):
    df = clean_event_view

    if equipment_code and equipment_code != "全部" and "equipment_code" in df.columns:
        df = df[df["equipment_code"] == str(equipment_code)]

    if state_name and state_name != "全部" and "state_name" in df.columns:
        df = df[df["state_name"] == str(state_name)]

    return df.copy()


def get_sop_by_equipment_state(equipment_code=None, state_name=None):
    events = get_matched_events(equipment_code, state_name)

    if events.empty:
        return pd.DataFrame()

    if "sop_id" not in events.columns or "sop_id" not in clean_sop_view.columns:
        return pd.DataFrame()

    sop_ids = events["sop_id"].dropna().unique().tolist()

    sop_df = clean_sop_view[
        clean_sop_view["sop_id"].isin(sop_ids)
    ].copy()

    if sop_df.empty:
        return sop_df

    if "sort_order" in sop_df.columns:
        return sop_df.sort_values(["sop_id", "sort_order"])

    return sop_df.sort_values(["sop_id"])


def get_sensor_by_events(event_ids):
    if "event_id" not in clean_sensor_view.columns:
        return pd.DataFrame()

    return clean_sensor_view[
        clean_sensor_view["event_id"].isin(event_ids)
    ].copy()




def _clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _escape_dot(value):
    text = _clean_text(value)
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    text = text.replace("\n", "\\n")
    return text


def classify_sop_stage(step_title, step_content):
    text = f"{step_title} {step_content}"

    if any(k in text for k in ["根節點", "確認是否", "判斷是否", "確認異常", "持續", "觀察", "是否仍"]):
        return "1. 異常確認"
    if any(k in text for k in ["安全", "停機", "隔離", "防護", "通知", "危險"]):
        return "2. 安全處置"
    if any(k in text for k in ["檢查", "確認", "比對", "量測", "查看", "判斷", "監控"]):
        return "3. 原因排查"
    if any(k in text for k in ["調整", "更換", "清除", "修正", "處理", "復歸", "退爐", "降速"]):
        return "4. 處置修正"
    if any(k in text for k in ["復機", "試軋", "恢復", "結案", "再確認", "OK"]):
        return "5. 復機確認"
    if any(k in text for k in ["紀錄", "回報", "上傳", "改善", "修訂", "回填"]):
        return "6. 紀錄回饋"
    return "3. 原因排查"


def build_sop_tree_dot(sop_df):
    """把 sop_step 的 parent_step_id / branch_label 轉成 Graphviz 樹狀圖。
    若原始資料只提供單一路徑，會補上「未完成 / 需升級」等合理分支，避免流程看起來像只有一條路。
    """
    if sop_df.empty or "step_id" not in sop_df.columns:
        return None

    df = sop_df.copy()
    if "sort_order" in df.columns:
        df = df.sort_values(["sort_order", "step_id"], na_position="last")

    valid_step_ids = set(df["step_id"].dropna().astype(str))

    lines = [
        "digraph SOP {",
        "rankdir=TB;",
        "graph [splines=ortho, nodesep=0.45, ranksep=0.65];",
        "node [shape=box, style=\"rounded,filled\", fillcolor=\"#EAF3FF\", color=\"#7EA6D8\", fontname=\"Microsoft JhengHei\", fontsize=11];",
        "edge [color=\"#6B7280\", fontname=\"Microsoft JhengHei\", fontsize=10];"
    ]

    fallback_ids = []
    row_by_id = {}
    child_labels_by_parent = {}

    for i, row in df.reset_index(drop=True).iterrows():
        step_id = _clean_text(row.get("step_id")) or f"STEP_AUTO_{i}"
        fallback_ids.append(step_id)
        row_by_id[step_id] = row
        title = _clean_text(row.get("step_title"))
        content = _clean_text(row.get("step_content"))
        stage = _clean_text(row.get("sop_stage")) or classify_sop_stage(title, content)
        check_type = _clean_text(row.get("check_method_label")) or _clean_text(row.get("check_type"))

        label_parts = [title]
        if content:
            label_parts.append(content[:38] + ("..." if len(content) > 38 else ""))
        if stage:
            label_parts.append(stage)
        if check_type:
            label_parts.append(f"判斷：{check_type}")

        label = "\\n".join([_escape_dot(x) for x in label_parts if x])
        lines.append(f'"{_escape_dot(step_id)}" [label="{label}"];')

    has_edge = False
    for _, row in df.iterrows():
        step_id = _clean_text(row.get("step_id"))
        parent = _clean_text(row.get("parent_step_id"))
        branch = _clean_text(row.get("branch_label"))

        if step_id and parent and parent in valid_step_ids and parent != step_id:
            label = f' [label="{_escape_dot(branch)}"]' if branch else ""
            lines.append(f'"{_escape_dot(parent)}" -> "{_escape_dot(step_id)}"{label};')
            child_labels_by_parent.setdefault(parent, []).append(branch)
            has_edge = True

    # 補齊缺漏分支：例如只有「是」沒有「否」、只有「完成」沒有「未完成」
    for parent, labels in child_labels_by_parent.items():
        if len(labels) == 1 and parent in row_by_id:
            missing_label = suggest_missing_branch_label(labels[0])
            virtual_id = f"VIRTUAL_{parent}_{abs(hash(missing_label)) % 100000}"
            parent_title = _clean_text(row_by_id[parent].get("step_title"))
            virtual_label = f"{missing_label}\\n需人工確認 / 升級處理\\n依終點分析結果處理"
            lines.append(f'"{_escape_dot(virtual_id)}" [label="{_escape_dot(virtual_label)}", fillcolor="#FFF7ED", color="#FDBA74"];')
            lines.append(f'"{_escape_dot(parent)}" -> "{_escape_dot(virtual_id)}" [label="{_escape_dot(missing_label)}"];')

    if not has_edge and len(fallback_ids) > 1:
        for a, b in zip(fallback_ids[:-1], fallback_ids[1:]):
            lines.append(f'"{_escape_dot(a)}" -> "{_escape_dot(b)}";')

    lines.append("}")
    return "\n".join(lines)

def _row_evidence(row):
    evidence = []
    if pd.notna(row.get("monitor_name")):
        evidence.append("系統參數截圖 / 感測數值")
    if str(row.get("image_needed", "")).lower() in ["true", "1", "yes", "y"]:
        evidence.append("現場照片")
    if pd.notna(row.get("safety_note")):
        evidence.append("安全確認紀錄")
    if not evidence:
        evidence.append("人工確認備註")
    return "、".join(evidence)


def _row_escalation(row):
    text = f"{row.get('step_title', '')} {row.get('step_content', '')}"
    if any(k in text for k in ["安全", "隔離", "停機", "危險"]):
        return "若涉及安全風險，立即停機並通知班長。"
    if any(k in text for k in ["軸承", "傳動", "馬達", "電流", "Guide", "Roll"]):
        return "若設備機構異常或處置後仍無法復機，通知設備工程師 / 保全單位。"
    if any(k in text for k in ["加熱", "溫度", "在爐", "鋼種", "製程"]):
        return "若製程條件與鋼種標準不一致，通知製程工程師確認。"
    if any(k in text for k in ["訊號", "PLC", "通訊", "sensor", "感測"]):
        return "若訊號異常或感測值不可信，通知自動化 / 設備單位確認。"
    return "若依此步驟確認後仍無法改善，通知班長並升級給製程或設備工程師。"


def _assist_match_score(row, state_name):
    state = _clean_text(state_name)
    keyword = _clean_text(row.get("state_keyword"))
    check_item = _clean_text(row.get("check_item"))
    human_action = _clean_text(row.get("human_action"))
    source_table = _clean_text(row.get("source_table"))

    score = 0
    if keyword == state:
        score += 100
    elif state and keyword and (state in keyword or keyword in state):
        score += 65
    if state and state in check_item:
        score += 25
    if state and state in human_action:
        score += 20
    if source_table == "12_sop_step" and keyword == state:
        score += 20
    return score


def build_sop_based_human_assist(equipment_code, state_name, max_items=8):
    sop_df = get_sop_by_equipment_state(equipment_code, state_name)
    if sop_df.empty:
        return pd.DataFrame(columns=HUMAN_ASSIST_COLS)

    records = []
    for _, row in sop_df.iterrows():
        title = _clean_text(row.get("step_title"))
        if title in ["", "根節點"]:
            continue
        check_type = _clean_text(row.get("check_type")).lower()
        is_manual = (
            check_type in ["manual", "hybrid"]
            or pd.isna(row.get("monitor_name"))
            or str(row.get("image_needed", "")).lower() in ["true", "1", "yes", "y"]
            or pd.notna(row.get("safety_note"))
        )
        if not is_manual:
            continue
        records.append({
            "source_table": "12_sop_step",
            "equipment_code": equipment_code,
            "state_keyword": state_name,
            "assist_type": _clean_text(row.get("sop_stage")) or classify_sop_stage(title, row.get("step_content", "")),
            "check_item": title,
            "human_action": _clean_text(row.get("step_content")),
            "related_monitor_id": row.get("monitor_id"),
            "evidence_required": row.get("evidence_required") if pd.notna(row.get("evidence_required")) else _row_evidence(row),
            "escalation_rule": _row_escalation(row),
            "source_note": f"來自 {row.get('sop_id', '')}｜{row.get('sop_name', '')}"
        })
    return pd.DataFrame(records).head(max_items)


def get_human_assistance(equipment_code, state_name, max_items=8):
    candidates = []

    if human_assist_view is not None and not human_assist_view.empty:
        df = human_assist_view.copy()
        df = df[df["equipment_code"].astype(str).str.upper() == str(equipment_code).upper()].copy()
        if not df.empty:
            df["match_score"] = df.apply(lambda row: _assist_match_score(row, state_name), axis=1)
            df = df[df["match_score"] > 0].sort_values("match_score", ascending=False)
            if not df.empty:
                candidates.append(df.drop(columns=["match_score"], errors="ignore"))

    # 永遠補上該設備、該異常的 SOP 人工確認項目，確保不同異常顯示不同內容
    sop_based = build_sop_based_human_assist(equipment_code, state_name, max_items=max_items)
    if not sop_based.empty:
        candidates.append(sop_based)

    if not candidates:
        return pd.DataFrame(columns=HUMAN_ASSIST_COLS)

    result = pd.concat(candidates, ignore_index=True)
    result = result.drop_duplicates(
        subset=["equipment_code", "state_keyword", "check_item", "human_action"],
        keep="first"
    )
    return result.head(max_items)


def human_assistance_agent(equipment_code, state_name, max_items=8):
    assist_df = get_human_assistance(equipment_code, state_name, max_items=max_items)

    lines = []
    lines.append("【Human Assistance Agent｜人工輔助檢查與處置】")
    lines.append(f"查詢設備：{equipment_code}")
    lines.append(f"異常狀況：{state_name}")
    lines.append("")

    if assist_df.empty:
        lines.append("目前沒有找到此設備與異常狀況的專屬人工輔助資料。")
        lines.append("建議先依 SOP 執行人工確認，並將本次現場處置結果回填到異常事件紀錄。")
        return {
            "agent": "Human Assistance Agent",
            "status": "no_data",
            "text": "\n".join(lines),
            "assist_df": assist_df
        }

    lines.append("此 Agent 會依目前選定的設備與異常狀況，整理原始檔與 SOP 中可由人工執行的檢查、處置、佐證與升級條件。")
    lines.append("")

    for idx, row in assist_df.reset_index(drop=True).iterrows():
        lines.append(f"【人工輔助項目 {idx + 1}】")
        lines.append(f"類型：{row.get('assist_type', '人工輔助')}")
        lines.append(f"人工需確認：{row.get('check_item', '依現場狀況確認')}")
        lines.append(f"建議人工處置：{row.get('human_action', '依班長或工程師指示處理')}")

        if pd.notna(row.get("related_monitor_id")) and _clean_text(row.get("related_monitor_id")):
            lines.append(f"可搭配監控項目：{row.get('related_monitor_id')}")
        else:
            lines.append("可搭配監控項目：無明確監控項目，偏人工經驗判斷")

        lines.append(f"建議留下佐證：{row.get('evidence_required', '人工確認紀錄')}")
        lines.append(f"升級通報條件：{row.get('escalation_rule', '若處置後仍未改善，需升級通報')}")
        lines.append(f"資料來源：{row.get('source_note', row.get('source_table', '原始資料'))}")
        lines.append("")

    return {
        "agent": "Human Assistance Agent",
        "status": "ok",
        "text": "\n".join(lines),
        "assist_df": assist_df
    }

def format_empty_message(agent_name):
    return f"【{agent_name}】目前資料不足，無法產生完整建議。"


# =========================================================
# Agent 1：Diagnosis Agent
# =========================================================

def diagnosis_agent(equipment_code, state_name, top_n_causes=3):
    events = get_matched_events(equipment_code, state_name)

    if events.empty:
        return {
            "agent": "Diagnosis Agent",
            "status": "no_data",
            "text": format_empty_message("Diagnosis Agent"),
            "possible_causes": [],
            "matched_event_count": 0,
            "avg_downtime": None
        }

    possible_causes = []

    if "root_cause_category" in events.columns:
        cause_counts = (
            events["root_cause_category"]
            .dropna()
            .value_counts()
            .head(top_n_causes)
        )

        for cause, count in cause_counts.items():
            possible_causes.append({
                "cause": cause,
                "count": int(count),
                "evidence": f"歷史相似事件中出現 {count} 次"
            })

    if "event_id" in events.columns:
        event_ids = events["event_id"].dropna().unique().tolist()
    else:
        event_ids = []

    sensor_df = get_sensor_by_events(event_ids)

    abnormal_sensors = pd.DataFrame()

    if not sensor_df.empty and "judgement" in sensor_df.columns:
        abnormal_sensors = sensor_df[
            sensor_df["judgement"].astype(str).str.contains(
                "異常|NG|超出|不正常|Fail|fail",
                case=False,
                na=False
            )
        ]

    if "downtime_min" in events.columns:
        avg_downtime = events["downtime_min"].mean()
        max_downtime = events["downtime_min"].max()
        min_downtime = events["downtime_min"].min()
    else:
        avg_downtime = None
        max_downtime = None
        min_downtime = None

    lines = []
    lines.append("【Diagnosis Agent｜異常診斷結果】")
    lines.append(f"查詢設備：{equipment_code}")
    lines.append(f"異常狀況：{state_name}")
    lines.append(f"找到相似歷史事件：{len(events)} 筆")

    if avg_downtime is not None and pd.notna(avg_downtime):
        lines.append(f"平均停機時間：約 {avg_downtime:.1f} 分鐘")
        lines.append(f"停機時間範圍：{min_downtime:.0f} 至 {max_downtime:.0f} 分鐘")
    else:
        lines.append("平均停機時間：目前無可用資料")

    lines.append("")
    lines.append("可能原因排序：")

    if possible_causes:
        for idx, item in enumerate(possible_causes, start=1):
            lines.append(f"{idx}. {item['cause']}：{item['evidence']}")
    else:
        lines.append("目前歷史資料中沒有明確的原因分類。")

    lines.append("")
    lines.append("生成式診斷推理：")

    if possible_causes:
        cause_text = "、".join([item["cause"] for item in possible_causes])
        lines.append(
            f"根據相似歷史事件的原因分類，此次「{state_name}」異常較可能與「{cause_text}」相關。"
            "建議現場不要只依單一原因判斷，而應搭配 SOP、感測器資料與現場觀察逐項排查。"
        )
    else:
        lines.append(
            "目前原因分類資料不足，因此系統無法提出明確主因。"
            "建議先確認設備狀態、感測器訊號、近期保養紀錄與現場操作條件。"
        )

    lines.append("")
    lines.append("感測器異常觀察：")

    if abnormal_sensors.empty:
        lines.append("目前未從相似事件中整理出明確的感測器異常紀錄。")
    else:
        if "monitor_name" in abnormal_sensors.columns:
            sensor_summary = (
                abnormal_sensors["monitor_name"]
                .dropna()
                .value_counts()
                .head(5)
            )

            for monitor_name, count in sensor_summary.items():
                lines.append(f"- {monitor_name}：曾出現 {count} 次異常判斷")
        else:
            lines.append("感測器資料中有異常判斷，但缺少 monitor_name 欄位。")

    lines.append("")
    lines.append("建議下一步檢查：")
    lines.append("1. 確認異常是否仍持續發生。")
    lines.append("2. 比對 SOP 中的關鍵檢查點。")
    lines.append("3. 檢查感測器資料是否有 loss、超出規格或缺漏。")
    lines.append("4. 查詢設備近期是否有維修、保養或更換零件紀錄。")

    return {
        "agent": "Diagnosis Agent",
        "status": "ok",
        "text": "\n".join(lines),
        "possible_causes": possible_causes,
        "matched_event_count": len(events),
        "avg_downtime": avg_downtime
    }


# =========================================================
# Agent 2：SOP Agent
# =========================================================

def sop_agent(equipment_code, state_name):
    sop_df = get_sop_by_equipment_state(equipment_code, state_name)

    if sop_df.empty:
        return {
            "agent": "SOP Agent",
            "status": "no_data",
            "text": format_empty_message("SOP Agent"),
            "sop_count": 0,
            "sop_df": sop_df
        }

    if "sop_stage" not in sop_df.columns:
        sop_df = sop_df.copy()
        sop_df["sop_stage"] = sop_df.apply(
            lambda row: classify_sop_stage(row.get("step_title", ""), row.get("step_content", "")),
            axis=1
        )

    lines = []
    lines.append("【SOP Agent｜階段式 SOP 流程建議】")
    lines.append(f"查詢設備：{equipment_code}")
    lines.append(f"異常狀況：{state_name}")
    lines.append("")

    sop_count = sop_df["sop_id"].nunique() if "sop_id" in sop_df.columns else 0
    lines.append(f"找到對應 SOP：{sop_count} 份")
    lines.append("此版本不只列出 SOP 表格，而是依異常確認、原因排查、處置修正、復機確認與紀錄回饋等階段整理。")
    lines.append("")

    group_iterator = sop_df.groupby("sop_id") if "sop_id" in sop_df.columns else [("未知 SOP", sop_df)]

    for sop_id, group in group_iterator:
        first = group.iloc[0]
        lines.append(f"【{sop_id}｜{first.get('sop_name', '')}】")

        if pd.notna(first.get("sop_desc", None)):
            lines.append(f"SOP 說明：{first.get('sop_desc')}")
        if pd.notna(first.get("owner_role", None)):
            lines.append(f"負責角色：{first.get('owner_role')}")
        lines.append("")

        if "sop_stage" in group.columns:
            stage_groups = group.sort_values(["sop_stage", "sort_order"], na_position="last").groupby("sop_stage", sort=True)
        else:
            stage_groups = [("3. 原因排查", group.sort_values("sort_order", na_position="last"))]

        for stage, stage_df in stage_groups:
            lines.append(f"【{stage}】")
            for _, step in stage_df.iterrows():
                title = step.get("step_title", "")
                if str(title).strip() == "根節點":
                    continue
                sort_order = step.get("sort_order", "")
                branch_label = step.get("branch_label", "")
                prefix = f"{int(sort_order)}." if pd.notna(sort_order) else "-"
                if pd.notna(branch_label) and str(branch_label).strip() not in ["", "nan", "None"]:
                    prefix += f" [{branch_label}]"

                lines.append(f"{prefix} {title}：{step.get('step_content', '')}")
                if pd.notna(step.get("check_method_label", None)):
                    lines.append(f"   - 判斷方式：{step.get('check_method_label')}")
                elif pd.notna(step.get("check_type", None)):
                    lines.append(f"   - 判斷方式：{step.get('check_type')}")
                if pd.notna(step.get("monitor_name", None)):
                    lines.append(f"   - 監控項目：{step.get('monitor_name')}")
                if pd.notna(step.get("standard_text", None)):
                    lines.append(f"   - 判斷標準：{step.get('standard_text')}")
                if pd.notna(step.get("evidence_required", None)):
                    lines.append(f"   - 需留存佐證：{step.get('evidence_required')}")
                if pd.notna(step.get("next_action_hint", None)):
                    lines.append(f"   - 異常時建議：{step.get('next_action_hint')}")
                if pd.notna(step.get("safety_note", None)):
                    lines.append(f"   - 安全提醒：{step.get('safety_note')}")
            lines.append("")

    return {
        "agent": "SOP Agent",
        "status": "ok",
        "text": "\n".join(lines),
        "sop_count": sop_count,
        "sop_df": sop_df
    }


# =========================================================
# Agent 3：Parts Agent
# =========================================================

def parts_agent(equipment_code, state_name):
    events = get_matched_events(equipment_code, state_name)
    sop_df = get_sop_by_equipment_state(equipment_code, state_name)

    if events.empty and sop_df.empty:
        return {
            "agent": "Parts Agent",
            "status": "no_data",
            "text": format_empty_message("Parts Agent"),
            "parts": []
        }

    text_pool = []

    if not events.empty:
        for col in ["state_name", "root_cause_category", "cause_summary", "action_summary"]:
            if col in events.columns:
                text_pool.extend(events[col].dropna().astype(str).tolist())

    if not sop_df.empty:
        for col in ["step_title", "step_content", "standard_text", "safety_note"]:
            if col in sop_df.columns:
                text_pool.extend(sop_df[col].dropna().astype(str).tolist())

    all_text = " ".join(text_pool)

    part_rules = {
        "軸承": ["軸承", "bearing"],
        "軋輥": ["軋輥", "roll", "Roll", "TC Roll"],
        "感測器": ["感測器", "sensor", "訊號", "signal", "loss"],
        "噴嘴": ["噴嘴", "水量", "冷卻", "堵塞"],
        "皮帶/傳動件": ["皮帶", "傳動", "打滑", "馬達", "鏈條"],
        "油壓/潤滑相關零件": ["油壓", "潤滑", "油", "壓力"],
        "電控元件": ["電流", "電壓", "PLC", "控制", "訊號"],
        "加熱爐相關零件": ["加熱", "爐", "溫度", "燃燒"]
    }

    recommended_parts = []

    for part_name, keywords in part_rules.items():
        matched_keywords = [kw for kw in keywords if kw in all_text]

        if matched_keywords:
            recommended_parts.append({
                "part": part_name,
                "matched_keywords": matched_keywords,
                "reason": f"相關資料中出現關鍵字：{', '.join(matched_keywords)}"
            })

    lines = []
    lines.append("【Parts Agent｜備件與物料確認建議】")
    lines.append(f"查詢設備：{equipment_code}")
    lines.append(f"異常狀況：{state_name}")
    lines.append("")

    if recommended_parts:
        lines.append("建議優先確認以下備件或物料：")

        for idx, item in enumerate(recommended_parts, start=1):
            lines.append(f"{idx}. {item['part']}")
            lines.append(f"   - 推估原因：{item['reason']}")
    else:
        lines.append("目前資料中沒有明確備件線索。")
        lines.append("建議至少確認：常用消耗品、設備專用備件、感測器與安全防護用品。")

    lines.append("")
    lines.append("生成式備件推理：")
    lines.append(
        "目前系統尚未接入正式庫存表，因此 Parts Agent 先根據 SOP 內容、歷史處置摘要與異常關鍵字推估可能需要確認的備件。"
        "此結果不是直接代表一定需要更換，而是作為維修人員進行現場確認與庫存查詢的優先方向。"
    )
    lines.append("")
    lines.append("若後續取得材料庫存資料，可再改成直接查詢庫存數量、安全庫存與補料需求。")

    return {
        "agent": "Parts Agent",
        "status": "ok",
        "text": "\n".join(lines),
        "parts": recommended_parts
    }


# =========================================================
# Agent 4：Notification Agent
# =========================================================

def notification_agent(equipment_code, state_name, diagnosis_result=None, sop_result=None):
    events = get_matched_events(equipment_code, state_name)

    if events.empty:
        severity = "未判定"
        avg_downtime = None
        event_count = 0
    else:
        if "severity" in events.columns and not events["severity"].dropna().empty:
            severity = events["severity"].mode().iloc[0]
        else:
            severity = "未判定"

        if "downtime_min" in events.columns:
            avg_downtime = events["downtime_min"].mean()
        else:
            avg_downtime = None

        event_count = len(events)

    possible_causes = []

    if diagnosis_result and diagnosis_result.get("possible_causes"):
        possible_causes = [
            item["cause"] for item in diagnosis_result["possible_causes"]
        ]

    cause_text = "、".join(possible_causes) if possible_causes else "目前資料不足，需現場進一步確認"

    sop_count = 0

    if sop_result:
        sop_count = sop_result.get("sop_count", 0)

    notify_units = ["設備維修單位", "製程工程單位"]

    if str(severity).upper() == "A":
        notify_units.append("產線主管")
        notify_units.append("品質工程單位")
    elif str(severity).upper() == "B":
        notify_units.append("班長")

    if avg_downtime is not None and pd.notna(avg_downtime):
        downtime_text = f"系統比對 {event_count} 筆相似歷史事件，平均停機時間約 {avg_downtime:.1f} 分鐘。"
    else:
        downtime_text = "目前系統尚未找到足夠相似歷史事件，停機影響仍需由現場確認。"

    if sop_count > 0:
        sop_text = f"系統已找到 {sop_count} 份可能對應 SOP，建議依標準流程逐項確認。"
    else:
        sop_text = "目前未找到明確對應 SOP，建議由現場工程人員補充處理流程。"

    field_lines = []
    field_lines.append("【現場人員版｜立即處理提醒】")
    field_lines.append(f"設備 {equipment_code} 發生「{state_name}」異常。")
    field_lines.append("請先確認異常是否仍持續發生，並依 SOP 優先檢查設備狀態、關鍵參數與安全條件。")
    field_lines.append(f"初步可能原因：{cause_text}。")
    field_lines.append(sop_text)
    field_lines.append("")
    field_lines.append("建議現場先做：")
    field_lines.append("1. 確認設備是否仍處於異常狀態。")
    field_lines.append("2. 檢查感測器數值是否異常或資料是否缺漏。")
    field_lines.append("3. 依 SOP 檢查關鍵部位、製程條件與安全狀態。")
    field_lines.append("4. 處理完成後記錄原因、處置方式與復機結果。")

    field_version = "\n".join(field_lines)

    manager_lines = []
    manager_lines.append("【主管通報版｜異常影響摘要】")
    manager_lines.append(f"目前 {equipment_code} 發生「{state_name}」異常，系統依歷史資料判斷常見嚴重度為 {severity}。")
    manager_lines.append(downtime_text)
    manager_lines.append(f"初步可能原因包含：{cause_text}。")
    manager_lines.append(sop_text)
    manager_lines.append("")
    manager_lines.append("建議主管關注：")
    manager_lines.append("1. 是否造成產線停機或延誤。")
    manager_lines.append("2. 是否需要設備、製程與品質單位共同處理。")
    manager_lines.append("3. 是否為重複發生異常，需要後續改善 SOP 或保養策略。")
    manager_lines.append("")
    manager_lines.append("建議通報單位：" + "、".join(notify_units) + "。")

    manager_version = "\n".join(manager_lines)

    maintenance_lines = []
    maintenance_lines.append("【維修單位版｜設備檢查建議】")
    maintenance_lines.append(f"異常設備：{equipment_code}")
    maintenance_lines.append(f"異常狀況：{state_name}")
    maintenance_lines.append(f"可能原因方向：{cause_text}")
    maintenance_lines.append("")
    maintenance_lines.append("建議維修單位優先確認：")
    maintenance_lines.append("1. 相關機構是否磨耗、鬆動、卡滯或破損。")
    maintenance_lines.append("2. 感測器訊號、PLC / 通訊狀態是否正常。")
    maintenance_lines.append("3. SOP 中列出的關鍵檢查點是否符合標準。")
    maintenance_lines.append("4. 若需更換零件，請同步確認備件與安全庫存。")
    maintenance_lines.append("")
    maintenance_lines.append(sop_text)

    maintenance_version = "\n".join(maintenance_lines)

    human_check_items = [
        "現場異常是否仍持續發生，或已經恢復正常。",
        "感測器資料是否可信，例如是否有訊號 loss、資料缺漏或異常跳動。",
        "設備近期是否已有保養、維修、更換零件或異常紀錄。",
        "目前 SOP 是否仍符合最新產線狀態與現場作業方式。",
        "AI 產生的原因推測是否與現場人員觀察一致。"
    ]

    human_check_lines = ["【仍需人工確認事項】"]

    for idx, item in enumerate(human_check_items, start=1):
        human_check_lines.append(f"{idx}. {item}")

    human_check_text = "\n".join(human_check_lines)

    lines = []
    lines.append("【Notification Agent｜多角色通報內容】")
    lines.append("")
    lines.append("本 Agent 依據異常設備、歷史事件、嚴重度、可能原因與 SOP 狀態，生成三種不同對象的通報版本。")
    lines.append("")
    lines.append(field_version)
    lines.append("")
    lines.append(manager_version)
    lines.append("")
    lines.append(maintenance_version)
    lines.append("")
    lines.append(human_check_text)

    return {
        "agent": "Notification Agent",
        "status": "ok",
        "text": "\n".join(lines),
        "notify_units": notify_units,
        "severity": severity,
        "role_versions": {
            "現場人員版": field_version,
            "主管通報版": manager_version,
            "維修單位版": maintenance_version
        },
        "human_check_items": human_check_items,
        "human_check_text": human_check_text
    }


# =========================================================
# 多 Agent 總控
# =========================================================

def multi_agent_diagnosis(equipment_code, state_name):
    diagnosis = diagnosis_agent(equipment_code, state_name)
    sop = sop_agent(equipment_code, state_name)
    human = human_assistance_agent(equipment_code, state_name)
    parts = parts_agent(equipment_code, state_name)
    notification = notification_agent(
        equipment_code,
        state_name,
        diagnosis_result=diagnosis,
        sop_result=sop
    )

    return {
        "diagnosis": diagnosis,
        "sop": sop,
        "human": human,
        "parts": parts,
        "notification": notification
    }


def final_demo_generative_summary(equipment_code, state_name):
    result = multi_agent_diagnosis(equipment_code, state_name)

    diagnosis = result["diagnosis"]
    sop = result["sop"]
    human = result["human"]
    parts = result["parts"]
    notification = result["notification"]

    if diagnosis["status"] == "ok" and diagnosis.get("possible_causes"):
        cause_text = "、".join([item["cause"] for item in diagnosis["possible_causes"]])
        matched_event_count = diagnosis.get("matched_event_count", 0)
        avg_downtime = diagnosis.get("avg_downtime", None)
    else:
        cause_text = "目前資料不足，尚無法判定明確主因"
        matched_event_count = 0
        avg_downtime = None

    if avg_downtime is not None and pd.notna(avg_downtime):
        downtime_sentence = f"系統找到 {matched_event_count} 筆相似歷史事件，平均停機時間約 {avg_downtime:.1f} 分鐘。"
    else:
        downtime_sentence = "目前相似事件資料不足，停機時間影響仍需由現場進一步確認。"

    if sop["status"] == "ok":
        sop_sentence = f"系統找到 {sop['sop_count']} 份可能對應 SOP，並已整理為階段式處理流程。"
    else:
        sop_sentence = "目前尚未找到明確 SOP，建議後續補建此異常的標準處理流程。"

    if human["status"] == "ok":
        assist_count = len(human.get("assist_df", pd.DataFrame()))
        human_sentence = f"系統另外找到 {assist_count} 項與此異常相關的人工輔助檢查 / 處置建議。"
    else:
        human_sentence = "目前缺少專屬人工輔助資料，建議將本次現場處置經驗回填成新案例。"

    if parts["status"] == "ok" and parts.get("parts"):
        part_text = "、".join([item["part"] for item in parts["parts"]])
    else:
        part_text = "目前沒有明確備件線索，需由維修單位依現場狀況確認"

    notify_units = notification.get("notify_units", [])
    notify_text = "、".join(notify_units) if notify_units else "設備維修單位、製程工程單位"

    lines = []
    lines.append("【AI 生成式分析摘要】")
    lines.append("")
    lines.append(
        f"系統針對設備「{equipment_code}」發生「{state_name}」的情境，"
        "整合歷史異常事件、SOP 流程、感測器資料、人工輔助知識與原因分類後，產生以下診斷建議。"
    )
    lines.append("")
    lines.append(f"從歷史資料來看，此類異常較可能與「{cause_text}」有關。{downtime_sentence}")
    lines.append("")
    lines.append(
        f"在處理流程上，{sop_sentence}建議現場人員依 SOP 樹狀流程先確認異常是否仍持續，"
        "再依原因排查、處置修正、復機確認與紀錄回饋的順序處理。"
    )
    lines.append("")
    lines.append(
        f"人工輔助方面，{human_sentence}人工輔助內容會依目前選擇的設備與異常狀況動態篩選，"
        "不會把所有異常都套用同一套確認事項。"
    )
    lines.append("")
    lines.append(
        f"備件與物料方面，系統推估可優先確認：{part_text}。"
        "若後續能加入正式庫存資料，系統可進一步提供庫存數量、安全庫存與是否需補料的建議。"
    )
    lines.append("")
    lines.append(
        f"通報方面，建議至少通知：{notify_text}。"
        "若事件嚴重度較高或已造成停機，應同步通知主管與相關支援單位，以降低處理延誤。"
    )
    lines.append("")
    lines.append(
        "補充說明：本摘要屬於生成式 AI 輔助判斷，主要用於協助現場快速整理線索，"
        "不應取代工程師的現場判斷與安全確認。"
    )

    return "\n".join(lines), result




# =========================================================
# SOP 互動呈現輔助函式
# =========================================================

def normalize_branch_label(label, default="下一步"):
    text = _clean_text(label)
    if text in ["", "None", "nan"]:
        return default
    return text


def suggest_missing_branch_label(existing_label):
    """依現有分支補上常見的相反分支，避免互動流程只有單一路徑。"""
    label = normalize_branch_label(existing_label, default="").replace(" ", "")
    mapping = {
        "是": "否 / 未完成",
        "否": "是 / 已確認",
        "正常": "異常 / 未完成",
        "異常": "正常 / 已排除",
        "完成": "未完成 / 需升級",
        "未完成": "完成 / 可結案",
        "可修復": "不可修復 / 需升級",
        "不可修復": "可修復 / 完成處置",
        "可處理": "不可處理 / 需升級",
        "不可處理": "可處理 / 依 SOP 處置",
        "可復機": "不可復機 / 需升級",
        "不可復機": "可復機 / 結案回饋",
    }
    if label in mapping:
        return mapping[label]
    if "是" in label:
        return "否 / 未完成"
    if "否" in label:
        return "是 / 已確認"
    if "正常" in label:
        return "異常 / 未完成"
    if "異常" in label:
        return "正常 / 已排除"
    if "完成" in label or "可" in label:
        return "未完成 / 需升級"
    return "未完成 / 需人工判斷"


def make_virtual_terminal_step(parent_row, missing_label, state_name=""):
    """建立補齊用的終點節點：代表此分支尚未完成、無法確認或需要升級。"""
    parent_id = _clean_text(parent_row.get("_node_id")) or _clean_text(parent_row.get("step_id")) or "PARENT"
    virtual_id = f"VIRTUAL_{parent_id}_{abs(hash(missing_label)) % 100000}"
    parent_title = _clean_text(parent_row.get("step_title"))
    parent_content = _clean_text(parent_row.get("step_content"))

    virtual = dict(parent_row)
    virtual.update({
        "step_id": virtual_id,
        "_node_id": virtual_id,
        "parent_step_id": parent_id,
        "_parent_node_id": parent_id,
        "branch_label": missing_label,
        "sort_order": 999,
        "step_title": "未完成 / 需升級處理",
        "step_content": f"此分支代表「{parent_title}」尚未完成、條件不符合或現場無法確認。請先保留現場紀錄，再依終點分析結果處理。",
        "sop_stage": "5. 復機確認 / 升級處理",
        "check_type": "manual",
        "check_method_label": "人工確認",
        "monitor_name": "無，需由現場人員確認",
        "standard_text": "確認此步驟尚未完成、無法判定或不符合預期。",
        "evidence_required": "現場照片 / 人工確認備註 / 異常截圖 / 通報紀錄",
        "safety_note": "若有安全疑慮或設備風險，先停機並通知班長。",
        "next_action_hint": "若此分支被選到，代表流程未能正常完成；請通知班長，並視情況升級給設備工程師或製程工程師。",
        "_virtual_terminal": True,
        "_virtual_reason": f"原始 SOP 只有單一分支，系統補上「{missing_label}」作為未完成或例外處理路徑。",
    })
    return pd.Series(virtual)


def prepare_sop_nodes(sop_group):
    """將 SOP 資料轉成便於樹狀圖與逐步導覽使用的節點結構。
    會自動補齊常見缺漏分支，例如只有「是」時補上「否 / 未完成」。
    """
    df = sop_group.copy().reset_index(drop=True)

    if "sop_stage" not in df.columns:
        df["sop_stage"] = df.apply(
            lambda row: classify_sop_stage(row.get("step_title", ""), row.get("step_content", "")),
            axis=1
        )

    if "check_method_label" not in df.columns:
        def _check_label(x):
            x = _clean_text(x).lower()
            if x == "auto":
                return "系統自動判斷"
            if x == "hybrid":
                return "系統輔助 + 人工確認"
            return "人工確認"
        df["check_method_label"] = df.get("check_type", pd.Series([""] * len(df))).apply(_check_label)

    df = df.sort_values(["sort_order", "step_id"], na_position="last").reset_index(drop=True)

    node_ids = []
    raw_to_node = {}
    for idx, row in df.iterrows():
        raw_id = _clean_text(row.get("step_id"))
        node_id = raw_id if raw_id else f"STEP_AUTO_{idx}"
        node_ids.append(node_id)
        if raw_id:
            raw_to_node[raw_id] = node_id
    df["_node_id"] = node_ids

    parent_ids = []
    for idx, row in df.iterrows():
        parent_raw = _clean_text(row.get("parent_step_id"))
        parent_ids.append(raw_to_node.get(parent_raw, ""))
    df["_parent_node_id"] = parent_ids

    step_map = {row["_node_id"]: row for _, row in df.iterrows()}
    children_map = {node_id: [] for node_id in df["_node_id"]}
    roots = []

    for _, row in df.iterrows():
        node_id = row["_node_id"]
        parent_id = row["_parent_node_id"]
        if parent_id and parent_id in children_map and parent_id != node_id:
            children_map[parent_id].append(row)
        else:
            roots.append(node_id)

    if not roots and len(df) > 0:
        roots = [df.iloc[0]["_node_id"]]

    for parent_id in list(children_map.keys()):
        children_map[parent_id] = sorted(
            children_map[parent_id],
            key=lambda r: (999999 if pd.isna(r.get("sort_order")) else r.get("sort_order"), _clean_text(r.get("_node_id")))
        )

    # 補齊只有單一路徑的節點，讓互動流程可以處理「未完成 / 例外 / 需升級」
    for parent_id in list(children_map.keys()):
        children = children_map.get(parent_id, [])
        if len(children) == 1 and parent_id in step_map:
            current_title = _clean_text(step_map[parent_id].get("step_title"))
            if current_title != "根節點":
                existing_label = normalize_branch_label(children[0].get("branch_label"), default="下一步")
                missing_label = suggest_missing_branch_label(existing_label)
                virtual = make_virtual_terminal_step(step_map[parent_id], missing_label)
                vid = virtual.get("_node_id")
                step_map[vid] = virtual
                children_map.setdefault(vid, [])
                children_map[parent_id].append(virtual)

    return df, step_map, children_map, roots


def find_related_human_assistance(step_row, assist_df, state_name="", max_items=3):
    if assist_df is None or assist_df.empty:
        return pd.DataFrame(columns=HUMAN_ASSIST_COLS)

    title = _clean_text(step_row.get("step_title"))
    content = _clean_text(step_row.get("step_content"))
    stage = _clean_text(step_row.get("sop_stage"))

    candidates = assist_df.copy()

    def _score(row):
        score = 0
        for field in ["check_item", "human_action", "assist_type", "source_note"]:
            text = _clean_text(row.get(field))
            if title and title in text:
                score += 90
            if content and content[:12] and content[:12] in text:
                score += 25
            if stage and stage in text:
                score += 10
        if state_name and _clean_text(row.get("state_keyword")) == _clean_text(state_name):
            score += 15
        return score

    candidates["_score"] = candidates.apply(_score, axis=1)
    matched = candidates[candidates["_score"] > 0].sort_values("_score", ascending=False)

    if matched.empty:
        matched = candidates.head(max_items)
    else:
        matched = matched.drop(columns=["_score"], errors="ignore").head(max_items)

    return matched


def render_human_assistance_cards(assist_df, section_title="Human Assistance Agent 建議"):
    if assist_df is None or assist_df.empty:
        st.info("目前沒有對應的人工輔助建議。")
        return

    st.markdown(f"**{section_title}**")

    for idx, row in assist_df.reset_index(drop=True).iterrows():
        with st.container(border=True):
            st.markdown(f"##### 建議 {idx + 1}｜{row.get('assist_type', '人工輔助')}")
            st.write(f"**人工需確認：** {row.get('check_item', '依現場狀況確認')}")
            st.write(f"**建議人工處置：** {row.get('human_action', '依班長或工程師指示處理')}")
            st.write(f"**建議留下佐證：** {row.get('evidence_required', '人工確認紀錄')}")
            st.write(f"**升級通報條件：** {row.get('escalation_rule', '若處置後仍未改善，需升級通報')}")
            source_note = row.get("source_note", row.get("source_table", "原始資料"))
            if pd.notna(source_note) and _clean_text(source_note):
                st.caption(f"資料來源：{source_note}")


def render_step_detail_card(step, equipment_code, state_name, assist_df=None, prefix_text=None, show_assist=True):
    title = _clean_text(step.get("step_title"))
    if title == "根節點":
        return

    sort_order = step.get("sort_order", "")
    branch_label = normalize_branch_label(step.get("branch_label"), default="")

    if prefix_text:
        header = prefix_text
    else:
        if bool(step.get("_virtual_terminal", False)):
            header = "例外分支"
        else:
            header = f"步驟 {int(sort_order)}" if pd.notna(sort_order) else "步驟"
        if branch_label:
            header += f"｜分支：{branch_label}"

    st.markdown(f"#### {header}：{title}")
    st.write(step.get("step_content", ""))

    if bool(step.get("_virtual_terminal", False)):
        st.warning(step.get("_virtual_reason", "此節點為系統補齊的未完成 / 例外處理分支。"))

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**判斷方式**")
        check_label = step.get("check_method_label", None)
        check_type = _clean_text(step.get("check_type")).lower()
        if pd.notna(check_label) and _clean_text(check_label):
            st.info(check_label)
        elif check_type == "auto":
            st.success("系統自動判斷")
        elif check_type == "hybrid":
            st.warning("系統輔助 + 人工確認")
        else:
            st.info("人工確認")

        st.markdown("**監控項目**")
        if pd.notna(step.get("monitor_name", None)) and _clean_text(step.get("monitor_name")):
            st.write(step.get("monitor_name"))
        else:
            st.write("無，需由現場人員確認")

        st.markdown("**建議佐證資料**")
        if pd.notna(step.get("evidence_required", None)) and _clean_text(step.get("evidence_required")):
            st.write(step.get("evidence_required"))
        else:
            st.write("人工確認備註")

    with c2:
        st.markdown("**判斷標準**")
        if pd.notna(step.get("standard_text", None)) and _clean_text(step.get("standard_text")):
            st.write(step.get("standard_text"))
        else:
            st.write("目前未提供標準，需依現場經驗或主管指示")

        st.markdown("**安全提醒**")
        if pd.notna(step.get("safety_note", None)) and _clean_text(step.get("safety_note")):
            st.warning(step.get("safety_note"))
        else:
            st.write("無特別安全提醒")

        st.markdown("**異常時下一步建議**")
        if pd.notna(step.get("next_action_hint", None)) and _clean_text(step.get("next_action_hint")):
            st.write(step.get("next_action_hint"))
        else:
            st.write(_row_escalation(step))

    if show_assist and assist_df is not None:
        related_assist = find_related_human_assistance(step, assist_df, state_name=state_name, max_items=2)
        render_human_assistance_cards(related_assist, section_title="對應人工輔助建議")


def render_version_one(group, equipment_code, state_name):
    overall_assist_df = get_human_assistance(equipment_code, state_name, max_items=8)

    st.markdown("### 版本一：樹狀架構＋詳細內容卡片")
    st.caption("先用樹狀圖看整體分支，再往下看每一步的詳細內容與對應人工輔助建議。")

    dot = build_sop_tree_dot(group)
    if dot:
        st.graphviz_chart(dot, use_container_width=True)
    else:
        st.warning("此 SOP 缺少 step_id / parent_step_id，無法繪製樹狀圖。")

    st.markdown("---")
    st.markdown("### 步驟詳細說明")

    step_df, _, _, _ = prepare_sop_nodes(group)
    step_df = step_df[step_df["step_title"].astype(str).str.strip() != "根節點"]

    for stage, stage_df in step_df.groupby("sop_stage", sort=True):
        st.markdown(f"## {stage}")
        for _, step in stage_df.iterrows():
            with st.container(border=True):
                render_step_detail_card(step, equipment_code, state_name, assist_df=overall_assist_df)

    st.markdown("---")
    st.markdown("### 全部人工輔助建議總覽")
    render_human_assistance_cards(overall_assist_df, section_title="此異常狀況的人工輔助建議")


def render_version_two(group, equipment_code, state_name, sop_id):
    st.markdown("### 逐步 SOP 導覽")
    st.caption("先確認當前問題，再依『是 / 否 / 正常 / 異常 / 完成 / 未完成』等分支往下走；只有走到沒有下一步時，才顯示分析結果與人工輔助建議。")

    step_df, step_map, children_map, roots = prepare_sop_nodes(group)
    if len(step_df) == 0:
        st.warning("此 SOP 沒有可用步驟。")
        return

    visible_roots = [r for r in roots if _clean_text(step_map[r].get("step_title")) != "根節點"]
    if visible_roots:
        root_id = visible_roots[0]
    elif roots and children_map.get(roots[0]):
        root_id = children_map[roots[0]][0].get("_node_id")
    else:
        root_id = roots[0]

    state_key = f"wizard_{sop_id}_current"
    path_key = f"wizard_{sop_id}_path"

    if state_key not in st.session_state:
        st.session_state[state_key] = root_id
    if path_key not in st.session_state:
        st.session_state[path_key] = [root_id]

    current_id = st.session_state[state_key]
    if current_id not in step_map:
        st.session_state[state_key] = root_id
        st.session_state[path_key] = [root_id]
        current_id = root_id

    current_step = step_map[current_id]
    overall_assist_df = get_human_assistance(equipment_code, state_name, max_items=8)
    path_ids = st.session_state[path_key]
    children = children_map.get(current_id, [])

    st.info(f"目前確認項目：{_clean_text(current_step.get('step_title'))}")

    # 先呈現問題確認 / 目前步驟，不先丟分析結果，避免還沒走完流程就給結論
    with st.container(border=True):
        render_step_detail_card(
            current_step,
            equipment_code,
            state_name,
            assist_df=None,
            prefix_text="問題確認 / 目前步驟" if len(path_ids) == 1 else None,
            show_assist=False
        )

    path_titles = [
        _clean_text(step_map[x].get("step_title"))
        for x in path_ids if x in step_map and _clean_text(step_map[x].get("step_title")) != "根節點"
    ]
    if len(path_titles) > 1:
        st.caption("目前路徑：" + " → ".join(path_titles))

    control_col1, control_col2 = st.columns([1, 1])
    with control_col1:
        if st.button("🔄 重新開始", key=f"restart_{sop_id}"):
            st.session_state[state_key] = root_id
            st.session_state[path_key] = [root_id]
            st.rerun()
    with control_col2:
        if len(path_ids) > 1:
            if st.button("⬅️ 回上一步", key=f"back_{sop_id}"):
                new_path = path_ids[:-1]
                st.session_state[path_key] = new_path
                st.session_state[state_key] = new_path[-1] if new_path else root_id
                st.rerun()

    st.markdown("---")
    if children:
        st.markdown("#### 請選擇下一步分支")
        cols = st.columns(min(3, max(1, len(children))))
        for idx, child in enumerate(children):
            branch = normalize_branch_label(child.get("branch_label"), default="下一步")
            next_title = _clean_text(child.get("step_title"))
            target_id = child.get("_node_id")
            if bool(child.get("_virtual_terminal", False)):
                btn_label = f"{branch}"
            else:
                btn_label = f"{branch} → {next_title}"
            if cols[idx % len(cols)].button(btn_label, key=f"goto_{sop_id}_{target_id}"):
                st.session_state[state_key] = target_id
                st.session_state[path_key] = path_ids + [target_id]
                st.rerun()
    else:
        st.success("此流程已走到終點，以下顯示分析結果。")
        st.markdown("### 分析結果")

        st.markdown("#### 1. 終點判斷")
        if pd.notna(current_step.get("next_action_hint", None)) and _clean_text(current_step.get("next_action_hint")):
            st.write(current_step.get("next_action_hint"))
        else:
            st.write(_row_escalation(current_step))

        st.markdown("#### 2. 對應人工輔助建議")
        related_assist = find_related_human_assistance(current_step, overall_assist_df, state_name=state_name, max_items=4)
        render_human_assistance_cards(related_assist, section_title="Human Assistance Agent 建議")

        path_rows = [step_map[x] for x in path_ids if x in step_map and _clean_text(step_map[x].get("step_title")) != "根節點"]
        if path_rows:
            st.markdown("#### 3. 本次判斷路徑摘要")
            for idx, step in enumerate(path_rows, start=1):
                branch = normalize_branch_label(step.get("branch_label"), default="")
                label = f"步驟 {idx}"
                if branch:
                    label += f"｜分支：{branch}"
                st.write(f"**{label}** {step.get('step_title', '')}：{step.get('step_content', '')}")

        st.markdown("#### 4. 回填建議")
        st.write("請將本次確認結果、照片 / 截圖、處置方式、是否復機與升級通報結果回填到異常事件紀錄，作為後續相似案例推薦與 SOP 改善依據。")



# =========================================================
# 加強版：多模態、互動圖譜、主動預警工具函式
# =========================================================

def _safe_text(value, default="未提供"):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in ["", "nan", "none", "nat"]:
        return default
    return text



def analyze_uploaded_photo_basic(uploaded_file):
    """
    展示版的照片基礎品質分析。
    這不是 Vision LLM，也不會判斷照片中的零件是什麼；
    但會依實際上傳圖片的尺寸、亮度、對比與清晰度產生不同提醒，
    避免「上傳任何照片結果都一樣」。
    """
    if uploaded_file is None:
        return {
            "has_photo": False,
            "filename": "未上傳照片",
            "width": None,
            "height": None,
            "brightness": None,
            "contrast": None,
            "sharpness": None,
            "quality_level": "未上傳",
            "quality_notes": [
                "尚未上傳照片，因此系統只能依設備、異常狀況、SOP 與歷史案例給出檢查方向。"
            ]
        }

    filename = getattr(uploaded_file, "name", "未命名照片")
    try:
        from PIL import Image
        import numpy as np

        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert("RGB")
        arr = np.asarray(img).astype("float32")
        gray = arr.mean(axis=2)

        width, height = img.size
        brightness = float(gray.mean())
        contrast = float(gray.std())
        # 不使用 OpenCV，改用 numpy gradient 做簡易清晰度估計。
        gy, gx = np.gradient(gray)
        sharpness = float((gx ** 2 + gy ** 2).mean())

        notes = []
        score = 0

        if width < 600 or height < 400:
            notes.append("照片解析度偏低，建議補拍近照或使用較高解析度照片。")
        else:
            score += 1
            notes.append("照片解析度足夠，可作為現場佐證。")

        if brightness < 55:
            notes.append("照片偏暗，可能看不清楚裂痕、磨耗或漏油位置。")
        elif brightness > 210:
            notes.append("照片偏亮，可能過曝而看不清楚表面細節。")
        else:
            score += 1
            notes.append("照片亮度大致正常。")

        if contrast < 22:
            notes.append("照片對比偏低，建議補拍有明確光線與角度的照片。")
        else:
            score += 1
            notes.append("照片對比尚可，有助於辨識邊界與異常區域。")

        if sharpness < 18:
            notes.append("照片可能模糊，建議補拍異常部位近照。")
        else:
            score += 1
            notes.append("照片清晰度尚可，可支援後續人工或 Vision 模型判讀。")

        if score >= 4:
            level = "高"
        elif score >= 2:
            level = "中"
        else:
            level = "低"

        uploaded_file.seek(0)
        return {
            "has_photo": True,
            "filename": filename,
            "width": width,
            "height": height,
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "quality_level": level,
            "quality_notes": notes
        }
    except Exception as exc:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        return {
            "has_photo": True,
            "filename": filename,
            "width": None,
            "height": None,
            "brightness": None,
            "contrast": None,
            "sharpness": None,
            "quality_level": "無法判讀",
            "quality_notes": [f"系統無法讀取此圖片的基礎資訊，請確認檔案格式是否正確。錯誤訊息：{exc}"]
        }


def infer_visual_focus_from_context(filename, equipment_code, state_name):
    """
    依檔名、設備與異常狀況推論「應該優先看照片哪裡」。
    這是展示版的情境推論，不是影像內容辨識。
    """
    filename_text = _safe_text(filename, "")
    equip_text = _safe_text(equipment_code, "未指定設備")
    state_text = _safe_text(state_name, "未指定異常")
    combined = f"{filename_text} {equip_text} {state_text}".lower()

    rules = [
        (
            ["斷", "裂", "break", "broken", "crack", "lever", "軸承破裂", "破裂"],
            "破損 / 斷裂類",
            [
                "優先看零件是否有裂痕、斷裂、變形、固定點鬆脫。",
                "補拍破損位置近照，並盡量拍到設備編號或相對位置。",
                "確認是否需要停機隔離，避免二次損壞或安全風險。"
            ]
        ),
        (
            ["磨", "耗", "wear", "knife", "切刀", "roll", "roller", "軋輥", "刮傷"],
            "磨耗 / 表面劣化類",
            [
                "優先看接觸面是否有磨耗、刮傷、缺角、表面剝落。",
                "建議補拍正常側與異常側對比照片，方便工程師判斷磨耗程度。",
                "同步確認更換週期、材料條件與近期是否重複發生。"
            ]
        ),
        (
            ["漏", "油", "water", "水", "leak", "液", "壓力", "流量"],
            "漏液 / 水油壓異常類",
            [
                "優先看現場是否有漏油、漏水、液體殘留或噴濺痕跡。",
                "補拍管線接頭、閥件、地面殘留與壓力表位置。",
                "同步檢查 sensor snapshot 中的水量、壓力、流量或溫度是否異常。"
            ]
        ),
        (
            ["偏", "卡", "cobble", "jam", "塞", "misalign", "打滑", "蛇行", "偏移"],
            "材料偏移 / 卡料 / 打滑類",
            [
                "優先看材料是否偏移、卡滯、打滑，或導引位置是否異常。",
                "補拍入口、出口、導板、夾輥與材料相對位置。",
                "同步確認加熱條件、速度設定與夾輥作動是否符合 SOP。"
            ]
        ),
        (
            ["異音", "vibration", "震", "振", "bearing", "軸承", "馬達"],
            "振動 / 異音 / 傳動類",
            [
                "照片本身不一定能判斷異音，建議搭配影片、聲音描述或振動數值。",
                "優先拍攝軸承座、馬達、聯軸器、固定螺絲與潤滑位置。",
                "同步檢查 sensor snapshot 中的振動、溫度與電流變化。"
            ]
        ),
    ]

    for keywords, label, hints in rules:
        if any(k in combined for k in keywords):
            return label, hints

    return "一般現場佐證類", [
        "請確認照片是否拍到異常部位、設備編號與周邊環境。",
        "建議至少補齊全景照、異常部位近照、正常/異常對比照。",
        "若照片無法直接說明原因，應以 SOP、sensor snapshot 與班長確認紀錄一起判斷。"
    ]


def generate_vision_agent_report(uploaded_file, equipment_code, state_name, related_cases=None, sop_df=None):
    """
    展示版 Vision Agent：
    不宣稱真正辨識照片內容；但會依「實際上傳照片品質」以及「檔名/設備/異常情境」產生不同檢查建議。
    未來可將 image_observation 區塊替換成 GPT-4o / Gemini Vision 的影像辨識結果。
    """
    photo_info = analyze_uploaded_photo_basic(uploaded_file)
    filename = photo_info["filename"]
    has_photo = photo_info["has_photo"]
    state_text = _safe_text(state_name, "未指定異常")
    equip_text = _safe_text(equipment_code, "未指定設備")
    focus_label, photo_clues = infer_visual_focus_from_context(filename, equipment_code, state_name)

    # SOP：優先抓需要照片/人工佐證的節點，避免文字爆量與重複。
    sop_hints = []
    if sop_df is not None and not sop_df.empty:
        temp = sop_df.copy()
        if "image_needed" in temp.columns:
            img_temp = temp[temp["image_needed"].astype(str).str.lower().isin(["y", "yes", "true", "1", "需要", "是"])]
            if not img_temp.empty:
                temp = img_temp

        for _, row in temp.head(3).iterrows():
            title = _safe_text(row.get("step_title"), "SOP 檢查節點")
            content = _safe_text(row.get("step_content"), "")
            evidence = _safe_text(row.get("evidence_required"), "照片、量測值或班長確認紀錄")
            owner = _safe_text(row.get("owner_role"), "現場人員")
            if content and content != "未提供":
                sop_hints.append(f"- {title}：{content}｜負責：{owner}｜建議佐證：{evidence}")
            else:
                sop_hints.append(f"- {title}｜負責：{owner}｜建議佐證：{evidence}")

    if not sop_hints:
        sop_hints = [
            "- 目前未找到對應 SOP 節點，建議先由班長確認異常部位並建立補充紀錄。",
            "- 後續可將本次照片、處置結果與實際原因回填，作為 SOP 改善資料。"
        ]

    # 歷史案例：只放最相近一筆，不重複展開過多案例。
    case_lines = []
    if related_cases is not None and not related_cases.empty:
        top_case = related_cases.iloc[0]
        case_lines.append(f"- 參考案例：{_safe_text(top_case.get('event_id'), '未知事件')}")
        case_lines.append(f"- 曾記錄原因：{_safe_text(top_case.get('cause_summary'), '未記錄')}")
        case_lines.append(f"- 曾採取處置：{_safe_text(top_case.get('action_summary'), '未記錄')}")
        if "downtime_min" in top_case.index:
            case_lines.append(f"- 該案例停機時間：{_safe_text(top_case.get('downtime_min'), '未記錄')} 分鐘")
    else:
        case_lines = ["- 目前沒有足夠相似案例可引用，建議以 SOP 與現場人工判斷為主。"]

    metric_lines = []
    if has_photo and photo_info.get("width"):
        metric_lines = [
            f"- 圖片尺寸：{photo_info['width']} × {photo_info['height']}",
            f"- 基礎品質等級：{photo_info['quality_level']}（依解析度、亮度、對比、清晰度估計）",
        ]
    else:
        metric_lines = [f"- 基礎品質等級：{photo_info['quality_level']}"]

    lines = [
        "【Vision Agent 綜合診斷摘要｜展示版】",
        "",
        "0. 目前定位",
        "- 這不是正式影像辨識模型，而是照片輔助診斷流程展示。",
        "- 本版會依實際照片品質、照片檔名、設備、異常狀況、SOP 與歷史案例產生不同建議。",
        "- 未來串接 GPT-4o / Gemini Vision 後，可把照片內容轉成結構化觀察，再交給 Agent 綜合判斷。",
        "",
        "1. 輸入資訊",
        f"- 照片狀態：{'已上傳' if has_photo else '未上傳'}",
        f"- 照片檔名：{filename}",
        f"- 設備：{equip_text}",
        f"- 異常狀況：{state_text}",
        f"- 情境推論類型：{focus_label}",
        *metric_lines,
        "",
        "2. 照片品質提醒",
        *[f"- {item}" for item in photo_info.get("quality_notes", [])],
        "",
        "3. 優先檢查方向",
        *[f"- {item}" for item in photo_clues],
        "",
        "4. 對應 SOP 確認項目",
        *sop_hints,
        "",
        "5. 歷史案例參考",
        *case_lines,
        "",
        "6. 建議處置順序",
        "1. 先判斷照片品質是否足夠；若偏暗、模糊或只拍局部，先補拍。",
        "2. 依情境推論類型確認重點部位，例如破損、磨耗、漏液、卡料、偏移或振動相關位置。",
        "3. 對照 SOP 節點，確認照片是否能支撐該步驟判斷。",
        "4. 同步保存 sensor snapshot、人工備註與處置結果，避免只靠照片下結論。",
        "5. 處理完成後回填實際原因，作為相似案例推薦與 SOP 改善資料。"
    ]
    return "\n".join(lines)


def _graph_columns(edges_df):
    src_col = "source" if "source" in edges_df.columns else edges_df.columns[0]
    tgt_col = "target" if "target" in edges_df.columns else edges_df.columns[1]
    rel_col = "relation" if "relation" in edges_df.columns else ("label" if "label" in edges_df.columns else None)
    return src_col, tgt_col, rel_col


def filter_graph_edges(edges_df, selected_node=None, relation_filter=None, max_edges=90):
    """把圖譜資料篩成適合展示的子圖，避免整張圖太大或空白。"""
    if edges_df is None or edges_df.empty:
        return pd.DataFrame()
    edf = edges_df.copy()
    src_col, tgt_col, rel_col = _graph_columns(edf)

    if selected_node and selected_node != "全部":
        selected = str(selected_node)
        mask = (edf[src_col].astype(str).str.contains(selected, case=False, na=False)) | (edf[tgt_col].astype(str).str.contains(selected, case=False, na=False))
        edf = edf[mask]

    if relation_filter and relation_filter != "全部" and rel_col:
        edf = edf[edf[rel_col].astype(str) == relation_filter]

    return edf.head(max_edges).copy()


def build_plotly_knowledge_graph(edges_df, nodes_df=None, selected_node=None, relation_filter=None, max_edges=90):
    """用 Plotly 畫穩定可顯示的知識圖譜；不用 pyvis，避免 Streamlit iframe 空白。"""
    if edges_df is None or edges_df.empty:
        return None, pd.DataFrame()

    edf = filter_graph_edges(edges_df, selected_node=selected_node, relation_filter=relation_filter, max_edges=max_edges)
    if edf.empty:
        return None, edf

    src_col, tgt_col, rel_col = _graph_columns(edf)
    node_list = sorted(set(edf[src_col].dropna().astype(str)) | set(edf[tgt_col].dropna().astype(str)))
    if not node_list:
        return None, edf

    # 先嘗試用 networkx spring layout；如果環境沒有 networkx，就用圓形配置。
    positions = {}
    try:
        import networkx as nx
        g = nx.Graph()
        for _, row in edf.iterrows():
            g.add_edge(str(row[src_col]), str(row[tgt_col]))
        positions = nx.spring_layout(g, seed=42, k=0.65, iterations=80)
    except Exception:
        n = len(node_list)
        for i, node in enumerate(node_list):
            angle = 2 * math.pi * i / max(n, 1)
            positions[node] = (math.cos(angle), math.sin(angle))

    label_lookup = {}
    type_lookup = {}
    if nodes_df is not None and not nodes_df.empty:
        id_col = "id" if "id" in nodes_df.columns else ("node_id" if "node_id" in nodes_df.columns else nodes_df.columns[0])
        label_col = "label" if "label" in nodes_df.columns else id_col
        type_col = "type" if "type" in nodes_df.columns else ("node_type" if "node_type" in nodes_df.columns else None)
        for _, row in nodes_df.iterrows():
            nid = str(row.get(id_col))
            label_lookup[nid] = _safe_text(row.get(label_col), nid)
            if type_col:
                type_lookup[nid] = _safe_text(row.get(type_col), "node")

    edge_x, edge_y, edge_hover = [], [], []
    mid_x, mid_y, mid_text = [], [], []
    for _, row in edf.iterrows():
        src, tgt = str(row[src_col]), str(row[tgt_col])
        if src not in positions or tgt not in positions:
            continue
        x0, y0 = positions[src]
        x1, y1 = positions[tgt]
        rel = _safe_text(row.get(rel_col), "關聯") if rel_col else "關聯"
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_hover.append(f"{src} → {tgt}: {rel}")
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        mid_text.append(rel[:12])

    node_x, node_y, node_text, node_hover, node_size = [], [], [], [], []
    degree_count = {node: 0 for node in node_list}
    for _, row in edf.iterrows():
        degree_count[str(row[src_col])] = degree_count.get(str(row[src_col]), 0) + 1
        degree_count[str(row[tgt_col])] = degree_count.get(str(row[tgt_col]), 0) + 1

    for node in node_list:
        x, y = positions[node]
        nlabel = label_lookup.get(node, node)
        ntype = type_lookup.get(node, "node")
        node_x.append(x)
        node_y.append(y)
        node_text.append(nlabel[:24])
        node_hover.append(f"節點：{nlabel}<br>類型：{ntype}<br>關聯數：{degree_count.get(node, 0)}")
        node_size.append(16 + min(degree_count.get(node, 0), 12) * 2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=1.4, color="rgba(90,90,90,0.45)"),
        hoverinfo="none", name="關係"
    ))
    fig.add_trace(go.Scatter(
        x=mid_x, y=mid_y, mode="text",
        text=mid_text, textfont=dict(size=10, color="rgba(70,70,70,0.75)"),
        hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_text, textposition="top center",
        hovertext=node_hover, hoverinfo="text",
        marker=dict(size=node_size, color="#4C78A8", line=dict(width=1.5, color="#FFFFFF")),
        name="節點"
    ))
    fig.update_layout(
        title="設備—異常—SOP—歷史事件關聯圖",
        height=650,
        margin=dict(l=10, r=10, t=45, b=10),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig, edf


def build_predictive_maintenance_alerts(sensor_df, event_df=None):
    """以現有 sensor snapshot 做簡化風險排序：超出規格、接近規格、異常 judgement、近期連續出現皆加分。"""
    if sensor_df is None or sensor_df.empty:
        return pd.DataFrame()

    df = sensor_df.copy()
    for col in ["actual_value", "spec_lower", "spec_upper"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "captured_at" in df.columns:
        df["captured_at"] = pd.to_datetime(df["captured_at"], errors="coerce")

    if event_df is not None and not event_df.empty and "event_id" in df.columns and "event_id" in event_df.columns:
        join_cols = [c for c in ["event_id", "equipment_code", "equipment_name", "state_name"] if c in event_df.columns]
        df = df.merge(event_df[join_cols].drop_duplicates("event_id"), on="event_id", how="left")

    group_cols = [c for c in ["equipment_code", "equipment_name", "monitor_name", "parameter_name", "unit"] if c in df.columns]
    if not group_cols:
        group_cols = [c for c in ["monitor_name", "parameter_name"] if c in df.columns]
    if not group_cols:
        return pd.DataFrame()

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        gg = g.copy()
        if "captured_at" in gg.columns:
            gg = gg.sort_values("captured_at")
        values = gg["actual_value"] if "actual_value" in gg.columns else pd.Series(dtype="float")
        lower = pd.to_numeric(gg["spec_lower"], errors="coerce") if "spec_lower" in gg.columns else pd.Series(dtype="float")
        upper = pd.to_numeric(gg["spec_upper"], errors="coerce") if "spec_upper" in gg.columns else pd.Series(dtype="float")
        judgement_text = gg["judgement"].fillna("").astype(str) if "judgement" in gg.columns else pd.Series([""] * len(gg))

        abnormal_count = judgement_text.str.contains("異常|NG|超|HIGH|LOW|ALARM|警", case=False, regex=True, na=False).sum()
        risk_score = abnormal_count * 3
        near_count = 0
        out_count = 0
        latest_value = None
        latest_upper = None
        latest_lower = None
        trend = "資料不足"

        if values.notna().any():
            latest_value = values.dropna().iloc[-1]
            if upper.notna().any():
                latest_upper = upper.dropna().iloc[-1]
            if lower.notna().any():
                latest_lower = lower.dropna().iloc[-1]

            valid = pd.DataFrame({"value": values, "lower": lower, "upper": upper}).dropna(subset=["value"])
            if not valid.empty:
                if "upper" in valid and valid["upper"].notna().any():
                    out_count += (valid["value"] > valid["upper"]).sum()
                    denom = (valid["upper"] - valid["lower"]).replace(0, pd.NA) if valid["lower"].notna().any() else valid["upper"].abs().replace(0, pd.NA)
                    ratio_to_upper = (valid["upper"] - valid["value"]) / denom
                    near_count += ((ratio_to_upper >= 0) & (ratio_to_upper <= 0.15)).sum()
                if "lower" in valid and valid["lower"].notna().any():
                    out_count += (valid["value"] < valid["lower"]).sum()
                risk_score += int(out_count) * 5 + int(near_count) * 2

            tail = values.dropna().tail(5)
            if len(tail) >= 3:
                delta = tail.iloc[-1] - tail.iloc[0]
                if abs(delta) < 1e-9:
                    trend = "持平"
                elif delta > 0:
                    trend = "上升"
                    risk_score += 1
                else:
                    trend = "下降"

        sample_count = len(gg)
        if sample_count >= 5:
            risk_score += 1

        if risk_score >= 8:
            level = "高"
        elif risk_score >= 4:
            level = "中"
        else:
            level = "低"

        reason_parts = []
        if abnormal_count:
            reason_parts.append(f"異常判定 {abnormal_count} 次")
        if out_count:
            reason_parts.append(f"超出規格 {int(out_count)} 次")
        if near_count:
            reason_parts.append(f"接近上下限 {int(near_count)} 次")
        if trend in ["上升", "下降"]:
            reason_parts.append(f"近期趨勢{trend}")
        if not reason_parts:
            reason_parts.append("目前未看到明顯超規或異常字樣")

        rows.append({
            **meta,
            "sample_count": sample_count,
            "latest_value": latest_value,
            "spec_lower": latest_lower,
            "spec_upper": latest_upper,
            "trend": trend,
            "abnormal_count": int(abnormal_count),
            "near_limit_count": int(near_count),
            "risk_score": round(float(risk_score), 2),
            "risk_level": level,
            "risk_reason": "、".join(reason_parts),
            "suggestion": "提前安排巡檢，確認軸承、潤滑、固定件、噴嘴或感測器狀態；若再次升高，建議停機檢查。" if level in ["中", "高"] else "維持觀察，保留趨勢紀錄並確認資料是否完整。"
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    risk_order = {"高": 3, "中": 2, "低": 1}
    result["_risk_order"] = result["risk_level"].map(risk_order).fillna(0)
    return result.sort_values(["_risk_order", "risk_score", "sample_count"], ascending=[False, False, False]).drop(columns=["_risk_order"])

# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("🛠️ 製造異常診斷系統")
st.sidebar.caption("生成式 AI × GraphRAG × Multi-Agent")
st.sidebar.markdown("---")

st.sidebar.markdown("### 建議展示順序")
st.sidebar.caption("先看系統與資料概況，再進入異常處理流程，最後查看預警、改善與設備原因關聯。")

MENU_GROUPS = {
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

if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "系統總覽"

st.sidebar.markdown("#### 請選擇功能頁面")

for group_title, group_pages in MENU_GROUPS.items():
    st.sidebar.markdown(f"**{group_title}**")
    for menu_item in group_pages:
        is_selected = st.session_state["selected_page"] == menu_item
        label = f"• {menu_item}"
        if st.sidebar.button(
            label,
            key=f"nav_{menu_item}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        ):
            st.session_state["selected_page"] = menu_item
    st.sidebar.markdown("")

page = st.session_state["selected_page"]

st.sidebar.markdown("---")
st.sidebar.subheader("資料概況")

st.sidebar.metric("異常事件數", f"{len(clean_event_view):,}")

if "equipment_code" in clean_event_view.columns:
    st.sidebar.metric("涉及設備數", f"{clean_event_view['equipment_code'].nunique():,}")

st.sidebar.metric("SOP 步驟數", f"{len(clean_sop_view):,}")
st.sidebar.metric("感測器快照數", f"{len(clean_sensor_view):,}")

if graph_nodes is not None:
    st.sidebar.metric("知識節點數", f"{len(graph_nodes):,}")

if graph_edges is not None:
    st.sidebar.metric("知識關係數", f"{len(graph_edges):,}")

if "occurred_at" in clean_event_view.columns:
    valid_dates = clean_event_view["occurred_at"].dropna()
    if len(valid_dates) > 0:
        st.sidebar.caption(
            f"資料期間：{valid_dates.min().date()} ～ {valid_dates.max().date()}"
        )


# =========================================================
# Page 1：系統總覽
# =========================================================

if page == "系統總覽":
    st.title("🛠️ 生成式 AI 製造異常診斷與 SOP 查詢輔助系統")
    st.caption("把製造異常資料、SOP、感測器快照與歷史案例串成知識圖譜，讓 AI 協助現場診斷、查 SOP、找相似案例與提出改善建議。")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("異常事件數", f"{len(clean_event_view):,}")

    if "equipment_code" in clean_event_view.columns:
        k2.metric("涉及設備數", f"{clean_event_view['equipment_code'].nunique():,}")
    else:
        k2.metric("涉及設備數", "無資料")

    k3.metric("SOP 步驟數", f"{len(clean_sop_view):,}")
    k4.metric("感測器快照數", f"{len(clean_sensor_view):,}")

    st.markdown("---")

    st.success("""
    **本專題的核心價值：** 不只是做異常 Dashboard，而是讓系統在異常發生時，
    依照「設備 → 異常狀況 → SOP → 感測器 → 歷史案例 → 人工確認」的順序整理線索，
    協助現場人員更快知道要查什麼、先處理什麼、哪些地方不能只靠 AI 判斷。
    """)

    tab_intro, tab_ai, tab_flow, tab_limit = st.tabs([
        "📌 專題背景",
        "🤖 生成式 AI 重點",
        "🧭 系統流程",
        "⚠️ 限制與未來改善"
    ])

    with tab_intro:
        st.markdown("""
        ### 專題背景與系統目的

        在傳統製造現場中，異常處理資訊常分散在 Excel、SOP 文件、歷史報告與工程師經驗中。
        當設備發生異常時，現場人員往往需要花時間查找過去案例、確認 SOP 流程、比對感測資料，
        並整理通報內容，整體處理流程容易受到經驗差異與資料分散影響。

        因此，本專題將異常事件、設備、異常狀況、SOP、感測器資料與歷史案例整理為結構化知識庫，
        並結合 GraphRAG 與 Multi-Agent 架構，讓系統能根據使用者選擇的設備與異常狀況，
        自動查詢相似事件、整理 SOP 步驟、推估可能原因、提醒人工確認項目，
        並產生初步診斷摘要與通報內容。
        """)

        st.markdown("### 展示時可以主打的 4 個亮點")
        st.markdown("""
        1. **資料可讀化**：把原本代碼型資料轉成設備、異常、SOP、感測器與歷史事件的可讀資料表。
        2. **GraphRAG 查詢**：不是讓 AI 直接亂猜，而是先取回與設備、異常、SOP、歷史案例相關的資料。
        3. **Multi-Agent 分工**：Diagnosis、SOP、Human Assistance、Parts、Notification 各自負責不同任務。
        4. **SOP 持續改善**：利用歷史事件與人工確認紀錄，找出最常出錯或最需要補資料的 SOP 節點。
        """)

    with tab_ai:
        st.markdown("""
        ### 生成式 AI 應用重點

        本系統的重點不只是資料查詢，而是將查詢到的歷史案例、SOP、感測器與原因分類資料，
        轉換成現場人員可閱讀的自然語言診斷摘要。
        """)

        agent_table = pd.DataFrame([
            {"Agent": "Diagnosis Agent", "負責內容": "根據歷史事件、停機時間、原因分類與感測器快照推估可能原因"},
            {"Agent": "SOP Agent", "負責內容": "查詢設備與異常狀況對應 SOP，整理為現場可理解的處理步驟"},
            {"Agent": "Human Assistance Agent", "負責內容": "找出需要現場目視、拍照、量測或人工補充的確認項目"},
            {"Agent": "Parts Agent", "負責內容": "根據 SOP 文字與歷史處置紀錄推估可能需要確認的備件或物料"},
            {"Agent": "Notification Agent", "負責內容": "依不同閱讀對象產生現場人員版、主管版與維修單位版通報"}
        ])
        st.dataframe(agent_table, use_container_width=True, hide_index=True)

        st.info("""
        **Human Assistance Agent 的意義：** 它不是表示 AI 失敗，而是表示製造現場有些判斷本來就必須由人確認，
        例如異音、現場照片、設備外觀、鋼材狀態、安全風險與復機條件。
        """)

    with tab_flow:
        st.markdown("### 系統展示流程")

        st.code("""
異常事件發生
↓
選擇設備與異常狀況
↓
GraphRAG 取回相關資料
↓
查到 SOP / 感測器 / 歷史案例
↓
Multi-Agent 分工診斷
↓
輸出處置建議、人工確認項目、備件提醒與通報內容
↓
回填處置結果，作為相似案例推薦與 SOP 改善依據
        """)

        st.markdown("### 系統架構")

        st.code("""
原始異常資料與 SOP 文件
↓
資料清理與欄位可讀化
↓
建立設備、異常狀況、SOP、感測器與歷史事件知識庫
↓
建立知識圖譜節點與關係
↓
依使用者輸入進行 GraphRAG 關聯查詢
↓
由多個 Agent 分工完成診斷、SOP、人工確認、備件與通報任務
↓
整合產生生成式 AI 診斷摘要與角色化通報內容
↓
透過 Streamlit UI 提供互動式查詢與展示
        """)

    with tab_limit:
        st.markdown("""
        ### 系統限制

        1. 部分資料屬於模擬或整理後資料，仍需更多真實事件驗證。
        2. AI 診斷目前是輔助整理線索，不能取代工程師與現場主管的判斷。
        3. 感測器判斷依賴資料完整性，若缺少即時資料仍需人工補充。
        4. 相似案例推薦目前以設備、異常狀況、原因與處置文字為主，未來可加入向量檢索與權重排序。
        5. 未來可把 SOP 改善建議回寫資料庫，形成持續學習機制。
        """)



# =========================================================
# Page 2：設備原因查詢
# =========================================================

elif page == "設備原因查詢":
    st.title("🔎 設備原因查詢")
    st.caption("用下拉選單直接查看某設備常見異常、可能原因與歷史處置，不需要使用者自己從表格中找線索。")

    st.info("""
    這頁保留知識圖譜的概念，但改成更實用的查詢方式：
    使用者先選設備，再看該設備過去常見的異常狀況、原因分類、停機時間與處置摘要。
    這比直接顯示一大張空白或難讀的圖更適合期末展示，也比較接近現場人員真正會使用的方式。
    """)

    if clean_event_view.empty or "equipment_code" not in clean_event_view.columns:
        st.warning("目前沒有異常事件資料，無法建立設備原因查詢。")
    else:
        eq_options = sorted(clean_event_view["equipment_code"].dropna().astype(str).unique().tolist())
        selected_eq = st.selectbox("選擇設備", eq_options, key="cause_lookup_equipment")

        eq_events = clean_event_view[clean_event_view["equipment_code"].astype(str) == selected_eq].copy()

        state_options = ["全部"]
        if "state_name" in eq_events.columns:
            state_options += sorted(eq_events["state_name"].dropna().astype(str).unique().tolist())
        selected_state_for_cause = st.selectbox("異常狀況", state_options, key="cause_lookup_state")

        if selected_state_for_cause != "全部" and "state_name" in eq_events.columns:
            eq_events = eq_events[eq_events["state_name"].astype(str) == selected_state_for_cause]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("歷史事件數", f"{len(eq_events):,}")
        if "downtime_min" in eq_events.columns and not eq_events["downtime_min"].dropna().empty:
            c2.metric("平均停機分鐘", f"{eq_events['downtime_min'].mean():.1f}")
            c3.metric("最高停機分鐘", f"{eq_events['downtime_min'].max():.0f}")
        else:
            c2.metric("平均停機分鐘", "無資料")
            c3.metric("最高停機分鐘", "無資料")
        if "state_name" in eq_events.columns and not eq_events.empty:
            most_state = eq_events["state_name"].value_counts().index[0]
            c4.metric("最常見異常", most_state)
        else:
            c4.metric("最常見異常", "無資料")

        if eq_events.empty:
            st.warning("目前這個篩選條件下沒有歷史事件。")
        else:
            tab_state, tab_cause, tab_action, tab_events = st.tabs([
                "常見異常",
                "原因整理",
                "處置整理",
                "歷史事件明細",
            ])

            with tab_state:
                if "state_name" in eq_events.columns:
                    state_count = eq_events["state_name"].fillna("未標記").value_counts().reset_index()
                    state_count.columns = ["異常狀況", "事件數"]
                    col_l, col_r = st.columns([1, 1.3])
                    with col_l:
                        st.dataframe(state_count, use_container_width=True, hide_index=True)
                    with col_r:
                        fig = px.bar(state_count.head(10), x="事件數", y="異常狀況", orientation="h", title=f"{selected_eq} 常見異常 Top 10")
                        fig.update_layout(yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("事件資料中沒有 state_name 欄位。")

            with tab_cause:
                if "root_cause_category" in eq_events.columns:
                    cause_cat = eq_events["root_cause_category"].fillna("未分類").astype(str).value_counts().reset_index()
                    cause_cat.columns = ["原因分類", "事件數"]
                    st.markdown("#### 原因分類統計")
                    st.dataframe(cause_cat, use_container_width=True, hide_index=True)

                if "cause_summary" in eq_events.columns:
                    st.markdown("#### 歷史原因摘要")
                    cause_df = eq_events[[c for c in ["event_id", "occurred_at", "state_name", "downtime_min", "root_cause_category", "cause_summary"] if c in eq_events.columns]].copy()
                    st.dataframe(cause_df.head(20), use_container_width=True, hide_index=True)

                    examples = cause_df["cause_summary"].dropna().astype(str).head(5).tolist()
                    if examples:
                        st.markdown("#### 可直接說明的重點")
                        for item in examples:
                            st.write(f"- {item}")
                else:
                    st.info("事件資料中沒有 cause_summary 欄位。")

            with tab_action:
                if "action_summary" in eq_events.columns:
                    action_df = eq_events[[c for c in ["event_id", "occurred_at", "state_name", "downtime_min", "action_summary"] if c in eq_events.columns]].copy()
                    st.dataframe(action_df.head(20), use_container_width=True, hide_index=True)
                    st.markdown("#### 使用方式")
                    st.write("這裡可以快速看過去同設備的處置方式，作為 SOP 查詢與 AI 診斷建議的參考；但仍需由現場人員確認本次狀況是否相同。")
                else:
                    st.info("事件資料中沒有 action_summary 欄位。")

            with tab_events:
                show_cols = [c for c in ["event_id", "occurred_at", "equipment_code", "state_name", "severity", "downtime_min", "root_cause_category", "cause_summary", "action_summary"] if c in eq_events.columns]
                st.dataframe(eq_events[show_cols].sort_values("occurred_at", ascending=False).head(100) if "occurred_at" in eq_events.columns else eq_events[show_cols].head(100), use_container_width=True, hide_index=True)

# =========================================================
# Page 3：異常分析看板
# =========================================================

elif page == "異常分析看板":
    st.title("📊 異常事件分析看板")
    st.caption("完整呈現異常分布、設備問題頻率、嚴重度、原因分類與時間趨勢。")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("總異常事件數", f"{dashboard_summary['total_events']:,}")

    if dashboard_summary["avg_downtime"] is not None:
        c2.metric("平均停機時間", f"{dashboard_summary['avg_downtime']:.1f} 分鐘")
        c3.metric("最高停機時間", f"{dashboard_summary['max_downtime']:.0f} 分鐘")
    else:
        c2.metric("平均停機時間", "無資料")
        c3.metric("最高停機時間", "無資料")

    c4.metric("涉及設備數", f"{dashboard_summary['equipment_count']:,}")

    st.markdown("---")

    st.markdown("### 異常分布統計")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 異常設備")

        fig1 = px.bar(
            top_equipment_df,
            x="equipment_code",
            y="count",
            title="Top 10 異常設備",
            color="count",
            color_continuous_scale="Blues",
            labels={
                "equipment_code": "設備代碼",
                "count": "異常次數"
            }
        )

        fig1.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig1, use_container_width=True)

        st.dataframe(
            top_equipment_df.rename(
                columns={
                    "equipment_code": "設備代碼",
                    "count": "異常次數"
                }
            ),
            use_container_width=True
        )

    with col2:
        st.subheader("Top 10 異常狀況")

        fig2 = px.bar(
            top_state_df,
            x="count",
            y="state_name",
            orientation="h",
            title="Top 10 異常狀況",
            color="count",
            color_continuous_scale="Viridis",
            labels={
                "state_name": "異常狀況",
                "count": "異常次數"
            }
        )

        fig2.update_layout(
            coloraxis_showscale=False,
            yaxis_title=""
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(
            top_state_df.rename(
                columns={
                    "state_name": "異常狀況",
                    "count": "異常次數"
                }
            ),
            use_container_width=True
        )

    st.markdown("---")

    st.markdown("### 嚴重度與原因分類")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("嚴重度分布")

        fig3 = px.pie(
            severity_df,
            names="severity",
            values="count",
            title="嚴重度分布",
            hole=0.4
        )

        fig3.update_traces(textinfo="percent+label")
        st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(
            severity_df.rename(
                columns={
                    "severity": "嚴重度",
                    "count": "事件數"
                }
            ),
            use_container_width=True
        )

    with col4:
        st.subheader("原因分類分布")

        fig4 = px.pie(
            cause_df,
            names="root_cause_category",
            values="count",
            title="原因分類分布",
            hole=0.4
        )

        fig4.update_traces(textinfo="percent+label")
        st.plotly_chart(fig4, use_container_width=True)

        st.dataframe(
            cause_df.rename(
                columns={
                    "root_cause_category": "原因分類",
                    "count": "事件數"
                }
            ),
            use_container_width=True
        )

    st.markdown("---")

    st.markdown("### 每月異常事件趨勢")
    st.caption("此圖可用來觀察不同月份的異常事件數量變化。")

    if len(monthly_count_df) > 0:
        fig_trend = px.area(
            monthly_count_df,
            x="month",
            y="count",
            title="每月異常事件趨勢",
            color_discrete_sequence=["#667eea"],
            labels={
                "month": "月份",
                "count": "異常事件數"
            }
        )

        fig_trend.update_xaxes(tickangle=45)
        st.plotly_chart(fig_trend, use_container_width=True)

        st.dataframe(
            monthly_count_df.rename(
                columns={
                    "month": "月份",
                    "count": "異常事件數"
                }
            ),
            use_container_width=True
        )
    else:
        st.warning("目前沒有可用的發生時間資料，因此無法產生時間趨勢圖。")


# =========================================================
# Page 4：異常事件查詢
# =========================================================

elif page == "異常事件查詢":
    st.title("🔍 歷史異常事件查詢")
    st.caption("依設備、異常狀況與嚴重度篩選歷史異常事件。")

    col1, col2, col3 = st.columns(3)

    equipment_options = ["全部"] + sorted(
        clean_event_view["equipment_code"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_equipment = col1.selectbox("設備", equipment_options)

    if selected_equipment == "全部":
        state_source_df = clean_event_view.copy()
    else:
        state_source_df = clean_event_view[
            clean_event_view["equipment_code"] == selected_equipment
        ].copy()

    state_options = ["全部"] + sorted(
        state_source_df["state_name"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_state = col2.selectbox("異常狀況", state_options)

    severity_source_df = state_source_df.copy()

    if selected_state != "全部":
        severity_source_df = severity_source_df[
            severity_source_df["state_name"] == selected_state
        ].copy()

    severity_options = ["全部"] + sorted(
        severity_source_df["severity"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_severity = col3.selectbox("嚴重度", severity_options)

    filtered = severity_source_df.copy()

    if selected_severity != "全部":
        filtered = filtered[
            filtered["severity"] == selected_severity
        ].copy()

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("符合條件事件數", f"{len(filtered):,}")

    if "downtime_min" in filtered.columns and len(filtered) > 0:
        c2.metric("平均停機時間", f"{filtered['downtime_min'].mean():.1f} 分鐘")
        c3.metric("最高停機時間", f"{filtered['downtime_min'].max():.0f} 分鐘")
    else:
        c2.metric("平均停機時間", "無資料")
        c3.metric("最高停機時間", "無資料")

    st.markdown("### 歷史異常事件列表")

    display_cols = [
        "event_id",
        "occurred_at",
        "equipment_code",
        "equipment_name",
        "state_name",
        "severity",
        "downtime_min",
        "root_cause_category",
        "action_summary"
    ]

    existing_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[existing_cols],
        use_container_width=True
    )

    with st.expander("查看完整欄位資料"):
        st.dataframe(filtered, use_container_width=True)


# =========================================================
# Page 5：SOP 處理流程查詢
# =========================================================


elif page == "SOP 處理流程查詢":
    st.title("📋 SOP 處理流程查詢｜逐步互動導覽")
    st.caption("依設備與異常狀況查詢對應 SOP，每個步驟以一頁一頁的方式呈現；走到無下一步時才顯示分析結果。")

    col1, col2 = st.columns(2)

    equipment_options = sorted(
        clean_event_view["equipment_code"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_equipment = col1.selectbox("選擇設備", equipment_options)

    related_states = (
        clean_event_view[
            clean_event_view["equipment_code"] == selected_equipment
        ]["state_name"]
        .dropna()
        .unique()
        .tolist()
    )

    related_states = sorted(related_states)

    if len(related_states) == 0:
        st.warning("此設備目前沒有對應的異常狀況資料。")
        st.stop()

    selected_state = col2.selectbox("選擇異常狀況", related_states)

    st.markdown("---")

    sop_df = get_sop_by_equipment_state(selected_equipment, selected_state)

    if sop_df.empty:
        st.warning("目前找不到對應 SOP。")
    else:
        st.success(f"找到 {sop_df['sop_id'].nunique()} 份 SOP")

        for sop_id, group in sop_df.groupby("sop_id"):
            first = group.iloc[0]

            with st.expander(f"{sop_id}｜{first.get('sop_name', '')}", expanded=True):
                c1, c2 = st.columns(2)
                c1.info(f"**SOP 說明**：{first.get('sop_desc', '')}")
                c2.info(f"**負責角色**：{first.get('owner_role', '')}")

                tab_v2, tab_table = st.tabs([
                    "逐步互動導覽",
                    "原始表格"
                ])

                with tab_v2:
                    render_version_two(group, selected_equipment, selected_state, sop_id)

                with tab_table:
                    display_cols = [
                        "sort_order",
                        "branch_label",
                        "sop_stage",
                        "step_title",
                        "step_content",
                        "check_method_label",
                        "monitor_name",
                        "standard_text",
                        "evidence_required",
                        "safety_note"
                    ]
                    existing_cols = [c for c in display_cols if c in group.columns]
                    st.dataframe(group[existing_cols], use_container_width=True)


# =========================================================
# Page 6：AI 異常診斷建議
# =========================================================

elif page == "AI 異常診斷建議":
    st.title("🤖 AI 異常診斷建議")
    st.caption("整合歷史案例、SOP、感測資料與通報規則，生成可供現場工程師參考的診斷摘要。")

    st.warning("""
    使用提醒：本頁面的 AI 診斷只負責整理歷史線索、SOP 與通報內容，不能取代現場判斷。
    涉及安全、設備實況、感測器是否可信、復機條件與最終結案，都必須由現場人員或工程師人工確認。
    """)

    col1, col2 = st.columns(2)

    equipment_options = sorted(
        clean_event_view["equipment_code"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_equipment = col1.selectbox(
        "選擇設備",
        equipment_options,
        key="diag_equipment"
    )

    related_states = (
        clean_event_view[
            clean_event_view["equipment_code"] == selected_equipment
        ]["state_name"]
        .dropna()
        .unique()
        .tolist()
    )

    related_states = sorted(related_states)

    if len(related_states) == 0:
        st.warning("此設備目前沒有對應的異常狀況資料。")
        st.stop()

    selected_state = col2.selectbox(
        "選擇異常狀況",
        related_states,
        key="diag_state"
    )

    st.markdown("---")

    if st.button("產生 AI 診斷建議", type="primary"):
        with st.spinner("系統正在整合歷史事件、SOP、感測資料與 Agent 結果..."):
            summary_text, result = final_demo_generative_summary(
                selected_equipment,
                selected_state
            )

        notification = result["notification"]
        human = result["human"]
        role_versions = notification.get("role_versions", {})

        formal_report = build_formal_ai_report(
            selected_equipment,
            selected_state,
            result
        )

        st.markdown("### 1. 正式 AI 異常診斷報告")
        st.info(formal_report)

        st.markdown("### 2. AI 生成式分析摘要")
        st.caption("這一段保留較口語化的整合摘要，方便放入報告或簡報說明。")
        st.info(summary_text)

        st.markdown("### 3. 需人工確認提醒")
        st.caption("這裡不是要再顯示一次 Human Assistance Agent，而是提醒使用此頁面的人：哪些地方不能只靠 AI 判斷。")

        manual_items = []
        if human.get("assist_df") is not None and not human["assist_df"].empty:
            for _, row in human["assist_df"].head(5).iterrows():
                check_item = row.get("check_item", "依現場狀況確認")
                evidence = row.get("evidence_required", "人工確認紀錄")
                escalation = row.get("escalation_rule", "若處置後仍未改善，需升級通報")
                manual_items.append({
                    "check_item": check_item,
                    "evidence": evidence,
                    "escalation": escalation
                })

        if not manual_items:
            manual_items = [
                {
                    "check_item": "確認現場異常是否仍持續發生，或是否已恢復正常。",
                    "evidence": "現場觀察紀錄、照片或班長確認備註。",
                    "escalation": "若異常持續或涉及安全風險，立即通知班長與工程師。"
                },
                {
                    "check_item": "確認感測器資料是否可信，例如是否有訊號 loss、資料缺漏或異常跳動。",
                    "evidence": "感測器截圖、PLC / SCADA 畫面或人工量測值。",
                    "escalation": "若資料不可信，不應直接採用 AI 判斷，需人工複核。"
                },
                {
                    "check_item": "確認目前 SOP 是否符合現場實際作業方式。",
                    "evidence": "SOP 執行紀錄、現場處置備註。",
                    "escalation": "若 SOP 與現場不符，需回饋製程或設備工程師修訂。"
                }
            ]

        for idx, item in enumerate(manual_items, start=1):
            with st.container(border=True):
                st.markdown(f"#### 人工確認 {idx}")
                st.write(f"**需要人工確認：** {item['check_item']}")
                st.write(f"**建議留下佐證：** {item['evidence']}")
                st.write(f"**升級通報條件：** {item['escalation']}")

        st.markdown("---")

        st.markdown("### 4. 多 Agent 分析細節")
        st.caption("此區把各 Agent 的輸出集中在同一個地方；角色化通報歸在 Notification Agent 底下，避免與前面的診斷摘要重複。")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Diagnosis Agent",
            "SOP Agent",
            "Parts Agent",
            "Notification Agent"
        ])

        with tab1:
            st.markdown("#### Diagnosis Agent｜可能原因判斷")
            st.caption("負責根據設備、異常狀況、歷史案例與感測資料，整理可能原因與風險說明。")
            st.markdown(result["diagnosis"]["text"].replace("\n", "  \n"))

        with tab2:
            st.markdown("#### SOP Agent｜標準流程對應")
            st.caption("負責找出可能對應的 SOP，並整理現場處理時應優先確認的步驟。")
            st.markdown(result["sop"]["text"].replace("\n", "  \n"))

        with tab3:
            st.markdown("#### Parts Agent｜備件與維修提醒")
            st.caption("負責提醒可能牽涉的備件、耗材或維修資源，供維修單位提前準備。")
            st.markdown(result["parts"]["text"].replace("\n", "  \n"))

        with tab4:
            st.markdown("#### Notification Agent｜角色化通報內容")
            st.caption("同一筆異常資料，系統會依不同閱讀對象生成不同版本的通報內容。這一塊原本獨立顯示，現在歸在 Notification Agent 底下，頁面邏輯會更清楚。")

            for role_name, role_desc in [
                ("現場人員版", "立即處理提醒，重點是先做什麼、如何確保安全與留下紀錄。"),
                ("主管通報版", "管理摘要，重點是嚴重度、影響範圍、目前進度與是否需要決策。"),
                ("維修單位版", "維修準備，重點是可能原因、檢修方向、備件與現場佐證。"),
            ]:
                with st.container(border=True):
                    st.markdown(f"##### {role_name}")
                    st.caption(role_desc)
                    st.markdown(role_versions.get(role_name, f"目前沒有{role_name}通報內容。").replace("\n", "  \n"))

        st.markdown("---")

        st.markdown("### 5. 相似歷史事件推薦")
        st.caption("系統優先推薦同設備、同異常、處置紀錄較完整，且停機時間較短的歷史案例。")

        matched_events = get_recommended_cases(selected_equipment, selected_state, top_n=8)

        display_cols = [
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
            "_recommend_score"
        ]

        existing_cols = [c for c in display_cols if c in matched_events.columns]

        st.dataframe(
            matched_events[existing_cols],
            use_container_width=True
        )

        manual_text_lines = []
        manual_text_lines.append("【需人工確認提醒】\n")
        for idx, item in enumerate(manual_items, start=1):
            manual_text_lines.append(f"{idx}. 需要人工確認：{item['check_item']}\n")
            manual_text_lines.append(f"   建議留下佐證：{item['evidence']}\n")
            manual_text_lines.append(f"   升級通報條件：{item['escalation']}\n")

        download_text = []
        download_text.append(formal_report)
        download_text.append("\n\n==============================\n")
        download_text.append(summary_text)
        download_text.append("\n\n==============================\n")
        download_text.append("".join(manual_text_lines))
        download_text.append("\n\n==============================\n")
        download_text.append("【角色化通報內容】")
        download_text.append("\n\n--- 現場人員版 ---\n")
        download_text.append(role_versions.get("現場人員版", ""))
        download_text.append("\n\n--- 主管通報版 ---\n")
        download_text.append(role_versions.get("主管通報版", ""))
        download_text.append("\n\n--- 維修單位版 ---\n")
        download_text.append(role_versions.get("維修單位版", ""))

        st.download_button(
            label="下載 AI 診斷與通報摘要 TXT",
            data="".join(download_text),
            file_name=f"{selected_equipment}_{selected_state}_AI_diagnosis.txt",
            mime="text/plain"
        )



# =========================================================
# Page 7：現場照片佐證紀錄
# =========================================================

elif page == "現場照片佐證紀錄":
    st.title("📷 現場照片佐證紀錄")
    st.caption("這頁只負責保存照片佐證與現場補充說明，不再產生重複或不可靠的診斷內容。")

    st.info("""
    **本頁定位：留下證據，不做照片判讀。**

    目前版本不判斷照片中的零件是否真的破裂、磨耗或漏油；照片只作為 SOP 人工確認、異常追蹤與事後改善的佐證。
    真正的原因判斷請回到「AI 異常診斷建議」或「設備原因查詢」頁面。
    """)

    st.markdown("### 建立照片佐證紀錄")
    col1, col2 = st.columns([1, 1])
    with col1:
        equipment_options = ["未指定"]
        if "equipment_code" in clean_event_view.columns:
            equipment_options += sorted(clean_event_view["equipment_code"].dropna().astype(str).unique().tolist())
        selected_equipment = st.selectbox("設備", equipment_options, key="photo_evidence_equipment")

    with col2:
        state_source = clean_event_view.copy()
        if selected_equipment not in ["未指定", "全部"] and "equipment_code" in state_source.columns:
            state_source = state_source[state_source["equipment_code"].astype(str) == selected_equipment]
        state_options = ["未指定"]
        if "state_name" in state_source.columns:
            state_options += sorted(state_source["state_name"].dropna().astype(str).unique().tolist())
        selected_state = st.selectbox("異常狀況", state_options, key="photo_evidence_state")

    col3, col4 = st.columns([1, 1])
    with col3:
        photo_type = st.selectbox(
            "照片用途",
            ["異常部位近照", "設備全景照", "正常/異常對照", "安全環境佐證", "其他"],
            key="photo_evidence_type",
        )
    with col4:
        confirm_role = st.selectbox(
            "確認人員角色",
            ["現場人員", "班長", "設備工程師", "製程工程師", "品質工程師"],
            key="photo_evidence_role",
        )

    uploaded_photo = st.file_uploader("上傳現場照片（jpg / png / jpeg）", type=["jpg", "jpeg", "png"], key="photo_evidence_upload")
    site_note = st.text_area(
        "現場補充說明",
        placeholder="例如：照片拍攝位置、異常發生時間、是否仍持續、是否已先停機、是否已通知班長。",
        height=110,
        key="photo_evidence_note",
    )

    photo_info = analyze_uploaded_photo_basic(uploaded_photo)

    st.markdown("### 照片與品質提醒")
    left, right = st.columns([0.9, 1.1])
    with left:
        if uploaded_photo is not None:
            st.image(uploaded_photo, caption=f"已上傳：{uploaded_photo.name}", use_container_width=True)
        else:
            st.warning("尚未上傳照片。")
            st.write("建議同時保留設備全景照與異常部位近照，避免事後無法判斷位置與細節。")
    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("照片品質", photo_info.get("quality_level", "未上傳"))
        c2.metric("照片用途", photo_type)
        c3.metric("確認角色", confirm_role)

        quality_notes = photo_info.get("notes", [])
        if quality_notes:
            st.markdown("#### 系統提醒")
            for note in quality_notes[:5]:
                st.write(f"- {note}")
        else:
            st.write("上傳照片後，系統會檢查解析度、亮度、對比與清晰度。")

    st.markdown("### 現場確認清單")
    checklist = [
        "照片是否能看出設備位置？",
        "照片是否能看出異常部位？",
        "是否需要補拍正常狀態作為對照？",
        "是否已記錄感測器數值、處置方式與復機結果？",
    ]
    for idx, item in enumerate(checklist, start=1):
        st.checkbox(item, key=f"photo_check_{idx}")

    st.markdown("### 產生照片佐證紀錄")
    filename = photo_info.get("filename", "未上傳照片")
    quality_notes = photo_info.get("notes", [])
    evidence_lines = [
        "【現場照片佐證紀錄】",
        f"設備：{selected_equipment}",
        f"異常狀況：{selected_state}",
        f"照片檔名：{filename}",
        f"照片用途：{photo_type}",
        f"確認角色：{confirm_role}",
        f"照片品質：{photo_info.get('quality_level', '未上傳')}",
        "",
        "一、現場補充說明",
        site_note.strip() if site_note.strip() else "尚未填寫現場補充說明。",
        "",
        "二、照片品質提醒",
        *([f"- {n}" for n in quality_notes[:5]] if quality_notes else ["- 尚未上傳照片或無明顯品質提醒。"]),
        "",
        "三、後續建議",
        "- 若照片看不清異常部位，請補拍近照與全景照。",
        "- 依 SOP 確認需要人工判斷的節點，並把照片、量測值與處置結果一起保存。",
        "- 原因判斷請搭配 AI 異常診斷建議與設備原因查詢，不要只依照片下結論。",
    ]
    evidence_text = "\n".join(evidence_lines)
    st.text_area("可複製的照片佐證紀錄", evidence_text, height=330)
    st.download_button(
        "下載照片佐證紀錄 TXT",
        data=evidence_text,
        file_name="photo_evidence_record.txt",
        mime="text/plain",
    )


# =========================================================
# Page 8：互動式知識圖譜（已整併）
# =========================================================

elif page == "互動式知識圖譜":
    st.title("互動式知識圖譜已整併")
    st.info("此版本已將原本容易空白的互動式圖譜頁移除，改由『設備原因查詢』提供更穩定、可讀的關聯查詢。")

# =========================================================
# Page 9：主動預警 Agent
# =========================================================

elif page == "主動預警 Agent":
    st.title("📡 主動預警 Agent｜Predictive Maintenance")
    st.caption("從 sensor snapshot 中找出超規、接近上下限或趨勢惡化的監控項目，將系統從被動處置延伸到主動巡檢。")

    alerts = build_predictive_maintenance_alerts(clean_sensor_view, clean_event_view)
    if alerts.empty:
        st.warning("目前 sensor_snapshot 欄位不足，無法建立主動預警表。請確認 clean_sensor_view.csv 是否包含 actual_value、spec_lower、spec_upper、judgement 等欄位。")
    else:
        eq_options = ["全部"]
        if "equipment_code" in alerts.columns:
            eq_options += sorted(alerts["equipment_code"].dropna().astype(str).unique().tolist())
        selected_eq = st.selectbox("選擇設備", eq_options, key="predictive_eq")

        show_alerts = alerts.copy()
        if selected_eq != "全部" and "equipment_code" in show_alerts.columns:
            show_alerts = show_alerts[show_alerts["equipment_code"].astype(str) == selected_eq]

        high_count = (show_alerts["risk_level"] == "高").sum() if "risk_level" in show_alerts.columns else 0
        mid_count = (show_alerts["risk_level"] == "中").sum() if "risk_level" in show_alerts.columns else 0
        low_count = (show_alerts["risk_level"] == "低").sum() if "risk_level" in show_alerts.columns else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("高風險項目", f"{high_count:,}")
        c2.metric("中風險項目", f"{mid_count:,}")
        c3.metric("低風險項目", f"{low_count:,}")
        c4.metric("監控項目數", f"{len(show_alerts):,}")

        st.markdown("### 預警排序表")
        display_cols = [c for c in [
            "equipment_code", "equipment_name", "monitor_name", "parameter_name", "latest_value", "unit",
            "spec_lower", "spec_upper", "trend", "abnormal_count", "near_limit_count", "risk_score", "risk_level", "risk_reason", "suggestion"
        ] if c in show_alerts.columns]
        st.dataframe(show_alerts[display_cols].head(50), use_container_width=True, hide_index=True)

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            if "risk_level" in show_alerts.columns:
                risk_df = show_alerts["risk_level"].value_counts().reset_index()
                risk_df.columns = ["risk_level", "count"]
                fig = px.bar(risk_df, x="risk_level", y="count", title="風險等級分布")
                st.plotly_chart(fig, use_container_width=True)
        with col_chart2:
            top_risk = show_alerts.head(10).copy()
            if "monitor_name" in top_risk.columns and "risk_score" in top_risk.columns:
                label_cols = [c for c in ["equipment_code", "monitor_name", "parameter_name"] if c in top_risk.columns]
                top_risk["label"] = top_risk[label_cols].astype(str).agg("｜".join, axis=1)
                fig = px.bar(top_risk, x="risk_score", y="label", orientation="h", title="Top 10 預警項目")
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 預警文字範例")
        if not show_alerts.empty:
            top = show_alerts.iloc[0]
            warning_text = f"""【主動預警 Agent 建議】

設備：{_safe_text(top.get('equipment_code'))} {_safe_text(top.get('equipment_name'), '')}
監控項目：{_safe_text(top.get('monitor_name'))} / {_safe_text(top.get('parameter_name'))}
目前風險等級：{_safe_text(top.get('risk_level'))}
近期趨勢：{_safe_text(top.get('trend'))}
判斷理由：{_safe_text(top.get('risk_reason'))}

建議：{_safe_text(top.get('suggestion'))}

注意：目前為簡化版風險排序，並非保證未來一定故障；正式版本需要連續時間序列資料、設備維修紀錄與模型驗證。"""
            st.text_area("可複製到報告 / 通報的預警文字", warning_text, height=280)
            st.download_button("下載主動預警 TXT", warning_text, file_name="predictive_maintenance_alert.txt", mime="text/plain")

        with st.expander("補充說明"):
            st.markdown("""
            本頁將系統從「異常發生後才查 SOP」延伸為「異常發生前先做風險提示」。
            Agent 會檢查 sensor snapshot 中是否有超出上下限、接近規格邊界、異常 judgement 或近期趨勢變化的項目，
            並產生高 / 中 / 低風險等級與巡檢建議。

            由於目前資料主要是異常事件快照，而不是完整連續感測器時間序列，因此本頁採用簡化版趨勢預警；
            未來若接入即時監控資料，可進一步訓練預測性維護模型，提前預測軸承、切刀、導輪或水量異常風險。
            """)


# =========================================================
# Page 10：SOP 改善建議
# =========================================================

elif page == "SOP 改善建議":
    st.title("🧩 SOP 改善建議")
    st.caption("利用歷史異常事件與 SOP 步驟資料，找出最值得優先改善的設備、異常與流程節點。")

    st.info("""
    這一頁的重點是：系統不只是查 SOP，而是能用歷史處置紀錄反推 SOP 哪些地方需要改善。
    例如：哪種設備異常平均停機時間最高、哪些 SOP 節點最常需要人工確認、哪些原因反覆出現。
    """)

    high_downtime, manual_steps, repeated_abnormal = build_sop_improvement_tables(
        clean_event_view,
        clean_sop_view
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "停機時間優先改善",
        "人工確認節點",
        "重複發生原因",
        "報告可用結論"
    ])

    with tab1:
        st.markdown("### 1. 平均停機時間較高的設備與異常")
        st.caption("平均停機時間越高，代表該類異常對產線影響越大，適合優先檢討 SOP 或設備維護策略。")

        if high_downtime.empty:
            st.warning("目前沒有足夠的停機時間資料。")
        else:
            st.dataframe(high_downtime, use_container_width=True, hide_index=True)

            if {"equipment_code", "state_name", "avg_downtime_min"}.issubset(high_downtime.columns):
                chart_df = high_downtime.copy()
                chart_df["equipment_state"] = chart_df["equipment_code"].astype(str) + "｜" + chart_df["state_name"].astype(str)
                fig = px.bar(
                    chart_df.head(10),
                    x="equipment_state",
                    y="avg_downtime_min",
                    text="event_count",
                    title="平均停機時間 Top 10"
                )
                fig.update_layout(xaxis_title="設備｜異常狀況", yaxis_title="平均停機時間（分鐘）")
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### 2. 最需要人工確認的 SOP 節點")
        st.caption("這些節點通常需要照片、目視、現場量測或工程師判斷，是 AI 無法完全自動化的地方。")

        if manual_steps.empty:
            st.warning("目前沒有整理出明確需要人工確認的 SOP 節點。")
        else:
            st.dataframe(manual_steps, use_container_width=True, hide_index=True)

            st.markdown("#### 改善方向")
            st.markdown("""
            - 若同一節點經常需要人工補資料，可以把「需要拍什麼、量什麼、回報什麼」寫進 SOP。
            - 若該節點其實可由感測器判斷，可以新增 monitor_id 或 parameter_spec。
            - 若現場常不知道如何判斷，應補上正常/異常照片或判斷範例。
            """)

    with tab3:
        st.markdown("### 3. 重複發生的設備、異常與原因")
        st.caption("如果同一類原因反覆出現，代表可能不是單次偶發，而是 SOP、設備保養或製程條件需要改善。")

        if repeated_abnormal.empty:
            st.warning("目前沒有足夠的原因分類資料。")
        else:
            st.dataframe(repeated_abnormal, use_container_width=True, hide_index=True)

            if {"equipment_code", "state_name", "root_cause_category", "event_count"}.issubset(repeated_abnormal.columns):
                chart_df = repeated_abnormal.copy()
                chart_df["label"] = (
                    chart_df["equipment_code"].astype(str)
                    + "｜"
                    + chart_df["state_name"].astype(str)
                    + "｜"
                    + chart_df["root_cause_category"].astype(str)
                )
                fig = px.bar(
                    chart_df.head(10),
                    x="label",
                    y="event_count",
                    title="重複發生原因 Top 10"
                )
                fig.update_layout(xaxis_title="設備｜異常｜原因", yaxis_title="事件數")
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### 4. 可直接放進報告的結論")
        st.markdown("""
        本系統除了提供 SOP 查詢與 AI 診斷外，也能透過歷史異常資料回饋 SOP 改善方向。
        若某些設備異常的平均停機時間較高，代表該類異常對產線影響較大，應優先檢討處置流程；
        若某些 SOP 節點經常需要人工確認，代表該節點可能需要補充照片、判斷標準或感測器監控；
        若同一原因分類反覆發生，則可進一步檢討設備保養、製程條件或教育訓練是否不足。

        因此，本專題的價值不只是「查 SOP」或「產生 AI 文字」，而是將異常處理資料轉成可持續改善的知識庫，
        讓 SOP 可以隨著歷史案例與現場經驗逐步更新。
        """)

        st.download_button(
            label="下載 SOP 改善結論 TXT",
            data="""【SOP 改善建議結論】

本系統除了提供 SOP 查詢與 AI 診斷外，也能透過歷史異常資料回饋 SOP 改善方向。
若某些設備異常的平均停機時間較高，代表該類異常對產線影響較大，應優先檢討處置流程；
若某些 SOP 節點經常需要人工確認，代表該節點可能需要補充照片、判斷標準或感測器監控；
若同一原因分類反覆發生，則可進一步檢討設備保養、製程條件或教育訓練是否不足。

因此，本專題的價值不只是查 SOP 或產生 AI 文字，而是將異常處理資料轉成可持續改善的知識庫，
讓 SOP 可以隨著歷史案例與現場經驗逐步更新。
""",
            file_name="SOP_improvement_conclusion.txt",
            mime="text/plain"
        )

