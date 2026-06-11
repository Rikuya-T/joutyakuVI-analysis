import re
import os
import sys
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def ensure_streamlit_context() -> None:
    """`python app.py` 実行時は Streamlit CLI で再起動する。"""
    runtime_exists = False
    try:
        from streamlit.runtime import exists as runtime_exists_fn

        runtime_exists = runtime_exists_fn()
    except Exception:
        runtime_exists = False

    if runtime_exists:
        return

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    raise SystemExit(stcli.main())


ensure_streamlit_context()
st.set_page_config(page_title="蒸着品質分析システム", layout="wide")


@dataclass
class FileMeta:
    file_path: Path
    file_name: str
    model_name: str
    equipment_no: str
    chamber_no: str
    batch_no: str
    measurement_surface: str
    measure_position: str
    measure_date: pd.Timestamp
    judgement: str


DATE_PATTERN = re.compile(r"_(\d{8})$")
CSV_ENCODINGS = ["utf-8-sig", "cp932", "shift_jis", "utf-8"]


def parse_file_meta(file_path: Path) -> Optional[FileMeta]:
    stem = file_path.stem.strip()
    # 区切りがスペース/アンダースコアのどちらでも読めるようにする
    parts = [p for p in re.split(r"[ _]+", stem) if p]
    if not parts:
        return None

    date_idx = -1
    for i in range(len(parts) - 1, -1, -1):
        if re.fullmatch(r"\d{8}", parts[i]):
            date_idx = i
            break

    if date_idx == -1:
        return None

    date_str = parts[date_idx]
    date_value = pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
    if pd.isna(date_value):
        return None

    judgement = "未指定"
    if date_idx + 1 < len(parts) and parts[date_idx + 1].upper() in {"OK", "NG"}:
        judgement = parts[date_idx + 1].upper()

    before_date = parts[:date_idx]

    equipment_no = "未指定"
    chamber_no = "未指定"
    batch_no = "未指定"
    measurement_surface = "未指定"

    if len(before_date) >= 5:
        model_tokens = before_date[:-4]
        model_name = "_".join(model_tokens).strip() if model_tokens else before_date[0].strip()
        equipment_no = before_date[-4].strip()
        chamber_no = before_date[-3].strip()
        batch_no = before_date[-2].strip()
        measurement_surface = before_date[-1].strip()
    else:
        if len(before_date) >= 2:
            model_name = "_".join(before_date[:-1]).strip()
            measurement_surface = before_date[-1].strip()
        elif len(before_date) == 1:
            model_name = before_date[0].strip()
        else:
            model_name = "未指定"

    if not model_name:
        model_name = "未指定"

    measure_position = f"設備{equipment_no}_槽{chamber_no}_バッチ{batch_no}_{measurement_surface}"

    return FileMeta(
        file_path=file_path,
        file_name=file_path.name,
        model_name=model_name,
        equipment_no=equipment_no,
        chamber_no=chamber_no,
        batch_no=batch_no,
        measurement_surface=measurement_surface,
        measure_position=measure_position,
        measure_date=date_value,
        judgement=judgement,
    )


def find_header_index_and_encoding(file_path: Path) -> Tuple[int, str]:
    for enc in CSV_ENCODINGS:
        try:
            preview = pd.read_csv(file_path, header=None, dtype=str, nrows=30, encoding=enc, engine="python")
        except Exception:
            continue

        for idx in range(len(preview)):
            row = preview.iloc[idx].fillna("").astype(str).str.strip().str.lower().tolist()
            if not row:
                continue
            first = row[0]
            second = row[1] if len(row) > 1 else ""
            if first in ["wavelength", "測定点", "x", "point"] and (
                second.startswith("data") or second in ["測定値", "b"]
            ):
                return idx, enc

        if len(preview) >= 2:
            first_row = str(preview.iloc[0, 0]).lower()
            second_row = str(preview.iloc[1, 0]).lower()
            if "datafile" in first_row and "wavelength" in second_row:
                return 1, enc

    return 1, "utf-8-sig"


def normalize_wave_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed: Dict[str, str] = {}
    cols = list(df.columns)
    if not cols:
        return df

    renamed[cols[0]] = "測定点"
    for i, col in enumerate(cols[1:21], start=1):
        renamed[col] = f"測定値{i}"

    converted = df.rename(columns=renamed)

    use_cols = ["測定点"] + [f"測定値{i}" for i in range(1, 21) if f"測定値{i}" in converted.columns]
    converted = converted[use_cols].copy()

    converted["測定点"] = pd.to_numeric(converted["測定点"], errors="coerce")
    for c in use_cols[1:]:
        converted[c] = pd.to_numeric(converted[c], errors="coerce")

    # 全行が0の列は「未測定データ」とみなして除外する
    measured_cols: List[str] = []
    for c in use_cols[1:]:
        series = converted[c]
        non_na = series.dropna()
        if non_na.empty:
            continue
        if (non_na == 0).all():
            continue
        measured_cols.append(c)

    converted = converted.dropna(subset=["測定点"]).sort_values("測定点").reset_index(drop=True)
    return converted[["測定点"] + measured_cols]


def load_folder_data(folder_path: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], List[str]]:
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError("指定されたフォルダが存在しないか、フォルダではありません。")

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise ValueError("CSVファイルが見つかりません。")

    meta_rows = []
    waves_by_file: Dict[str, pd.DataFrame] = {}
    warnings: List[str] = []

    for file_path in csv_files:
        meta = parse_file_meta(file_path)
        if meta is None:
            warnings.append(f"メタ情報を読めないためスキップ: {file_path.name}")
            continue

        try:
            header_idx, enc = find_header_index_and_encoding(file_path)
            raw = pd.read_csv(file_path, header=header_idx, encoding=enc, engine="python")
            wave_df = normalize_wave_columns(raw)
            if wave_df.empty:
                warnings.append(f"波形データが空のためスキップ: {file_path.name}")
                continue

            file_key = file_path.name
            waves_by_file[file_key] = wave_df
            meta_rows.append(
                {
                    "ファイル名": file_key,
                    "機種名": meta.model_name,
                    "設備No": meta.equipment_no,
                    "チャンバーNo": meta.chamber_no,
                    "バッチNo": meta.batch_no,
                    "測定面": meta.measurement_surface,
                    "測定位置": meta.measure_position,
                    "判定": meta.judgement,
                    "測定日": meta.measure_date,
                    "データ点数": len(wave_df),
                }
            )
        except Exception:
            warnings.append(f"読込失敗のためスキップ: {file_path.name}")

    if not meta_rows:
        raise ValueError("有効なCSVを読み込めませんでした。")

    meta_df = pd.DataFrame(meta_rows).sort_values(["測定日", "機種名", "設備No", "チャンバーNo", "バッチNo"]).reset_index(drop=True)
    return meta_df, waves_by_file, warnings


def load_uploaded_files(uploaded_files: List) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], List[str]]:
    """Streamlit ファイルアップロードから読込"""
    if not uploaded_files:
        raise ValueError("CSVファイルがアップロードされていません。")

    meta_rows = []
    waves_by_file: Dict[str, pd.DataFrame] = {}
    warnings: List[str] = []

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        if not file_name.lower().endswith(".csv"):
            warnings.append(f"CSV以外のためスキップ: {file_name}")
            continue

        meta = parse_file_meta(Path(file_name))
        if meta is None:
            warnings.append(f"メタ情報を読めないためスキップ: {file_name}")
            continue

        try:
            header_idx, enc = find_header_index_and_encoding_from_bytes(uploaded_file.getvalue())
            raw = pd.read_csv(
                uploaded_file,
                header=header_idx,
                encoding=enc,
                engine="python",
            )
            wave_df = normalize_wave_columns(raw)
            if wave_df.empty:
                warnings.append(f"波形データが空のためスキップ: {file_name}")
                continue

            file_key = file_name
            waves_by_file[file_key] = wave_df
            meta_rows.append(
                {
                    "ファイル名": file_key,
                    "機種名": meta.model_name,
                    "設備No": meta.equipment_no,
                    "チャンバーNo": meta.chamber_no,
                    "バッチNo": meta.batch_no,
                    "測定面": meta.measurement_surface,
                    "測定位置": meta.measure_position,
                    "判定": meta.judgement,
                    "測定日": meta.measure_date,
                    "データ点数": len(wave_df),
                }
            )
        except Exception:
            warnings.append(f"読込失敗のためスキップ: {file_name}")

    if not meta_rows:
        raise ValueError("有効なCSVを読み込めませんでした。")

    meta_df = pd.DataFrame(meta_rows).sort_values(["測定日", "機種名", "設備No", "チャンバーNo", "バッチNo"]).reset_index(drop=True)
    return meta_df, waves_by_file, warnings


def find_header_index_and_encoding_from_bytes(file_bytes: bytes) -> Tuple[int, str]:
    """バイト列からヘッダ行と文字コードを検出"""
    for enc in CSV_ENCODINGS:
        try:
            preview = pd.read_csv(
                io.BytesIO(file_bytes),
                header=None,
                dtype=str,
                nrows=30,
                encoding=enc,
                engine="python",
            )
        except Exception:
            continue

        for idx in range(len(preview)):
            row = preview.iloc[idx].fillna("").astype(str).str.strip().str.lower().tolist()
            if not row:
                continue
            first = row[0]
            second = row[1] if len(row) > 1 else ""
            if first in ["wavelength", "測定点", "x", "point"] and (
                second.startswith("data") or second in ["測定値", "b"]
            ):
                return idx, enc

        if len(preview) >= 2:
            first_row = str(preview.iloc[0, 0]).lower()
            second_row = str(preview.iloc[1, 0]).lower()
            if "datafile" in first_row and "wavelength" in second_row:
                return 1, enc

    return 1, "utf-8-sig"


def build_specification_bounds(
    x_values: np.ndarray,
    spec_points_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    規格値（下限/上限）を測定点全体に補間。
    上限または下限のどちらか1点以上あればOK。
    戻り値: (lower, upper, (x_min, x_max))
    """
    points = spec_points_df.copy()
    points["測定点"] = pd.to_numeric(points["測定点"], errors="coerce")
    points["下限"] = pd.to_numeric(points["下限"], errors="coerce")
    points["上限"] = pd.to_numeric(points["上限"], errors="coerce")
    
    # 上限または下限のどちらかを指定している行を抜き貼り
    points = points[points["測定点"].notna() & (points["下限"].notna() | points["上限"].notna())]
    
    if len(points) < 1:
        raise ValueError("有効な規格値点がありません。上限または下限を入力してください。")
    
    if len(points) < 2:
        # 1点だけの場合、その点の値を常数で使用
        x_spec = points["測定点"].iloc[0]
        lower_val = points["下限"].iloc[0] if pd.notna(points["下限"].iloc[0]) else -np.inf
        upper_val = points["上限"].iloc[0] if pd.notna(points["上限"].iloc[0]) else np.inf
        xp = np.array([x_spec])
        lower_vals = np.array([lower_val])
        upper_vals = np.array([upper_val])
        x_range = (x_spec, x_spec)
    else:
        points = points.sort_values("測定点")
        xp = points["測定点"].to_numpy()
        lower_vals = points["下限"].fillna(-np.inf).to_numpy()
        upper_vals = points["上限"].fillna(np.inf).to_numpy()
        x_range = (xp.min(), xp.max())

    lower = np.interp(x_values, xp, lower_vals)
    upper = np.interp(x_values, xp, upper_vals)
    return lower, upper, x_range


def calc_file_violation_rate(
    wave_df: pd.DataFrame,
    selected_series: List[str],
    upper: np.ndarray,
    lower: np.ndarray,
) -> float:
    if not selected_series:
        return 0.0

    total_count = 0
    ng_count = 0
    for col in selected_series:
        if col not in wave_df.columns:
            continue
        y = wave_df[col].to_numpy()
        valid = ~np.isnan(y)
        total_count += int(valid.sum())
        ng_count += int(((y > upper) | (y < lower)) & valid).sum()

    if total_count == 0:
        return 0.0
    return ng_count / total_count * 100.0


def find_csv_folders(base_dirs: List[Path], max_depth: int = 3, max_results: int = 30) -> List[str]:
    found: List[str] = []

    for base_dir in base_dirs:
        if not base_dir.exists() or not base_dir.is_dir():
            continue

        for root, dirs, files in os.walk(base_dir):
            root_path = Path(root)

            try:
                rel_depth = len(root_path.relative_to(base_dir).parts)
            except Exception:
                rel_depth = 0

            if rel_depth >= max_depth:
                dirs[:] = []

            has_csv = any(name.lower().endswith(".csv") for name in files)
            if has_csv:
                found.append(str(root_path))
                if len(found) >= max_results:
                    return sorted(set(found))

    return sorted(set(found))


def folder_picker() -> str:
    st.sidebar.subheader("データ読込")
    default_path = st.session_state.get("folder_path", "")
    folder_path = st.sidebar.text_input("CSVフォルダパス", value=default_path, key="folder_path")

    if st.sidebar.button("フォルダ選択ダイアログを開く"):
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askdirectory()
            root.destroy()
            if selected:
                st.session_state["folder_path"] = selected
                folder_path = selected
                st.session_state["folder_candidates"] = [selected]
        except Exception:
            st.sidebar.warning("ダイアログが利用できないため、候補フォルダ検索または直接入力を使用してください。")

    if st.sidebar.button("CSVフォルダ候補を検索"):
        user_home = Path.home()
        search_bases = [
            user_home / "Desktop",
            user_home / "Documents",
            Path.cwd(),
        ]
        st.session_state["folder_candidates"] = find_csv_folders(search_bases)

    candidates: List[str] = st.session_state.get("folder_candidates", [])
    if candidates:
        selected_candidate = st.sidebar.selectbox("候補フォルダ", options=[""] + candidates, index=0)
        if selected_candidate:
            st.session_state["folder_path"] = selected_candidate
            folder_path = selected_candidate

    st.sidebar.caption(f"現在の作業フォルダ: {Path.cwd()}")

    return folder_path


def main() -> None:
    st.title("蒸着品質分析システム")
    st.caption("CSVフォルダから機種・測定位置・日付を抽出し、波形分析と規格アラートを可視化します。")

    st.sidebar.subheader("データ読込")
    input_mode = st.sidebar.radio("入力方法を選択", ["ファイルアップロード", "フォルダパス入力"])

    meta_df = None
    waves_by_file = None
    warn_list = []

    if input_mode == "ファイルアップロード":
        uploaded_files = st.sidebar.file_uploader(
            "CSVファイルを選択（複数選択可能）",
            type="csv",
            accept_multiple_files=True,
        )
        if uploaded_files:
            try:
                meta_df, waves_by_file, warn_list = load_uploaded_files(uploaded_files)
            except Exception as ex:
                st.error(f"データ読込エラー: {ex}")
                st.stop()
        else:
            st.info("左サイドバーでCSVファイルを選択してください。")
            st.stop()
    else:
        folder_path = folder_picker()
        if not folder_path:
            st.info("左サイドバーでCSVフォルダを指定してください。")
            st.stop()

        try:
            meta_df, waves_by_file, warn_list = load_folder_data(folder_path)
        except Exception as ex:
            st.error(f"データ読込エラー: {ex}")
            st.stop()

    for msg in warn_list:
        st.warning(msg)

    st.subheader("読込サマリー")
    c1, c2, c3 = st.columns(3)
    c1.metric("読込ファイル数", f"{len(meta_df):,}")
    c2.metric("機種数", f"{meta_df['機種名'].nunique():,}")
    c3.metric("測定位置数", f"{meta_df['測定位置'].nunique():,}")

    with st.expander("読込ファイル一覧"):
        st.dataframe(meta_df, use_container_width=True)

    st.sidebar.subheader("絞り込み条件")
    model_list = sorted(meta_df["機種名"].dropna().unique().tolist())
    pos_list = sorted(meta_df["測定位置"].dropna().unique().tolist())

    selected_models = st.sidebar.multiselect("機種名", model_list, default=model_list)
    selected_positions = st.sidebar.multiselect("測定位置", pos_list, default=pos_list)

    date_min = meta_df["測定日"].min().date()
    date_max = meta_df["測定日"].max().date()
    selected_dates = st.sidebar.date_input("測定日範囲", value=(date_min, date_max), min_value=date_min, max_value=date_max)

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = date_min, date_max

    filtered_meta = meta_df[
        meta_df["機種名"].isin(selected_models)
        & meta_df["測定位置"].isin(selected_positions)
        & (meta_df["測定日"].dt.date >= start_date)
        & (meta_df["測定日"].dt.date <= end_date)
    ].copy()

    if filtered_meta.empty:
        st.warning("絞り込み条件に一致するデータがありません。")
        st.stop()

    st.subheader("波形表示")
    target_files = filtered_meta["ファイル名"].tolist()
    selected_file = st.selectbox("表示対象ファイル", options=target_files, index=0)

    wave_df = waves_by_file[selected_file]
    series_cols = [c for c in wave_df.columns if c.startswith("測定値")]
    if not series_cols:
        st.warning("このファイルは全測定列が未測定（全て0）として判定されたため、波形表示の対象がありません。")
        st.stop()
    selected_series = st.multiselect("表示系列（B～U相当）", series_cols, default=series_cols)

    with st.expander("規格値設定", expanded=True):
        st.write("重要測定点での下限・上限を指定してください。規格値は線形補間で生成されます。")
        default_points = pd.DataFrame(
            {
                "測定点": [wave_df["測定点"].min(), wave_df["測定点"].median(), wave_df["測定点"].max()],
                "下限": [0.15, 0.15, 0.15],
                "上限": [0.25, 0.25, 0.25],
            }
        )
        key_points = st.data_editor(
            default_points,
            num_rows="dynamic",
            use_container_width=True,
            key="key_points_editor",
        )

    fig = go.Figure()
    x = wave_df["測定点"].to_numpy()

    for col in selected_series:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=wave_df[col],
                mode="lines",
                name=col,
                line={"width": 1.2},
                opacity=0.8,
            )
        )

    standard_ok = False
    upper = lower = None
    x_spec_range = None
    try:
        lower, upper, x_spec_range = build_specification_bounds(x, key_points)
        standard_ok = True
        
        # 指定範囲内のみ的線を描画
        if x_spec_range:
            x_min, x_max = x_spec_range
            x_mask = (x >= x_min) & (x <= x_max)
            x_filtered = x[x_mask]
            upper_filtered = upper[x_mask]
            lower_filtered = lower[x_mask]
            
            fig.add_trace(
                go.Scatter(
                    x=x_filtered,
                    y=upper_filtered,
                    mode="lines",
                    name="上限",
                    line={"width": 2, "dash": "dash", "color": "red"},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_filtered,
                    y=lower_filtered,
                    mode="lines",
                    name="下限",
                    line={"width": 2, "dash": "dash", "color": "red"},
                    fill="tonexty",
                    fillcolor="rgba(255, 0, 0, 0.1)",
                )
            )
    except Exception as ex:
        st.info(f"規格値は未作成です: {ex}")

    x_min_default = float(np.nanmin(x))
    x_max_default = float(np.nanmax(x))

    y_values = wave_df[selected_series].to_numpy(dtype=float).ravel() if selected_series else np.array([])
    y_values = y_values[~np.isnan(y_values)]
    if y_values.size == 0:
        y_min_default, y_max_default = 0.0, 1.0
    else:
        y_min_default = float(np.min(y_values))
        y_max_default = float(np.max(y_values))

    with st.expander("グラフ表示範囲設定", expanded=False):
        st.caption("測定点(X)と測定値(Y)の最小/最大を入力して拡大縮小できます。")
        x_col1, x_col2 = st.columns(2)
        x_input_min = x_col1.number_input("X最小（測定点）", value=x_min_default, format="%.4f")
        x_input_max = x_col2.number_input("X最大（測定点）", value=x_max_default, format="%.4f")

        y_col1, y_col2 = st.columns(2)
        y_input_min = y_col1.number_input("Y最小（測定値）", value=y_min_default, format="%.6f")
        y_input_max = y_col2.number_input("Y最大（測定値）", value=y_max_default, format="%.6f")

    x_range_valid = x_input_min < x_input_max
    y_range_valid = y_input_min < y_input_max

    if x_range_valid:
        fig.update_xaxes(range=[x_input_min, x_input_max])
    else:
        st.warning("X軸の範囲が不正です。X最小はX最大より小さくしてください。")

    if y_range_valid:
        fig.update_yaxes(range=[y_input_min, y_input_max])
    else:
        st.warning("Y軸の範囲が不正です。Y最小はY最大より小さくしてください。")

    fig.update_layout(
        title=f"2D波形: {selected_file}",
        xaxis_title="測定点",
        yaxis_title="測定値",
        height=560,
        legend_title="系列",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("系列別コーティング日トレンド（同一機種）")
    trend_model_list = sorted(filtered_meta["機種名"].dropna().unique().tolist())
    trend_model = st.selectbox("トレンド対象機種", options=trend_model_list, index=0, key="trend_model")
    model_meta = filtered_meta[filtered_meta["機種名"] == trend_model].copy()

    equipment_list = sorted(model_meta["設備No"].dropna().astype(str).unique().tolist())
    chamber_list = sorted(model_meta["チャンバーNo"].dropna().astype(str).unique().tolist())

    trend_col1, trend_col2 = st.columns(2)
    selected_equipment = trend_col1.multiselect(
        "設備号機",
        options=equipment_list,
        default=equipment_list,
        key="trend_equipment_filter",
    )
    selected_chamber = trend_col2.multiselect(
        "チャンバー",
        options=chamber_list,
        default=chamber_list,
        key="trend_chamber_filter",
    )

    model_meta = model_meta[
        model_meta["設備No"].astype(str).isin(selected_equipment)
        & model_meta["チャンバーNo"].astype(str).isin(selected_chamber)
    ].sort_values("測定日")

    if model_meta.empty:
        st.warning("選択した設備号機・チャンバーに一致するデータがありません。")
        st.stop()

    series_master: set = set()
    for file_name in model_meta["ファイル名"]:
        cols = [c for c in waves_by_file[file_name].columns if c.startswith("測定値")]
        series_master.update(cols)

    trend_series_all = sorted(series_master, key=lambda s: int(s.replace("測定値", "")))
    default_trend_series = trend_series_all[: min(3, len(trend_series_all))]
    trend_series_selected = st.multiselect(
        "系列を選択（例: 測定値1=B列, 測定値2=C列, 測定値3=D列）",
        trend_series_all,
        default=default_trend_series,
        key="trend_series_selected",
    )

    if trend_series_selected:
        series_tabs = st.tabs(trend_series_selected)
        for tab, series_name in zip(series_tabs, trend_series_selected):
            with tab:
                series_fig = go.Figure()
                for _, meta_row in model_meta.iterrows():
                    file_name = meta_row["ファイル名"]
                    wave = waves_by_file[file_name]
                    if series_name not in wave.columns:
                        continue

                    date_text = meta_row["測定日"].strftime("%Y-%m-%d")
                    judgement_text = str(meta_row.get("判定", "未指定"))
                    eq_text = str(meta_row.get("設備No", "-"))
                    ch_text = str(meta_row.get("チャンバーNo", "-"))
                    label = f"{date_text} / 設備{eq_text} / 槽{ch_text} / {judgement_text} / {file_name}"

                    series_fig.add_trace(
                        go.Scatter(
                            x=wave["測定点"],
                            y=wave[series_name],
                            mode="lines",
                            name=label,
                            line={"width": 1.4},
                        )
                    )

                # メイン波形で設定した規格値（上限/下限）を系列別トレンドにも反映
                if standard_ok and upper is not None and lower is not None and x_spec_range:
                    x_min_spec, x_max_spec = x_spec_range
                    x_mask_spec = (x >= x_min_spec) & (x <= x_max_spec)
                    x_spec = x[x_mask_spec]
                    upper_spec = upper[x_mask_spec]
                    lower_spec = lower[x_mask_spec]

                    series_fig.add_trace(
                        go.Scatter(
                            x=x_spec,
                            y=upper_spec,
                            mode="lines",
                            name="上限",
                            line={"width": 2, "dash": "dash", "color": "red"},
                        )
                    )
                    series_fig.add_trace(
                        go.Scatter(
                            x=x_spec,
                            y=lower_spec,
                            mode="lines",
                            name="下限",
                            line={"width": 2, "dash": "dash", "color": "red"},
                            fill="tonexty",
                            fillcolor="rgba(255, 0, 0, 0.1)",
                        )
                    )

                series_fig.update_layout(
                    title=f"{trend_model} - {series_name}（コーティング日別2D波形）",
                    xaxis_title="測定点",
                    yaxis_title=series_name,
                    height=500,
                    legend_title="コーティング日 / 判定 / ファイル",
                    template="plotly_white",
                )

                # 「グラフ表示範囲設定」の入力値を系列別トレンド波形にも一括適用
                if x_range_valid:
                    series_fig.update_xaxes(range=[x_input_min, x_input_max])
                if y_range_valid:
                    series_fig.update_yaxes(range=[y_input_min, y_input_max])

                st.plotly_chart(series_fig, use_container_width=True)
    else:
        st.info("系列を1つ以上選択してください。")

    st.subheader("規格逸脱アラート")
    if not standard_ok:
        st.warning("規格値が未設定のためアラート判定を実行できません。")
        return

    alert_threshold = st.slider("アラートしきい値（逸脱率 %）", 0.0, 100.0, 5.0, 0.5)

    trend_rows = []
    for _, row in filtered_meta.iterrows():
        file_name = row["ファイル名"]
        target_df = waves_by_file[file_name]
        x_target = target_df["測定点"].to_numpy()

        if standard_ok and upper is not None and lower is not None:
            upper_t = np.interp(x_target, x, upper)
            lower_t = np.interp(x_target, x, lower)
        else:
            upper_t = np.full_like(x_target, np.inf)
            lower_t = np.full_like(x_target, -np.inf)

        violation_rate = calc_file_violation_rate(target_df, selected_series, upper_t, lower_t)
        trend_rows.append(
            {
                "測定日": row["測定日"],
                "ファイル名": file_name,
                "機種名": row["機種名"],
                "測定位置": row["測定位置"],
                "逸脱率(%)": violation_rate,
                "アラート": "要確認" if violation_rate >= alert_threshold else "正常",
            }
        )

    trend_df = pd.DataFrame(trend_rows).sort_values("測定日").reset_index(drop=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.dataframe(trend_df, use_container_width=True)
    with col_b:
        alert_count = (trend_df["アラート"] == "要確認").sum()
        st.metric("アラート件数", f"{alert_count:,}")
        st.metric("全件数", f"{len(trend_df):,}")

    st.subheader("日付トレンド（逸脱率）")
    trend_fig = go.Figure()

    for name, grp in trend_df.groupby(["機種名", "測定位置"]):
        label = f"{name[0]} / {name[1]}"
        trend_fig.add_trace(
            go.Scatter(
                x=grp["測定日"],
                y=grp["逸脱率(%)"],
                mode="lines+markers",
                name=label,
            )
        )

    trend_fig.add_hline(y=alert_threshold, line_dash="dash", line_color="red", annotation_text="アラートしきい値")
    trend_fig.update_layout(
        xaxis_title="測定日",
        yaxis_title="逸脱率(%)",
        height=420,
        template="plotly_white",
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    csv_bytes = trend_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="トレンド判定結果CSVを保存",
        data=csv_bytes,
        file_name="蒸着品質トレンド判定結果.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
