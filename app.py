import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==============================================================================
# CẤU HÌNH TRANG WEB
# ==============================================================================
st.set_page_config(
    page_title="Hệ thống Phân tích & Phát hiện Gian lận Ngân hàng",
    page_icon="🛡️",
    layout="wide"
)

os.makedirs("data", exist_ok=True)

# ==============================================================================
# HÀM XỬ LÝ & CHUẨN HÓA ĐA DATASET
# ==============================================================================
def detect_target_column(df: pd.DataFrame) -> str:
    """Tự động nhận diện cột nhãn gian lận nếu người dùng nạp dataset khác nhau."""
    candidates = ["isFraud", "Class", "fraud", "is_fraud", "target", "label"]
    for col in candidates:
        if col in df.columns:
            return col
    # Nếu không trùng tên chuẩn, tìm cột nhị phân (chỉ gồm 0 và 1) ở cuối bảng
    for col in reversed(df.columns):
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0, 1}) and len(unique_vals) == 2:
            return col
    return df.columns[-1]


@st.cache_data(show_spinner=False)
def load_and_merge_datasets(file_paths: list) -> pd.DataFrame:
    """Đọc một hoặc nhiều file CSV, chuẩn hóa cột và gộp thành tập dữ liệu thống nhất."""
    dfs = []
    for path in file_paths:
        # Tối ưu kiểu dữ liệu để đọc file hàng triệu dòng mượt mà
        dtypes = {
            "step": "int32",
            "amount": "float32",
            "oldbalanceOrg": "float32",
            "newbalanceOrig": "float32",
            "oldbalanceDest": "float32",
            "newbalanceDest": "float32",
            "isFraud": "int8",
            "isFlaggedFraud": "int8",
        }
        try:
            temp_df = pd.read_csv(path, dtype=dtypes, low_memory=False)
        except Exception:
            temp_df = pd.read_csv(path, low_memory=False)

        target_col = detect_target_column(temp_df)
        if target_col != "isFraud":
            temp_df = temp_df.rename(columns={target_col: "isFraud"})

        # Chuẩn hóa các cột cơ bản nếu người dùng nạp bộ dữ liệu cấu trúc khác
        if "type" not in temp_df.columns:
            temp_df["type"] = "GENERAL_TX"
        if "amount" not in temp_df.columns:
            num_cols = temp_df.select_dtypes(include=[np.number]).columns.tolist()
            num_cols = [c for c in num_cols if c != "isFraud"]
            temp_df["amount"] = temp_df[num_cols[0]].abs() if num_cols else 0.0
        if "step" not in temp_df.columns:
            temp_df["step"] = np.arange(1, len(temp_df) + 1)

        temp_df["dataset_source"] = os.path.basename(path)
        dfs.append(temp_df)

    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df


def classify_fraud_branch(row):
    """Phân loại từng giao dịch gian lận vào 4 nhánh rủi ro nghiệp vụ ngân hàng."""
    tx_type = str(row.get("type", ""))
    old_org = float(row.get("oldbalanceOrg", 0))
    new_org = float(row.get("newbalanceOrig", 0))
    old_dest = float(row.get("oldbalanceDest", 0))
    amount = float(row.get("amount", 0))

    if old_org > 0 and new_org == 0 and abs(old_org - amount) < 1.0:
        return "Nhánh 1: Chiếm đoạt tài khoản (Rút sạch số dư)"
    elif tx_type == "TRANSFER" and old_dest == 0:
        return "Nhánh 2: Chuyển khoản rửa tiền (Tài khoản đích rỗng)"
    elif tx_type == "CASH_OUT":
        return "Nhánh 3: Rút tiền mặt tẩu tán (Cash-out)"
    else:
        return "Nhánh 4: Thanh toán / Biến động bất thường khác"


def train_and_predict_on_dataset(df: pd.DataFrame, algo_name: str, max_normal_samples: int):
    """Huấn luyện mô hình AI từ dataset đã chọn và chạy dự đoán hàng loạt trên mẫu kiểm thử."""
    df_fraud = df[df["isFraud"] == 1].copy()
    df_normal = df[df["isFraud"] == 0].copy()

    if len(df_fraud) == 0:
        return None

    n_normal = min(len(df_normal), max_normal_samples)
    df_normal_sampled = df_normal.sample(n=n_normal, random_state=42)
    df_train_pool = pd.concat([df_fraud, df_normal_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Chọn các đặc trưng số và mã hóa cột 'type'
    exclude_cols = {"isFraud", "isFlaggedFraud", "nameOrig", "nameDest", "dataset_source"}
    feature_cols = [c for c in df_train_pool.columns if c not in exclude_cols]

    X = df_train_pool[feature_cols].copy()
    if "type" in X.columns:
        X["type"] = X["type"].astype("category").cat.codes

    # Chỉ giữ lại các cột dạng số cho mô hình học máy
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = df_train_pool["isFraud"].astype(int)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_train_pool.index, test_size=0.2, random_state=42, stratify=y
    )

    if algo_name == "Random Forest (Khuyên dùng)":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif algo_name == "Decision Tree (Cây quyết định)":
        model = DecisionTreeClassifier(max_depth=12, random_state=42)
    else:
        model = LogisticRegression(max_iter=500, class_weight="balanced")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] * 100

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred) * 100,
        "precision": precision_score(y_test, y_pred, zero_division=0) * 100,
        "recall": recall_score(y_test, y_pred, zero_division=0) * 100,
        "f1": f1_score(y_test, y_pred, zero_division=0) * 100,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "used_features": list(X.columns),
    }

    # Tạo bảng kết quả dự đoán đầu ra trực tiếp từ tập kiểm thử
    result_df = df_train_pool.loc[idx_test].copy()
    result_df["AI_Du_Doan"] = np.where(y_pred == 1, "🚨 Gian lận", "✅ Hợp lệ")
    result_df["Thuc_Te"] = np.where(y_test == 1, "Gian lận (1)", "Hợp lệ (0)")
    result_df["Ty_Le_Rui_Ro (%)"] = np.round(y_prob, 2)
    result_df["Phan_Loai_4_Nhanh"] = result_df.apply(
        lambda r: classify_fraud_branch(r) if r["AI_Du_Doan"] == "🚨 Gian lận" else "An toàn (Không thuộc nhóm rủi ro)",
        axis=1
    )

    return metrics, result_df


# ==============================================================================
# SIDEBAR: QUẢN LÝ ĐA DATASET & CẤU HÌNH AI HỌC TỰ ĐỘNG
# ==============================================================================
with st.sidebar:
    st.header("📂 Quản lý Dataset & AI")
    st.write("Nạp thêm nhiều file `.csv` khác nhau để AI tự động học và phân tích trực quan.")

    # 1. Upload thêm nhiều dataset mới
    uploaded_files = st.file_uploader(
        "Tải lên thêm Dataset mới (.csv)",
        type=["csv"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for uf in uploaded_files:
            save_path = os.path.join("data", uf.name)
            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(uf.getbuffer())
        st.success(f"Đã lưu {len(uploaded_files)} file vào thư mục `data/`!")

    # 2. Quét toàn bộ file CSV trong thư mục data/
    available_files = sorted(glob.glob("data/*.csv"))
    if not available_files:
        st.warning("Chưa có file `.csv` nào trong thư mục `data/`. Hãy đặt `paysim.csv` vào `data/` hoặc tải lên ở trên.")
        st.stop()

    selected_files = st.multiselect(
        "Chọn các Dataset đưa vào hệ thống:",
        options=available_files,
        default=[available_files[0]],
        format_func=lambda x: os.path.basename(x)
    )

    if not selected_files:
        st.warning("Vui lòng chọn ít nhất 1 file dataset để tiếp tục.")
        st.stop()

    st.divider()
    st.subheader("⚙️ Cấu hình Huấn luyện AI")
    algo_choice = st.selectbox(
        "Thuật toán Học máy:",
        [
            "Random Forest (Khuyên dùng)",
            "Decision Tree (Cây quyết định)",
            "Logistic Regression (Hồi quy Logistic)"
        ]
    )
    sample_limit = st.slider(
        "Số mẫu hợp lệ lấy để cân bằng học:",
        min_value=5000,
        max_value=100000,
        value=40000,
        step=5000,
        help="Giữ nguyên toàn bộ mẫu gian lận và lấy mẫu ngẫu nhiên giao dịch hợp lệ để AI học nhanh, tránh lệch nhãn."
    )

# ==============================================================================
# NẠP DỮ LIỆU & CHẠY PIPELINE
# ==============================================================================
with st.spinner("Đang đọc dữ liệu, tổng hợp thống kê và huấn luyện mô hình AI..."):
    df = load_and_merge_datasets(selected_files)
    ai_output = train_and_predict_on_dataset(df, algo_choice, sample_limit)

st.title("🛡️ Hệ Thống Phân Tích Trực Quan & Phát Hiện Gian Lận Ngân Hàng")
st.caption(f"Đang khai thác **{len(selected_files)}** bộ dữ liệu: `{', '.join([os.path.basename(f) for f in selected_files])}`")

# ==============================================================================
# 4 TAB CHỨC NĂNG CHÍNH
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Thống kê & Biểu đồ Gian lận",
    "🌳 2. Sơ đồ 4 nhánh & Điểm chạm Rủi ro",
    "🔄 3. Lưu đồ Dòng chảy Dữ liệu",
    "🤖 4. Kết quả AI Học & Dự đoán Dataset"
])

# ------------------------------------------------------------------------------
# TAB 1: BIỂU ĐỒ TỶ LỆ % GIAN LẬN & TỔNG HỢP THEO LOẠI HÌNH GIAO DỊCH
# ------------------------------------------------------------------------------
with tab1:
    total_tx = len(df)
    fraud_df = df[df["isFraud"] == 1]
    normal_df = df[df["isFraud"] == 0]

    fraud_count = len(fraud_df)
    normal_count = len(normal_df)
    fraud_pct = (fraud_count / total_tx * 100) if total_tx > 0 else 0
    normal_pct = (normal_count / total_tx * 100) if total_tx > 0 else 0

    total_amount_all = float(df["amount"].sum())
    total_amount_fraud = float(fraud_df["amount"].sum())
    fraud_amount_pct = (total_amount_fraud / total_amount_all * 100) if total_amount_all > 0 else 0

    # Hàng thẻ chỉ số KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tổng số giao dịch", f"{total_tx:,}")
    k2.metric("Giao dịch Gian lận (1)", f"{fraud_count:,}", f"{fraud_pct:.3f}% tổng số GD", delta_color="inverse")
    k3.metric("Giao dịch Hợp lệ (0)", f"{normal_count:,}", f"{normal_pct:.3f}% tổng số GD")
    k4.metric("Tổng tiền bị Gian lận", f"{total_amount_fraud:,.2f}", f"Chiếm {fraud_amount_pct:.2f}% tổng dòng tiền", delta_color="inverse")

    st.divider()

    # Hàng 1: Biểu đồ tỷ lệ % Gian lận vs Hợp lệ + Bảng chi tiết
    col_pie, col_detail = st.columns([1.1, 1.2])

    with col_pie:
        st.subheader("Tỷ lệ phần trăm Gian lận vs Hợp lệ")
        pie_data = pd.DataFrame({
            "Trạng thái": ["Hợp lệ (0)", "Gian lận (1)"],
            "Số lượng": [normal_count, fraud_count]
        })
        fig_pie = px.pie(
            pie_data,
            names="Trạng thái",
            values="Số lượng",
            hole=0.48,
            color="Trạng thái",
            color_discrete_map={"Hợp lệ (0)": "#2ECC71", "Gian lận (1)": "#E74C3C"}
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label+value")
        fig_pie.update_layout(height=380, margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_detail:
        st.subheader("Bảng thống kê chi tiết toàn tập dữ liệu")
        summary_overview = pd.DataFrame([
            {
                "Phân loại": "✅ Giao dịch Hợp lệ (isFraud = 0)",
                "Số lượng giao dịch": f"{normal_count:,}",
                "Tỷ lệ số lượng (%)": f"{normal_pct:.4f}%",
                "Tổng giá trị tiền": f"{float(normal_df['amount'].sum()):,.2f}",
                "Trung bình tiền / GD": f"{float(normal_df['amount'].mean()) if normal_count > 0 else 0:,.2f}",
            },
            {
                "Phân loại": "🚨 Giao dịch Gian lận (isFraud = 1)",
                "Số lượng giao dịch": f"{fraud_count:,}",
                "Tỷ lệ số lượng (%)": f"{fraud_pct:.4f}%",
                "Tổng giá trị tiền": f"{total_amount_fraud:,.2f}",
                "Trung bình tiền / GD": f"{float(fraud_df['amount'].mean()) if fraud_count > 0 else 0:,.2f}",
            },
            {
                "Phân loại": "📌 TỔNG CỘNG TOÀN BỘ",
                "Số lượng giao dịch": f"{total_tx:,}",
                "Tỷ lệ số lượng (%)": "100.0000%",
                "Tổng giá trị tiền": f"{total_amount_all:,.2f}",
                "Trung bình tiền / GD": f"{float(df['amount'].mean()) if total_tx > 0 else 0:,.2f}",
            }
        ])
        st.dataframe(summary_overview, use_container_width=True, hide_index=True)

        st.info(
            f"💡 **Nhận xét nhanh:** Dù số vụ gian lận chỉ chiếm **{fraud_pct:.3f}%** tổng số giao dịch, "
            f"nhưng giá trị trung bình mỗi vụ gian lận lên tới **{float(fraud_df['amount'].mean()) if fraud_count > 0 else 0:,.2f}** "
            f"(cao gấp nhiều lần giao dịch thông thường)."
        )

    st.divider()

    # Hàng 2: Tính tổng của từng loại hình giao dịch gian lận
    st.subheader("Thống kê tổng hợp theo từng Loại hình Giao dịch (Type)")

    type_stats = df.groupby("type", observed=False).agg(
        Tong_So_GD=("isFraud", "count"),
        So_GD_Gian_Lan=("isFraud", "sum"),
        Tong_Tien_Tat_Ca=("amount", "sum"),
    ).reset_index()

    fraud_amount_by_type = fraud_df.groupby("type", observed=False)["amount"].sum().reset_index().rename(
        columns={"amount": "Tong_Tien_Gian_Lan"}
    )
    type_stats = pd.merge(type_stats, fraud_amount_by_type, on="type", how="left").fillna({"Tong_Tien_Gian_Lan": 0.0})
    type_stats["Ty_Le_Gian_Lan (%)"] = (type_stats["So_GD_Gian_Lan"] / type_stats["Tong_So_GD"] * 100).round(4)
    type_stats["Ty_Trong_Vu_Gian_Lan (%)"] = (
        (type_stats["So_GD_Gian_Lan"] / fraud_count * 100).round(2) if fraud_count > 0 else 0
    )

    c_bar1, c_bar2 = st.columns(2)
    with c_bar1:
        fig_bar_count = px.bar(
            type_stats,
            x="type",
            y="So_GD_Gian_Lan",
            color="type",
            text="So_GD_Gian_Lan",
            title="Tổng số vụ GIAN LẬN theo loại hình giao dịch",
            labels={"type": "Loại giao dịch", "So_GD_Gian_Lan": "Số vụ gian lận"}
        )
        fig_bar_count.update_traces(textposition="outside")
        fig_bar_count.update_layout(showlegend=False, height=370)
        st.plotly_chart(fig_bar_count, use_container_width=True)

    with c_bar2:
        fig_bar_amt = px.bar(
            type_stats,
            x="type",
            y="Tong_Tien_Gian_Lan",
            color="type",
            text_auto=".2s",
            title="Tổng giá trị TIỀN GIAN LẬN theo loại hình giao dịch",
            labels={"type": "Loại giao dịch", "Tong_Tien_Gian_Lan": "Tổng tiền gian lận"}
        )
        fig_bar_amt.update_traces(textposition="outside")
        fig_bar_amt.update_layout(showlegend=False, height=370)
        st.plotly_chart(fig_bar_amt, use_container_width=True)

    # Bảng tổng hợp số liệu chi tiết từng loại hình giao dịch
    display_type_table = type_stats.copy()
    display_type_table.columns = [
        "Loại hình giao dịch",
        "Tổng số GD",
        "Tổng số GD Gian lận",
        "Tổng tiền toàn bộ GD",
        "Tổng tiền GD Gian lận",
        "Tỷ lệ gian lận nội bộ (%)",
        "Tỷ trọng trên tổng vụ gian lận (%)"
    ]
    st.dataframe(display_type_table, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 2: SƠ ĐỒ 4 NHÁNH PHÂN LOẠI GIAN LẬN & ĐIỂM CHẠM RỦI RO GIAO DỊCH
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("🌳 Sơ đồ 4 Nhánh Phân Loại Gian Lận & Điểm Chạm Rủi Ro Giao Dịch Ngân Hàng")
    st.write("Mô hình hóa các kịch bản tấn công tài chính và xác định các điểm chạm (Touchpoints) cần kiểm soát trên hành trình giao dịch.")

    # Vẽ sơ đồ 4 nhánh trực quan bằng HTML/CSS tương thích cả Dark/Light Mode
    branch_diagram_html = """
    <style>
        .tree-root {
            text-align: center;
            background: linear-gradient(135deg, #1E3A8A, #2563EB);
            color: white;
            padding: 16px 24px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 18px;
            margin: 0 auto 10px auto;
            max-width: 620px;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        }
        .tree-connector {
            text-align: center;
            font-size: 22px;
            color: #64748B;
            margin-bottom: 12px;
            letter-spacing: 4px;
        }
        .branch-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin-bottom: 20px;
        }
        .branch-card {
            border-radius: 12px;
            padding: 16px;
            border-top: 6px solid;
            background-color: rgba(148, 163, 184, 0.1);
            border-left: 1px solid rgba(148, 163, 184, 0.25);
            border-right: 1px solid rgba(148, 163, 184, 0.25);
            border-bottom: 1px solid rgba(148, 163, 184, 0.25);
        }
        .b1 { border-top-color: #EF4444; }
        .b2 { border-top-color: #F59E0B; }
        .b3 { border-top-color: #8B5CF6; }
        .b4 { border-top-color: #06B6D4; }
        .branch-title {
            font-weight: 700;
            font-size: 15px;
            margin-bottom: 8px;
        }
        .branch-desc {
            font-size: 13px;
            line-height: 1.5;
            margin-bottom: 10px;
            opacity: 0.9;
        }
        .touchpoint-box {
            background-color: rgba(239, 68, 68, 0.12);
            border-left: 3px solid #EF4444;
            padding: 8px 10px;
            border-radius: 6px;
            font-size: 12.5px;
            line-height: 1.45;
        }
        @media (max-width: 900px) {
            .branch-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>

    <div class="tree-root">
        🛡️ HỆ THỐNG PHÂN LOẠI GIAN LẬN GIAO DỊCH NGÂN HÀNG (4 NHÁNH RỦI RO)
    </div>
    <div class="tree-connector">⬇ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ⬇ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ⬇ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ⬇</div>

    <div class="branch-grid">
        <div class="branch-card b1">
            <div class="branch-title">1️⃣ Chiếm đoạt tài khoản (Account Takeover)</div>
            <div class="branch-desc">
                <b>Dấu hiệu dữ liệu:</b><br>
                • Rút sạch 100% số dư hiện có (<code>oldbalanceOrg == amount</code>, <code>newbalanceOrig == 0</code>).<br>
                • Giao dịch diễn ra dồn dập trong khung giờ bất thường.
            </div>
            <div class="touchpoint-box">
                <b>🎯 Điểm chạm rủi ro:</b><br>
                • Cổng đăng nhập Mobile Banking (thiết bị lạ / IP mới).<br>
                • Điểm chạm đổi mật khẩu, thay đổi số điện thoại nhận Smart OTP.
            </div>
        </div>

        <div class="branch-card b2">
            <div class="branch-title">2️⃣ Chuyển khoản rửa tiền (Mule Transfer)</div>
            <div class="branch-desc">
                <b>Dấu hiệu dữ liệu:</b><br>
                • Loại giao dịch <code>TRANSFER</code> giá trị lớn.<br>
                • Tài khoản đích có số dư trước bằng 0 (<code>oldbalanceDest == 0</code>) và không lưu số dư.
            </div>
            <div class="touchpoint-box">
                <b>🎯 Điểm chạm rủi ro:</b><br>
                • Cổng chuyển tiền nhanh liên ngân hàng Napas 24/7.<br>
                • Tài khoản nhận tiền mới mở (tài khoản rác / trung gian).
            </div>
        </div>

        <div class="branch-card b3">
            <div class="branch-title">3️⃣ Rút tiền mặt tẩu tán (Cash-Out Fraud)</div>
            <div class="branch-desc">
                <b>Dấu hiệu dữ liệu:</b><br>
                • Loại giao dịch <code>CASH_OUT</code> xảy ra ngay sau lệnh chuyển khoản chiếm đoạt.<br>
                • Nhằm chuyển đổi tiền số thành tiền mặt để xóa dấu vết.
            </div>
            <div class="touchpoint-box">
                <b>🎯 Điểm chạm rủi ro:</b><br>
                • Cây ATM / Điểm giao dịch đại lý rút tiền (Merchant Cash-out).<br>
                • Xác thực sinh trắc học khuôn mặt đối với giao dịch hạn mức cao.
            </div>
        </div>

        <div class="branch-card b4">
            <div class="branch-title">4️⃣ Gian lận Thanh toán & Lách ngưỡng (Payment/Smurfing)</div>
            <div class="branch-desc">
                <b>Dấu hiệu dữ liệu:</b><br>
                • Các giao dịch <code>PAYMENT</code>, <code>DEBIT</code> hoặc chia nhỏ số tiền dưới ngưỡng 200.000 để né cờ cảnh báo tĩnh.
            </div>
            <div class="touchpoint-box">
                <b>🎯 Điểm chạm rủi ro:</b><br>
                • Cổng thanh toán trực tuyến (E-commerce / POS / QR Pay).<br>
                • Điểm chạm liên kết ví điện tử & xác thực 3D-Secure.
            </div>
        </div>
    </div>
    """
    st.markdown(branch_diagram_html, unsafe_allow_html=True)

    st.divider()

    # Thống kê thực tế từ dataset đang nạp vào 4 nhánh trên
    st.subheader("📊 Phân bổ thực tế các vụ gian lận từ Dataset vào 4 nhánh trên")
    if fraud_count > 0:
        fraud_with_branch = fraud_df.copy()
        fraud_with_branch["Nhom_Gian_Lan"] = fraud_with_branch.apply(classify_fraud_branch, axis=1)

        branch_summary = fraud_with_branch.groupby("Nhom_Gian_Lan").agg(
            So_Vu_Gian_Lan=("isFraud", "count"),
            Tong_Thiet_Hai=("amount", "sum")
        ).reset_index()
        branch_summary["Ty_Le (%)"] = (branch_summary["So_Vu_Gian_Lan"] / fraud_count * 100).round(2)

        b_col1, b_col2 = st.columns([1.2, 1])
        with b_col1:
            fig_branch = px.bar(
                branch_summary,
                y="Nhom_Gian_Lan",
                x="So_Vu_Gian_Lan",
                orientation="h",
                color="Nhom_Gian_Lan",
                text="So_Vu_Gian_Lan",
                title="Số lượng giao dịch gian lận phân theo 4 nhánh",
                labels={"Nhom_Gian_Lan": "Nhánh phân loại", "So_Vu_Gian_Lan": "Số vụ phát hiện"}
            )
            fig_branch.update_traces(textposition="outside")
            fig_branch.update_layout(showlegend=False, height=340)
            st.plotly_chart(fig_branch, use_container_width=True)

        with b_col2:
            st.dataframe(branch_summary, use_container_width=True, hide_index=True)
    else:
        st.warning("Tập dữ liệu hiện tại không có bản ghi gian lận (`isFraud = 1`) để phân nhánh.")

# ------------------------------------------------------------------------------
# TAB 3: LƯU ĐỒ MÔ TẢ DÒNG CHẢY DỮ LIỆU (DATA FLOW PIPELINE)
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("🔄 Lưu Đồ Mô Tả Dòng Chảy Dữ Liệu (Data Flow Diagram - DFD)")
    st.write("Kiến trúc luồng xử lý dữ liệu khép kín từ bước nạp đa tập dữ liệu đầu vào cho đến khi xuất dự báo rủi ro từ AI.")

    train_n = ai_output[0]["train_rows"] if ai_output else 0
    test_n = ai_output[0]["test_rows"] if ai_output else 0

    dfd_html = f"""
    <style>
        .pipeline-container {{
            display: flex;
            align-items: stretch;
            justify-content: space-between;
            gap: 10px;
            margin: 18px 0;
            flex-wrap: wrap;
        }}
        .pipe-step {{
            flex: 1;
            min-width: 190px;
            background-color: rgba(148, 163, 184, 0.1);
            border: 1px solid rgba(59, 130, 246, 0.4);
            border-radius: 12px;
            padding: 16px 14px;
            position: relative;
        }}
        .pipe-badge {{
            display: inline-block;
            background: #2563EB;
            color: white;
            font-weight: 700;
            font-size: 12px;
            padding: 3px 10px;
            border-radius: 20px;
            margin-bottom: 8px;
        }}
        .pipe-title {{
            font-weight: 700;
            font-size: 15px;
            margin-bottom: 8px;
        }}
        .pipe-body {{
            font-size: 12.5px;
            line-height: 1.5;
            opacity: 0.92;
        }}
        .pipe-metric {{
            margin-top: 10px;
            padding-top: 8px;
            border-top: 1px dashed rgba(148, 163, 184, 0.4);
            font-weight: 600;
            font-size: 12px;
            color: #10B981;
        }}
        .pipe-arrow {{
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            font-weight: bold;
            color: #3B82F6;
        }}
    </style>

    <div class="pipeline-container">
        <div class="pipe-step">
            <div class="pipe-badge">BƯỚC 1: THU THẬP</div>
            <div class="pipe-title">📥 Nạp Đa Dataset (.CSV)</div>
            <div class="pipe-body">
                • Đọc đồng thời nhiều file dữ liệu lịch sử giao dịch.<br>
                • Tự động nhận diện cột nhãn (<code>isFraud</code> / <code>Class</code>) và chuẩn hóa cấu trúc.
            </div>
            <div class="pipe-metric">Dữ liệu thô: {total_tx:,} dòng</div>
        </div>

        <div class="pipe-arrow">➔</div>

        <div class="pipe-step">
            <div class="pipe-badge">BƯỚC 2: TIỀN XỬ LÝ</div>
            <div class="pipe-title">🧹 Làm sạch & Cân bằng</div>
            <div class="pipe-body">
                • Loại bỏ mã ID nhiễu (<code>nameOrig</code>, <code>nameDest</code>) và cột chống rò rỉ (<code>isFlaggedFraud</code>).<br>
                • Xử lý mất cân bằng lớp (Undersampling) & mã hóa biến phân loại.
            </div>
            <div class="pipe-metric">Tập học cân bằng: {train_n + test_n:,} dòng</div>
        </div>

        <div class="pipe-arrow">➔</div>

        <div class="pipe-step">
            <div class="pipe-badge">BƯỚC 3: CHIA TẬP DỮ LIỆU</div>
            <div class="pipe-title">✂️ Phân tách Train / Test</div>
            <div class="pipe-body">
                • Chia phân tầng (Stratified Split) tỷ lệ <b>80% Huấn luyện</b> và <b>20% Kiểm thử</b>.<br>
                • Bảo toàn tỷ lệ mẫu gian lận ở cả hai tập.
            </div>
            <div class="pipe-metric">Train: {train_n:,} | Test: {test_n:,}</div>
        </div>

        <div class="pipe-arrow">➔</div>

        <div class="pipe-step">
            <div class="pipe-badge">BƯỚC 4: HỌC MÁY (AI)</div>
            <div class="pipe-title">🧠 Huấn luyện Mô hình</div>
            <div class="pipe-body">
                • Thuật toán: <b>{algo_choice.split(' (')[0]}</b>.<br>
                • Học quy luật biến động số dư trước/sau và loại hình giao dịch.
            </div>
            <div class="pipe-metric">Độ chính xác: {ai_output[0]['accuracy']:.2f}%</div>
        </div>

        <div class="pipe-arrow">➔</div>

        <div class="pipe-step">
            <div class="pipe-badge">BƯỚC 5: ĐẦU RA DỰ BÁO</div>
            <div class="pipe-title">📤 Cảnh báo & Trực quan</div>
            <div class="pipe-body">
                • Xuất nhãn dự đoán gian lận hàng loạt kèm xác suất rủi ro (%).<br>
                • Phân nhánh rủi ro & xuất file báo cáo CSV.
            </div>
            <div class="pipe-metric">Sẵn sàng xuất báo cáo</div>
        </div>
    </div>
    """
    st.markdown(dfd_html, unsafe_allow_html=True)

    st.divider()
    st.subheader("📋 Nhật ký luồng dữ liệu đang chạy trực tiếp")
    if ai_output:
        st.write(f"- **Các file nguồn đầu vào:** `{', '.join([os.path.basename(f) for f in selected_files])}`")
        st.write(f"- **Các đặc trưng (Features) mô hình AI đã trích xuất để học:** `{', '.join(ai_output[0]['used_features'])}`")
        st.write(f"- **Tỷ lệ phân chia dữ liệu:** 80% Huấn luyện (`{train_n:,}` dòng) — 20% Kiểm thử & Dự đoán (`{test_n:,}` dòng).")

# ------------------------------------------------------------------------------
# TAB 4: KẾT QUẢ AI HỌC & DỰ ĐOÁN TRỰC TIẾP TRÊN DATASET (KHÔNG CẦN NHẬP TAY)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("🤖 Kết Quả Huấn Luyện AI & Đầu Ra Dự Đoán Trên Tập Dữ Liệu")
    st.write("Mô hình AI tự động học từ các dataset bạn đã chọn ở thanh bên trái và xuất kết quả dự đoán hàng loạt dưới đây:")

    if ai_output is None:
        st.error("Dataset được chọn không có mẫu gian lận (`isFraud = 1`) nên không thể huấn luyện mô hình phân loại nhị phân.")
    else:
        metrics, result_df = ai_output

        # 4 chỉ số đánh giá chất lượng mô hình AI
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Độ chính xác (Accuracy)", f"{metrics['accuracy']:.2f}%")
        m2.metric("Độ chuẩn xác (Precision)", f"{metrics['precision']:.2f}%")
        m3.metric("Độ phủ bắt gian lận (Recall)", f"{metrics['recall']:.2f}%")
        m4.metric("Điểm cân bằng (F1-Score)", f"{metrics['f1']:.2f}%")

        st.divider()

        # Bộ lọc xem kết quả dự đoán trên dataset
        filter_pred = st.radio(
            "Lọc danh sách giao dịch sau khi AI dự đoán:",
            [
                "Hiển thị tất cả kết quả dự đoán",
                "Chỉ xem các giao dịch AI dự đoán GIAN LẬN (🚨)",
                "Chỉ xem các giao dịch AI dự đoán HỢP LỆ (✅)"
            ],
            horizontal=True
        )

        filtered_res = result_df
        if filter_pred == "Chỉ xem các giao dịch AI dự đoán GIAN LẬN (🚨)":
            filtered_res = result_df[result_df["AI_Du_Doan"] == "🚨 Gian lận"]
        elif filter_pred == "Chỉ xem các giao dịch AI dự đoán HỢP LỆ (✅)":
            filtered_res = result_df[result_df["AI_Du_Doan"] == "✅ Hợp lệ"]

        # Sắp xếp cột quan trọng lên đầu cho dễ quan sát
        priority_cols = [
            "AI_Du_Doan",
            "Ty_Le_Rui_Ro (%)",
            "Thuc_Te",
            "Phan_Loai_4_Nhanh",
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "dataset_source"
        ]
        ordered_cols = [c for c in priority_cols if c in filtered_res.columns] + [
            c for c in filtered_res.columns if c not in priority_cols and c != "isFraud"
        ]

        st.dataframe(filtered_res[ordered_cols].head(500), use_container_width=True, height=400)

        # Nút tải file kết quả dự đoán CSV về máy để nộp kèm báo cáo
        csv_export = filtered_res[ordered_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Tải xuống kết quả dự đoán của AI (.CSV)",
            data=csv_export,
            file_name="ket_qua_du_doan_gian_lan.csv",
            mime="text/csv",
            type="primary"
        )