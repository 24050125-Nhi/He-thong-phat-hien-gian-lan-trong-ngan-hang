import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("1. Đang đọc dữ liệu từ data/paysim.csv...")
df = pd.read_csv("data/paysim.csv")

# Lấy toàn bộ giao dịch gian lận (isFraud == 1) và 50.000 giao dịch hợp lệ (isFraud == 0)
df_fraud = df[df["isFraud"] == 1]
df_normal = df[df["isFraud"] == 0].sample(n=50000, random_state=42)

# Gộp lại và xáo trộn ngẫu nhiên
df_balanced = pd.concat([df_fraud, df_normal]).sample(frac=1, random_state=42).reset_index(drop=True)

# Chọn 7 đặc trưng quan trọng nhất để đưa vào mô hình và giao diện nhập liệu
features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
target = "isFraud"

X = df_balanced[features].copy()
y = df_balanced[target]

# Mã hóa cột 'type' từ dạng chữ sang dạng số
type_mapping = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4,
}
X["type"] = X["type"].map(type_mapping)

# Chia dữ liệu: 80% để học (Train), 20% để kiểm tra (Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("2. Đang huấn luyện mô hình Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Đánh giá độ chính xác trên tập kiểm tra
y_pred = model.predict(X_test)
print(f"\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ---")
print(f"Độ chính xác tổng thể (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nBáo cáo chi tiết:")
print(classification_report(y_test, y_pred, target_names=["Hợp lệ (0)", "Gian lận (1)"]))

# Lưu mô hình ra file để ứng dụng sử dụng
joblib.dump(model, "fraud_model.pkl")
print("3. Đã lưu mô hình thành công vào file 'fraud_model.pkl'!")