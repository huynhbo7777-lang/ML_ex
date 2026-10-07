"""
BÀI TẬP LỚN GIỮA KỲ MÔN HỌC MÁY (MACHINE LEARNING)
File: midterm_ml_project.py
Dataset: IBM HR Analytics Employee Attrition & Performance
Định dạng dữ liệu đầu vào: WA_Fn-UseC_-HR-Employee-Attrition.csv (đặt cùng thư mục)
Đầu ra thực nghiệm: artifacts/ml_results.json
"""

import os
import json
import inspect
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

# =====================================================================
# BƯỚC 1: ĐIỀN MSSV ĐỂ SINH CẤU HÌNH CÁ NHÂN HÓA (BẮT BUỘC)
# =====================================================================
STUDENT_ID = "24701841"  # <-- THÍ SINH ĐIỀN CHÍNH XÁC MSSV CỦA MÌNH VÀO ĐÂY


def get_student_ml_config(sid: str):
    sid = str(sid).strip()
    if not sid.isdigit():
        raise ValueError("MSSV không hợp lệ: chỉ được chứa chữ số.")

    digits = [int(c) for c in sid]
    
    last = digits[-1]
    second_last = digits[-2] if len(digits) >= 2 else 0
    third_last = digits[-3] if len(digits) >= 3 else 0

    # 1. Thuật toán Chuẩn hóa đặc trưng (Scaler)
    scalers = ["standard", "minmax", "robust"]
    scaler_type = scalers[last % 3]

    # 2. Chiến lược xử lý khuyết thiếu (Imputer)
    impute_strategy = "mean" if (second_last % 2 == 0) else "median"

    # 3. Loại Hồi quy & Hệ số phạt Alpha (Regression)
    reg_type = "ridge" if (last % 2 == 0) else "lasso"
    reg_alpha = round(0.1 + (second_last % 5) * 0.2, 2)

    # 4. Siêu tham số Phân loại (Classification)
    max_depth = 3 + (third_last % 4)  # Miền giá trị: [3, 4, 5, 6]
    logreg_c = round(0.5 + (last % 4) * 0.5, 2)  # Miền giá trị: [0.5, 1.0, 1.5, 2.0]

    # 5. Hạt giống ngẫu nhiên cá nhân hóa theo 4 số cuối MSSV
    student_seed = int(sid[-4:]) if len(sid) >= 4 else int(sid)

    return {
        "student_id": str(sid).strip(),
        "scaler_type": scaler_type,
        "impute_strategy": impute_strategy,
        "reg_type": reg_type,
        "reg_alpha": reg_alpha,
        "max_depth": max_depth,
        "logreg_c": logreg_c,
        "student_seed": student_seed,
        "test_size": 0.25
    }

CONFIG = get_student_ml_config(STUDENT_ID)


# =====================================================================
# BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU & PHÂN NHÁNH THỰC NGHIỆM
# =====================================================================
def load_and_preprocess_data(csv_path="WA_Fn-UseC_-HR-Employee-Attrition.csv", config=CONFIG):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Không tìm thấy file '{csv_path}'! "
            "Vui lòng tải tập dữ liệu IBM HR Attrition từ Kaggle về và đặt cùng thư mục với script."
        )

    df_raw = pd.read_csv(csv_path)

    # 1. Lấy mẫu 80% dữ liệu gốc theo seed cá nhân hóa.
    # Đây là một biến thể dữ liệu phụ thuộc MSSV, không đảm bảo các sinh viên hoàn toàn không trùng mẫu.
    df_sampled = df_raw.sample(
        frac=0.8,
        random_state=config["student_seed"]
    ).reset_index(drop=True)

    # 2. Tách biến mục tiêu
    y_reg = df_sampled["MonthlyIncome"].to_numpy()
    y_clf = (
        df_sampled["Attrition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
        .astype(int)
        .to_numpy()
    )

    # 3. Loại bỏ target, ID và các cột hằng
    drop_cols = [
        "MonthlyIncome",
        "Attrition",
        "EmployeeCount",
        "Over18",
        "StandardHours",
        "EmployeeNumber",
    ]
    X_features = df_sampled.drop(columns=drop_cols)

    # 4. Chèn 5% NaN trên 3 cột số đầu tiên để kiểm thử imputation
    try:
        rng = np.random.default_rng(config["student_seed"])
    except AttributeError:
        # NumPy cũ không có default_rng. RandomState là API tương thích cũ.
        rng = np.random.RandomState(config["student_seed"])
    numeric_cols_all = X_features.select_dtypes(include=[np.number]).columns.tolist()
    injected_nan_cols = numeric_cols_all[:3]

    for col in injected_nan_cols:
        mask = rng.random(len(X_features)) < 0.05
        X_features.loc[mask, col] = np.nan

    # 5. Split trước khi fit mọi phép tiền xử lý.
    # stratify theo Attrition để giữ tỷ lệ lớp giữa train/test.
    X_train_df, X_test_df, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X_features,
        y_reg,
        y_clf,
        test_size=config["test_size"],
        random_state=config["student_seed"],
        stratify=y_clf,
    )

    # 6. Xác định nhóm cột dựa trên tập train
    numeric_cols = X_train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # 7. Chọn scaler
    if config["scaler_type"] == "standard":
        scaler = StandardScaler()
    elif config["scaler_type"] == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()

    # 8. Pipeline cho biến số: impute -> scale
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=config["impute_strategy"])), # xử lý các dữ liệu khuyết thiếu, SimpleImputer có thể điền mean, median hoặc config khác
            ("scaler", scaler),
        ]
    )

    # 9. One-hot encoding chỉ fit trên train; handle_unknown tránh lỗi khi test có category mới
    # Tương thích nhiều phiên bản scikit-learn:
    # - sklearn mới dùng sparse_output=False
    # - sklearn cũ dùng sparse=False
    onehot_params = {
        "drop": "first",
        "handle_unknown": "ignore",
    }

    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        onehot_params["sparse_output"] = False
    else:
        onehot_params["sparse"] = False

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")), # điền dữ liệu khuyết thiếu bằng giá trị có tần suất lớn nhất
            ("onehot", OneHotEncoder(**onehot_params)), # mã hóa nhị phân với drop=first và handle_unknown=ignore tránh cho test có thể sẽ xuất hiện 1 giá trị hoàn toàn mới
        ]
    )

    # Không truyền verbose_feature_names_out để tránh lỗi trên sklearn cũ.
    # Phần lấy tên feature bên dưới đã có cơ chế tương thích riêng.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    # 10. Lấy các object đã fit để xuất artifact
    fitted_num_pipeline = preprocessor.named_transformers_["num"]
    imputer = fitted_num_pipeline.named_steps["imputer"]
    scaler = fitted_num_pipeline.named_steps["scaler"]

    # 11. Tên đặc trưng sau preprocessing, tương thích nhiều phiên bản sklearn.
    # Ưu tiên API mới -> API cũ -> fallback cuối cùng.
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except (AttributeError, NotImplementedError):
        feature_names = list(numeric_cols)

        if categorical_cols:
            fitted_cat_pipeline = preprocessor.named_transformers_["cat"]
            fitted_onehot = fitted_cat_pipeline.named_steps["onehot"]

            if hasattr(fitted_onehot, "get_feature_names_out"):
                try:
                    cat_feature_names = fitted_onehot.get_feature_names_out(categorical_cols).tolist()
                except TypeError:
                    cat_feature_names = fitted_onehot.get_feature_names_out().tolist()
            elif hasattr(fitted_onehot, "get_feature_names"):
                try:
                    cat_feature_names = fitted_onehot.get_feature_names(categorical_cols).tolist()
                except TypeError:
                    cat_feature_names = fitted_onehot.get_feature_names().tolist()
            else:
                # Fallback cuối cùng nếu phiên bản sklearn quá cũ.
                n_cat_features = X_train.shape[1] - len(numeric_cols)
                cat_feature_names = [
                    "cat_feature_{}".format(i)
                    for i in range(n_cat_features)
                ]

            feature_names.extend(cat_feature_names)

    # Đảm bảo số lượng tên feature luôn khớp số cột ma trận sau preprocessing.
    if len(feature_names) != X_train.shape[1]:
        feature_names = ["feature_{}".format(i) for i in range(X_train.shape[1])]

    # Lưu metadata cần thiết để diễn giải feature index 0
    preprocessing_meta = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "first_numeric_feature": numeric_cols[0] if numeric_cols else None,
        "injected_nan_cols": injected_nan_cols,
    }

    return (
        X_train,
        X_test,
        y_reg_train,
        y_reg_test,
        y_clf_train,
        y_clf_test,
        imputer,
        scaler,
        feature_names,
        preprocessing_meta,
    )


# =====================================================================
# BƯỚC 3: MÔ HÌNH HỒI QUY (REGRESSION)
# =====================================================================
def train_regression_model(X_train, y_train, config=CONFIG):
    if config["reg_type"] == "ridge":
        model = Ridge(alpha=config["reg_alpha"], random_state=config["student_seed"])
    else:
        # Lasso sử dụng Coordinate Descent
        model = Lasso(alpha=config["reg_alpha"], random_state=config["student_seed"], max_iter=2000)
    
    model.fit(X_train, y_train)
    return model


# =====================================================================
# BƯỚC 4: MÔ HÌNH PHÂN LOẠI (CLASSIFICATION)
# =====================================================================
def train_classification_models(X_train, y_train, config=CONFIG):
    # Model 1: Logistic Regression với chuẩn phạt L2
    logreg = LogisticRegression(
        C=config["logreg_c"], 
        random_state=config["student_seed"], 
        max_iter=1000, 
        solver="lbfgs"
    )
    logreg.fit(X_train, y_train)

    # Model 2: Decision Tree với giới hạn độ sâu max_depth
    tree = DecisionTreeClassifier(
        max_depth=config["max_depth"], 
        criterion="gini", 
        random_state=config["student_seed"]
    )
    tree.fit(X_train, y_train)

    return logreg, tree


# =====================================================================
# BƯỚC 5: TRÍCH XUẤT DẤU VẾT THỰC NGHIỆM ĐỂ LÀM BÀI THI (ARTIFACTS)
# =====================================================================
def export_student_artifacts(X_train, X_test, y_reg_train, y_reg_test,
                              y_clf_train, y_clf_test, reg_model, logreg_m,
                              tree_m, imputer, scaler, feature_names,
                              preprocessing_meta, config=CONFIG):
    
    # 1. Kích thước dữ liệu thực tế
    n_train = int(X_train.shape[0])
    n_test = int(X_test.shape[0])
    n_features = int(X_train.shape[1])

    # 2. Thống kê tiền xử lý trên đặc trưng số đầu tiên
    # Giữ lại các key cũ để tương thích với ngân hàng câu hỏi,
    # nhưng scaler_scale_0 luôn là đúng thuộc tính scale_ của scaler.
    first_numeric_feature = preprocessing_meta["first_numeric_feature"]
    imputer_val_0 = round(float(imputer.statistics_[0]), 4)

    scaler_center_0 = None
    scaler_scale_0 = None
    scaler_range_0 = None
    scaler_min_0 = None

    if isinstance(scaler, StandardScaler):
        scaler_center_0 = round(float(scaler.mean_[0]), 4)
        scaler_scale_0 = round(float(scaler.scale_[0]), 4)
    elif isinstance(scaler, MinMaxScaler):
        scaler_center_0 = round(float(scaler.data_min_[0]), 4)
        scaler_scale_0 = round(float(scaler.scale_[0]), 6)
        scaler_range_0 = round(float(scaler.data_range_[0]), 4)
        scaler_min_0 = round(float(scaler.min_[0]), 6)
    elif isinstance(scaler, RobustScaler):
        scaler_center_0 = round(float(scaler.center_[0]), 4)
        scaler_scale_0 = round(float(scaler.scale_[0]), 4)

    # 3. Kết quả Hồi quy thực tế
    reg_train_preds = reg_model.predict(X_train)
    reg_test_preds = reg_model.predict(X_test)
    train_mse = round(float(mean_squared_error(y_reg_train, reg_train_preds)), 4)
    test_mse = round(float(mean_squared_error(y_reg_test, reg_test_preds)), 4)
    intercept = round(float(reg_model.intercept_), 4)
    
    # Đếm hệ số gần 0. Chỉ Lasso mới có ý nghĩa sparsity trực tiếp.
    zero_coef_count = int(np.sum(np.abs(reg_model.coef_) < 1e-5))
    is_sparse_model = bool(config["reg_type"] == "lasso")

    # 4. Kết quả Phân loại thực tế
    logreg_train_acc = round(float(accuracy_score(y_clf_train, logreg_m.predict(X_train))), 4)
    logreg_test_acc = round(float(accuracy_score(y_clf_test, logreg_m.predict(X_test))), 4)
    logreg_test_f1 = round(float(f1_score(y_clf_test, logreg_m.predict(X_test), zero_division=0)), 4)

    tree_train_acc = round(float(accuracy_score(y_clf_train, tree_m.predict(X_train))), 4)
    tree_test_acc = round(float(accuracy_score(y_clf_test, tree_m.predict(X_test))), 4)
    tree_test_f1 = round(float(f1_score(y_clf_test, tree_m.predict(X_test), zero_division=0)), 4)

    tree_actual_depth = int(tree_m.get_depth())
    tree_n_leaves = int(tree_m.get_n_leaves())

    # Xác định đặc trưng rẽ nhánh tại nút gốc (Root node feature)
    root_feature_idx = tree_m.tree_.feature[0]
    root_feature_name = feature_names[root_feature_idx] if root_feature_idx >= 0 else "None"

    results = {
        "student_id": config["student_id"],
        "configuration": {
            "scaler_type": config["scaler_type"],
            "impute_strategy": config["impute_strategy"],
            "reg_type": config["reg_type"],
            "reg_alpha": config["reg_alpha"],
            "max_depth": config["max_depth"],
            "logreg_c": config["logreg_c"],
            "student_seed": config["student_seed"]
        },
        "dataset_shape": {
            "n_train": n_train,
            "n_test": n_test,
            "n_features": n_features
        },
        "preprocessing_stats": {
            "first_numeric_feature": first_numeric_feature,
            "imputer_val_0": imputer_val_0,
            "scaler_center_0": scaler_center_0,
            "scaler_scale_0": scaler_scale_0,
            "scaler_range_0": scaler_range_0,
            "scaler_min_0": scaler_min_0
        },
        "regression_results": {
            "train_mse": train_mse,
            "test_mse": test_mse,
            "intercept": intercept,
            "zero_coef_count": zero_coef_count,
            "zero_coef_count_interpretation": (
                "sparsity_count" if is_sparse_model else "near_zero_count_only"
            )
        },
        "classification_results": {
            "logreg_train_acc": logreg_train_acc,
            "logreg_test_acc": logreg_test_acc,
            "logreg_test_f1": logreg_test_f1,
            "tree_train_acc": tree_train_acc,
            "tree_test_acc": tree_test_acc,
            "tree_test_f1": tree_test_f1,
            "tree_depth": tree_actual_depth,
            "tree_leaves": tree_n_leaves,
            "root_feature": root_feature_name
        }
    }

    os.makedirs("artifacts", exist_ok=True)
    output_path = os.path.join("artifacts", "ml_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    return results, output_path


# =====================================================================
# HÀM THỰC THI CHÍNH (MAIN ENTRYPOINT)
# =====================================================================
def main():
    print("=" * 70)
    print(f"BẮT ĐẦU CHẠY DỰ ÁN HỌC MÁY CHO MSSV: [{CONFIG['student_id']}]")
    print("=" * 70)

    # 1. Tải và tiền xử lý
    print("\n[1/4] Đang xử lý dữ liệu bảng và phân nhánh theo seed...")
    (X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te,
     imputer, scaler, feat_names, preprocessing_meta) = load_and_preprocess_data()
    print(f" -> Kích thước X_train: {X_tr.shape} | X_test: {X_te.shape}")
    print(f" -> Scaler áp dụng: {CONFIG['scaler_type'].upper()} | Imputer: {CONFIG['impute_strategy'].upper()}")

    # 2. Huấn luyện Hồi quy
    print(f"\n[2/4] Đang huấn luyện mô hình Hồi quy ({CONFIG['reg_type'].upper()})...")
    reg_model = train_regression_model(X_tr, yr_tr)

    # 3. Huấn luyện Phân loại
    print("\n[3/4] Đang huấn luyện các mô hình Phân loại (Logistic Regression & Decision Tree)...")
    logreg_m, tree_m = train_classification_models(X_tr, yc_tr)

    # 4. Trích xuất file kết quả thực nghiệm
    print("\n[4/4] Đang trích xuất log thực nghiệm vào thư mục artifacts/...")
    res, out_file = export_student_artifacts(
        X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te,
        reg_model, logreg_m, tree_m, imputer, scaler, feat_names,
        preprocessing_meta
    )

    print("\n" + "=" * 70)
    print("BẢNG TÓM TẮT DẤU VẾT THỰC NGHIỆM ĐỂ MANG VÀO PHÒNG THI:")
    print("=" * 70)
    print(f" * Kích thước X_train (n_train, n_features) : ({res['dataset_shape']['n_train']}, {res['dataset_shape']['n_features']})")
    print(f" * Imputer Stat Feature 0 (imputer_val_0)   : {res['preprocessing_stats']['imputer_val_0']}")
    print(f" * Scaler Center 0 (scaler_center_0)        : {res['preprocessing_stats']['scaler_center_0']}")
    print(f" * Regression Intercept                     : {res['regression_results']['intercept']}")
    print(f" * Regression Zero Coefs Count              : {res['regression_results']['zero_coef_count']}")
    print(f" * Regression Test MSE                      : {res['regression_results']['test_mse']}")
    print(f" * Tree Depth & Leaves Count                : Depth = {res['classification_results']['tree_depth']}, Leaves = {res['classification_results']['tree_leaves']}")
    print(f" * Tree Root Split Feature                  : '{res['classification_results']['root_feature']}'")
    print(f" * Logistic Test Acc vs Tree Test Acc       : {res['classification_results']['logreg_test_acc']} vs {res['classification_results']['tree_test_acc']}")
    print("-" * 70)
    print(f"[HOÀN TẤT] File log lưu tại: '{out_file}'")
    print("HƯỚNG DẪN: Sinh viên in bảng kết quả này kèm bản in mã nguồn để sử dụng trong phòng thi!")
    print("=" * 70)


if __name__ == "__main__":
    main()