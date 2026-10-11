"""
Pipeline hồi quy dự đoán giá xe ô tô.

Khung tổng thể: đọc -> làm sạch -> tách đặc trưng -> huấn luyện nhiều mô hình
-> đánh giá MAE/MSE/RMSE/R2 -> vẽ biểu đồ (gồm SHAP) -> lưu outputs/.

Tiền xử lý phản ánh EDA.ipynb và ĐƯỢC THIẾT KẾ RIÊNG THEO TỪNG HỌ MÔ HÌNH:
    - Họ cần khoảng cách/tuyến tính (Linear, KNN):
        impute -> log(Mileage, Levy) -> OneHot/Target -> StandardScaler toàn bộ.
    - Họ cây (Decision Tree, Random Forest):
        impute -> OneHot/Target, KHÔNG scale, KHÔNG log đặc trưng (vô ích với cây).
    - Với mô hình họ tuyến tính (Linear, KNN), cho huấn luyện trên log(Price) (kết luận từ EDA.ipynb),
    - Với mô hình họ cây (Decision Tree, Random Forest), huấn luyện trực tiếp trên Price (vì cây không bị ảnh hưởng bởi phân phối lệch).
    - Dự đoán/đánh giá được quy về USD gốc để dễ diễn giải.
"""

from pathlib import Path
import os
import tempfile

matplotlib_cache_dir = Path(tempfile.gettempdir()) / "lab_car_matplotlib_cache"
matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)
from sklearn.tree import DecisionTreeRegressor

# ----------------------------------------------------------------------------
# Cấu hình
# ----------------------------------------------------------------------------

TARGET = "Price"
ID_COLUMN = "ID"

# Ngưỡng lọc outlier (theo EDA.ipynb). Chỉ áp dụng cho TẬP TRAIN.
MILEAGE_MAX = 1_000_000
ENGINE_VOLUME_MAX = 10
PRICE_LOWER_QUANTILE = 0.000
PRICE_UPPER_QUANTILE = 0.995

# Năm tham chiếu để tính tuổi xe (EDA dùng năm hiện tại)
REFERENCE_YEAR = pd.Timestamp.now().year

RANDOM_STATE = 42
# Tỉ lệ chia train/test (dùng cho đánh giá mô hình, KHÔNG áp dụng cho test.csv không nhãn)
TEST_SIZE = 0.2

# Nhóm đặc trưng
NUMERIC_FEATURES = ["Levy", "Engine volume", "Mileage", "Age", "Mileage per year"]
SKEWED_FEATURES = ["Levy", "Mileage", "Mileage per year"]  # lệch phải mạnh (EDA) -> log cho mô hình tuyến tính/khoảng cách
NON_SKEWED_FEATURES = [c for c in NUMERIC_FEATURES if c not in SKEWED_FEATURES]
BINARY_FEATURES = ["Leather interior", "Turbo", "Levy missing"]  # -> SimpleImputer + giữ nguyên 0/1
HIGH_CARD_FEATURES = ["Manufacturer", "Model", "Color"]  # -> Target Encoding
LOW_CARD_FEATURES = [
    "Category", "Fuel type", "Cylinders", "Gear box type",
    "Drive wheels", "Wheel", "Doors", "Airbags",
]  # -> OneHotEncoder

# Mô hình nào thuộc họ nào
SCALED_MODELS = {"Linear Regression", "KNN"}  # cần scale + log đặc trưng
TREE_MODELS = {"Decision Tree", "Decision Tree No Limit", "Random Forest"}


# ----------------------------------------------------------------------------
# 1. Đọc dữ liệu
# ----------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    df = pd.read_csv(path)
    print(f"Đã đọc {path.name}: {df.shape}")
    return df


# ----------------------------------------------------------------------------
# 2. Làm sạch dữ liệu
# ----------------------------------------------------------------------------

def _transform_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Levy: "-" -> NaN -> 0, kèm cờ đánh dấu thiếu
    df["Levy"] = pd.to_numeric(df["Levy"], errors="coerce")
    df["Levy missing"] = df["Levy"].isna().astype(int)
    df["Levy"] = df["Levy"].fillna(0)

    # Engine volume: tách cờ Turbo rồi chuyển về số
    engine_str = df["Engine volume"].astype(str)
    df["Turbo"] = engine_str.str.contains("Turbo", case=False, na=False).astype(int)
    df["Engine volume"] = pd.to_numeric(
        engine_str.str.replace("Turbo", "", regex=False).str.strip(), errors="coerce"
    )

    # Mileage: bỏ đơn vị " km"
    df["Mileage"] = pd.to_numeric(
        df["Mileage"].astype(str).str.replace("km", "", regex=False).str.strip(),
        errors="coerce",
    )

    # Leather interior: Yes/No -> 1/0
    df["Leather interior"] = (
        df["Leather interior"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    )

    # Doors: gộp nhãn cho dễ hiểu
    door_mapping = {"02-Mar": "2-3", "04-May": "4-5", ">5": "5+"}
    df["Doors"] = df["Doors"].astype(str).replace(door_mapping)

    # Tạo biến Age (tuổi xe)
    df["Age"] = REFERENCE_YEAR - pd.to_numeric(df["Prod. year"], errors="coerce")

    # Thêm feature "Mileage per Year" để cung cấp thêm thông tin về mức độ sử dụng xe, có thể hữu ích cho mô hình.
    df["Mileage per year"] = df["Mileage"] / (df["Age"]+1) # tránh chia cho 0 nếu Age=0 (xe mới sản xuất năm nay)

    # Ép các đặc trưng số về numeric   
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Cylinders/Airbags về chuỗi để xử lý như biến phân loại trong OneHot
    for col in ["Cylinders", "Airbags"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].astype("Int64").astype(str).replace("<NA>", np.nan)

    return df


def clean_data(df: pd.DataFrame, drop_rows: bool = True) -> pd.DataFrame:
    """
    Làm sạch dữ liệu.

    drop_rows=True  -> dùng cho TRAIN: bỏ dòng trùng + bỏ outlier.
    drop_rows=False -> dùng cho TEST : giữ nguyên mọi dòng (chỉ biến đổi cột),
                       NaN sẽ được imputer trong pipeline xử lý sau.
    """
    original_rows = len(df)
    df = _transform_columns(df)

    if drop_rows:
        # Bỏ dòng trùng
        df = df.drop_duplicates()
        
        # Target phải hợp lệ
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df[df[TARGET] > 0]

        # Lọc outlier do lỗi/đặc thù (theo EDA)
        before_outlier = len(df)
        df = df[
            (df["Mileage"] <= MILEAGE_MAX)
            & (df["Engine volume"] <= ENGINE_VOLUME_MAX)& 
            (df[TARGET] >= df[TARGET].quantile(PRICE_LOWER_QUANTILE)) &
            (df[TARGET] <= df[TARGET].quantile(PRICE_UPPER_QUANTILE))
        ]
        print(f"Kích thước train sau làm sạch: {df.shape}")
        print(f"  - Số dòng đã loại (trùng/giá lỗi): {original_rows - before_outlier}")
        print(f"  - Số dòng outlier đã loại: {before_outlier - len(df)}")
    else:
        print(f"Kích thước test sau biến đổi (giữ nguyên số dòng): {df.shape}")

    return df


# ----------------------------------------------------------------------------
# 3. Tách đặc trưng
# ----------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame):
    drop_cols = [c for c in [ID_COLUMN, "Prod. year", TARGET] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET] if TARGET in df.columns else None

    # Chỉ giữ các cột đặc trưng đã định nghĩa, đúng thứ tự
    feature_cols = (
        NUMERIC_FEATURES + BINARY_FEATURES + HIGH_CARD_FEATURES + LOW_CARD_FEATURES
    )
    return X[feature_cols], y


# ----------------------------------------------------------------------------
# 4. Bộ tiền xử lý theo từng họ mô hình
# ----------------------------------------------------------------------------

def make_one_hot_encoder():
    """OneHotEncoder gộp nhãn hiếm (min_frequency) để giảm nhiễu."""
    try:
        return OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, min_frequency=0.01
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _log1p_transformer():
    return FunctionTransformer(np.log1p, feature_names_out="one-to-one")


def build_preprocessor(family: str):
    """
    family = "scaled" -> cho Linear/KNN (log biến lệch + StandardScaler toàn bộ)
    family = "tree"   -> cho Decision Tree/Random Forest (gọn, không scale/log đặc trưng)
    """
    bin_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    low_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    high_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("target", TargetEncoder(cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE))),
        ]
    )

    if family == "scaled":
        skew_pipe = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("log", _log1p_transformer())]
        )
        plain_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        column_transformer = ColumnTransformer(
            transformers=[
                ("num_skew", skew_pipe, SKEWED_FEATURES),
                ("num_plain", plain_pipe, NON_SKEWED_FEATURES),
                ("bin", bin_pipe, BINARY_FEATURES),
                ("low", low_pipe, LOW_CARD_FEATURES),
                ("high", high_pipe, HIGH_CARD_FEATURES),
            ]
        )
        # Scale toàn bộ ma trận sau khi mã hóa (quan trọng cho KNN/Linear)
        return Pipeline(
            [("columns", column_transformer), ("scaler", StandardScaler())]
        )

    # family == "tree"
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERIC_FEATURES),
            ("bin", bin_pipe, BINARY_FEATURES),
            ("low", low_pipe, LOW_CARD_FEATURES),
            ("high", high_pipe, HIGH_CARD_FEATURES),
        ]
    )


def model_family(model_name: str) -> str:
    return "scaled" if model_name in SCALED_MODELS else "tree"


# ----------------------------------------------------------------------------
# 5. Mô hình
# ----------------------------------------------------------------------------

def get_model_estimators():
    return {
        "Linear Regression": LinearRegression(),
        "KNN": KNeighborsRegressor(n_neighbors=10),
        "Decision Tree": DecisionTreeRegressor(max_depth=7, random_state=RANDOM_STATE),
        "Decision Tree No Limit": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def make_model_pipeline(model_name: str, estimator):
    """
    Tiền xử lý theo họ mô hình.
    Log target (log1p) CHỈ áp dụng cho họ scaled (Linear/KNN);
    họ cây huấn luyện trực tiếp trên Price.
    """
    base_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(model_family(model_name))),
            ("model", clone(estimator)),
        ]
    )
    if model_family(model_name) == "scaled":
        return TransformedTargetRegressor(
            regressor=base_pipeline, func=np.log1p, inverse_func=np.expm1
        )
    return base_pipeline


# ----------------------------------------------------------------------------
# 6. Đánh giá
# ----------------------------------------------------------------------------

def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    return {
        "Model": model_name,
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred),
        "Train MSE": train_mse,
        "Test MSE": test_mse,
        "Train RMSE": np.sqrt(train_mse),
        "Test RMSE": np.sqrt(test_mse),
        "Train R2": r2_score(y_train, y_train_pred),
        "Test R2": r2_score(y_test, y_test_pred),
    }


def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    trained_models, results = {}, []
    for model_name, estimator in get_model_estimators().items():
        print(f"Đang huấn luyện: {model_name} (họ: {model_family(model_name)})")
        model = make_model_pipeline(model_name, estimator)
        model.fit(X_train, y_train)
        trained_models[model_name] = model
        results.append(evaluate_model(model_name, model, X_train, y_train, X_test, y_test))
    results_df = pd.DataFrame(results).sort_values(by="Test RMSE").reset_index(drop=True)
    return results_df, trained_models, X_train, y_train, X_test, y_test


# ----------------------------------------------------------------------------
# 7. Biểu đồ & kết quả
# ----------------------------------------------------------------------------

def save_comparison_chart(results_df, figures_dir):
    for metric, fname in [("Test RMSE", "model_comparison_rmse.png"),
                          ("Test R2", "model_comparison_r2.png")]:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=results_df, x="Model", y=metric)
        plt.title(f"So sánh {metric} giữa các mô hình")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(figures_dir / fname, dpi=150)
        plt.close()


def _unwrap_pipeline(model):
    """Trả về Pipeline bên trong, dù model có bị bọc TransformedTargetRegressor hay không."""
    return getattr(model, "regressor_", model)


def save_feature_importance(model, output_dir, figures_dir):
    pipeline = _unwrap_pipeline(model)
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["model"]
    if not hasattr(regressor, "feature_importances_"):
        return
    feature_names = preprocessor.get_feature_names_out()
    fi_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": regressor.feature_importances_}
    ).sort_values(by="Importance", ascending=False)
    fi_df.to_csv(output_dir / "feature_importance.csv", index=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=fi_df.head(15), x="Importance", y="Feature")
    plt.title("Top 15 đặc trưng quan trọng nhất - Random Forest")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance.png", dpi=150)
    plt.close()


def save_shap_summary(model, model_name, X_sample, output_dir, figures_dir):
    """
    SHAP cho mô hình cây tốt nhất. Giải thích trên đặc trưng SAU tiền xử lý.
    Thang của giá trị SHAP phụ thuộc việc target có bị log hay không:
    - Mô hình bọc TransformedTargetRegressor (log) -> SHAP ở thang log-Price.
    - Mô hình huấn luyện thẳng trên Price          -> SHAP ở thang USD gốc.
    """
    pipeline = _unwrap_pipeline(model)
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["model"]

    if not hasattr(regressor, "feature_importances_"):
        print(f"Bỏ qua SHAP: {model_name} không phải mô hình cây.")
        return

    try:
        import shap
    except ImportError:
        print("Bỏ qua SHAP: chưa cài thư viện 'shap' (pip install shap).")
        return

    # Xác định thang giá của SHAP: log nếu model bị bọc TransformedTargetRegressor
    target_logged = hasattr(model, "regressor_")
    scale_label = "thang log-Price" if target_logged else "thang USD gốc"

    sample = X_sample.sample(min(1500, len(X_sample)), random_state=RANDOM_STATE)
    X_trans = preprocessor.transform(sample)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_trans)

    fi_df = pd.DataFrame(
        {"Feature": feature_names, "Mean |SHAP|": np.abs(shap_values).mean(axis=0)}
    ).sort_values(by="Mean |SHAP|", ascending=False)
    fi_df.to_csv(output_dir / "shap_importance.csv", index=False)

    plt.figure()
    shap.summary_plot(
        shap_values, X_trans, feature_names=feature_names,
        max_display=15, show=False,
    )
    plt.title(f"SHAP summary - {model_name} ({scale_label})")
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu SHAP: shap_summary.png + shap_importance.csv ({scale_label})")

def save_actual_vs_predicted(model, X_test, y_test, model_name, figures_dir):
    y_pred = model.predict(X_test)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.3, s=12)
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", label="Dự đoán hoàn hảo")
    plt.xlabel("Giá thực tế"); plt.ylabel("Giá dự đoán")
    plt.title(f"Actual vs Predicted - {model_name}")
    plt.legend(); plt.tight_layout()
    plt.savefig(figures_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()


def save_residual_plot(model, X_test, y_test, model_name, figures_dir):
    y_pred = model.predict(X_test)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, y_test - y_pred, alpha=0.3, s=12)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Giá dự đoán"); plt.ylabel("Sai số (thực tế - dự đoán)")
    plt.title(f"Residual plot - {model_name}")
    plt.tight_layout()
    plt.savefig(figures_dir / "residual_plot.png", dpi=150)
    plt.close()


def save_sample_predictions(model, X_test, y_test, output_dir):
    sample_X, actual = X_test.head(10), y_test.head(10)
    predicted = model.predict(sample_X)
    pd.DataFrame(
        {
            "Actual Price": actual.values,
            "Predicted Price": predicted,
            "Absolute Error": np.abs(actual.values - predicted),
        }
    ).to_csv(output_dir / "sample_predictions.csv", index=False)


def make_filename_slug(text):
    return text.lower().replace(" ", "_")


def save_learning_curve_charts(X_train, y_train, X_test, y_test, output_dir, figures_dir):
    rows = []
    for model_name, estimator in get_model_estimators().items():
        print(f"Đang tạo learning curve: {model_name}")
        for ratio in np.linspace(0.1, 1.0, 10):
            n = max(20, int(len(X_train) * ratio))
            X_sub, y_sub = X_train.iloc[:n], y_train.iloc[:n]
            model = make_model_pipeline(model_name, estimator)
            model.fit(X_sub, y_sub)
            res = evaluate_model(model_name, model, X_sub, y_sub, X_test, y_test)
            rows.append({
                "Model": model_name, "Train Size": n,
                "Train RMSE": res["Train RMSE"], "Test RMSE": res["Test RMSE"],
                "Train R2": res["Train R2"], "Test R2": res["Test R2"],
            })
    lc_df = pd.DataFrame(rows)
    lc_df.to_csv(output_dir / "learning_curves_all_models.csv", index=False)
    for model_name in lc_df["Model"].unique():
        mdf = lc_df[lc_df["Model"] == model_name]
        slug = make_filename_slug(model_name)
        for metric in ["RMSE", "R2"]:
            plt.figure(figsize=(10, 5))
            plt.plot(mdf["Train Size"], mdf[f"Train {metric}"], marker="o", label=f"Train {metric}")
            plt.plot(mdf["Train Size"], mdf[f"Test {metric}"], marker="o", label=f"Test {metric}")
            plt.xlabel("Số lượng mẫu train"); plt.ylabel(metric)
            plt.title(f"Learning Curve {metric} - {model_name}")
            plt.legend(); plt.tight_layout()
            plt.savefig(figures_dir / f"learning_curve_{slug}_{metric.lower()}.png", dpi=150)
            plt.close()


def save_overfit_underfit_charts(X_train, y_train, X_test, y_test, output_dir, figures_dir):
    rows = []

    models_to_test = {
        "Decision Tree": {
            "estimator": DecisionTreeRegressor(random_state=RANDOM_STATE),
            "param_name": "max_depth",
            "param_values": [1, 2, 3, 5, 7, 10, 15, 20, None]
        },
        "Random Forest": {
            "estimator": RandomForestRegressor(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),
            "param_name": "max_depth",
            "param_values": [1, 2, 3, 5, 7, 10, 15, 20, None]
        }
    }

    for model_name, cfg in models_to_test.items():
        for val in cfg["param_values"]:

            params = {cfg["param_name"]: val}

            model = make_model_pipeline(
                model_name,
                clone(cfg["estimator"]).set_params(**params)
            )

            model.fit(X_train, y_train)

            res = evaluate_model(
                f"{model_name} depth={val}",
                model,
                X_train, y_train,
                X_test, y_test
            )

            rows.append({
                "Model": model_name,
                "max_depth": "None" if val is None else val,
                "complexity_index": len(rows) + 1,
                "Train RMSE": res["Train RMSE"],
                "Test RMSE": res["Test RMSE"],
                "Train R2": res["Train R2"],
                "Test R2": res["Test R2"],
            })

    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(output_dir / "overfit_underfit_tree_family.csv", index=False)

    # plot riêng từng model
    for model_name in curve_df["Model"].unique():
        mdf = curve_df[curve_df["Model"] == model_name]

        for metric in ["RMSE", "R2"]:
            plt.figure(figsize=(10, 5))

            plt.plot(mdf["complexity_index"], mdf[f"Train {metric}"], marker="o", label=f"Train {metric}")
            plt.plot(mdf["complexity_index"], mdf[f"Test {metric}"], marker="o", label=f"Test {metric}")

            plt.xticks(mdf["complexity_index"], mdf["max_depth"])
            plt.xlabel("max_depth")
            plt.ylabel(metric)

            plt.title(f"Overfit/Underfit Curve - {model_name} ({metric})")
            plt.legend()
            plt.tight_layout()

            plt.savefig(
                figures_dir / f"overfit_underfit_{model_name.lower().replace(' ', '_')}_{metric.lower()}.png",
                dpi=150
            )
            plt.close()

# ----------------------------------------------------------------------------
# 8. Dự đoán cho test.csv (không nhãn)
# ----------------------------------------------------------------------------

def predict_test_set(best_name, best_estimator, X_full, y_full, test_path, output_dir):
    if not test_path.exists():
        print(f"Bỏ qua dự đoán test: không tìm thấy {test_path}")
        return
    test_raw = load_data(test_path)
    test_clean = clean_data(test_raw, drop_rows=False)
    X_test_final, _ = prepare_features(test_clean)

    # Huấn luyện lại mô hình tốt nhất trên TOÀN BỘ train rồi dự đoán
    final_model = make_model_pipeline(best_name, best_estimator)
    final_model.fit(X_full, y_full)
    predictions = final_model.predict(X_test_final)
    ids = test_raw[ID_COLUMN] if ID_COLUMN in test_raw.columns else range(len(predictions))
    pd.DataFrame({ID_COLUMN: ids, "Predicted Price": predictions}).to_csv(
        output_dir / "test_predictions.csv", index=False
    )
    print(f"Đã lưu dự đoán test: {len(predictions)} dòng -> test_predictions.csv")


# ----------------------------------------------------------------------------
# 9. Điều phối
# ----------------------------------------------------------------------------

def run_regression_experiment(train_path, test_path, output_dir):
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Đọc + làm sạch train
    train_raw = load_data(train_path)
    train_clean = clean_data(train_raw, drop_rows=True)
    X, y = prepare_features(train_clean)

    # Huấn luyện & đánh giá (chia nội bộ 80/20)
    results_df, trained_models, X_train, y_train, X_test, y_test = train_and_evaluate_models(X, y)

    print("\nBảng so sánh mô hình:")
    print(results_df.to_string(index=False))
    results_df.to_csv(output_dir / "results.csv", index=False)

    # Biểu đồ so sánh
    save_comparison_chart(results_df, figures_dir)

    # Phân tích sâu trên mô hình tốt nhất (theo Test RMSE)
    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    print(f"\nMô hình tốt nhất theo Test RMSE: {best_name}")

    save_actual_vs_predicted(best_model, X_test, y_test, best_name, figures_dir)
    save_residual_plot(best_model, X_test, y_test, best_name, figures_dir)
    save_sample_predictions(best_model, X_test, y_test, output_dir)

    # Feature importance + SHAP trên mô hình cây tốt nhất (ưu tiên Random Forest nếu có)
    fi_name = "Random Forest" if "Random Forest" in trained_models else best_name
    fi_model = trained_models[fi_name]
    save_feature_importance(fi_model, output_dir, figures_dir)
    save_shap_summary(fi_model, fi_name, X_test, output_dir, figures_dir)

    # Learning curve & validation curve
    save_learning_curve_charts(X_train, y_train, X_test, y_test, output_dir, figures_dir)
    save_overfit_underfit_charts(X_train, y_train, X_test, y_test, output_dir, figures_dir)
    
    # Dự đoán test.csv (không nhãn) bằng mô hình tốt nhất, refit trên toàn bộ train
    predict_test_set(best_name, get_model_estimators()[best_name], X, y, test_path, output_dir)

    print(f"\nĐã lưu toàn bộ kết quả vào: {output_dir}/")
    return results_df
