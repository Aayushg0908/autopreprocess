"""
AutoPreprocess web UI (Flask + Jinja).

Run from the repository root:
    pip install -r ui/requirements.txt
    python ui/app.py
Then open http://127.0.0.1:5000
"""

import io
import os
import sys
import time
import uuid
import tempfile
import contextlib
from collections import OrderedDict

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, abort

# Make the library importable when running from a source checkout
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from autopreprocess import AutoClean  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB uploads

# Recent results kept in memory for downloads (token -> files)
RESULTS: "OrderedDict[str, dict]" = OrderedDict()
MAX_RESULTS = 10

SCALING_METHODS = ["standard", "minmax", "robust"]
SELECTION_METHODS = ["variance", "correlation", "importance", "mutual_info"]

DEFAULTS = {
    "target": "",
    "test_size": 0.2,
    "random_state": 42,
    "scaling": True,
    "scaling_method": "standard",
    "handle_missing": True,
    "handle_outliers": True,
    "remove_duplicates": True,
    "drop_useless": True,
    "feature_selection": False,
    "selection_method": "importance",
    "top_k": "",
    "baseline": True,
}


def sample_dataset(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Small synthetic house-price dataset with messy real-world features."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "house_id": np.arange(1, n + 1),
        "living_area": rng.integers(500, 4000, n).astype(float),
        "lot_area": rng.integers(2000, 20000, n),
        "quality": rng.integers(1, 11, n),
        "year_built": rng.integers(1900, 2015, n),
        "neighborhood": rng.choice(["Oakwood", "Riverside", "Hilltop", "Downtown", "Lakeside"], n),
        "garage": rng.choice(["Attached", "Detached", None], n, p=[0.55, 0.3, 0.15]),
        "zoning": rng.choice(["RL", "RM", "FV"], n),
        "sold_on": pd.date_range("2019-01-01", periods=n, freq="2D").strftime("%Y-%m-%d"),
    })
    df["price"] = (
        df["living_area"] * 85 + df["quality"] * 14000
        + (df["year_built"] - 1900) * 300 + rng.normal(0, 18000, n)
    ).round(-2)
    df.loc[rng.choice(n, 30, replace=False), "living_area"] = np.nan
    return pd.concat([df, df.iloc[:6]], ignore_index=True)  # a few duplicates


def _bool(form, key):
    return form.get(key) in ("on", "true", "1")


def read_options(form) -> dict:
    opts = dict(DEFAULTS)
    opts["target"] = form.get("target", "").strip()
    try:
        opts["test_size"] = min(max(float(form.get("test_size", 0.2)), 0.05), 0.5)
    except ValueError:
        pass
    try:
        opts["random_state"] = int(form.get("random_state", 42))
    except ValueError:
        pass
    if form.get("scaling_method") in SCALING_METHODS:
        opts["scaling_method"] = form["scaling_method"]
    if form.get("selection_method") in SELECTION_METHODS:
        opts["selection_method"] = form["selection_method"]
    opts["top_k"] = form.get("top_k", "").strip()
    for key in ("scaling", "handle_missing", "handle_outliers", "remove_duplicates",
                "drop_useless", "feature_selection", "baseline"):
        opts[key] = _bool(form, key)
    return opts


def baseline_model(X_train, X_test, y_train, y_test, problem_type):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

    if problem_type == "classification":
        model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = [
            ("Accuracy", f"{accuracy_score(y_test, pred):.3f}"),
            ("F1 (weighted)", f"{f1_score(y_test, pred, average='weighted'):.3f}"),
        ]
    else:
        model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = [
            ("R² score", f"{r2_score(y_test, pred):.3f}"),
            ("Mean abs. error", f"{mean_absolute_error(y_test, pred):,.2f}"),
        ]
    imp = sorted(zip(X_train.columns, model.feature_importances_), key=lambda t: -t[1])[:8]
    top = imp[0][1] if imp and imp[0][1] > 0 else 1
    importances = [{"name": n, "value": v, "pct": 100 * v / top} for n, v in imp]
    return {"model": type(model).__name__, "metrics": metrics, "importances": importances}


def column_profile(df: pd.DataFrame, target: str):
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "name": col,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": 100 * s.isna().mean(),
            "unique": int(s.nunique(dropna=True)),
            "is_target": col == target,
        })
    return rows


def run_pipeline(df: pd.DataFrame, opts: dict) -> dict:
    top_k = int(opts["top_k"]) if opts["top_k"].isdigit() else None
    log = io.StringIO()
    start = time.time()
    with contextlib.redirect_stdout(log):
        pipeline = AutoClean(
            data=df,
            target=opts["target"],
            test_size=opts["test_size"],
            random_state=opts["random_state"],
            scaling=opts["scaling"],
            scaling_method=opts["scaling_method"],
            handle_missing=opts["handle_missing"],
            handle_outliers=opts["handle_outliers"],
            remove_duplicates=opts["remove_duplicates"],
            drop_useless=opts["drop_useless"],
            feature_selection=opts["feature_selection"],
            selection_method=opts["selection_method"],
            top_k=top_k,
            verbose=True,
        )
        X_train, X_test, y_train, y_test = pipeline.preprocess()
    elapsed = time.time() - start

    baseline = None
    if opts["baseline"] and len(X_train) and X_train.shape[1]:
        try:
            baseline = baseline_model(X_train, X_test, y_train, y_test, pipeline.problem_type_)
        except Exception as exc:  # show, don't crash the page
            baseline = {"error": str(exc)}

    # Prepare downloadable files
    token = uuid.uuid4().hex
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pipeline.pkl")
        with contextlib.redirect_stdout(io.StringIO()):
            pipeline.save(path)
        with open(path, "rb") as fh:
            pkl_bytes = fh.read()
    RESULTS[token] = {
        "train.csv": X_train.assign(**{opts["target"]: y_train}).to_csv(index=False).encode(),
        "test.csv": X_test.assign(**{opts["target"]: y_test}).to_csv(index=False).encode(),
        "pipeline.pkl": pkl_bytes,
    }
    while len(RESULTS) > MAX_RESULTS:
        RESULTS.popitem(last=False)

    preview = X_train.head(8).copy()
    for col in preview.select_dtypes("number").columns:
        preview[col] = preview[col].astype("float64").round(3)

    return {
        "token": token,
        "problem_type": pipeline.problem_type_,
        "elapsed": elapsed,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "features_in": df.shape[1] - 1,
        "features_out": X_train.shape[1],
        "dropped": list(pipeline.dropped_cols_),
        "categorical": list(pipeline.categorical_cols_),
        "scaled": list(getattr(pipeline, "scaled_cols_", []) or []),
        "nan_left": int(X_train.isna().sum().sum() + X_test.isna().sum().sum()),
        "preview_cols": list(preview.columns),
        "preview_rows": preview.values.tolist(),
        "baseline": baseline,
        "log": log.getvalue(),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {
        "opts": dict(DEFAULTS),
        "scaling_methods": SCALING_METHODS,
        "selection_methods": SELECTION_METHODS,
        "result": None,
        "error": None,
        "dataset": None,
        "profile": None,
    }
    if request.method == "GET":
        return render_template("index.html", **ctx)

    opts = read_options(request.form)
    ctx["opts"] = opts
    try:
        if request.form.get("use_sample"):
            df = sample_dataset()
            name = "sample_houses.csv"
            if opts["target"] not in df.columns:
                opts["target"] = "price"
        else:
            upload = request.files.get("file")
            if not upload or not upload.filename:
                raise ValueError("Choose a CSV file to upload, or try the sample dataset.")
            name = upload.filename
            df = pd.read_csv(upload)
        if df.empty:
            raise ValueError("The uploaded file has no rows.")
        if opts["target"] not in df.columns:
            raise ValueError(f"Target column '{opts['target']}' was not found in {name}.")

        ctx["dataset"] = {"name": name, "rows": len(df), "cols": df.shape[1],
                          "missing": int(df.isna().sum().sum()),
                          "duplicates": int(df.duplicated().sum())}
        ctx["profile"] = column_profile(df, opts["target"])
        ctx["result"] = run_pipeline(df, opts)
    except Exception as exc:
        ctx["error"] = str(exc) or type(exc).__name__
    return render_template("index.html", **ctx)


@app.route("/download/<token>/<name>")
def download(token, name):
    files = RESULTS.get(token)
    if not files or name not in files:
        abort(404)
    mime = "text/csv" if name.endswith(".csv") else "application/octet-stream"
    return send_file(io.BytesIO(files[name]), mimetype=mime,
                     as_attachment=True, download_name=f"autopreprocess_{name}")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
