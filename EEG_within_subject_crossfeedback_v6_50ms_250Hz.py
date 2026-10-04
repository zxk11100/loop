# -*- coding: utf-8 -*-
"""被试内跨反馈/跨时间解码 v6：新增线性 SVC / SVR。

默认 CLASSIFIER="svc"、REGRESSOR="svr"；可切换回 "logistic"、"ridge"。
输入为重新提取的50ms特征；不改变试次筛选、分折或评分。
同类仅同时间窗测试，跨类保留20x20泛化；迁移损失仅同时间比较。
PE_valence：balanced SVC，固定 C，逐折 AUC 后平均。
PE_magnitude / update：SVR，逐训练时间窗在内层 CV 选择 C。
SVR 的 X、y 均由各训练折标准化；epsilon 按训练目标的标准差计。
群体统计保留 v3：被试分数符号翻转，每张 GAT 图内部的双侧簇 FWER。
本脚本没有新增 Absolute / Relative 身份分类，不进行跨图/跨算法校正。

在 Spyder 中运行整个文件；也可在交互环境运行，不要求存在 __file__。
"""

from pathlib import Path
import os
import sys
import json
import hashlib
import platform

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# 使用当前Spyder/Python环境的依赖，避免加载其他Python版本的二进制包。

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, t as tdist
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR

# ============================================================
# 配置
# ============================================================

BASE = Path(r"D:\LOOP脑电数据\EEG_main_analysis")
INPUT = BASE / "results" / "feedback_patterns_50ms_250Hz_20261004"
# 两项可以独立切换；同一套输入和种子对应相同的外层分折。
CLASSIFIER = "svc"       # "svc" 或 "logistic"
REGRESSOR = "svr"        # "svr" 或 "ridge"
CODE_VERSION = "v6_50ms_250Hz_diagonalwithin_20261004_01"
OUTPUT = BASE / "results" / f"within_subject_crossfeedback_v6_50ms_250Hz_{CLASSIFIER}_{REGRESSOR}"

ALL_BANDS = ["delta", "theta", "alpha", "beta", "lowbeta", "highbeta"]

# 主频段与补充频段分别标记；只做每张图内部簇校正
MAIN_BANDS = ["delta", "theta", "alpha", "beta"]
SECONDARY_BANDS = ["lowbeta", "highbeta"]
RUN_SECONDARY_BANDS = True

# 真正50ms不重叠特征：先运行prepare_crossfeedback_50ms_250Hz.py。
TIME_SCHEME = "50ms_nonoverlap"
WINDOW_STARTS_MS = np.arange(0, 1000, 50, dtype=int)
WINDOW_WIDTH_MS = 50

N_FOLDS = 5
SEED = 20261001

# PE valence classifier
LOGISTIC_C = 1.0
SVC_C = 1.0             # 分类器固定 C；不会在目标反馈的数据上调参
SVM_MAX_ITER = -1       # libsvm 不设迭代次数上限
SVM_TOL = 1e-3
MIN_CLASS_TRAIN = 2

# Continuous outcomes
LAMBDAS = np.array([0.01, 0.1, 1.0, 10.0, 100.0], dtype=float)
SVR_C_GRID = np.array([0.01, 0.1, 1.0, 10.0], dtype=float)
SVR_EPSILON = 0.1       # y 在每个训练折内部 z 标准化，故 0.1 表示 0.1 个训练SD
# 固定旧脚本的 C=0.1：将 SVR_C_GRID 改为 np.array([0.1])。
# 即使固定相同 C，本脚本仍然使用功率特征、训练折标准化，而非旧脚本的原始电压。
MIN_OUTER_TRAIN = 20
MIN_INNER_TRAIN = 12

# Scoring
MIN_SCORE_TRIALS = 8
MIN_SCORE_CLASS = 3
MIN_VALID_AUC_FOLDS = 3  # 至少3折各有两个类别；每折AUC等权平均

# Group inference
MIN_GROUP_N = 20
N_SIGNFLIPS = 1000

# cluster-forming threshold:
# two-sided: |t| > t(.975)  -> pointwise two-sided p < .05
# one-sided positive: t > t(.95) -> pointwise one-sided p < .05
TAILS = ["two_sided"]

# 评分全部保存；正式群体推断按 outcome 选择
INFERENCE_SUBSETS = {
    "PE_valence": ["all", "ability_high", "ability_low"],
    "PE_magnitude": ["all", "ability_high", "ability_low", "PE_positive", "PE_negative",
                     "high_positive", "high_negative", "low_positive", "low_negative"],
    "update": ["all", "ability_high", "ability_low", "PE_positive", "PE_negative",
                     "high_positive", "high_negative", "low_positive", "low_negative"],
}

SMOKE_TEST = False

FEEDBACKS = ["absolute", "relative"]
OUTCOMES = ["PE_valence", "PE_magnitude", "update"]

# ============================================================
# 基础工具
# ============================================================

def write_json(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def band_family(band):
    if band in MAIN_BANDS:
        return "main_bands"
    return "beta_subbands"

def auc_score_binary(y, score):
    """
    y 必须为 0/1。
    """
    y = np.asarray(y, float)
    score = np.asarray(score, float)

    good = np.isfinite(y) & np.isfinite(score)
    y = y[good]
    score = score[good]

    pos = y == 1
    n1 = int(pos.sum())
    n0 = int(len(y) - n1)

    if min(n1, n0) < MIN_SCORE_CLASS:
        return np.nan

    ranks = rankdata(score)
    return float(
        (ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    )

# ============================================================
# 被试内固定 5-fold
# ============================================================

def make_subject_folds(d, seed):
    """
    普通随机 5-fold。

    为保证 Absolute / Relative 内部的 PE valence 尽量均衡，
    按 feedback_type × ability × PE_sign 分层后随机打散，再轮流分配到 5 个 fold。

    PE == 0 单独作为 zero stratum；它不进入 valence 分类，
    但仍可进入 magnitude / update 回归。
    """
    fold = np.full(len(d), -1, dtype=int)

    pe = d["PE"].to_numpy()
    sign = np.where(pe > 0, "positive", np.where(pe < 0, "negative", "zero"))
    fb = d["feedback_type"].astype(str).to_numpy()

    ability = d["ability"].astype(str).str.lower().to_numpy()
    strata = np.array([f"{f}__{a}__{v}" for f, a, v in zip(fb, ability, sign)])

    rng = np.random.default_rng(seed)

    for stratum in np.unique(strata):
        rows = np.flatnonzero(strata == stratum)
        rows = rng.permutation(rows)

        # 每个 stratum 尽量均匀分到 5 折
        labels = np.arange(len(rows), dtype=int) % N_FOLDS

        # 每个 stratum 的起始 fold 随机旋转，避免所有小 stratum 都偏向 fold 0
        shift = int(rng.integers(0, N_FOLDS))
        labels = (labels + shift) % N_FOLDS

        fold[rows] = labels

    assert np.all(fold >= 0)
    return fold

# ============================================================
# 标准化 + Logistic Regression
# ============================================================

def fit_logistic(X, y):
    """
    PE valence:
    - 训练集内部 z 标准化
    - balanced class weights
    - L2 logistic regression
    """
    X = np.asarray(X, float)
    y = np.asarray(y, int)

    if len(X) < MIN_OUTER_TRAIN:
        return None

    counts = np.bincount(y, minlength=2)
    if counts.min() < MIN_CLASS_TRAIN:
        return None

    xm = X.mean(axis=0)
    xs = X.std(axis=0)
    xs[xs < 1e-8] = 1.0

    Z = (X - xm) / xs

    clf = LogisticRegression(
        C=LOGISTIC_C,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=SEED,
    )
    clf.fit(Z, y)
    if np.max(clf.n_iter_) >= clf.max_iter:
        raise RuntimeError("LogisticRegression未收敛；请检查该模型，不保存为成功拟合。")

    return xm, xs, clf

def predict_logistic(fit, X):
    if fit is None:
        return np.full(len(X), np.nan, dtype=float)

    xm, xs, clf = fit
    Z = (np.asarray(X, float) - xm) / xs
    return clf.decision_function(Z).astype(float)

# ============================================================
# Ridge regression
# ============================================================

def fit_ridge_all_lambdas(X, Y):
    """
    X: n_trials × n_features
    Y: n_trials × n_continuous_outcomes

    X 和 Y 都只根据当前训练数据标准化。
    一次返回所有 lambda 的预测权重。
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)

    xm = X.mean(axis=0)
    xs = X.std(axis=0)
    xs[xs < 1e-8] = 1.0
    Z = (X - xm) / xs

    ym = Y.mean(axis=0)
    ys = Y.std(axis=0)
    valid = ys > 1e-8
    ys_safe = ys.copy()
    ys_safe[ys_safe < 1e-8] = 1.0

    R = (Y - ym) / ys_safe

    # Z.T @ Z / n
    gram = Z.T @ Z / len(Z)
    eigvals, eigvecs = np.linalg.eigh(gram)
    eigvals = np.maximum(eigvals, 0.0)

    # eigvecs.T @ Z.T @ R / n
    proj = eigvecs.T @ (Z.T @ R / len(Z))

    # W: feature × outcome × lambda
    denom = eigvals[:, None, None] + LAMBDAS[None, None, :]
    W = np.einsum(
        "ij,jok->iok",
        eigvecs,
        proj[:, :, None] / denom,
    )

    return xm, xs, ym, ys_safe, W, valid

def predict_ridge_all_lambdas(fit, X):
    xm, xs, ym, ys, W, valid = fit
    Z = (np.asarray(X, float) - xm) / xs

    # n_trial × outcome × lambda
    pred_std = np.einsum("ni,iok->nok", Z, W)
    pred = ym[None, :, None] + ys[None, :, None] * pred_std
    pred[:, ~valid, :] = np.nan
    return pred

# ============================================================
# SVC / SVR 与原模型的统一接口
# ============================================================

def fit_classifier(X: np.ndarray, y: np.ndarray):
    """相同训练折标准化；两种分类器都返回连续 decision score。"""
    if CLASSIFIER == "logistic":
        return fit_logistic(X, y)
    if CLASSIFIER != "svc":
        raise ValueError("CLASSIFIER 必须为 svc 或 logistic。")

    X, y = np.asarray(X, float), np.asarray(y, int)
    if len(X) < MIN_OUTER_TRAIN or np.bincount(y, minlength=2).min() < MIN_CLASS_TRAIN:
        return None
    xm, xs = X.mean(axis=0), X.std(axis=0)
    xs[xs < 1e-8] = 1.0
    clf = SVC(
        kernel="linear", C=SVC_C, class_weight="balanced",
        probability=False, tol=SVM_TOL, max_iter=SVM_MAX_ITER,
        cache_size=200, random_state=SEED,
    )
    clf.fit((X - xm) / xs, y)
    if clf.fit_status_ != 0:
        raise RuntimeError("SVC 未收敛。")
    return xm, xs, clf

def predict_classifier(fit, X: np.ndarray) -> np.ndarray:
    return predict_logistic(fit, X)

def regression_grid() -> np.ndarray:
    return LAMBDAS if REGRESSOR == "ridge" else SVR_C_GRID

def fit_regression_grid(X: np.ndarray, Y: np.ndarray, selected_indices=None):
    """内层拟合候选参数；外层 SVR 仅重拟合各结局选中的 C。

    线性 SVR 拟合后保存 coef_ 和 intercept_，预测时与 sklearn.predict 等价。
    这里只压缩已拟合的线性函数，不更改 libsvm 的训练算法。
    """
    if REGRESSOR == "ridge":
        return fit_ridge_all_lambdas(X, Y)
    if REGRESSOR != "svr":
        raise ValueError("REGRESSOR 必须为 svr 或 ridge。")

    X, Y = np.asarray(X, float), np.asarray(Y, float)
    xm, xs = X.mean(axis=0), X.std(axis=0)
    xs[xs < 1e-8] = 1.0
    Z = (X - xm) / xs
    ym, ys = Y.mean(axis=0), Y.std(axis=0)
    valid = ys > 1e-8
    ys[~valid] = 1.0
    R = (Y - ym) / ys

    W = np.full((X.shape[1], Y.shape[1], len(SVR_C_GRID)), np.nan)
    intercept = np.full((Y.shape[1], len(SVR_C_GRID)), np.nan)
    for o in range(Y.shape[1]):
        if not valid[o]:
            continue
        candidates = (range(len(SVR_C_GRID)) if selected_indices is None
                      else [int(selected_indices[o])])
        for ci in candidates:
            if ci < 0:
                continue
            reg = SVR(
                kernel="linear", C=float(SVR_C_GRID[ci]), epsilon=SVR_EPSILON,
                tol=SVM_TOL, max_iter=SVM_MAX_ITER, cache_size=200,
            )
            reg.fit(Z, R[:, o])
            if reg.fit_status_ != 0:
                raise RuntimeError("SVR 未收敛。")
            W[:, o, ci] = reg.coef_[0]
            intercept[o, ci] = reg.intercept_[0]
    return dict(kind="svr", xm=xm, xs=xs, ym=ym, ys=ys,
                W=W, intercept=intercept, valid=valid)

def predict_regression_grid(fit, X: np.ndarray) -> np.ndarray:
    if not isinstance(fit, dict):
        return predict_ridge_all_lambdas(fit, X)
    Z = (np.asarray(X, float) - fit["xm"]) / fit["xs"]
    pred = np.einsum("ni,iok->nok", Z, fit["W"])
    pred += fit["intercept"][None, :, :]
    pred = fit["ym"][None, :, None] + fit["ys"][None, :, None] * pred
    pred[:, ~fit["valid"], :] = np.nan
    return pred

def choose_parameter_by_train_time(d, X, Y_cont, outer_train, outer_fold, train_time_index):
    """
    对一个 source、一个 outer fold、一个 training-time：
    分别为 PE_magnitude 和 update 选择 Ridge 的 lambda 或 SVR 的 C。

    inner CV 直接使用 outer training set 中剩余的原始 fold 标签，
    因此外层测试 fold 从不参与参数选择。
    """
    losses = np.zeros((Y_cont.shape[1], len(regression_grid())), dtype=float)
    counts = np.zeros(Y_cont.shape[1], dtype=int)

    inner_fold_labels = sorted(np.unique(outer_fold[outer_train]))

    for j in inner_fold_labels:
        val = outer_train[outer_fold[outer_train] == j]
        inner = outer_train[outer_fold[outer_train] != j]

        if len(inner) < MIN_INNER_TRAIN or len(val) < 2:
            continue

        fit = fit_regression_grid(
            X[inner, train_time_index],
            Y_cont[inner],
        )
        pred = predict_regression_grid(
            fit,
            X[val, train_time_index],
        )

        for o in range(Y_cont.shape[1]):
            good = np.isfinite(pred[:, o, :]).all(axis=1) & np.isfinite(Y_cont[val, o])
            if not good.any():
                continue

            err = (
                pred[good, o, :]
                - Y_cont[val[good], o, None]
            ) ** 2

            losses[o] += err.sum(axis=0)
            counts[o] += int(good.sum())

    chosen = np.full(Y_cont.shape[1], -1, dtype=int)

    for o in range(Y_cont.shape[1]):
        if counts[o] > 0:
            chosen[o] = int(np.argmin(losses[o] / counts[o]))

    return chosen, losses, counts

# ============================================================
# 评分
# ============================================================

def subset_masks(d):
    """
    训练时不拆 ability / PE valence；
    评分时拆分。
    """
    pe = d["PE"].to_numpy()
    ability = d["ability"].astype(str).str.lower().to_numpy()

    return {
        "all": np.ones(len(d), dtype=bool),
        "ability_high": ability == "high",
        "ability_low": ability == "low",
        "PE_positive": pe > 0,
        "PE_negative": pe < 0,
        "high_positive": (ability == "high") & (pe > 0),
        "high_negative": (ability == "high") & (pe < 0),
        "low_positive": (ability == "low") & (pe > 0),
        "low_negative": (ability == "low") & (pe < 0),
    }

def score_vectors(y, pred, outcome):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)

    good = np.isfinite(y) & np.isfinite(pred)
    y = y[good]
    pred = pred[good]

    result = dict(
        n_trials=len(y),
        primary=np.nan,
        r=np.nan,
        fisher_z=np.nan,
        AUC=np.nan,
        accuracy=np.nan,
        balanced_accuracy=np.nan,
        R2=np.nan,
        MAE=np.nan,
        MSE=np.nan,
    )

    if len(y) < MIN_SCORE_TRIALS:
        return result

    if outcome == "PE_valence":
        aa = auc_score_binary(y, pred)
        result["AUC"] = aa

        if np.isfinite(aa):
            result["primary"] = aa - 0.5

        # decision_function > 0 对应 class 1
        hard = pred > 0
        result["accuracy"] = float(np.mean(hard == (y == 1)))

        if np.any(y == 1) and np.any(y == 0):
            sens = np.mean(hard[y == 1])
            spec = np.mean(~hard[y == 0])
            result["balanced_accuracy"] = float((sens + spec) / 2.0)

        return result

    if y.std() > 1e-8 and pred.std() > 1e-8:
        r = float(np.corrcoef(y, pred)[0, 1])
        z = float(np.arctanh(np.clip(r, -0.999999, 0.999999)))
        result["r"] = r
        result["fisher_z"] = z
        result["primary"] = z

    result["MSE"] = float(np.mean((y - pred) ** 2))
    result["MAE"] = float(np.mean(np.abs(y - pred)))

    if y.std() > 1e-8:
        denom = np.sum((y - y.mean()) ** 2)
        result["R2"] = float(
            1.0 - np.sum((y - pred) ** 2) / denom
        )

    return result

def score_oof(y, pred, outcome, folds):
    """分类避免不同外折decision score跨尺度比较；连续预测已还原到原单位。
    单个子条件试次较少：连续r用合并OOF预测，不能把折当作独立被试。
    """
    y, pred, folds = np.asarray(y), np.asarray(pred), np.asarray(folds)
    result = score_vectors(y, pred, outcome)
    result["n_valid_folds"] = 0
    result["n_scored_trials"] = result["n_trials"]
    result["status"] = "ok" if np.isfinite(result["primary"]) else "insufficient_trials_or_variance"
    if outcome != "PE_valence":
        result["n_valid_folds"] = len(np.unique(folds[np.isfinite(y) & np.isfinite(pred)]))
        return result
    result["AUC"] = result["primary"] = np.nan
    good = np.isfinite(y) & np.isfinite(pred)
    yy = y[good]
    if len(yy) < MIN_SCORE_TRIALS or min(np.sum(yy == 0), np.sum(yy == 1)) < MIN_SCORE_CLASS:
        result["status"] = "insufficient_class_counts"
        return result
    values, n_scored = [], 0
    for k in np.unique(folds):
        m = good & (folds == k)
        a, b = int(np.sum(y[m] == 1)), int(np.sum(y[m] == 0))
        if min(a, b) < 1:
            continue
        ranks = rankdata(pred[m])
        values.append((ranks[y[m] == 1].sum() - a*(a+1)/2) / (a*b))
        n_scored += a+b
    result["n_valid_folds"] = len(values)
    result["n_scored_trials"] = n_scored
    if len(values) >= MIN_VALID_AUC_FOLDS:
        result["AUC"] = float(np.mean(values))
        result["primary"] = result["AUC"] - 0.5
        result["status"] = "ok"
    else:
        result["status"] = "insufficient_valid_auc_folds"
    return result

SCORE_KEY = ["SubjectID", "band", "outcome", "source", "target",
             "train_start_ms", "test_start_ms"]

def condition_contrasts(frame):
    """同一人、同一方向、同一时间格点内做对比；保留不可估计行。"""
    specs = {
        "ability_high_minus_low": {"ability_high": 1, "ability_low": -1},
        "PE_positive_minus_negative": {"PE_positive": 1, "PE_negative": -1},
        "ability_diff_positive": {"high_positive": 1, "low_positive": -1},
        "ability_diff_negative": {"high_negative": 1, "low_negative": -1},
        "valence_diff_high": {"high_positive": 1, "high_negative": -1},
        "valence_diff_low": {"low_positive": 1, "low_negative": -1},
        "ability_x_valence": {"high_positive": 1, "low_positive": -1,
                              "high_negative": -1, "low_negative": 1},
    }
    assert not frame.duplicated(SCORE_KEY + ["subset"]).any()
    piv = frame.set_index(SCORE_KEY + ["subset"])["primary"].unstack("subset")
    results = []
    for name, weights in specs.items():
        out = piv.index.to_frame(index=False)
        vals = piv.reindex(columns=list(weights))
        out["primary"] = vals.mul(pd.Series(weights)).sum(axis=1, min_count=len(weights)).to_numpy()
        out["subset"] = name
        out["definition"] = json.dumps(weights, ensure_ascii=False)
        if name != "ability_high_minus_low":
            out = out[out.outcome != "PE_valence"]
        out["status"] = np.where(np.isfinite(out.primary), "ok", "missing_component_score")
        results.append(out)
    return pd.concat(results, ignore_index=True)

# ============================================================
# 单个被试 × 单频段
# ============================================================

def run_person_band(d, X, band, dest, fingerprint):
    marker = dest / "complete.json"

    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["fingerprint"] != fingerprint:
            raise RuntimeError(
                "配置或输入已经改变；请使用新的 OUTPUT 目录。"
            )
        return

    dest.mkdir(parents=True, exist_ok=True)

    sid = int(d["SubjectID"].iloc[0])

    # 每位被试固定一套普通随机 5-fold
    fold = make_subject_folds(d, seed=SEED + sid)

    pe = d["PE"].to_numpy(dtype=float)

    # zero PE 在 valence 中记为 NaN，因此不会进入分类训练或评分
    y_valence = np.where(
        pe > 0,
        1.0,
        np.where(pe < 0, 0.0, np.nan),
    )

    y_magnitude = np.abs(pe)
    y_update = d["update"].to_numpy(dtype=float)

    assert np.allclose(
        y_update,
        d["next_same_task_expectation"].to_numpy(dtype=float)
        - d["expectation"].to_numpy(dtype=float),
    )

    Y = np.column_stack([y_valence, y_magnitude, y_update])
    Y_cont = np.column_stack([y_magnitude, y_update])

    nw = X.shape[1]
    no = len(OUTCOMES)

    # prediction[source_feedback, outcome, train_time, test_time, trial]
    P = np.full(
        (2, no, nw, nw, len(d)),
        np.nan,
        dtype=np.float32,
    )

    coverage = np.zeros((2, len(d)), dtype=int)

    audits = []
    tuning = []
    splits = []

    fb = d["feedback_type"].astype(str).to_numpy()

    for k in range(N_FOLDS):
        held = np.flatnonzero(fold == k)

        for fi, source in enumerate(FEEDBACKS):
            # 不匹配 Absolute / Relative 的训练试次数；
            # 该 source 的所有非测试折 trial 都用于训练
            train = np.flatnonzero(
                (fold != k) & (fb == source)
            )

            assert not set(train) & set(held)
            assert np.all(fb[train] == source)

            audit = dict(
                SubjectID=sid,
                band=band,
                fold=k,
                source=source,
                n_train_total=len(train),
                n_train_positive=int(np.sum(pe[train] > 0)),
                n_train_negative=int(np.sum(pe[train] < 0)),
                n_train_zero=int(np.sum(pe[train] == 0)),
                n_test_absolute=int(np.sum(fb[held] == "absolute")),
                n_test_relative=int(np.sum(fb[held] == "relative")),
                overlap=0,
            )

            split = dict(
                SubjectID=sid,
                band=band,
                fold=k,
                source=source,
                train_row_ids=d.loc[train, "row_id"].astype(int).tolist(),
                test_row_ids=d.loc[held, "row_id"].astype(int).tolist(),
            )

            # ------------------------------------------------
            # 1) PE valence: balanced SVC / Logistic
            # ------------------------------------------------
            train_val = train[np.isfinite(y_valence[train])]

            if len(train_val) >= MIN_OUTER_TRAIN:
                yv = y_valence[train_val].astype(int)

                if np.bincount(yv, minlength=2).min() >= MIN_CLASS_TRAIN:
                    for ti in range(nw):
                        fit_log = fit_classifier(
                            X[train_val, ti],
                            yv,
                        )

                        if fit_log is None:
                            continue

                        for tj in range(nw):
                            test_rows = held if ti == tj else held[fb[held] != source]
                            P[
                                fi,
                                OUTCOMES.index("PE_valence"),
                                ti,
                                tj,
                                test_rows,
                            ] = predict_classifier(
                                fit_log,
                                X[test_rows, tj],
                            )

            # ------------------------------------------------
            # 2) PE magnitude / update: SVR / Ridge
            #    C / lambda 每个 training-time 单独 inner-CV
            # ------------------------------------------------
            if len(train) >= MIN_OUTER_TRAIN:
                chosen_by_time = np.full(
                    (nw, Y_cont.shape[1]),
                    -1,
                    dtype=int,
                )

                for ti in range(nw):
                    chosen, losses, counts = choose_parameter_by_train_time(
                        d=d,
                        X=X,
                        Y_cont=Y_cont,
                        outer_train=train,
                        outer_fold=fold,
                        train_time_index=ti,
                    )

                    chosen_by_time[ti] = chosen

                    for o_cont, outcome in enumerate(["PE_magnitude", "update"]):
                        for li, param in enumerate(regression_grid()):
                            tuning.append(
                                dict(
                                    SubjectID=sid,
                                    band=band,
                                    outer_fold=k,
                                    source=source,
                                    outcome=outcome,
                                    train_start_ms=int(WINDOW_STARTS_MS[ti]),
                                    parameter_name=("lambda" if REGRESSOR == "ridge" else "C"),
                                    parameter_value=float(param),
                                    lambda_value=(float(param) if REGRESSOR == "ridge" else np.nan),
                                    C_value=(float(param) if REGRESSOR == "svr" else np.nan),
                                    epsilon=(SVR_EPSILON if REGRESSOR == "svr" else np.nan),
                                    validation_MSE=(
                                        losses[o_cont, li] / counts[o_cont]
                                        if counts[o_cont] > 0
                                        else np.nan
                                    ),
                                    validation_trial_count=int(counts[o_cont]),
                                    selected=(li == chosen[o_cont]),
                                )
                            )

                for ti in range(nw):
                    fit_ridge = fit_regression_grid(
                        X[train, ti], Y_cont[train],
                        selected_indices=chosen_by_time[ti],
                    )

                    for tj in range(nw):
                        test_rows = held if ti == tj else held[fb[held] != source]
                        pred_all = predict_regression_grid(
                            fit_ridge,
                            X[test_rows, tj],
                        )

                        # continuous outcome 0 = PE magnitude
                        # continuous outcome 1 = update
                        for o_cont, outcome in enumerate(["PE_magnitude", "update"]):
                            li = chosen_by_time[ti, o_cont]
                            if li < 0:
                                continue

                            oi = OUTCOMES.index(outcome)
                            P[fi, oi, ti, tj, test_rows] = pred_all[:, o_cont, li]

            coverage[fi, held] += 1
            audits.append(audit)
            splits.append(split)

    assert np.all(coverage == 1)

    np.savez_compressed(
        dest / "oof_predictions.npz",
        prediction=P,
        outcome=Y,
        row_id=d["row_id"].to_numpy(),
        fold=fold,
        SubjectID=np.array(sid),
        window_starts_ms=np.asarray(WINDOW_STARTS_MS),
        window_width_ms=np.array(WINDOW_WIDTH_MS),
        source_labels=np.array(FEEDBACKS),
        outcome_labels=np.array(OUTCOMES),
        classifier=np.array(CLASSIFIER), regressor=np.array(REGRESSOR),
    )

    d.assign(
        outer_fold=fold,
        PE_valence_binary=y_valence,
    ).to_csv(
        dest / "trial_labels.csv",
        index=False,
    )

    pd.DataFrame(audits).to_csv(
        dest / "split_audit.csv",
        index=False,
    )

    pd.DataFrame(tuning, columns=[
        "SubjectID", "band", "outer_fold", "source", "outcome", "train_start_ms",
        "parameter_name", "parameter_value", "lambda_value", "C_value", "epsilon",
        "validation_MSE", "validation_trial_count", "selected",
    ]).to_csv(
        dest / "tuning.csv",
        index=False,
    )

    write_json(
        dest / "split_indices.json",
        splits,
    )

    # --------------------------------------------------------
    # 被试级 decoding score
    # --------------------------------------------------------
    score_rows = []
    gap_rows = []

    masks = subset_masks(d)

    for oi, outcome in enumerate(OUTCOMES):
        for target in FEEDBACKS:
            target_mask = fb == target

            for subset_name, subset_mask in masks.items():
                # valence 分类不能在只有 positive 或只有 negative 的 subset 中评分
                if outcome == "PE_valence" and subset_name not in ["all", "ability_high", "ability_low"]:
                    continue

                mask = target_mask & subset_mask

                for ti, train_start in enumerate(WINDOW_STARTS_MS):
                    for tj, test_start in enumerate(WINDOW_STARTS_MS):

                        pair = {}

                        # 只有两个 source 都有预测的 trial
                        common = (
                            mask
                            & np.isfinite(P[0, oi, ti, tj])
                            & np.isfinite(P[1, oi, ti, tj])
                        )

                        for fi, source in enumerate(FEEDBACKS):
                            if source == target and ti != tj:
                                continue
                            r = score_oof(
                                Y[mask, oi],
                                P[fi, oi, ti, tj, mask],
                                outcome, fold[mask],
                            )

                            score_rows.append(
                                dict(
                                    SubjectID=sid,
                                    band=band,
                                    outcome=outcome,
                                    source=source,
                                    target=target,
                                    subset=subset_name,
                                    train_start_ms=int(train_start),
                                    test_start_ms=int(test_start),
                                    **r,
                                )
                            )

                            pair[source] = score_oof(
                                Y[common, oi],
                                P[fi, oi, ti, tj, common],
                                outcome, fold[common],
                            )

                        if ti != tj:
                            continue  # 条件内参照只在同时间窗存在。

                        # within - cross
                        other = (
                            "relative"
                            if target == "absolute"
                            else "absolute"
                        )

                        within = pair[target]
                        cross = pair[other]

                        gap_rows.append(
                            dict(
                                SubjectID=sid,
                                band=band,
                                outcome=outcome,
                                source="within_minus_cross",
                                target=target,
                                subset=subset_name,
                                train_start_ms=int(train_start),
                                test_start_ms=int(test_start),
                                n_trials=within["n_trials"],
                                n_valid_folds=within["n_valid_folds"],
                                status=("ok" if np.isfinite(within["primary"]) and np.isfinite(cross["primary"]) else "missing_paired_score"),
                                primary=(
                                    within["primary"] - cross["primary"]
                                    if np.isfinite(within["primary"])
                                    and np.isfinite(cross["primary"])
                                    else np.nan
                                ),
                                within_primary=within["primary"],
                                cross_primary=cross["primary"],
                                R2=(
                                    within["R2"] - cross["R2"]
                                    if np.isfinite(within["R2"])
                                    and np.isfinite(cross["R2"])
                                    else np.nan
                                ),
                            )
                        )

    scores = pd.DataFrame(score_rows)
    gaps = pd.DataFrame(gap_rows)
    for frame in (scores, gaps):
        frame["classifier"] = CLASSIFIER
        frame["regressor"] = REGRESSOR

    scores.to_csv(
        dest / "subject_scores.csv",
        index=False,
    )
    gaps.to_csv(
        dest / "paired_transfer_scores.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 直接条件差异：
    # high - low
    # positive - negative（仅 continuous outcomes）
    # --------------------------------------------------------
    condition_contrasts(scores).to_csv(dest / "subject_condition_contrasts.csv", index=False)
    condition_contrasts(gaps).to_csv(dest / "transfer_condition_contrasts.csv", index=False)

    write_json(
        marker,
        dict(
            fingerprint=fingerprint,
            SubjectID=sid,
            band=band,
            finite_prediction_fraction=float(np.isfinite(P).mean()),
        ),
    )

# ============================================================
# 2D temporal-generalization cluster permutation
# ============================================================

def build_time_graph(train_times, test_times):
    """
    只连接 train-time / test-time 上相邻的格子。
    """
    train_times = list(sorted(train_times))
    test_times = list(sorted(test_times))

    nodes = []
    lookup = {}

    idx = 0
    for tr in train_times:
        for te in test_times:
            lookup[(tr, te)] = idx
            nodes.append((tr, te))
            idx += 1

    graph = sparse.lil_matrix((len(nodes), len(nodes)), dtype=int)

    train_pos = {v: i for i, v in enumerate(train_times)}
    test_pos = {v: i for i, v in enumerate(test_times)}

    for (tr, te), i in lookup.items():
        it = train_pos[tr]
        jt = test_pos[te]

        neighbors = []

        if it > 0:
            neighbors.append((train_times[it - 1], te))
        if it < len(train_times) - 1:
            neighbors.append((train_times[it + 1], te))
        if jt > 0:
            neighbors.append((tr, test_times[jt - 1]))
        if jt < len(test_times) - 1:
            neighbors.append((tr, test_times[jt + 1]))

        for nb in neighbors:
            j = lookup[nb]
            graph[i, j] = 1

    return nodes, graph.tocsr()

def one_matrix_group_test(
    g,
    outcome,
    analysis_family,
    subset,
    band,
    source,
    target,
    tail,
):
    """
    一个固定矩阵：
      outcome × family × subset × band × source × target
    只对 train-time × test-time 做 cluster correction。
    """
    keys = ["train_start_ms", "test_start_ms"]

    pvt = g.pivot_table(
        index="SubjectID",
        columns=keys,
        values="primary",
        aggfunc="first",
    ).sort_index(axis=1)

    # 完整网格顺序
    train_times = sorted(g["train_start_ms"].unique())
    test_times = sorted(g["test_start_ms"].unique())

    diagonal_only = source == target or source == "within_minus_cross"
    if diagonal_only:
        assert (g.train_start_ms == g.test_start_ms).all()
        full_cols = pd.MultiIndex.from_tuples([(t, t) for t in train_times], names=keys)
    else:
        full_cols = pd.MultiIndex.from_product([train_times, test_times], names=keys)
    pvt = pvt.reindex(columns=full_cols)

    valid = pvt.notna().to_numpy()
    Z = pvt.fillna(0.0).to_numpy(dtype=float)

    n = valid.sum(axis=0)
    eligible = n >= MIN_GROUP_N

    denom = np.maximum(n, 1)
    mean = Z.sum(axis=0) / denom

    variance = np.maximum(
        (
            np.sum(Z ** 2, axis=0)
            - n * mean ** 2
        )
        / np.maximum(n - 1, 1),
        0.0,
    )

    se = np.sqrt(variance / denom)
    eligible &= se > 1e-12  # 零方差的t统计量不可估计
    T = np.divide(
        mean,
        se,
        out=np.zeros_like(mean),
        where=se > 0,
    )
    T[~eligible] = 0.0

    df = np.maximum(n - 1, 1)

    if tail == "two_sided":
        threshold = tdist.ppf(0.975, df)
        p_raw = 2.0 * tdist.sf(np.abs(T), df)
    elif tail == "positive_one_sided":
        threshold = tdist.ppf(0.95, df)
        p_raw = tdist.sf(T, df)
    else:
        raise ValueError(tail)

    # threshold 每个 cell 的 df 可能略不同。
    # 为了 cluster formation，逐 cell 比较，而不是用单一 t threshold。
    if tail == "two_sided":
        supra_pos = T > threshold
        supra_neg = -T > threshold
    else:
        supra_pos = T > threshold
        supra_neg = np.zeros_like(supra_pos, dtype=bool)

    if diagonal_only:
        edges = np.ones(max(len(train_times)-1, 0))
        graph = sparse.diags([edges, edges], [-1, 1],
                             shape=(len(train_times), len(train_times)), format="csr")
    else:
        nodes, graph = build_time_graph(train_times, test_times)

    def extract_clusters_from_masks(tvals, pos_mask, neg_mask):
        out = []

        for sign, mask in [(1, pos_mask), (-1, neg_mask)]:
            ix = np.flatnonzero(mask)

            if len(ix) == 0:
                continue

            n_comp, labels = connected_components(
                graph[ix][:, ix],
                directed=False,
            )

            for comp in range(n_comp):
                node_ix = ix[labels == comp]
                mass = float(np.sum(np.abs(tvals[node_ix])))
                out.append((mass, sign, node_ix))

        return out

    observed_clusters = extract_clusters_from_masks(
        T,
        supra_pos,
        supra_neg,
    )

    # sign-flip null
    rng = np.random.default_rng(SEED)
    null_max = np.zeros(N_SIGNFLIPS, dtype=float)

    for b in range(N_SIGNFLIPS):
        if not eligible.any():
            break  # 无可估计格点时保留零null，不执行无意义置换。
        signs = rng.choice(
            [-1.0, 1.0],
            size=len(Z),
        )

        mu = signs @ Z / denom

        var = np.maximum(
            (
                np.sum(Z ** 2, axis=0)
                - n * mu ** 2
            )
            / np.maximum(n - 1, 1),
            1e-15,
        )

        perm_t = mu / np.sqrt(var / denom)
        perm_t[~eligible] = 0.0

        if tail == "two_sided":
            pmask = perm_t > threshold
            nmask = -perm_t > threshold
        else:
            pmask = perm_t > threshold
            nmask = np.zeros_like(pmask, dtype=bool)

        perm_clusters = extract_clusters_from_masks(
            perm_t,
            pmask,
            nmask,
        )

        null_max[b] = max(
            [0.0] + [c[0] for c in perm_clusters]
        )

    # point table
    point = pd.DataFrame(
        [tuple(x) for x in pvt.columns],
        columns=keys,
    )

    point["map_dimension"] = "1D_same_time" if diagonal_only else "2D_cross_time"
    point["outcome"] = outcome
    point["analysis_family"] = analysis_family
    point["subset"] = subset
    point["band"] = band
    point["band_family"] = band_family(band)
    point["source"] = source
    point["target"] = target
    point["tail"] = tail
    point["n_subjects"] = n
    point["mean_primary"] = np.where(eligible, mean, np.nan)
    point["SE"] = np.where(eligible, se, np.nan)
    point["t"] = np.where(eligible, T, np.nan)
    point["df"] = np.where(eligible, n - 1, np.nan)
    point["p_raw"] = np.where(eligible, p_raw, np.nan)

    ci = tdist.ppf(0.975, df) * se
    point["lower_primary"] = np.where(eligible, mean - ci, np.nan)
    point["upper_primary"] = np.where(eligible, mean + ci, np.nan)

    if analysis_family == "decoding":
        if outcome == "PE_valence":
            point["performance"] = point["mean_primary"] + 0.5
            point["performance_lower"] = point["lower_primary"] + 0.5
            point["performance_upper"] = point["upper_primary"] + 0.5
        else:
            point["performance"] = np.tanh(point["mean_primary"])
            point["performance_lower"] = np.tanh(point["lower_primary"])
            point["performance_upper"] = np.tanh(point["upper_primary"])

    point["cluster_id"] = ""
    point["p_cluster"] = np.where(eligible, 1.0, np.nan)

    cluster_rows = []

    matrix_stub = (
        f"{analysis_family}"
        f"__{outcome}"
        f"__{subset}"
        f"__{band}"
        f"__{source}"
        f"__{target}"
        f"__{tail}"
    )

    for ci_idx, (mass, sign, ix) in enumerate(observed_clusters):
        p_cluster = (
            1.0
            + np.sum(null_max >= mass)
        ) / (N_SIGNFLIPS + 1.0)

        cid = f"{matrix_stub}__cluster{ci_idx}"

        sub = point.iloc[ix]
        peak_local = ix[np.nanargmax(np.abs(T[ix]))]
        peak = point.iloc[peak_local]

        cluster_rows.append(
            dict(
                cluster_id=cid,
                outcome=outcome,
                analysis_family=analysis_family,
                subset=subset,
                band=band,
                band_family=band_family(band),
                source=source,
                target=target,
                tail=tail,
                sign=int(sign),
                p_cluster=float(p_cluster),
                mass=float(mass),
                n_nodes=int(len(ix)),
                train_start_min=int(sub["train_start_ms"].min()),
                train_start_max=int(sub["train_start_ms"].max()),
                test_start_min=int(sub["test_start_ms"].min()),
                test_start_max=int(sub["test_start_ms"].max()),
                peak_train_start=int(peak["train_start_ms"]),
                peak_test_start=int(peak["test_start_ms"]),
                peak_mean_primary=float(peak["mean_primary"]),
                peak_t=float(peak["t"]),
                n_subjects_peak=int(peak["n_subjects"]),
            )
        )

        point.loc[ix, "cluster_id"] = cid
        point.loc[ix, "p_cluster"] = p_cluster

    null_name = (
        f"null"
        f"__{analysis_family}"
        f"__{outcome}"
        f"__{subset}"
        f"__{band}"
        f"__{source}"
        f"__{target}"
        f"__{tail}.npy"
    )

    return point, pd.DataFrame(cluster_rows), null_name, null_max

def run_group_inference(frame, analysis_family, inference_dir):
    all_points = []
    all_clusters = []

    for outcome in sorted(frame["outcome"].unique()):
        allowed_subsets = INFERENCE_SUBSETS.get(outcome, ["all"])

        # condition contrast frame 自己带 contrast 名称
        if analysis_family in ["condition_contrast", "transfer_condition_contrast"]:
            allowed_subsets = sorted(frame["subset"].unique())

        for subset in allowed_subsets:
            sub0 = frame[
                (frame["outcome"] == outcome)
                & (frame["subset"] == subset)
            ].copy()

            if sub0.empty:
                continue

            matrix_groups = sub0.groupby(
                ["band", "source", "target"],
                sort=True,
            )

            for (band, source, target), g in matrix_groups:
                if g.empty:
                    continue

                for tail in TAILS:
                    point, clusters, null_name, null_max = one_matrix_group_test(
                        g=g,
                        outcome=outcome,
                        analysis_family=analysis_family,
                        subset=subset,
                        band=band,
                        source=source,
                        target=target,
                        tail=tail,
                    )

                    all_points.append(point)

                    if not clusters.empty:
                        all_clusters.append(clusters)

                    np.save(
                        inference_dir / null_name,
                        null_max,
                    )

    if all_points:
        points = pd.concat(all_points, ignore_index=True)
    else:
        points = pd.DataFrame()

    if all_clusters:
        clusters = pd.concat(all_clusters, ignore_index=True)
    else:
        clusters = pd.DataFrame()

    return points, clusters

def main():
    global OUTPUT, N_SIGNFLIPS

    if CLASSIFIER not in {"svc", "logistic"} or REGRESSOR not in {"svr", "ridge"}:
        raise ValueError("模型设置：CLASSIFIER=svc/logistic；REGRESSOR=svr/ridge。")
    if len(regression_grid()) == 0 or np.any(regression_grid() <= 0):
        raise ValueError("回归候选参数必须为正数。")
    if SVR_EPSILON < 0:
        raise ValueError("SVR_EPSILON 必须非负。")

    bands = MAIN_BANDS.copy()
    if RUN_SECONDARY_BANDS:
        bands += SECONDARY_BANDS

    if SMOKE_TEST:
        OUTPUT = OUTPUT.with_name(OUTPUT.name + "_SMOKE")
        bands = ["alpha"]
        # 保留1000次；SMOKE仅使用两人，群体最低人数仍为20

    OUTPUT.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((INPUT / "feature_spec.json").read_text(encoding="utf-8"))
    if metadata["windows_ms"] != [[int(t), int(t+50)] for t in WINDOW_STARTS_MS]:
        raise ValueError("输入不是0–1000ms的20个50ms窗口。先重新提取特征。")
    if metadata.get("sampling_hz") != 250:
        raise ValueError("必须使用真实250Hz重采样后重新提取的特征，不能使用旧100Hz输入。")
    if metadata["bands"] != ALL_BANDS:
        raise ValueError("频段顺序不一致。")
    d = pd.read_csv(INPUT / "trials.csv")
    if metadata["trials_hash"] != hashlib.sha256((INPUT / "trials.csv").read_bytes()).hexdigest():
        raise ValueError("特征和试次表不匹配。")
    neural = np.load(
        INPUT / "neural.npy",
        mmap_mode="r",
    )

    if len(neural) != len(d):
        raise ValueError("neural.npy 与 trials.csv 行数不一致。")

    if neural.shape[1] != len(ALL_BANDS):
        raise ValueError(
            f"neural.npy 频段维={neural.shape[1]}，"
            f"但 ALL_BANDS={len(ALL_BANDS)}。"
        )

    if neural.shape[2] != len(WINDOW_STARTS_MS):
        raise ValueError(
            f"当前 TIME_SCHEME={TIME_SCHEME} 需要 "
            f"{len(WINDOW_STARTS_MS)} 个时间窗，"
            f"但 neural.npy 有 {neural.shape[2]} 个。"
        )

    required_cols = [
        "SubjectID",
        "row_id",
        "feedback_type",
        "ability",
        "PE",
        "update",
        "expectation",
        "next_same_task_expectation",
    ]

    missing = [c for c in required_cols if c not in d.columns]
    if missing:
        raise ValueError(f"trials.csv 缺少列: {missing}")

    if not set(d["feedback_type"]).issubset(set(FEEDBACKS)):
        raise ValueError("feedback_type必须为absolute/relative。")
    if not set(d["ability"]).issubset({"high", "low"}):
        raise ValueError("ability必须为high/low。")
    if not d["row_id"].is_unique:
        raise ValueError("row_id 必须唯一。")

    continuous_check = d[
        [
            "PE",
            "update",
            "expectation",
            "next_same_task_expectation",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(continuous_check).all():
        raise ValueError("PE / update / expectation 中存在非有限值。")

    ids = sorted(d["SubjectID"].unique())
    if SMOKE_TEST:
        ids = ids[:2]

    config = dict(
        code_version=CODE_VERSION,
        classifier=CLASSIFIER,
        regressor=REGRESSOR,
        svc_C=SVC_C,
        svr_C_grid=SVR_C_GRID.tolist(),
        svr_epsilon=SVR_EPSILON,
        svr_target_scaling="within-training-fold z score; inverse transform predictions",
        svm_kernel="linear",
        svm_tol=SVM_TOL,
        svm_max_iter=SVM_MAX_ITER,
        subjects=list(map(int, ids)),
        main_bands=MAIN_BANDS,
        secondary_bands=SECONDARY_BANDS,
        run_secondary_bands=RUN_SECONDARY_BANDS,
        actual_bands=bands,
        within_time="diagonal_only",
        transfer_cost_time="diagonal_only",
        cross_time="all_train_test_pairs",
        time_scheme=TIME_SCHEME,
        window_starts_ms=list(map(int, WINDOW_STARTS_MS)),
        window_width_ms=int(WINDOW_WIDTH_MS),
        outcomes=OUTCOMES,
        folds=N_FOLDS,
        logistic_C=LOGISTIC_C,
        lambdas=LAMBDAS.tolist(),
        min_outer_train=MIN_OUTER_TRAIN,
        min_inner_train=MIN_INNER_TRAIN,
        min_class_train=MIN_CLASS_TRAIN,
        min_score_trials=MIN_SCORE_TRIALS,
        min_score_class=MIN_SCORE_CLASS,
        min_valid_auc_folds=MIN_VALID_AUC_FOLDS,
        correction="per-map two-sided max cluster; no across-map adjustment",
        continuous_scoring="pooled original-unit OOF predictions",
        classification_scoring="equal-weight mean within-fold AUC",
        min_group_n=MIN_GROUP_N,
        inference_subsets=INFERENCE_SUBSETS,
        tails=TAILS,
        signflips=N_SIGNFLIPS,
        seed=SEED,
        smoke=SMOKE_TEST,
        trials_hash=hashlib.sha256(
            (INPUT / "trials.csv").read_bytes()
        ).hexdigest(),
        power_size=(INPUT / "neural.npy").stat().st_size,
        power_mtime=(INPUT / "neural.npy").stat().st_mtime_ns,
        # 脚本运行时校验文件；交互运行时使用手动版本标记。
        # 交互运行中修改分析代码后，同步更新下面的版本标记。
        script_hash=(
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            if globals().get("__file__") and Path(__file__).is_file()
            else CODE_VERSION
        ),
    )

    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()

    plan = OUTPUT / "configuration.json"

    if plan.exists():
        old = json.loads(plan.read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint:
            raise RuntimeError(
                "当前 OUTPUT 已存在不同配置；请更换 OUTPUT 目录。"
            )

    write_json(
        plan,
        dict(
            fingerprint=fingerprint,
            **config,
        ),
    )

    import sklearn

    write_json(
        OUTPUT / "software.json",
        dict(
            python=platform.python_version(),
            numpy=np.__version__,
            pandas=pd.__version__,
            scipy=scipy.__version__,
            sklearn=sklearn.__version__,
        ),
    )

    subject_band_folders = []

    for sid in ids:
        indices = np.flatnonzero(
            d["SubjectID"].to_numpy() == sid
        )

        person = (
            d.iloc[indices]
            .copy()
            .reset_index(drop=True)
        )

        # 保持 trials.csv 当前顺序；
        # 普通随机 CV 不再依赖 trial 的连续时间位置
        for band in bands:
            folder = (
                OUTPUT
                / "subjects"
                / f"{int(sid)}_{band}"
            )
            subject_band_folders.append(folder)

            bi = ALL_BANDS.index(band)

            power = np.asarray(
                neural[indices, bi, :, :],
                dtype=float,
            )

            if not np.isfinite(power).all():
                raise ValueError(
                    f"Subject {sid}, band {band} 存在非有限 power。"
                )

            run_person_band(
                person,
                power,
                band,
                folder,
                fingerprint,
            )

            print(
                "完成",
                int(sid),
                band,
                flush=True,
            )

    # ========================================================
    # 汇总被试级文件
    # ========================================================

    file_specs = [
        ("subject_scores.csv", "all_subject_scores.csv"),
        ("paired_transfer_scores.csv", "all_paired_transfer_scores.csv"),
        ("subject_condition_contrasts.csv", "all_subject_condition_contrasts.csv"),
        ("transfer_condition_contrasts.csv", "all_transfer_condition_contrasts.csv"),
        ("split_audit.csv", "all_split_audit.csv"),
        ("tuning.csv", "all_tuning.csv"),
    ]

    # 大型被试级CSV逐文件写出，不同时把全部频段/所有结局装入内存。
    for src_name, out_name in file_specs:
        first = True
        for folder in subject_band_folders:
            frame = pd.read_csv(folder / src_name)
            frame.to_csv(OUTPUT / out_name, mode="w" if first else "a",
                         header=first, index=False)
            first = False

    inference_dir = OUTPUT / "inference"
    inference_dir.mkdir(exist_ok=True)
    point_frames, cluster_frames = [], []
    families = [
        ("subject_scores.csv", "decoding"),
        ("paired_transfer_scores.csv", "transfer_cost"),
        ("subject_condition_contrasts.csv", "condition_contrast"),
        ("transfer_condition_contrasts.csv", "transfer_condition_contrast"),
    ]
    # 推断仍为每图独立校正；仅分批加载，不改变统计模型。
    for band in bands:
        folders = [f for f in subject_band_folders if f.name.endswith("_" + band)]
        for filename, family in families:
            frame = pd.concat([pd.read_csv(f / filename) for f in folders], ignore_index=True)
            print("群体推断", CLASSIFIER, REGRESSOR, band, family, flush=True)
            p, c = run_group_inference(frame, family, inference_dir)
            if not p.empty:
                point_frames.append(p)
            if not c.empty:
                cluster_frames.append(c)
            del frame

    if point_frames:
        all_points = pd.concat(
            point_frames,
            ignore_index=True,
        )
    else:
        all_points = pd.DataFrame()

    if cluster_frames:
        all_clusters = pd.concat(
            cluster_frames,
            ignore_index=True,
        )
    else:
        all_clusters = pd.DataFrame(columns=["cluster_id", "outcome", "analysis_family", "subset", "band", "source", "target", "tail", "p_cluster", "mass", "n_nodes"])

    all_points.to_csv(
        OUTPUT / "all_group_points.csv",
        index=False,
    )

    all_clusters.to_csv(
        OUTPUT / "all_clusters.csv",
        index=False,
    )
    # 仅提取已有二维检验的对角线，不另做一维显著性检验。
    all_points[all_points.train_start_ms == all_points.test_start_ms].to_csv(
        OUTPUT / "all_diagonal_points.csv", index=False,
    )

    # ========================================================
    # 结果使用说明
    # ========================================================

    notes = {
        "time_design": "within diagonal20; cross20x20; transfer differences diagonal20",
        "classifier": CLASSIFIER,
        "regressor": REGRESSOR,
        "classification_parameter": "C fixed; AUC uses decision_function, not hard labels",
        "regression_parameter": "Ridge lambda / SVR C selected per source, fold, train-time, outcome by inner-CV MSE",
        "svr_epsilon": "0.1 by default in training-y SD units; fixed during tuning",
        "comparison": "Same folds, features, outcomes and scoring as v3. Models output into separate directories.",
        "feedback_type_classification": "Not included. feedback_type selects training/testing domains, not class labels.",
        "inference_limits": "Subject-score sign flips assume symmetric null; not trial-label permutation/retraining. No causal or identical-mechanism inference.",
        "classification_scoring": "Mean within-fold AUC; minimum 3 valid folds, each with both classes. Overall class counts minimum 3 each.",
        "continuous_scoring": "Pooled OOF r in original target units; not fold-mean r. Primary = Fisher z.",
        "transfer_condition_contrast": "Condition contrasts of within-minus-cross score; ability_x_valence=(HP-LP)-(HN-LN).",
        "decoding_primary_metric": {
            "PE_valence": "AUC - 0.5; chance = 0",
            "PE_magnitude": "Fisher z of Pearson r; chance = 0",
            "update": "Fisher z of Pearson r; chance = 0",
        },
        "transfer_cost": "within-feedback primary - cross-feedback primary",
        "condition_contrast": {
            "ability_high_minus_low": "high subset primary - low subset primary",
            "PE_positive_minus_negative": "positive-PE subset primary - negative-PE subset primary",
        },
        "cluster_statistics": {
            "p_cluster": "FWER-corrected within one train-time × test-time matrix",
            "across_maps": "not corrected; report all maps as exploratory families",
            "two_sided": "cluster-forming pointwise p < .05, two-sided",
        },
        "PE_zero": {
            "PE_valence": "excluded",
            "PE_magnitude": "retained",
            "update": "retained",
        },
        "training": {
            "ability": "pooled during training",
            "feedback": "absolute and relative trained separately",
            "training_count_matching": False,
            "cross_validation": "random stratified 5-fold by feedback × ability × PE sign",
            "embargo": False,
        },
    }

    write_json(
        OUTPUT / "analysis_notes.json",
        notes,
    )

    print(
        "完成。输出目录：",
        OUTPUT,
        flush=True,
    )

if __name__ == "__main__":
    main()




