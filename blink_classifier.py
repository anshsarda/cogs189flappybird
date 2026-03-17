import os
import sys
import joblib
import numpy as np

from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


FS = 250
EEG_CH = [0, 1]
MODEL_FILE = "blink_model.joblib"

TRAIN_RUNS = ["1", "2"]
TEST_RUNS  = ["3"]


def bandpass(x, low=1, high=15):
    nyq = FS / 2
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x, axis=1)


def make_epochs(eeg, events, timestamps):
    eeg = eeg[EEG_CH, :]
    eeg = bandpass(eeg)

    win = int(0.8 * FS)

    blink = []
    noblink = []

    total_events = 0

    for ev in events:
        try:
            name = str(ev["event"]).lower()
            t_sec = float(ev["time"])
        except:
            try:
                name = str(ev[1]).lower()
                t_sec = float(ev[2])
            except:
                continue

        if "blink" not in name:
            continue

        total_events += 1

        idx = np.argmin(np.abs(timestamps - t_sec))

        a = idx - win // 2
        b = idx + win // 2

        c = idx + int(1.5 * FS)
        d = c + win

        if a < 0 or b > eeg.shape[1]:
            continue
        if d > eeg.shape[1]:
            continue

        e1 = eeg[:, a:b]
        e2 = eeg[:, c:d]

        if e1.shape[1] != win or e2.shape[1] != win:
            continue

        e1 = e1 - e1.mean(axis=1, keepdims=True)
        e2 = e2 - e2.mean(axis=1, keepdims=True)

        blink.append(e1)
        noblink.append(e2)

    print("events found:", total_events)
    print("epochs created:", len(blink))

    X = blink + noblink
    y = np.array([1]*len(blink) + [0]*len(noblink))

    return X, y


def load_runs(folder):
    runs = {}

    for f in os.listdir(folder):
        if f.startswith("eeg_run"):
            run = f.split("-")[1].split(".")[0]

            eeg = np.load(os.path.join(folder, f))
            events = np.load(os.path.join(folder, f"events_run-{run}.npy"), allow_pickle=True)
            timestamps = np.load(os.path.join(folder, f"timestamp_run-{run}.npy"))

            X, y = make_epochs(eeg, events, timestamps)
            runs[run] = (X, y)

    return runs


def balance(X, y):
    X = np.array(X, dtype=object)

    idx1 = np.where(y == 1)[0]
    idx0 = np.where(y == 0)[0]

    if len(idx1) == 0 or len(idx0) == 0:
        return X, y

    n = min(len(idx1), len(idx0))
    idx = np.concatenate([idx1[:n], idx0[:n]])

    return X[idx], y[idx]


def extract_features(X):
    feats = []

    for ep in X:
        ep = np.array(ep)

        ch_feats = []
        for ch in ep:
            ch_feats.extend([
                np.max(ch),
                np.min(ch),
                np.ptp(ch),
                np.std(ch),
                np.sum(np.abs(ch))
            ])

        feats.append(ch_feats)

    return np.array(feats)


def train_model(X, y):
    Xf = extract_features(X)

    if len(Xf) == 0:
        raise ValueError("No training data — check event parsing.")

    scaler = StandardScaler()
    Xf = scaler.fit_transform(Xf)

    clf = LinearDiscriminantAnalysis()
    clf.fit(Xf, y)

    return {"scaler": scaler, "clf": clf}


def predict(model, ep):
    ep = np.array(ep)

    feats = []
    for ch in ep:
        feats.extend([
            np.max(ch),
            np.min(ch),
            np.ptp(ch),
            np.std(ch),
            np.sum(np.abs(ch))
        ])

    feats = np.array(feats).reshape(1, -1)
    feats = model["scaler"].transform(feats)

    p = model["clf"].predict_proba(feats)[0, 1]
    return int(p > 0.5), p


def evaluate(model, X, y):
    if len(X) == 0:
        print("no test data")
        return

    preds = []

    for ep in X:
        p, _ = predict(model, ep)
        preds.append(p)

    preds = np.array(preds)

    print("\nresults")
    print("total accuracy:", (preds == y).mean())

    blink_acc = ((preds == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
    noblink_acc = ((preds == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)

    print("blink accuracy:", blink_acc)
    print("no-blink accuracy:", noblink_acc)

    print("\nsample probabilities:")
    for i in range(min(10, len(X))):
        _, prob = predict(model, X[i])
        print(round(prob, 3))


if __name__ == "__main__":
    folder = sys.argv[1]

    runs = load_runs(folder)

    X_train, y_train = [], []
    X_test, y_test = [], []

    for r in TRAIN_RUNS:
        X_train.extend(runs[r][0])
        y_train.extend(runs[r][1])

    for r in TEST_RUNS:
        X_test.extend(runs[r][0])
        y_test.extend(runs[r][1])

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("training on:", TRAIN_RUNS)
    print("testing on:", TEST_RUNS)

    print("\nbefore balancing")
    print("blink:", (y_train == 1).sum())
    print("no-blink:", (y_train == 0).sum())

    X_train, y_train = balance(X_train, y_train)

    print("\nafter balancing")
    print("blink:", (y_train == 1).sum())
    print("no-blink:", (y_train == 0).sum())

    model = train_model(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    print("\nmodel saved")

    evaluate(model, X_test, y_test)
