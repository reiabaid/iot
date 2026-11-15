"""Trains and compares the models for SHIELD-X Glove and exports a small decision tree for the dashboard.

Usage:
    python tools/train_models.py                       # every CSV in data/ (the simulated sample first)
    python tools/train_models.py data/a.csv data/b.csv # chosen CSVs, in that order
    python tools/train_models.py data/my_live.csv --live-only   # keep only rows whose source is "live"

With more than one CSV, the LAST file is the test set and the others are used for training.

Two tasks:
    gas       clean / smoke / alcohol / combustible from the MQ-2, MQ-7, temperature and humidity
    activity  fall / walk / run / sit / wave / jump from 3 second windows of the MPU6050

Split: the last CSV (or the last session when there is only one) is the test set, so a test run
never comes from data used for training. Models: decision tree, random forest, SVM, kNN and a
threshold rule as the baseline. The decision trees are written to dashboard/models.js.
"""
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
args = [a for a in sys.argv[1:] if not a.startswith('--')]
data_dir = os.path.join(ROOT, 'data')
if args: files = args
else:
    others = sorted(f for f in os.listdir(data_dir) if f.endswith('.csv') and f != 'simulated_sample.csv')
    files = [os.path.join(data_dir, 'simulated_sample.csv')] + [os.path.join(data_dir, f) for f in others]
parts = []
for i, f in enumerate(files):
    part = pd.read_csv(f); part['file'] = i; part['session'] = i * 100 + part['session']; parts.append(part)
    print(f'  {os.path.basename(f)}: {len(part)} rows')
df = pd.concat(parts, ignore_index=True)
if '--live-only' in sys.argv:
    df = df[df['source'] == 'live']
SOURCE = ', '.join(sorted(df['source'].unique()))
print(f'Loaded {len(df)} rows (source: {SOURCE})')
if len(files) > 1: TEST_FILE = len(files) - 1

GAS = ['clean', 'smoke', 'alcohol', 'combustible']
ACT = ['fall', 'walk', 'run', 'sit', 'wave', 'jump']
WIN = 6                                              # 6 packets = 3 seconds at 2 packets per second


def gas_features(d):
    q = (d['mq2_ratio'] - 1) / np.maximum(d['mq7_ratio'] - 1, 0.05)
    return np.column_stack([d['mq2_ratio'], d['mq7_ratio'], q, d['temp_c'], d['humidity_pct']])


GAS_FEATS = ['mq2_ratio', 'mq7_ratio', 'mq2_over_mq7_excess', 'temp_c', 'humidity_pct']
ACT_FEATS = ['acc_mean', 'acc_std', 'acc_min', 'acc_max', 'pitch_std', 'roll_std']


def window_features(d):
    X, y, sess, fil = [], [], [], []
    d = d.copy()
    d['label'] = d['label'].replace({'clean': 'sit', 'high_hr': 'sit', 'panic': 'sit', 'weak_link': 'sit'})   # resting rows
    d = d[d['label'].isin(ACT)].reset_index(drop=True)
    grp = (d['label'] != d['label'].shift()) | (d['session'] != d['session'].shift())
    for _, g in d.groupby(grp.cumsum()):
        a, p, r = g['accel_g'].to_numpy(), g['pitch_deg'].to_numpy(), g['roll_deg'].to_numpy()
        for i in range(0, len(g) - WIN + 1, 2):
            s = slice(i, i + WIN)
            if g['label'].iloc[0] == 'fall' and a[s].max() < 2.0: continue      # lying still after the impact is not a fall window
            X.append([a[s].mean(), a[s].std(), a[s].min(), a[s].max(), p[s].std(), r[s].std()])
            y.append(g['label'].iloc[0]); sess.append(g['session'].iloc[0]); fil.append(g['file'].iloc[0])
    return np.array(X), np.array(y), np.array(sess), np.array(fil)


def threshold_gas(X):                                # baseline: the rule the dashboard used before
    out = []
    for a, b, q, _, _ in X:
        if max(a - 1, b - 1) < 0.4: out.append('clean')
        else: out.append('smoke' if q > 3 else ('alcohol' if q < 0.8 else 'combustible'))
    return np.array(out)


def threshold_act(X):                                # baseline: a fall is only "acc above 2.5 g"
    return np.array(['fall' if r[3] > 2.5 else ('run' if r[1] > 0.4 else ('walk' if r[1] > 0.15 else 'sit')) for r in X])


def models():
    return {
        'Decision tree': DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=0),
        'Random forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=0),
        'SVM (RBF)': make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight='balanced')),
        'kNN (k=5)': make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    }


def evaluate(name, task, X_tr, y_tr, X_te, y_te, labels, baseline):
    rows, cms, trees = [], {}, None
    def record(model_name, pred, ms):
        p, r, f, _ = precision_recall_fscore_support(y_te, pred, labels=labels, average='macro', zero_division=0)
        rows.append({'task': task, 'model': model_name, 'accuracy': accuracy_score(y_te, pred), 'precision': p, 'recall': r, 'f1': f, 'ms_per_row': ms})
        cms[model_name] = pd.DataFrame(confusion_matrix(y_te, pred, labels=labels), index=labels, columns=labels)
    t = time.perf_counter(); pred = baseline(X_te); record('Threshold rule', pred, (time.perf_counter() - t) / len(X_te) * 1000)
    fitted = {}
    for m_name, m in models().items():
        m.fit(X_tr, y_tr); t = time.perf_counter(); pred = m.predict(X_te)
        record(m_name, pred, (time.perf_counter() - t) / len(X_te) * 1000); fitted[m_name] = m
    return rows, cms, fitted


def export_tree(tree, classes, feats):
    t = tree.tree_
    nodes = []
    for i in range(t.node_count):
        if t.children_left[i] == -1: nodes.append({'c': int(np.argmax(t.value[i]))})
        else: nodes.append({'f': int(t.feature[i]), 't': round(float(t.threshold[i]), 5), 'l': int(t.children_left[i]), 'r': int(t.children_right[i])})
    return {'classes': [str(c) for c in tree.classes_], 'features': feats, 'nodes': nodes}


results, cms, export = [], {}, {}
last = df['session'].max()
split_by_file = len(files) > 1

# ---- gas ----
g = df[df['label'].isin(GAS)].copy()
# below 0.4 above clean air the gas is too weak to see, so those rows count as clean air (the same alert level as the dashboard)
excess = np.maximum(g['mq2_ratio'] - 1, g['mq7_ratio'] - 1)
g.loc[(g['label'] != 'clean') & (excess < 0.4), 'label'] = 'clean'
Xg, yg = gas_features(g), g['label'].to_numpy()
te = ((g['file'] == TEST_FILE) if split_by_file else (g['session'] == last)).to_numpy()
rows, cm, fit = evaluate('gas', 'gas', Xg[~te], yg[~te], Xg[te], yg[te], GAS, threshold_gas)
results += rows; cms.update({('gas', k): v for k, v in cm.items()})
export['gas'] = export_tree(fit['Decision tree'], GAS, GAS_FEATS)
print(f'gas: train {int((~te).sum())} rows, test {int(te.sum())} rows (' + ('last file' if split_by_file else f'session {last}') + ')')

# ---- activity ----
Xa, ya, sa, fa = window_features(df)
te = (fa == TEST_FILE) if split_by_file else (sa == last)
rows, cm, fit = evaluate('activity', 'activity', Xa[~te], ya[~te], Xa[te], ya[te], ACT, threshold_act)
results += rows; cms.update({('activity', k): v for k, v in cm.items()})
export['activity'] = export_tree(fit['Decision tree'], ACT, ACT_FEATS)
print(f'activity: train {int((~te).sum())} windows, test {int(te.sum())} windows (' + ('last file' if split_by_file else f'session {last}') + ')')

# ---- outputs ----
out_dir = os.path.join(ROOT, 'results'); os.makedirs(out_dir, exist_ok=True)
res = pd.DataFrame(results); res.to_csv(os.path.join(out_dir, 'model_results.csv'), index=False, float_format='%.4f')
for (task, name), m in cms.items():
    m.to_csv(os.path.join(out_dir, f'confusion_{task}_{name.split()[0].lower()}.csv'))
pd.set_option('display.width', 140)
print('\n' + res.round(3).to_string(index=False))
export['meta'] = {'trainedOn': SOURCE, 'rows': int(len(df)), 'window': WIN}
js = os.path.join(ROOT, 'dashboard', 'models.js')
with open(js, 'w') as f:
    f.write('// Written by tools/train_models.py. Decision trees used by the dashboard for prediction.\n')
    f.write('window.SHIELD_MODELS = ' + json.dumps(export, separators=(',', ':')) + ';\n')
print(f'\nSaved results to {out_dir} and the dashboard model to {js}')
if 'simulation' in SOURCE:
    print('NOTE: trained on SIMULATED data. Scores are not valid results for the report.')
