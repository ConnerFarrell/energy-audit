"""Fit the Arousal Index ridge and save its coefficients to data/.

Ported from the scoring notebook (notebooks/scoring.ipynb cells 0 and 3,
identical to the canonical Kaggle run): DEAM song-level static arousal,
mapped 1-9 -> 0-11; decorrelated CORE features; StandardScaler + Ridge(1.0)
fit on all DEAM songs. Only the scaler + ridge are saved: they are all that
the Tier-1 statistics need (pre-calibration "raw ridge" score =
standardized CORE features @ coef + intercept). The isotonic calibrator is
not needed for any banked statistic and is not saved.

Inputs (the two Kaggle datasets the scoring notebook attaches; no audio):
  <DEAM_DIR>/DEAM_Annotations/annotations/annotations averaged per song/
      song_level/static_annotations_averaged_songs_1_2000.csv
      song_level/static_annotations_averaged_songs_2000_2058.csv
  <FEATURES_CSV>  deam_librosa_features.csv (pre-extracted; recipe in
                  notebooks/scoring.ipynb `extract_six`)

Usage:
  python scripts/fit_arousal_ridge.py --deam-dir DIR --features CSV \
      [--out data/arousal_index_ridge.json]
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score

CORE = ['zcr_mean', 'zcr_std', 'loudness_mean', 'loudness_std', 'hnr_mean', 'hnr_std']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deam-dir', required=True)
    ap.add_argument('--features', required=True)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'data',
                                                  'arousal_index_ridge.json'))
    a = ap.parse_args()

    ann = os.path.join(a.deam_dir, 'DEAM_Annotations', 'annotations',
                       'annotations averaged per song', 'song_level')
    d1 = pd.read_csv(os.path.join(ann, 'static_annotations_averaged_songs_1_2000.csv'))
    d2 = pd.read_csv(os.path.join(ann, 'static_annotations_averaged_songs_2000_2058.csv'))
    d1.columns = d1.columns.str.strip(); d2.columns = d2.columns.str.strip()
    shared = [c for c in d1.columns if c in d2.columns]
    deam = pd.concat([d1[shared], d2[shared]], ignore_index=True)
    deam['target'] = (deam['arousal_mean'] - 1.0) / (9.0 - 1.0) * 11.0

    deam_feat = pd.read_csv(a.features)
    deam_m = deam_feat.merge(deam[['song_id', 'target']], on='song_id')
    assert not any('arousal' in c or 'target' in c for c in CORE), 'LEAK!'

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pred = cross_val_predict(model, deam_m[CORE].values, deam_m['target'].values, cv=cv)
    r2 = r2_score(deam_m['target'].values, pred)
    model.fit(deam_m[CORE].values, deam_m['target'].values)
    sc, rg = model.named_steps['standardscaler'], model.named_steps['ridge']

    out = {
        'description': 'Arousal Index ridge (StandardScaler + Ridge alpha=1.0) fit on all DEAM songs; '
                       'raw score = ((x[features] - scaler_mean) / scaler_scale) @ coef + intercept',
        'features': CORE,
        'scaler_mean': [float(v) for v in sc.mean_],
        'scaler_scale': [float(v) for v in sc.scale_],
        'coef': [float(v) for v in rg.coef_],
        'intercept': float(rg.intercept_),
        'alpha': 1.0,
        'n_train_songs': int(len(deam_m)),
        'deam_5fold_r2_oof': float(r2),
        'fitted_with': {'python': sys.version.split()[0], 'numpy': np.__version__,
                        'pandas': pd.__version__, 'scikit-learn': sklearn.__version__},
        'inputs': ['DEAM static_annotations_averaged_songs_1_2000.csv',
                   'DEAM static_annotations_averaged_songs_2000_2058.csv',
                   'deam_librosa_features.csv'],
    }
    with open(a.out, 'w') as fh:
        json.dump(out, fh, indent=2)   # json floats are repr -> exact round-trip
        fh.write('\n')
    print(f'n={len(deam_m)}  DEAM 5-fold R2 (OOF) = {r2:.4f}  -> wrote {a.out}')

if __name__ == '__main__':
    main()
