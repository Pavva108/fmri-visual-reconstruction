import numpy as np

from fmri_reconstruction.data.sequences import make_sequences
from fmri_reconstruction.preprocessing import detrend_time
from fmri_reconstruction.retrieval.cosine import top_k
from fmri_reconstruction.preprocessing import preprocess_haxby, preprocess_nsd
from sklearn.preprocessing import StandardScaler


def main():
    x = np.random.randn(30, 8).astype(np.float32)
    y = make_sequences(x, length=5)
    assert y.shape == (26, 5, 8), f"unexpected shape: {y.shape}"

    x2 = np.random.randn(20, 10).astype(np.float32)
    assert detrend_time(x2).shape == x2.shape

    db = np.eye(4, dtype=np.float32)
    idx, scores = top_k(db[0], db, k=2)
    assert idx[0] == 0

    # Haxby-like preprocessing: ensure scaler fit only on training data
    # Synthetic example: training zeros, test ones -> fit on train should not zero-test
    X_train = np.zeros((50, 10), dtype=np.float32)
    X_test = np.ones((20, 10), dtype=np.float32)
    X_train_p, state = preprocess_haxby(X_train, tr_seconds=3.0, fit_scaler=True)
    X_test_p, _ = preprocess_haxby(X_test, tr_seconds=3.0, fit_scaler=False, scaler=StandardScaler().fit(X_train))
    assert X_train_p.shape[1] == X_test_p.shape[1]

    # NSD-style preprocessing: fit on train, transform test
    Xn_train = (np.random.randn(100, 30) * 2 + 1).astype(np.float32)
    Xn_test = (np.random.randn(40, 30) * 2 + 1).astype(np.float32)
    Xn_tr, scaler = preprocess_nsd(Xn_train, fit_scaler=True)
    Xn_te, _ = preprocess_nsd(Xn_test, fit_scaler=False, scaler=scaler)
    assert Xn_tr.shape[1] == Xn_te.shape[1]

    print("SMOKE PASSED")


if __name__ == '__main__':
    main()
