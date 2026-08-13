"""The response surface: a logistic model in the six axes plus band structure.

Used twice: fitted on Part F alone to choose where Part G refines, then
refitted on Parts F and G combined to become the delivered selection function.

Design-matrix convention, fixed here so the two fits cannot drift:
  axes enter as z-scored log values where the axis is sampled in log, and as
  z-scored linear values otherwise,  all six main effects, all fifteen pairwise
  interactions, and band structure as a main effect with its own interactions.
"""
import numpy as np

AXES = ["n_cyc", "eta_x", "snr", "tau_over_p", "a2_over_a1",
        "samples_per_cycle"]
LOG_AXES = {"n_cyc", "snr", "tau_over_p", "samples_per_cycle"}
ALL = AXES + ["band_structure"]


def raw_matrix(df):
    """The seven columns, log-transformed where the axis is sampled in log."""
    cols = []
    for a in AXES:
        v = df[a].to_numpy(dtype=float)
        cols.append(np.log(np.maximum(v, 1e-12)) if a in LOG_AXES else v)
    cols.append(df["band_structure"].to_numpy(dtype=float))
    return np.column_stack(cols)


class Surface:
    """Logistic response surface with a documented predict interface."""

    def __init__(self, fap, outcome, background="DRW", cut_mode=None,
                 grid_choice=None, duty_cycle=None, snr_convention=None):
        self.fap = float(fap)
        self.outcome = outcome
        self.background = background
        self.cut_mode = cut_mode
        self.grid_choice = grid_choice
        self.duty_cycle = duty_cycle
        self.snr_convention = snr_convention
        self.names = None
        self.mu = None
        self.sd = None
        self.model = None
        self.n_train = 0

    def _expand(self, R):
        """z-score, then main effects + all pairwise interactions."""
        Z = (R - self.mu) / self.sd
        cols, names = [], []
        for i, a in enumerate(ALL):
            cols.append(Z[:, i])
            names.append(a)
        for i in range(len(ALL)):
            for j in range(i + 1, len(ALL)):
                cols.append(Z[:, i] * Z[:, j])
                names.append(f"{ALL[i]}:{ALL[j]}")
        self.names = names
        return np.column_stack(cols)

    def fit(self, df, y, C=1.0):
        from sklearn.linear_model import LogisticRegression
        R = raw_matrix(df)
        self.mu = R.mean(axis=0)
        self.sd = R.std(axis=0)
        self.sd[self.sd == 0] = 1.0
        X = self._expand(R)
        self.model = LogisticRegression(C=C, max_iter=5000, solver="lbfgs")
        self.model.fit(X, y)
        self.n_train = int(len(y))
        self._X_train, self._y_train = X, np.asarray(y)
        return self

    def predict_df(self, df):
        return self.model.predict_proba(self._expand(raw_matrix(df)))[:, 1]

    def predict(self, n_cyc, eta_x, snr, tau_over_p, a2_over_a1,
                samples_per_cycle, band_structure, fap=None):
        """P(chirp recovery) and its uncertainty.

        Parameters are the dimensionless axes of Section 8.1.  `band_structure`
        is 0 for one band and 1 for two bands with free phase.  `fap` must equal
        the false-alarm probability the surface was calibrated at, an efficiency
        without an operating point is not a number this object will return.

        Returns (probability, (lo, hi)) where the interval is the delta-method
        95% interval on the logit, propagated from the fitted coefficient
        covariance.
        """
        if fap is not None and not np.isclose(fap, self.fap):
            raise ValueError(
                f"this surface is calibrated at FAP {self.fap:g}, "
                f"{fap:g} was requested.  Refit at that operating point.")
        import pandas as pd
        df = pd.DataFrame({"n_cyc": np.atleast_1d(n_cyc),
                           "eta_x": np.atleast_1d(eta_x),
                           "snr": np.atleast_1d(snr),
                           "tau_over_p": np.atleast_1d(tau_over_p),
                           "a2_over_a1": np.atleast_1d(a2_over_a1),
                           "samples_per_cycle": np.atleast_1d(samples_per_cycle),
                           "band_structure": np.atleast_1d(band_structure)})
        X = self._expand(raw_matrix(df))
        eta = X @ self.model.coef_[0] + self.model.intercept_[0]
        p = 1.0 / (1.0 + np.exp(-eta))
        se = self._logit_se(X)
        lo = 1.0 / (1.0 + np.exp(-(eta - 1.96 * se)))
        hi = 1.0 / (1.0 + np.exp(-(eta + 1.96 * se)))
        if p.size == 1:
            return float(p[0]), (float(lo[0]), float(hi[0]))
        return p, (lo, hi)

    def _logit_se(self, X):
        """Delta-method standard error on the logit, from the observed
        information of the fitted logistic model."""
        if not hasattr(self, "_cov"):
            Xt = np.column_stack([self._X_train, np.ones(len(self._X_train))])
            p = self.model.predict_proba(self._X_train)[:, 1]
            W = p * (1 - p)
            F = Xt.T @ (Xt * W[:, None])
            self._cov = np.linalg.pinv(F + 1e-9 * np.eye(F.shape[0]))
        Xt = np.column_stack([X, np.ones(len(X))])
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Xt, self._cov, Xt), 0))

    def coefficients(self):
        """Coefficients with standard errors, on the z-scored scale."""
        self._logit_se(self._X_train[:1])
        se = np.sqrt(np.diag(self._cov))
        out = []
        for k, nm in enumerate(self.names):
            out.append(dict(term=nm, coef=float(self.model.coef_[0][k]),
                            se=float(se[k]),
                            z=float(self.model.coef_[0][k] / se[k])
                            if se[k] > 0 else np.nan))
        out.append(dict(term="(intercept)", coef=float(self.model.intercept_[0]),
                        se=float(se[-1]), z=np.nan))
        return out

    def deviance_ranking(self, df, y):
        """Axis ranking by deviance explained: refit without each axis (its main
        effect and every interaction it appears in) and report the increase in
        deviance.  That is the axis's unique contribution, not its marginal one.
        """
        from sklearn.linear_model import LogisticRegression
        R = raw_matrix(df)
        X = self._expand(R)
        y = np.asarray(y)

        def dev(Xs):
            m = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
            m.fit(Xs, y)
            p = np.clip(m.predict_proba(Xs)[:, 1], 1e-12, 1 - 1e-12)
            return float(-2 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

        p0 = np.clip(np.full(len(y), y.mean()), 1e-12, 1 - 1e-12)
        dev_null = float(-2 * np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0)))
        dev_full = dev(X)
        rows = []
        for a in ALL:
            keep = [k for k, nm in enumerate(self.names)
                    if nm != a and a not in nm.split(":")]
            d = dev(X[:, keep]) if keep else dev_null
            rows.append(dict(axis=a, deviance_increase=d - dev_full,
                             frac_of_explained=(d - dev_full)
                             / max(dev_null - dev_full, 1e-12)))
        rows.sort(key=lambda r: -r["deviance_increase"])
        return dict(deviance_null=dev_null, deviance_full=dev_full,
                    deviance_explained=dev_null - dev_full,
                    pseudo_r2=1.0 - dev_full / dev_null, ranking=rows)


def calibration(y_true, p_pred, n_bins=10):
    """Observed against predicted in equal-width probability bins, plus Brier."""
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p_pred, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            rows.append(dict(bin=b, lo=edges[b], hi=edges[b + 1], n=0,
                             mean_predicted=np.nan, observed=np.nan,
                             observed_lo=np.nan, observed_hi=np.nan))
            continue
        obs = float(y_true[m].mean())
        se = float(np.sqrt(max(obs * (1 - obs), 1e-12) / n))
        rows.append(dict(bin=b, lo=float(edges[b]), hi=float(edges[b + 1]), n=n,
                         mean_predicted=float(p_pred[m].mean()), observed=obs,
                         observed_lo=max(0.0, obs - 1.96 * se),
                         observed_hi=min(1.0, obs + 1.96 * se)))
    brier = float(np.mean((p_pred - y_true) ** 2))
    ok = [r for r in rows if r["n"] > 0]
    max_dev = max((abs(r["observed"] - r["mean_predicted"]) for r in ok),
                  default=np.nan)
    n_out = sum(1 for r in ok
                if not (r["observed_lo"] <= r["mean_predicted"] <= r["observed_hi"]))
    return dict(bins=rows, brier=brier, n_bins_occupied=len(ok),
                max_abs_deviation=max_dev, n_bins_predicted_outside_ci=n_out)
