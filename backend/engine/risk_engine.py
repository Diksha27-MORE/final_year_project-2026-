import os
from pathlib import Path

from services.text_analyzer import TextAnalyzer
from services.email_checker import EmailChecker
from services.url_checker import URLChecker
from services.urgency_detector import UrgencyDetector


class RiskEngine:
    """
    Fraud detection engine that combines several independent signals into
    one 0-100 risk score and a matching Low / Medium / High level.

    Scoring (see `_combine_signals`):
    - Each detector returns a 0-100 score.
    - Scores are combined with a "noisy-OR" rule instead of a weighted
      average. A weighted average dilutes strong evidence: a posting with
      a fee request, urgency, a suspicious site and a mismatched email can
      still average out to ~35. With noisy-OR, every additional independent
      red flag raises the risk, a single weak flag stays low, and nothing
      is force-set to HIGH - the score is always a smooth function of the
      evidence.
    - The logistic-regression probability is one more signal in that same
      combination (it no longer overrides everything via max()).

    Levels (integer score, single source of truth):
        0-30 Low, 31-60 Medium, 61-100 High
    """

    DATASET_FILENAME = 'Fake_Real_Job_Posting_Cleaned.csv'

    # Reporting weights (kept for API/UI compatibility; they describe the
    # relative importance of each signal, sum = 1.0).
    WEIGHTS = {
        'text': 0.40,
        'email': 0.20,
        'url': 0.20,
        'urgency': 0.10,
        'company': 0.10
    }

    # Noisy-OR strength per signal: the largest share of "risk probability"
    # a single signal can contribute when its own score is 100. Values < 1
    # mean no single signal can push the result to 100 on its own.
    SIGNAL_STRENGTH = {
        'text': 0.60,
        'email': 0.50,
        'url': 0.50,
        'urgency': 0.60,
        'company': 0.25,
        'model': 0.50,
    }

    # Risk level thresholds on the INTEGER risk score (inclusive bounds).
    LOW_MAX = 30       # 0-30   -> Low
    MEDIUM_MAX = 60    # 31-60  -> Medium, 61-100 -> High
    RISK_LEVELS = {
        'low': (0, LOW_MAX),
        'medium': (LOW_MAX + 1, MEDIUM_MAX),
        'high': (MEDIUM_MAX + 1, 100)
    }

    # Human-readable labels for the urgency detector's pattern categories
    URGENCY_REASON_LABELS = {
        'registration_fee': 'Payment or registration fee requested',
        'immediate_joining': 'Pressure for immediate joining detected',
        'limited_seats': 'Urgent hiring / limited seats tactic detected',
        'salary_pressure': 'Unrealistic salary or "easy money" promises detected',
        'too_good': 'Guaranteed placement / no-interview claim detected',
    }

    # The ML probability only influences the score if it beats this ROC-AUC
    # on a held-out split (it is still reported as fraud_probability either way).
    MIN_MODEL_AUC = 0.65
    MAX_TRAINING_SAMPLES = 800

    def __init__(self):
        """Initialize all detection services and train the logistic model."""
        self.text_analyzer = TextAnalyzer()
        self.email_checker = EmailChecker()
        self.url_checker = URLChecker()
        self.urgency_detector = UrgencyDetector()

        self.model = None            # trained pipeline (None = unavailable)
        self.model_reliable = False  # True if held-out AUC >= MIN_MODEL_AUC
        self.model_metrics = None
        self.model_error = None      # why the model is unavailable, if it is
        self._train_logistic_model()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def analyze(self, title: str, description: str, email: str, website: str, company_name: str = "") -> dict:
        """
        Comprehensive fraud detection analysis.

        Returns a dict with (among others):
            risk_score / final_risk_score : int 0-100 (identical)
            risk_level                    : 'Low' | 'Medium' | 'High'
                                            (always derived from risk_score)
            fraud_probability             : float 0-1 (only if model active)
            signals, reasons, recommendation, prediction, is_safe
        """
        title = title or ''
        description = description or ''

        text_score = self._analyze_text(title, description)
        email_score = self._analyze_email(email, company_name)
        url_score = self._analyze_url(website)
        urgency_score = self._analyze_urgency(title, description)
        company_score = self._analyze_company(company_name)

        signal_scores = {
            'text': text_score,
            'email': email_score,
            'url': url_score,
            'urgency': urgency_score,
            'company': company_score,
        }

        # Heuristic-only score (no ML), kept for transparency.
        heuristic_score = self._combine_signals(signal_scores)

        # Old weighted average, reported only for comparison/debugging.
        weighted_average = sum(
            signal_scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS
        )

        reasons = self._collect_reasons(
            email=email,
            website=website,
            description=description,
            title=title,
            company_name=company_name,
            text_score=text_score
        )

        # ML probability. It is reported whenever a trained model exists
        # (model_available), independent of whether it is trusted enough to
        # influence the score (model_used_in_score).
        # Features come from the SAME function used for training.
        prob_fraud = None
        model_signal = None
        if self.model is not None:
            try:
                features = self._extract_features(title, description)
                probabilities = self.model.predict_proba([features])[0]
                fraud_idx = list(self.model.classes_).index(1)
                prob_fraud = float(probabilities[fraud_idx])
                if self.model_reliable:
                    # Only probability ABOVE the neutral 0.5 counts as risk
                    # evidence, so an uncertain model (~0.5) adds nothing
                    # to otherwise clean postings.
                    model_signal = max(0.0, (prob_fraud - 0.5) / 0.5) * 100.0
            except Exception as e:
                self.model_error = f"Prediction failed: {e}"
                print(f"Error computing logistic model probability: {e}")
                prob_fraud = None
                model_signal = None

        final_scores = dict(signal_scores)
        if model_signal is not None:
            final_scores['model'] = model_signal

        # ONE final score, ONE integer, ONE level derived from it.
        risk_score = self._to_int_score(self._combine_signals(final_scores))
        risk_level = self._get_risk_level(risk_score)

        result = {
            'final_risk_score': risk_score,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'confidence': round(max(risk_score, 100 - risk_score), 2),
            'heuristic_risk_score': self._to_int_score(heuristic_score),
            'weighted_average_score': round(weighted_average, 2),
            'model_confidence': None,
            'reasons': reasons,
            'signals': {
                'text': {
                    'score': round(text_score, 2),
                    'weight': self.WEIGHTS['text'],
                    'description': 'Similarity to known scam patterns'
                },
                'email': {
                    'score': round(email_score, 2),
                    'weight': self.WEIGHTS['email'],
                    'description': 'Email address risk assessment'
                },
                'url': {
                    'score': round(url_score, 2),
                    'weight': self.WEIGHTS['url'],
                    'description': 'Website URL and domain reputation'
                },
                'urgency': {
                    'score': round(urgency_score, 2),
                    'weight': self.WEIGHTS['urgency'],
                    'description': 'Pressure and urgency tactics detected'
                },
                'company': {
                    'score': round(company_score, 2),
                    'weight': self.WEIGHTS['company'],
                    'description': 'Company validation and legitimacy'
                }
            }
        }

        result['model_available'] = prob_fraud is not None
        result['model_used_in_score'] = model_signal is not None
        result['model_error'] = None if prob_fraud is not None else (
            self.model_error or 'Model is not trained')
        result['fraud_probability'] = None

        if prob_fraud is not None:
            result['fraud_probability'] = round(prob_fraud, 4)
            result['safe_confidence'] = round(1.0 - prob_fraud, 4)
            result['model_risk_score'] = round(prob_fraud * 100, 2)
            result['model_confidence'] = round(max(prob_fraud, 1.0 - prob_fraud) * 100, 2)

        # Everything below is derived from risk_level / risk_score only.
        result['is_safe'] = risk_level == 'Low'
        result['prediction'] = self._prediction_label(risk_level)
        result['recommendation'] = self._get_recommendation(risk_level)

        return result

    # ------------------------------------------------------------------
    # Score combination and levels
    # ------------------------------------------------------------------
    def _combine_signals(self, scores: dict) -> float:
        """
        Noisy-OR combination of 0-100 signal scores -> 0-100 risk.

            risk = 1 - prod(1 - strength_i * score_i / 100)

        - No evidence (all scores 0) -> 0.
        - One weak signal -> stays low.
        - Several strong, independent signals -> high, without any hard
          override or forced level.
        """
        survive = 1.0
        for name, score in scores.items():
            strength = self.SIGNAL_STRENGTH.get(name, 0.0)
            s = min(max(float(score), 0.0), 100.0) / 100.0
            survive *= (1.0 - strength * s)
        return (1.0 - survive) * 100.0

    @staticmethod
    def _to_int_score(score: float) -> int:
        """Clamp to 0-100 and round to an integer (the displayed score)."""
        return int(round(min(max(score, 0.0), 100.0)))

    def _get_risk_level(self, score: float) -> str:
        """
        Map the (integer) risk score to a level. Bounds are inclusive and
        match RISK_LEVELS: 0-30 Low, 31-60 Medium, 61-100 High.
        """
        score = self._to_int_score(score)
        if score <= self.LOW_MAX:
            return 'Low'
        if score <= self.MEDIUM_MAX:
            return 'Medium'
        return 'High'

    def _prediction_label(self, risk_level: str) -> str:
        """Human-readable verdict shown to the student."""
        return {
            'Low': 'Trustworthy',
            'Medium': 'Proceed with Caution',
            'High': 'Not Trustworthy',
        }.get(risk_level, 'Unknown')

    def _get_recommendation(self, risk_level: str) -> str:
        """Plain-language recommendation matched to the risk level."""
        return {
            'Low': 'This internship looks legitimate based on our checks. You may proceed, but always stay alert.',
            'Medium': 'Some suspicious signs were detected. Proceed with caution and verify the company independently before sharing personal details or paying any fee.',
            'High': 'Do not proceed. This posting shows strong indicators of being fraudulent.',
        }.get(risk_level, 'Review the details carefully before proceeding.')

    # ------------------------------------------------------------------
    # Reasons
    # ------------------------------------------------------------------
    def _collect_reasons(self, email: str, website: str, description: str,
                          title: str, company_name: str, text_score: float) -> list:
        """
        Turn the existing detectors' own internal signals into a short,
        human-readable list of reasons.
        """
        reasons = []

        try:
            if email:
                if self.email_checker._is_disposable_domain(email):
                    reasons.append('Disposable or temporary email domain detected')
                if self.email_checker._has_random_username(email):
                    reasons.append('Suspicious, randomly generated-looking email address')
                if company_name and self.email_checker._is_domain_mismatch(email, company_name):
                    reasons.append('Email domain does not match the company name')
        except Exception:
            pass

        try:
            if website:
                normalized = self.url_checker._normalize_url(website)
                if normalized:
                    if self.url_checker._has_suspicious_tld(normalized):
                        reasons.append('Suspicious website domain extension')
                    if not self.url_checker._is_reachable(normalized):
                        reasons.append('Website could not be verified or reached')
                    if self.url_checker._looks_temporary(normalized):
                        reasons.append('Website looks like a temporary or throwaway site')
                else:
                    reasons.append('Website/URL could not be parsed')
        except Exception:
            pass

        try:
            combined_text = f"{title} {description}"
            for pattern in self.urgency_detector.get_detected_patterns(combined_text):
                label = self.URGENCY_REASON_LABELS.get(pattern)
                if label:
                    reasons.append(label)
        except Exception:
            pass

        try:
            if text_score >= 50:
                reasons.append('Message closely matches known scam job-posting patterns')
        except Exception:
            pass

        try:
            if not company_name or len(company_name.strip()) < 2:
                reasons.append('Company name is missing or could not be validated')
        except Exception:
            pass

        return reasons

    # ------------------------------------------------------------------
    # Individual signal wrappers
    # ------------------------------------------------------------------
    def _analyze_text(self, title: str, description: str) -> float:
        """Analyze text using sentence embeddings."""
        try:
            combined_text = f"{title} {description}"
            return float(self.text_analyzer.analyze(combined_text))
        except Exception as e:
            print(f"Error in text analysis: {e}")
            return 0.0

    def _analyze_email(self, email: str, company_name: str = "") -> float:
        """Analyze email for suspicious patterns."""
        try:
            return float(self.email_checker.analyze(email, company_name))
        except Exception as e:
            print(f"Error in email analysis: {e}")
            return 0.0

    def _analyze_url(self, website: str) -> float:
        """Analyze URL for suspicious patterns."""
        try:
            return float(self.url_checker.analyze(website))
        except Exception as e:
            print(f"Error in URL analysis: {e}")
            return 0.0

    def _analyze_urgency(self, title: str, description: str = "") -> float:
        """
        Detect urgency and pressure tactics on title + description (the
        same text used for the reasons list and for the ML features).
        """
        try:
            combined_text = f"{title or ''} {description or ''}".strip()
            return float(self.urgency_detector.analyze(combined_text))
        except Exception as e:
            print(f"Error in urgency detection: {e}")
            return 0.0

    def _analyze_company(self, company_name: str) -> float:
        """
        Validate company legitimacy (placeholder for a future API).
        """
        try:
            if not company_name or len(company_name.strip()) == 0:
                return 40.0
            if len(company_name.strip()) < 2:
                return 50.0
            return 0.0
        except Exception as e:
            print(f"Error in company validation: {e}")
            return 20.0

    # ------------------------------------------------------------------
    # Logistic regression
    # ------------------------------------------------------------------
    def _extract_features(self, title: str, text: str) -> list:
        """
        Feature vector for the logistic regression model. Used for BOTH
        training and prediction, so the two are always identical:
            [text_similarity_score, urgency_score]

        Text length is deliberately NOT a feature. In this dataset fake
        rows are short (~160 chars on average) and real rows are long
        (~540), so length would only teach the model "short = scam,
        long = real", which fails on real-world scam posts of any length.
        """
        combined_text = f"{title or ''} {text or ''}".strip()

        try:
            text_score = float(self.text_analyzer.analyze(combined_text))
        except Exception:
            text_score = 0.0

        try:
            urgency_score = float(self.urgency_detector.analyze(combined_text))
        except Exception:
            urgency_score = 0.0

        return [text_score, urgency_score]

    # ------------------------------------------------------------------
    # Dataset loading (shared by training AND evaluation)
    # ------------------------------------------------------------------
    @classmethod
    def find_dataset_path(cls):
        """
        Locate backend/data/Fake_Real_Job_Posting_Cleaned.csv regardless of
        whether this file sits in backend/ or backend/services/ (or the
        server is started from another working directory).
        Returns (path_or_None, list_of_locations_checked).
        An explicit path can be given via RISK_ENGINE_DATASET.
        """
        checked = []
        override = os.environ.get('RISK_ENGINE_DATASET')
        if override:
            checked.append(Path(override))

        here = Path(__file__).resolve().parent
        for base in [here, *here.parents, Path.cwd(), *Path.cwd().parents]:
            checked.append(base / 'backend' / 'data' / cls.DATASET_FILENAME)
            checked.append(base / 'data' / cls.DATASET_FILENAME)

        seen = []
        for path in checked:
            if path not in seen:
                seen.append(path)
        for path in seen:
            if path.exists():
                return path, seen
        return None, seen

    @staticmethod
    def _label_to_target(value):
        """Map the CSV 'fraudulent' label to 1 (fake) / 0 (real) / None."""
        v = str(value).strip().lower()
        if v in {'real', '0', 'false', 'legitimate'}:
            return 0
        if v in {'fake', '1', 'true', 'fraud', 'fraudulent', 'scam'}:
            return 1
        return None

    @classmethod
    def load_dataset_rows(cls, csv_path=None):
        """
        Read and clean the dataset. Returns (rows, stats) where rows is a
        list of (title, text, target) with target 1 = fake, 0 = real, in a
        DETERMINISTIC SHUFFLED order.

        The shuffle matters: the CSV is sorted (rows 0-99 are 93 Real / 7
        Fake, rows 100-182 are all Fake), so taking the first N rows or an
        unshuffled split gives a badly skewed sample.
        """
        import csv
        import random

        if csv_path is None:
            csv_path, checked = cls.find_dataset_path()
            if csv_path is None:
                raise FileNotFoundError(
                    f"{cls.DATASET_FILENAME} not found. Expected at "
                    f"backend/data/{cls.DATASET_FILENAME}. Checked: "
                    f"{[str(c) for c in checked[:6]]} ..."
                )

        rows, seen = [], set()
        stats = {'path': str(csv_path), 'raw_rows': 0, 'unknown_label': 0,
                 'empty_text': 0, 'duplicates': 0}
        with open(csv_path, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or 'fraudulent' not in reader.fieldnames:
                raise ValueError(
                    f"Label column 'fraudulent' missing; columns: {reader.fieldnames}")
            for row in reader:
                stats['raw_rows'] += 1
                target = cls._label_to_target(row.get('fraudulent', ''))
                if target is None:
                    stats['unknown_label'] += 1
                    continue
                title_val = (row.get('title') or '').strip()
                # This CSV has no 'description' column, only 'requirements'.
                text_val = (row.get('description') or row.get('requirements') or '').strip()
                # Placeholder text carries no signal.
                if text_val.lower() in {'', 'not mentioned', 'nan', 'none'}:
                    stats['empty_text'] += 1
                    continue
                key = (title_val.lower(), text_val.lower(), target)
                if key in seen:
                    stats['duplicates'] += 1
                    continue
                seen.add(key)
                rows.append((title_val, text_val, target))

        random.Random(42).shuffle(rows)
        stats['usable_rows'] = len(rows)
        stats['fake'] = sum(1 for r in rows if r[2] == 1)
        stats['real'] = stats['usable_rows'] - stats['fake']
        return rows, stats

    # ------------------------------------------------------------------
    # Logistic regression
    # ------------------------------------------------------------------
    @staticmethod
    def build_model_pipeline():
        """The one definition of the model, used by training and evaluation."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight='balanced')
        )

    def get_model_status(self) -> dict:
        """Diagnostic info; safe to expose from a /health or /model-status route."""
        return {
            'model_available': self.model is not None,
            'model_used_in_score': self.model is not None and self.model_reliable,
            'model_error': self.model_error,
            'holdout_metrics': self.model_metrics,
        }

    def reload_model(self) -> dict:
        """Retrain (e.g. after fixing the dataset path) without restarting."""
        self._train_logistic_model()
        return self.get_model_status()

    def _train_logistic_model(self):
        """
        Train logistic regression on backend/data/Fake_Real_Job_Posting_Cleaned.csv.

        self.model is set whenever training succeeds, so `fraud_probability`
        is always returned. Whether that probability is also allowed to
        influence the score is decided separately by `model_reliable`
        (held-out AUC >= MIN_MODEL_AUC). Any failure is stored in
        self.model_error (and printed) instead of being silently swallowed.
        """
        self.model = None
        self.model_reliable = False
        self.model_metrics = None
        self.model_error = None
        try:
            from sklearn.metrics import accuracy_score, roc_auc_score
            from sklearn.model_selection import train_test_split
        except Exception as e:
            self.model_error = f"scikit-learn is not installed/importable: {e}"
            print(f"Logistic model unavailable: {self.model_error}")
            return

        try:
            rows, stats = self.load_dataset_rows()
            print(f"Dataset {stats['path']}: raw={stats['raw_rows']}, "
                  f"usable={stats['usable_rows']} (fake={stats['fake']}, "
                  f"real={stats['real']}), skipped: "
                  f"unknown_label={stats['unknown_label']}, "
                  f"empty_text={stats['empty_text']}, "
                  f"duplicates={stats['duplicates']}")

            if stats['usable_rows'] < 20 or min(stats['fake'], stats['real']) < 5:
                raise ValueError(
                    f"Not enough data / class diversity: fake={stats['fake']}, "
                    f"real={stats['real']}")

            # Balanced, capped sample (rows are already shuffled).
            cap = max(1, self.MAX_TRAINING_SAMPLES // 2)
            fake_rows = [r for r in rows if r[2] == 1][:cap]
            real_rows = [r for r in rows if r[2] == 0][:cap]
            sampled = fake_rows + real_rows

            X = [self._extract_features(t, x) for t, x, _ in sampled]
            y = [target for _, _, target in sampled]

            # Honest held-out evaluation (stratified, shuffled).
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y)
            check = self.build_model_pipeline().fit(X_tr, y_tr)
            fraud_col = list(check.classes_).index(1)
            proba = check.predict_proba(X_te)[:, fraud_col]
            auc = float(roc_auc_score(y_te, proba))
            acc = float(accuracy_score(y_te, check.predict(X_te)))
            self.model_metrics = {
                'holdout_auc': round(auc, 4),
                'holdout_accuracy': round(acc, 4),
                'train_rows': len(X_tr),
                'test_rows': len(X_te),
                'test_fake': int(sum(y_te)),
                'test_real': int(len(y_te) - sum(y_te)),
            }
            print(f"Logistic model holdout: AUC={auc:.3f}, accuracy={acc:.3f} "
                  f"({len(X_te)} test rows, {sum(y_te)} fake)")

            # Deployed model: refit on all sampled rows.
            self.model = self.build_model_pipeline().fit(X, y)
            self.model_reliable = auc >= self.MIN_MODEL_AUC
            if not self.model_reliable:
                print(f"Model AUC below {self.MIN_MODEL_AUC}: fraud_probability is "
                      f"still reported, but it does NOT influence risk_score.")
            print(f"Logistic regression model trained with {len(X)} rows "
                  f"(features: text_score, urgency_score)")
        except Exception as e:
            self.model = None
            self.model_reliable = False
            self.model_error = f"{type(e).__name__}: {e}"
            print(f"Failed to train logistic regression model: {self.model_error}")