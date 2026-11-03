"""Model choice from request features and frozen calibration; no expected answers."""

import copy

if __package__:
    from .adaptive_router import bucket_for, SMALL, LARGE, PROMPT_VERSION, MIN_SAMPLES
else:
    from adaptive_router import bucket_for, SMALL, LARGE, PROMPT_VERSION, MIN_SAMPLES


class CoveragePolicy:
    def __init__(self, profile):
        if profile.get("prompt_version") != PROMPT_VERSION:
            raise ValueError("Calibration prompt mismatch")
        self.metrics = copy.deepcopy(profile["buckets"])

    def select(self, question):
        bucket = bucket_for(question)
        if bucket is None:
            return {"model": None, "bucket": None, "reason": "unsupported_request"}
        small = self.metrics.get(bucket, {}).get(SMALL, {})
        eligible = small.get("samples", 0) >= MIN_SAMPLES and small.get("lower_accuracy", 0) >= 0.8
        return {"model": SMALL if eligible else LARGE, "bucket": bucket,
                "reason": "calibrated_small" if eligible else "small_confidence_insufficient",
                "requires_runtime_verification": True,
                "large_prior": self.metrics.get(bucket, {}).get(LARGE),
                "note": "Large is a verification-gated candidate, not assumed reliable"}
