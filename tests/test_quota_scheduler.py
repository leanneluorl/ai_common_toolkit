"""ai_client: pure scheduling logic (buckets, limits, key discovery). No API calls are made."""
import ai_client


def test_bucket_is_ready_immediately_when_idle():
    b = ai_client._Bucket("gemini-x-flash")
    assert b.ready_at(100.0, {"rpm": 5, "tpm": 1000}, est_tokens=10) <= 100.0


def test_rpm_limit_delays_next_request_until_window_slides():
    b = ai_client._Bucket("gemini-x-flash")
    for t in (10.0, 11.0, 12.0):
        b.request_times.append(t)
    ready = b.ready_at(13.0, {"rpm": 3}, est_tokens=0)
    assert ready >= 10.0 + 60.0  # oldest request leaves the 60 s window first


def test_requests_older_than_a_minute_are_pruned():
    b = ai_client._Bucket("gemini-x-flash")
    for t in (1.0, 2.0, 3.0):
        b.request_times.append(t)
    assert b.ready_at(200.0, {"rpm": 3}, est_tokens=0) <= 200.0


def test_tpm_limit_accounts_for_estimated_tokens():
    b = ai_client._Bucket("gemini-x-flash")
    b.token_events.append((50.0, 900))
    assert b.ready_at(51.0, {"tpm": 1000}, est_tokens=200) >= 50.0 + 60.0
    assert b.ready_at(51.0, {"tpm": 1000}, est_tokens=50) <= 51.0


def test_cooldown_after_429_is_respected():
    b = ai_client._Bucket("gemini-x-flash")
    b.cooldown_until = 500.0
    assert b.ready_at(100.0, {}, est_tokens=0) == 500.0


def test_rpd_left_counts_down_and_zero_means_no_quota():
    b = ai_client._Bucket("gemini-x-flash")
    assert b.rpd_left({"rpd": 20}) == 20
    b.day_requests = 7
    assert b.rpd_left({"rpd": 20}) == 13
    assert b.rpd_left({"rpd": 0}) == 0          # explicit 0 = model has no free quota
    assert b.rpd_left({}) > 10 ** 6             # unset = unlimited
    b.day_blocked = True
    assert b.rpd_left({"rpd": 20}) == 0


def test_buckets_are_independent_per_model_and_owner():
    owner_a = ai_client._QuotaOwner("key-1", "fp-a")
    owner_b = ai_client._QuotaOwner("key-2", "fp-b")
    owner_a.bucket("m1").day_requests = 5
    assert owner_a.bucket("m2").day_requests == 0
    assert owner_b.bucket("m1").day_requests == 0
    assert owner_a.bucket("m1") is owner_a.bucket("m1")


def test_model_tier_classification():
    assert ai_client.get_model_tier("gemini-3-flash") == "high"
    assert ai_client.get_model_tier("gemini-3-flash-lite") == "low"


def test_numbered_keys_are_discovered_in_numeric_order(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY-10", "k10")
    monkeypatch.setenv("GEMINI_API_KEY-2", "k2")
    monkeypatch.setenv("GEMINI_API_KEY-1", "k1")
    monkeypatch.setenv("GEMINI_API_KEY-3", "   ")   # blank values are ignored
    keys = ai_client._discover_api_keys()
    assert [v for _, v in keys] == ["k1", "k2", "k10"]


def test_legacy_single_key_is_used_when_no_numbered_keys(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "legacy")
    assert ai_client._discover_api_keys() == [("GOOGLE_API_KEY", "legacy")]


def test_no_keys_found_returns_empty_list():
    assert ai_client._discover_api_keys() == []
