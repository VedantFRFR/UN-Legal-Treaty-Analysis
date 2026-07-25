import os
import sys
import json
import time
import math
import random
import asyncio
import warnings
from typing import Optional, List, Any

import httpx

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if sys.path and os.path.abspath(sys.path[0]) == _THIS_DIR:
    sys.path.pop(0)
sys.path.insert(0, _PROJECT_ROOT)

warnings.filterwarnings("ignore", message=".*langchain-community.*is being sunset.*", category=DeprecationWarning)
import langchain_community.chat_models
from unittest.mock import MagicMock
_mock = MagicMock()
_mock.ChatVertexAI = MagicMock()
langchain_community.chat_models.vertexai = _mock
sys.modules["langchain_community.chat_models.vertexai"] = _mock

import pandas as pd
from datasets import Dataset

from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    Faithfulness,
    ResponseRelevancy,
)
from ragas import evaluate, RunConfig
from ragas.llms.base import LangchainLLMWrapper

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

COL_PRECISION = "llm_context_precision_with_reference"
COL_RECALL = "context_recall"
COL_FAITHFULNESS = "faithfulness"
COL_RELEVANCY = "answer_relevancy"
COL_SHORT = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]

os.chdir(os.path.dirname(os.path.abspath(__file__)))

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(EVAL_DIR, "eval_dataset.json")
RESULTS_PATH = os.path.join(EVAL_DIR, "eval_results.json")
REPORT_PATH = os.path.join(EVAL_DIR, "eval_report.md")

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_RPM_LIMIT = 30
GROQ_TPM_LIMIT = 8_000
GROQ_TPD_LIMIT = 200_000
GROQ_RPD_LIMIT = 1_000
GROQ_TPD_HEADROOM = 5_000
TARGET_TPM = 4_000
TARGET_RPM = 25

MAX_CONTEXTS = 4
MAX_CONTEXT_CHARS = 800
MAX_ANSWER_CHARS = 1_200

EMBED_MODEL = "BAAI/bge-m3"

MAX_WORKERS = 1
MAX_RETRIES = 3
METRIC_RETRIES = 2

RETRY_BACKOFF_BASE = 8
RETRY_BACKOFF_MAX = 45

CALL_DELAY_BASE = 10.0
ADAPTIVE_SLEEP_MAX = 75.0

CLIENT_CONNECT_TIMEOUT = 15.0
CLIENT_READ_TIMEOUT = 180.0

JUDGE_REASONING_EFFORT = "low"
JUDGE_MAX_TOKENS = 3_072

RUN_CONFIG_TIMEOUT = 540

THRESHOLD_FAITHFULNESS = 0.5
THRESHOLD_RECALL = 0.4
THRESHOLD_PRECISION = 0.4
THRESHOLD_RELEVANCY = 0.4


class _SharedState:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_calls: int = 0
    start_time: Optional[float] = None
    last_request_time: Optional[float] = None

_shared = _SharedState()


class DailyLimitReached(RuntimeError):
    pass


def _is_daily_limit_error(error_text: str) -> bool:
    error_text = error_text.lower()
    daily_markers = (
        "tokens per day",
        "requests per day",
        "daily token",
        "daily request",
        "daily quota",
        "tpd limit",
        "rpd limit",
        " tpd",
        " tpd_",
        " rpd",
        " rpd_",
    )
    return any(marker in error_text for marker in daily_markers)


class _RateLimitAwareGroq(ChatGroq):
    def _call_with_retries(self, method, *args, **kwargs):
        kwargs.pop("n", None)

        if _shared.start_time is None:
            _shared.start_time = time.time()

        self._tpd_preflight_check()

        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES + 1):
            attempt_start = time.time()
            try:
                result = method(*args, **kwargs)
                attempt_elapsed = time.time() - attempt_start

                self._track_usage(result)
                _shared.total_calls += 1
                self._log_tpm()
                print(f"  [TIMING] attempt {attempt + 1} succeeded in {attempt_elapsed:.1f}s",
                      file=sys.stderr)

                self._adaptive_sleep()

                return result

            except Exception as e:
                attempt_elapsed = time.time() - attempt_start
                last_error = e
                error_text = str(e).lower()

                if _is_daily_limit_error(error_text):
                    raise DailyLimitReached(
                        "Groq daily limit reached. Completed scores were saved; "
                        "run again after the Groq reset to resume missing metrics."
                    ) from e

                status_code = getattr(e, "status_code", None)
                error_type = type(e).__name__.lower()
                is_rate_limit = "429" in error_text or "rate_limit" in error_text or status_code == 429
                is_timeout = (
                    isinstance(e, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException))
                    or "timeout" in error_text
                    or "timed out" in error_text
                    or "timeout" in error_type
                )
                is_connection_error = "connectionerror" in error_type or "connection error" in error_text
                is_server_error = isinstance(status_code, int) and 500 <= status_code < 600
                is_retryable = is_rate_limit or is_timeout or is_connection_error or is_server_error

                print(
                    f"  [TIMING] attempt {attempt + 1} failed after {attempt_elapsed:.1f}s "
                    f"({type(e).__name__}: retryable={is_retryable})",
                    file=sys.stderr,
                )

                if is_retryable:
                    delay = min(
                        RETRY_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1),
                        RETRY_BACKOFF_MAX,
                    )
                    reason = "Rate limited" if is_rate_limit else "Transient API failure"
                    print(
                        f"\n[RETRY {attempt + 1}/{MAX_RETRIES}] "
                        f"{reason}. Retrying in {delay:.1f}s...",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                    continue

                raise

        print(
            f"\n[FAIL] All {MAX_RETRIES} retries exhausted. "
            f"Last error: {last_error}",
            file=sys.stderr,
        )
        raise last_error

    def generate_prompt(
        self,
        prompts: List[Any],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        if len(prompts) > 1:
            print(f"  [LIMITER] Collapsing {len(prompts)} duplicate judge prompts to one", file=sys.stderr)
            prompts = prompts[:1]
        saved_n = self.n
        self.n = 1
        try:
            return self._call_with_retries(
                super().generate_prompt, prompts,
                stop=stop, callbacks=callbacks,
                **kwargs,
            )
        finally:
            self.n = saved_n

    async def agenerate_prompt(
        self,
        prompts: List[Any],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        kwargs.pop("n", None)
        if len(prompts) > 1:
            print(f"  [LIMITER] Collapsing {len(prompts)} duplicate judge prompts to one", file=sys.stderr)
            prompts = prompts[:1]

        if _shared.start_time is None:
            _shared.start_time = time.time()

        self._tpd_preflight_check()

        saved_n = self.n
        self.n = 1

        last_error: Optional[Exception] = None

        try:
            for attempt in range(MAX_RETRIES + 1):
                attempt_start = time.time()
                try:
                    result = await super().agenerate_prompt(
                        prompts, stop=stop, callbacks=callbacks, **kwargs
                    )
                    attempt_elapsed = time.time() - attempt_start
                    self._track_usage(result)
                    _shared.total_calls += 1
                    self._log_tpm()
                    print(f"  [TIMING] attempt {attempt + 1} succeeded in {attempt_elapsed:.1f}s",
                          file=sys.stderr)
                    await asyncio.sleep(self._adaptive_pause_seconds())
                    return result

                except Exception as e:
                    attempt_elapsed = time.time() - attempt_start
                    last_error = e
                    error_text = str(e).lower()

                    if _is_daily_limit_error(error_text):
                        raise DailyLimitReached(
                            "Groq daily limit reached. Completed scores were saved; "
                            "run again after the Groq reset to resume missing metrics."
                        ) from e

                    status_code = getattr(e, "status_code", None)
                    error_type = type(e).__name__.lower()
                    is_rate_limit = "429" in error_text or "rate_limit" in error_text or status_code == 429
                    is_timeout = (
                        isinstance(e, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException))
                        or "timeout" in error_text
                        or "timed out" in error_text
                        or "timeout" in error_type
                    )
                    is_connection_error = "connectionerror" in error_type or "connection error" in error_text
                    is_server_error = isinstance(status_code, int) and 500 <= status_code < 600
                    is_retryable = is_rate_limit or is_timeout or is_connection_error or is_server_error

                    print(
                        f"  [TIMING] attempt {attempt + 1} failed after {attempt_elapsed:.1f}s "
                        f"({type(e).__name__}: retryable={is_retryable})",
                        file=sys.stderr,
                    )

                    if is_retryable:
                        delay = min(
                            RETRY_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1),
                            RETRY_BACKOFF_MAX,
                        )
                        reason = "Rate limited" if is_rate_limit else "Transient API failure"
                        print(
                            f"\n[RETRY {attempt + 1}/{MAX_RETRIES}] "
                            f"{reason}. Retrying in {delay:.1f}s...",
                            file=sys.stderr,
                        )
                        await asyncio.sleep(delay)
                        continue

                    raise

            print(
                f"\n[FAIL] All {MAX_RETRIES} retries exhausted. "
                f"Last error: {last_error}",
                file=sys.stderr,
            )
            raise last_error
        finally:
            self.n = saved_n

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("n", None)
        saved_n = self.n
        self.n = 1
        try:
            return self._call_with_retries(super().generate, *args, **kwargs)
        finally:
            self.n = saved_n

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("n", None)
        saved_n = self.n
        self.n = 1
        try:
            return self._call_with_retries(super().invoke, *args, **kwargs)
        finally:
            self.n = saved_n

    def _track_usage(self, result: Any) -> None:
        tokens_in = 0
        tokens_out = 0
        usage: dict = {}

        llm_out = getattr(result, "llm_output", None)
        if isinstance(llm_out, dict):
            usage = llm_out.get("token_usage", {}) or {}

        if not usage:
            generations = getattr(result, "generations", None)
            if isinstance(generations, list):
                for gen_list in generations:
                    if isinstance(gen_list, list):
                        for gen in gen_list:
                            info = getattr(gen, "generation_info", None) or {}
                            usage = info.get("token_usage", {}) or {}
                            if usage:
                                break
                    if usage:
                        break

        if not usage:
            meta = getattr(result, "response_metadata", None)
            if isinstance(meta, dict):
                usage = meta.get("token_usage", {}) or {}

        if isinstance(usage, dict):
            try:
                tokens_in = int(usage.get("prompt_tokens", 0) or 0)
                tokens_out = int(usage.get("completion_tokens", 0) or 0)
            except (TypeError, ValueError):
                tokens_in = 0
                tokens_out = 0

        _shared.total_input_tokens += tokens_in
        _shared.total_output_tokens += tokens_out

    def _log_tpm(self) -> None:
        start = _shared.start_time
        if start is None:
            start = time.time()
            _shared.start_time = start
        elapsed = time.time() - start
        total = (
            _shared.total_input_tokens
            + _shared.total_output_tokens
        )
        tpm = (total / elapsed) * 60 if elapsed > 0 else 0
        tpd_remaining = GROQ_TPD_LIMIT - total

        print(
            f"  [USAGE] Calls: {_shared.total_calls} | "
            f"Tokens: {total:,} | "
            f"TPD left: {tpd_remaining:,} | "
            f"TPM: {tpm:.0f}",
            file=sys.stderr,
        )

    def _adaptive_pause_seconds(self) -> float:
        total_tokens = _shared.total_input_tokens + _shared.total_output_tokens
        calls = _shared.total_calls
        avg_tokens_per_call = total_tokens / max(calls, 1)

        tpm_pause = avg_tokens_per_call / max(TARGET_TPM, 1) * 60
        rpm_pause = 60.0 / max(TARGET_RPM, 1)
        pause = max(tpm_pause, rpm_pause, CALL_DELAY_BASE)
        pause = min(pause, ADAPTIVE_SLEEP_MAX)
        return pause

    def _adaptive_sleep(self) -> None:
        time.sleep(self._adaptive_pause_seconds())

    def _tpd_preflight_check(self) -> None:
        total_so_far = _shared.total_input_tokens + _shared.total_output_tokens
        if total_so_far >= GROQ_TPD_LIMIT - GROQ_TPD_HEADROOM:
            remaining = GROQ_TPD_LIMIT - total_so_far
            print(
                f"\n[FATAL] Groq TPD limit reached ({total_so_far:,} / "
                f"{GROQ_TPD_LIMIT:,} tokens used). "
                f"Remaining budget: {remaining:,} tokens.\n"
                f"Wait for daily reset or use a model with higher TPD.",
                file=sys.stderr,
            )
            raise DailyLimitReached(
                f"Groq TPD limit exhausted ({total_so_far:,} / {GROQ_TPD_LIMIT:,})"
            )
        if _shared.total_calls >= GROQ_RPD_LIMIT - 10:
            print(
                f"\n[FATAL] Groq RPD limit nearly reached "
                f"({_shared.total_calls} / {GROQ_RPD_LIMIT} requests).",
                file=sys.stderr,
            )
            raise DailyLimitReached(
                f"Groq RPD limit exhausted ({_shared.total_calls} / {GROQ_RPD_LIMIT})"
            )


def _make_judge_llm() -> _RateLimitAwareGroq:
    from dotenv import load_dotenv

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    load_dotenv(os.path.join(project_root, ".env"))

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found in .env. "
            "Set GROQ_API_KEY=your_groq_api_key in the project root .env file."
        )

    client_timeout = httpx.Timeout(
        connect=CLIENT_CONNECT_TIMEOUT,
        read=CLIENT_READ_TIMEOUT,
        write=CLIENT_CONNECT_TIMEOUT,
        pool=CLIENT_CONNECT_TIMEOUT,
    )
    no_keepalive = httpx.Limits(max_keepalive_connections=0)

    return _RateLimitAwareGroq(
        model=GROQ_MODEL,
        api_key=api_key,
        temperature=0.1,
        reasoning_effort=JUDGE_REASONING_EFFORT,
        max_tokens=JUDGE_MAX_TOKENS,
        timeout=client_timeout,
        http_client=httpx.Client(timeout=client_timeout, limits=no_keepalive),
        http_async_client=httpx.AsyncClient(timeout=client_timeout, limits=no_keepalive),
        max_retries=0,
    )


def preflight_check() -> bool:
    from dotenv import load_dotenv

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    load_dotenv(os.path.join(project_root, ".env"))
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[PREFLIGHT] GROQ_API_KEY not set.", file=sys.stderr)
        return False

    try:
        import groq as _groq
    except ImportError:
        print(
            "[PREFLIGHT] 'groq' SDK not installed \u2014 install with: pip install groq",
            file=sys.stderr,
        )
        return False
    client = _groq.Groq(api_key=api_key)
    print(f"[PREFLIGHT] Testing Groq API with {GROQ_MODEL}...")

    try:
        raw = client.chat.completions.with_raw_response.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        resp = raw.parse()
        headers = raw.headers

        usage = resp.usage
        used_tokens = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
        remaining = GROQ_TPD_LIMIT - used_tokens

        limit_names = [
            "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
            "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens",
            "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens",
        ]
        real_limits = {h: headers.get(h) for h in limit_names if headers.get(h) is not None}

        print(
            f"[PREFLIGHT] OK - Groq API reachable. "
            f"Test call used {used_tokens} tokens. "
            f"Configured limits: {GROQ_TPM_LIMIT:,} TPM, {GROQ_TPD_LIMIT:,} TPD, "
            f"{GROQ_RPD_LIMIT:,} RPD."
        )
        print(
            "[PREFLIGHT] Note: Groq's response headers report remaining TPM and RPD, "
            "not remaining daily tokens. The local TPD counter starts at zero for each run."
        )
        if real_limits:
            print("[PREFLIGHT] REAL rate-limit headers from Groq (ground truth \u2014 "
                  "use these to correct GROQ_TPD_LIMIT / TARGET_TPM at the top of this file):")
            for k, v in real_limits.items():
                print(f"    {k}: {v}")
        else:
            print("[PREFLIGHT] No x-ratelimit-* headers found in response \u2014 "
                  "check your Groq dashboard directly for real limits.", file=sys.stderr)
        return True
    except Exception as e:
        print(f"[PREFLIGHT] FAIL - Groq API error: {e}", file=sys.stderr)
        return False


def load_dataset() -> list[dict]:
    with open(DATASET_PATH, encoding="utf-8") as f:
        samples = json.load(f)

    required = {"sample_id", "strategy", "question", "ground_truth", "relevant_paths"}
    for s in samples:
        missing = required - s.keys()
        if missing:
            raise ValueError(
                f"Sample {s.get('sample_id', '?')} missing required fields: {missing}"
            )

    print(f"Loaded {len(samples)} samples from {DATASET_PATH}")
    return samples


def run_pipeline_for_sample(query: str) -> tuple[str, list[str], list[dict]]:
    from src.agent import run_pipeline, build_retrieval_graph, _make_initial_state

    graph = build_retrieval_graph()
    initial = _make_initial_state(query)
    state = graph.invoke(initial)

    answer = state.get("answer", "")
    chunks = state.get("chunks", [])

    contexts = [c.get("text", "") for c in chunks if c.get("text")]
    context_metadata = [
        {
            "path": c.get("id", ""),
            "treaty": c.get("doc_name", "").replace(".pdf", ""),
            "article": c.get("article_no", c.get("parent_article_no", "")),
            "clause": "",
        }
        for c in chunks
    ]

    return answer, contexts, context_metadata


def enrich_samples(samples: list[dict]) -> list[dict]:
    for i, s in enumerate(samples):
        qid = s["sample_id"]
        try:
            answer, contexts, meta = run_pipeline_for_sample(s["question"])
            s["answer"] = answer
            s["contexts"] = contexts
            s["context_metadata"] = meta
            s["status"] = "no_context" if not contexts else "success"
            s["failure_reason"] = None
        except Exception as e:
            s["answer"] = ""
            s["contexts"] = []
            s["context_metadata"] = []
            s["status"] = "failed"
            s["failure_reason"] = str(e)
        print(f"  [{i + 1}/{len(samples)}] {qid}: {s['status']}")
    return samples


def build_ragas_dataset(samples: list[dict]) -> tuple[Dataset, list[dict]]:
    success = [s for s in samples if s["status"] == "success"]
    failed = [s for s in samples if s["status"] != "success"]

    print(f"Building RAGAS dataset: {len(success)} successful, {len(failed)} failed/no-context")

    ds = Dataset.from_list([
        {
            "question": s["question"],
            "answer": s["answer"],
            "contexts": s["contexts"],
            "ground_truth": s["ground_truth"],
        }
        for s in success
    ])
    return ds, failed


_METRICS_CACHE: dict[str, Any] = {}

_RUN_CONFIG = RunConfig(max_workers=1, max_retries=0, timeout=RUN_CONFIG_TIMEOUT)

METRIC_SPECS: list[tuple[str, str, Any]] = [
    ("context_precision", COL_PRECISION, LLMContextPrecisionWithReference()),
    ("context_recall", COL_RECALL, LLMContextRecall()),
    ("faithfulness", COL_FAITHFULNESS, Faithfulness()),
    ("answer_relevancy", COL_RELEVANCY, ResponseRelevancy(strictness=1)),
]


def _make_ragas_llm(raw_llm: _RateLimitAwareGroq) -> LangchainLLMWrapper:
    return LangchainLLMWrapper(
        raw_llm,
        run_config=_RUN_CONFIG,
        bypass_n=True,
    )


def _get_judge_llm_and_embeddings() -> tuple:
    if "llm" not in _METRICS_CACHE:
        _METRICS_CACHE["llm"] = _make_ragas_llm(_make_judge_llm())
    if "embeddings" not in _METRICS_CACHE:
        _METRICS_CACHE["embeddings"] = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _METRICS_CACHE["llm"], _METRICS_CACHE["embeddings"]


def _prepare_ragas_row(sample: dict) -> dict:
    contexts = [c.strip() for c in sample.get("contexts", []) if c and c.strip()]
    contexts = contexts[:MAX_CONTEXTS]
    contexts = [c[:MAX_CONTEXT_CHARS] for c in contexts]
    answer = (sample.get("answer") or "")[:MAX_ANSWER_CHARS]
    return {
        "question": sample["question"],
        "answer": answer,
        "contexts": contexts,
        "ground_truth": sample["ground_truth"],
    }


def _is_valid_score(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def _is_pipeline_error_answer(answer: Any) -> bool:
    text = (answer or "").strip().lower()
    return not text or text.startswith("failed to generate an answer")


def _missing_score_keys(sample: dict, keys: set[str]) -> set[str]:
    return {k for k in keys if not _is_valid_score(sample.get(k))}


def _has_scores(s: dict, keys: set[str]) -> bool:
    return not _missing_score_keys(s, keys)


def _evaluate_one_metric(
    row: dict,
    metric: Any,
    llm: LangchainLLMWrapper,
    embeddings: HuggingFaceEmbeddings,
    col_name: str,
) -> float:
    ds = Dataset.from_list([row])
    last_error: Optional[Exception] = None

    for attempt in range(METRIC_RETRIES + 1):
        try:
            result = evaluate(
                dataset=ds,
                metrics=[metric],
                llm=llm,
                embeddings=embeddings,
                run_config=_RUN_CONFIG,
                show_progress=False,
                raise_exceptions=True,
            )
            val = float(result.to_pandas().iloc[0].get(col_name, float("nan")))
            if _is_valid_score(val):
                return val
            last_error = ValueError(f"{col_name} returned NaN")
            print(
                f"    [METRIC RETRY {attempt + 1}/{METRIC_RETRIES}] "
                f"{col_name} returned NaN",
                file=sys.stderr,
            )
        except Exception as e:
            last_error = e
            print(
                f"    [METRIC RETRY {attempt + 1}/{METRIC_RETRIES}] "
                f"{col_name} failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )

        if attempt < METRIC_RETRIES:
            delay = min(
                RETRY_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1),
                RETRY_BACKOFF_MAX,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Metric {col_name} failed after {METRIC_RETRIES + 1} attempts: {last_error}"
    )


def compute_sample_metrics(
    sample: dict,
    only_keys: Optional[set[str]] = None,
) -> dict[str, float]:
    llm, embeddings = _get_judge_llm_and_embeddings()
    row = _prepare_ragas_row(sample)
    n_ctx = len(row["contexts"])
    if n_ctx == 0:
        raise ValueError(f"Sample {sample.get('sample_id')} has no contexts for RAGAS")

    scores: dict[str, float] = {}
    metric_errors: dict[str, str] = {}
    specs = METRIC_SPECS
    if only_keys:
        specs = [spec for spec in METRIC_SPECS if spec[0] in only_keys]

    for short_key, col_name, metric in specs:
        print(f"    metric={short_key} (contexts={n_ctx})", file=sys.stderr)
        try:
            scores[short_key] = _evaluate_one_metric(
                row, metric, llm, embeddings, col_name
            )
            sample.update(scores)
        except DailyLimitReached:
            sample.update(scores)
            raise
        except Exception as e:
            metric_errors[short_key] = f"{type(e).__name__}: {e}"
            print(
                f"    [METRIC FAILED] {short_key}: {metric_errors[short_key]}",
                file=sys.stderr,
            )

    if metric_errors:
        sample["metric_errors"] = metric_errors
    else:
        sample.pop("metric_errors", None)

    return scores


def _save_results(samples: list[dict]) -> None:
    samples.sort(key=lambda x: x["sample_id"])
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


def generate_report(success_samples: list[dict], failed_samples: list[dict]) -> None:
    out: list[str] = []
    L = out.append

    def get_score(s: dict, col: str) -> float | None:
        v = s.get(col)
        if isinstance(v, (int, float)) and _is_valid_score(v):
            return float(v)
        return None

    def format_score(value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "N/A"

    L("# RAGAS Evaluation Report")
    L("")
    L(f"Total samples: {len(success_samples) + len(failed_samples)}")
    L(f"Success: {len(success_samples)}")
    L(f"Failed: {len(failed_samples)}")

    L("")
    L("## 5.1 Overall Averages")
    L("")
    L("| Metric | Average Score |")
    L("|---|---|")
    for col in COL_SHORT:
        vals = [get_score(s, col) for s in success_samples]
        vals = [v for v in vals if v is not None]
        if vals:
            L(f"| {col.replace('_', ' ').title()} | {sum(vals) / len(vals):.3f} |")
    L("")

    strategy_map: dict[str, list[dict]] = {
        "CONCEPTUAL": [],
        "DEFINITIONAL": [],
        "PROVISION": [],
    }
    for s in success_samples:
        key = s.get("strategy", "")
        if key in strategy_map:
            strategy_map[key].append(s)

    L("## 5.2 Per-Strategy Breakdown")
    L("")
    for strat, s_list in strategy_map.items():
        if not s_list:
            continue
        L(f"### {strat} ({len(s_list)} samples)")
        L("")
        L("| Metric | Average |")
        L("|---|---|")
        for col in COL_SHORT:
            vals = [get_score(s, col) for s in s_list]
            vals = [v for v in vals if v is not None]
            if vals:
                L(f"| {col.replace('_', ' ').title()} | {sum(vals) / len(vals):.3f} |")
        L("")

    L("## 5.3 Failures")
    L("")
    if not failed_samples:
        L("No failures recorded.")
    else:
        for s in failed_samples:
            L(f"**[{s['sample_id']}]** | Strategy: {s['strategy']} | Status: {s['status']}")
            L(f"Question: {s['question']}")
            L(f"Reason: {s.get('failure_reason', 'N/A')}")
            L("")
    L("")

    L("## 5.4 Edge Cases (Scores Below Threshold)")
    L("")
    edge_count = 0
    for s in success_samples:
        scores = {col: get_score(s, col) for col in COL_SHORT}
        triggers: list[str] = []
        for k, v in scores.items():
            if v is None:
                continue
            if k == "faithfulness" and v < THRESHOLD_FAITHFULNESS:
                triggers.append("faithfulness")
            elif k == "context_recall" and v < THRESHOLD_RECALL:
                triggers.append("context_recall")
            elif k == "context_precision" and v < THRESHOLD_PRECISION:
                triggers.append("context_precision")
            elif k == "answer_relevancy" and v < THRESHOLD_RELEVANCY:
                triggers.append("answer_relevancy")

        if not triggers:
            continue

        edge_count += 1
        L(f"**[{s['sample_id']}]** | Strategy: {s['strategy']}")
        L(f"Question: {s['question']}")
        L(f"Answer: {s['answer'][:300]}")
        L(
            f"Scores: precision={format_score(scores.get('context_precision'))}, "
            f"recall={format_score(scores.get('context_recall'))}, "
            f"faithfulness={format_score(scores.get('faithfulness'))}, "
            f"relevancy={format_score(scores.get('answer_relevancy'))}"
        )
        for ctx in s.get("contexts", [])[:3]:
            L(f"  - {ctx[:200]}")

        if "faithfulness" in triggers and "context_precision" in triggers:
            L("Diagnosis: Retriever returned off-topic chunks; LLM may have hallucinated")
        elif "context_recall" in triggers:
            L("Diagnosis: Answer correct but retriever missed expected paths")
        elif "answer_relevancy" in triggers:
            L("Diagnosis: Answer does not address the question directly")
        else:
            L("Diagnosis: Low score on one or more metrics")
        L("")

    if edge_count == 0:
        L("No edge cases below thresholds.")
        L("")

    L("## 5.5 Retrieved Paths vs Expected Paths")
    L("")
    hit_rates: list[float] = []
    for s in success_samples:
        expected = set(s.get("relevant_paths", []))
        retrieved = set(m.get("path", "") for m in s.get("context_metadata", []))
        if not expected:
            continue
        matches = expected & retrieved
        hr = len(matches) / len(expected)
        hit_rates.append(hr)
        if hr < 0.5:
            missed = expected - retrieved
            L(
                f"**[{s['sample_id']}]** | Expected: {len(expected)} paths | "
                f"Retrieved: {len(matches)} matches | "
                f"Hit rate: {hr * 100:.0f}%"
            )
            for p in sorted(missed)[:5]:
                L(f"  - {p}")
            L("")

    if hit_rates:
        L(
            f"**Overall average path hit rate:** "
            f"{sum(hit_rates) / len(hit_rates) * 100:.1f}%"
        )
        L("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    ok = preflight_check()
    if not ok:
        sys.exit(1)

    samples = load_dataset()

    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8") as f:
            samples = json.load(f)

        if all(s.get("answer") is not None for s in samples):
            print(f"Loaded {len(samples)} cached results from {RESULTS_PATH}")
        else:
            samples = enrich_samples(samples)
    else:
        samples = enrich_samples(samples)

    for s in samples:
        if s.get("status") == "success" and _is_pipeline_error_answer(s.get("answer")):
            s["status"] = "failed"
            s["failure_reason"] = "RAG pipeline did not produce an answer; metrics skipped."

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    success = [s for s in samples if s["status"] == "success"]
    failed = [s for s in samples if s["status"] != "success"]
    all_score_keys = {"context_precision", "context_recall", "faithfulness", "answer_relevancy"}

    if success:
        remaining = [s for s in success if _missing_score_keys(s, all_score_keys)]
        done = len(success) - len(remaining)

        if remaining:
            print(f"Computing RAGAS metrics: {done} fully scored, {len(remaining)} need work")
            print(f"  Using model: {GROQ_MODEL}")
            print(f"  max_workers: 1 (sequential, one metric at a time)")
            print(f"  Adaptive delay: {CALL_DELAY_BASE}s base, targets {TARGET_TPM} TPM / {TARGET_RPM} RPM")
            print(f"  Groq limits: {GROQ_TPM_LIMIT:,} TPM, {GROQ_TPD_LIMIT:,} TPD, {GROQ_RPD_LIMIT:,} RPD")
            print(f"  Context trim: max {MAX_CONTEXTS} chunks \u00d7 {MAX_CONTEXT_CHARS} chars")
            print()

            daily_limit_reached = False
            for idx, s in enumerate(remaining):
                qid = s["sample_id"]
                missing = _missing_score_keys(s, all_score_keys)
                try:
                    scores = compute_sample_metrics(s, only_keys=missing)
                    s.update(scores)
                    _save_results(samples)
                    def _fmt(key: str) -> str:
                        v = s.get(key)
                        return f"{v:.3f}" if _is_valid_score(v) else "nan"

                    print(f"  [{done + idx + 1}/{len(success)}] {qid}: "
                          f"precision={_fmt('context_precision')}, "
                          f"recall={_fmt('context_recall')}, "
                          f"faithfulness={_fmt('faithfulness')}, "
                          f"relevancy={_fmt('answer_relevancy')}")
                except DailyLimitReached as e:
                    _save_results(samples)
                    daily_limit_reached = True
                    print(
                        f"  [{done + idx + 1}/{len(success)}] {qid}: PAUSED - {e}",
                        file=sys.stderr,
                    )
                    break
                except Exception as e:
                    print(f"  [{done + idx + 1}/{len(success)}] {qid}: FAILED - {e}", file=sys.stderr)
                    _save_results(samples)
                    raise

            if daily_limit_reached:
                print(
                    "Evaluation paused at Groq's daily limit. Results were saved; "
                    "run this command again after the limit resets to resume."
                )
            else:
                fully_scored = sum(_has_scores(s, all_score_keys) for s in success)
                print(
                    f"  Evaluation attempts finished: {fully_scored}/{len(success)} fully scored. "
                    f"Results saved to {RESULTS_PATH}"
                )
        else:
            print(f"RAGAS scores already cached for all {len(success)} samples \u2014 skipping metric computation.")
    else:
        print("No successful samples \u2014 skipping metric computation.")

    generate_report(success, failed)
