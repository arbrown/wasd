+++
title = "Observability in the Agent Storybook Project: Adding Traces to ADK"
date = 2026-09-28T11:15:00-06:00
draft = false
categories = ["AI and Machine Learning", "Kubernetes", "Observability"]
tags = ["gke", "kubernetes", "ai", "agents", "google-adk", "gemini", "opentelemetry", "cloud-trace", "observability", "asp"]
description = "Peeking inside the black box: how instrumenting an ADK-based system with OpenTelemetry and Google Cloud Trace exposed bugs in deterministic logic, rate limiting, and model choice."
+++

![](/images/observability-asp/hero.png)

In case you missed it, in the [first post](/posts/introducing-agent-storybook-project/) of this series, I introduced the [Agent Storybook Project (ASP)](https://github.com/arbrown/asp)—an open-source multi-agent system running on [GKE](https://cloud.google.com/kubernetes-engine?utm_campaign=CDR_0x145aeba1_default_b559176373&utm_medium=external&utm_source=blog) that adapts classic public-domain literature into illustrated children's books.

The goal was always to use it as a testbed for showing different lessons around running agents on GKE, and I have my first topic: Observability!

The initial implementation had some basic logging and even pretty good tracing out of the box thanks to ADK, but it wasn't quite enough to answer a few important questions: 

* Why does this take so long (or more precisely, where was all the time being spent)?
* How do all the different agents fit together for one giant pipeline run?
* What about special calls like generating images (a big part of this agent's job!)?

These are the kinds of questions that I can answer with tracing! 

I instrumented the entire pipeline with [OpenTelemetry (OTel)](https://opentelemetry.io/) and [Google Cloud Trace](https://cloud.google.com/trace?utm_campaign=CDR_0x145aeba1_default_b559176373&utm_medium=external&utm_source=blog). Here is the [PR for the tracing instrumentation](https://github.com/arbrown/asp/pull/1) and the [follow-up PR with the fixes](https://github.com/arbrown/asp/pull/2) it inspired. Let's walk through what changed, and what I did after seeing observability in action!

---

## Instrumenting an ADK Application: The Code

Since this "agent" is really a multi-agent pipeline with several sub-agents, and each of those sub-agents might have several steps, I wanted a way to wrap this whole pipeline in a root trace, with each step in a child span. I set up a `SpanContextManager` to make sure each span has the appropriate context about where it is in the pipeline as well as the right metadata for GenAI-specific attributes like model used and token counts.

### Tracing the whole pipeline

First, I created some helper utilities to keep the pipeline code simpler:
*(From [`src/storybook/tracing.py`](https://github.com/arbrown/asp/blob/main/src/storybook/tracing.py))*

```python
# OpenTelemetry GenAI semantic conventions
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"


class SpanContextManager:
    def __init__(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        tracer: trace.Tracer | None = None,
    ) -> None:
        self.name = name
        self.attributes = attributes or {}
        self.tracer = tracer
        self._cm: Any = None
        self.span: trace.Span | None = None

    def __enter__(self) -> trace.Span:
        tracer = self.tracer or get_tracer()
        self._cm = tracer.start_as_current_span(self.name, attributes=self.attributes)
        self.span = self._cm.__enter__()
        return self.span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        if self.span is not None and exc_val is None:
            if self.span.is_recording():
                status = getattr(self.span, "status", None)
                if status is not None and getattr(status, "status_code", None) == trace.StatusCode.UNSET:
                    self.span.set_status(trace.StatusCode.OK)
        if self._cm is not None:
            return self._cm.__exit__(exc_type, exc_val, exc_tb)
        return False

    async def __aenter__(self) -> trace.Span:
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        return self.__exit__(exc_type, exc_val, exc_tb)


def trace_stage(stage_name: str, **attributes: Any) -> SpanContextManager:
    """Wrap a pipeline stage with proper span hierarchy and error recording."""
    return SpanContextManager(stage_name, attributes=attributes)


def trace_agent_call(
    agent_name: str,
    model: str | None = None,
    **attributes: Any,
) -> SpanContextManager:
    """Wrap model/agent calls with standard GenAI semantic conventions."""
    span_name = (
        agent_name
        if ("." in agent_name or agent_name.startswith("agent."))
        else f"agent.{agent_name}"
    )
    system = attributes.pop(GEN_AI_SYSTEM, "gemini")
    operation_name = attributes.pop(GEN_AI_OPERATION_NAME, "generate_content")
    span_attrs: dict[str, Any] = {
        GEN_AI_SYSTEM: system,
        GEN_AI_OPERATION_NAME: operation_name,
        "agent.name": agent_name,
    }
    if model:
        span_attrs[GEN_AI_REQUEST_MODEL] = model
    span_attrs.update(attributes)
    return SpanContextManager(span_name, attributes=span_attrs)


def set_span_token_usage(
    span: trace.Span | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> None:
    """Attach token usage metrics to a span if available."""
    target_span = span or trace.get_current_span()
    if target_span is not None and target_span.is_recording():
        if prompt_tokens is not None:
            target_span.set_attribute(GEN_AI_USAGE_PROMPT_TOKENS, prompt_tokens)
        if completion_tokens is not None:
            target_span.set_attribute(GEN_AI_USAGE_COMPLETION_TOKENS, completion_tokens)
```

Then, I use those helpers in setting up the full [agent pipeline](https://github.com/arbrown/asp/blob/main/src/storybook/agents/pipeline.py#L893-L903):

```python
        with trace_stage("stage.generate_illustrations", session_id=sid, total_spreads=total_spreads):
            log.info(
                "Starting stage.generate_illustrations for session %s (total_spreads=%d, model=%s)",
                sid,
                total_spreads,
                settings.model_image,
            )
            image_sem = asyncio.Semaphore(settings.image_concurrency)
            llm_sem = asyncio.Semaphore(settings.llm_concurrency)
            ref_ready: asyncio.Event = asyncio.Event()
            ref_image: list[bytes] = []
```

And so on for each of the stages.

This gives us a rich view of a full pipeline run, which I can expand into more detailed views of each stage.

![A recently instrumented pipeline run](/images/observability-asp/trace-1.png)

---

## What the Traces Revealed: Architecture & Logic Improvements

In that full trace, I uncovered a couple of spots that I didn't realize had been slowing down runs of the full agent pipeline. 

### 1. Fine-Grained 429 Retries at the LLM Call Level (Not the Workflow Level)

**What the trace showed:** LLMs are in high demand. Even (maybe especially!) for Google employees, sometimes the system is too overwhelmed with higher priority requests. I noticed that `429 RESOURCE_EXHAUSTED` errors were triggering retries at the *agent* level instead of the *model call* level—aborting an entire `LoopAgent` pass and throwing away perfectly good work!

**The fix:** Rather than retrying the whole workflow stage, I wrapped ADK's `Gemini.generate_content_async` with per-call exponential backoff retries. Now individual model calls self-heal in place without aborting the parent span or session.

```python
# src/storybook/agents/pipeline.py
from google.adk.models.google_llm import Gemini

_original_generate_content_async = Gemini.generate_content_async


async def _retryable_generate_content_async(
    self: Gemini, llm_request: Any, stream: bool = False
) -> Any:
    """Wrap Gemini.generate_content_async with fine-grained 429 retry.

    Retries transient 429 RESOURCE_EXHAUSTED errors at the individual LLM call
    level with exponential backoff. This prevents an entire LoopAgent (e.g.
    craft_adapt_validate) or multi-step workflow from aborting and restarting
    from scratch when a single validation or generation call hits rate limits.
    """
    max_retries = 5
    for attempt in range(max_retries):
        yielded = False
        try:
            async for item in _original_generate_content_async(self, llm_request, stream=stream):
                yielded = True
                yield item
            return
        except Exception as exc:
            msg = str(exc)
            if not yielded and ("RESOURCE_EXHAUSTED" in msg or "429" in msg) and attempt < max_retries - 1:
                wait = 5 * (2 ** attempt)
                log.warning(
                    "429 RESOURCE_EXHAUSTED on model %s; retrying LLM call in %ds (attempt %d/%d)",
                    getattr(llm_request, "model", "model"),
                    wait,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(wait)
                continue
            raise


Gemini.generate_content_async = _retryable_generate_content_async
```

---

### 2. Flaky Text Matching vs. Deterministic ADK Session State

**What the trace showed:** Another flaky error! I was using a "model-as-a-judge" technique to make sure the generated images matched the story and our "character bible" for visual consistency. Even though my validator agent had `approve_image` and `reject_image` tools that recorded the verdict in ADK session state, the pipeline orchestrator was still naively checking `"approved" in result.lower()` on the model's final text response. Whenever the model approved an image via the tool call without repeating the literal word `"approved"` in its reply, we threw the image away and regenerated it!

**The fix:** I switched the orchestrator from scraping output strings to inspecting deterministic ADK session state (`vsession.state["validation.passed"]` and `vsession.state["revised_image_prompt"]`). Now, when an image is rejected, it's for a good reason!

*(From [`src/storybook/agents/pipeline.py`](https://github.com/arbrown/asp/blob/main/src/storybook/agents/pipeline.py#L1343-L1388))*

```python
# src/storybook/agents/pipeline.py
with check_image_validation(
    attempt=attempt,
    max_attempts=settings.image_max_retries + 1,
    spread_number=spread_number,
    image_index=image_index,
) as val_span:
    async with llm_sem:
        result = await _run_agent(
            validator_runner,
            val_sid,
            validate_input,
            subject_image=img_bytes,
            reference_image=style_ref,
            prev_spread_image=prev_img,
        )

    # Inspect deterministic ADK session state instead of scraping raw output text!
    vsession = None
    if hasattr(validator_runner, "session_service") and hasattr(validator_runner.session_service, "get_session"):
        session_coro = validator_runner.session_service.get_session(
            app_name=validator_runner.app_name,
            user_id="pipeline",
            session_id=val_sid,
        )
        vsession = await session_coro if inspect.isawaitable(session_coro) else session_coro

    if vsession and "validation.passed" in vsession.state:
        passed = bool(vsession.state["validation.passed"])
        score = float(vsession.state.get("validation.score", 1.0 if passed else 0.0))
        reasons = vsession.state.get("validation.reasons") or ([] if passed else [result.strip()])
        revised_prompt = vsession.state.get("revised_image_prompt")
        if revised_prompt and not passed:
            current_prompt = revised_prompt  # Self-healing retry with revised prompt
    else:
        passed = "approved" in result.lower()
        score = 1.0 if passed else 0.0
        reasons = [] if passed else [result.strip()]

    record_validation_result(val_span, passed=passed, score=score, attempt=attempt, reasons=reasons)
```

---

### 3. Different Models for Different Purposes: Heavy, Fast, and Image

**What the trace showed:** Using a heavy reasoning model for every single step was overkill. It dragged out latency on simple, mechanical tasks.

**The fix:** I right-sized the models to the actual job in [`src/storybook/config.py`](https://github.com/arbrown/asp/blob/main/src/storybook/config.py):

* **Heavy (`gemini-3.1-pro-preview`):** Creative storytelling and the polished literary craft pass.
* **Fast (`gemini-3.5-flash`):** Rapid, structured tasks like structural drafting, character bibles, spread layout planning, and vision validation.
* **Image (`gemini-3.1-flash-image`):** Dedicated illustration generation.

Splitting these cut pipeline cycle times by over 40%.

---

### 4. Adapter Prompt Hardening & Gutenberg Fallbacks

**Eliminating validation churn:** Waterfalls of the story adaptation stage showed repeated 2nd and 3rd retry passes. The traces exposed the exact validation feedback: the adapter frequently duplicated text between facing pages (verso and recto—that's a fun rabbit hole for another post) or failed strict sentence count constraints. Strengthening the instructions in [`src/storybook/agents/story_adapter.py`](https://github.com/arbrown/asp/blob/main/src/storybook/agents/story_adapter.py) with explicit "Zero-Tolerance Structural Compliance" rules immediately eliminated unnecessary retry cycles.

**Graceful fallback when Gutenberg misses:** Spans on `gutenberg.search` also highlighted external HTTP latency and empty results for titles missing from Gutendex. I added an automated fallback so that when a Gutenberg search comes up empty, the pipeline generates the story directly from Gemini's parametric weights—tracked cleanly as its own span in the trace.

---
![A faster trace on a larger project!](/images/observability-asp/trace-2.png)

---

## What's Next: Distributed Agent Tracing Across Pods & Sandboxes

A few big takeaways from this exercise:

* **Observability is a tuning tool, not just a crash debugger:** In multi-agent systems, traces show you where agents are spinning their wheels, retrying behind your back, or using too much model for the job.
* **Measure the non-deterministic stuff:** Tracking token counts and validation retries per trace turns fuzzy LLM behavior into concrete engineering metrics you can actually optimize.
* **Trust tool state over text scraping:** Rely on ADK session state and call-level retries rather than fragile string matching in your orchestrator.

As the Agent Storybook Project grows, I'll be decoupling these agents from a single container into separate Kubernetes Jobs and [GKE Agent Sandboxes](https://cloud.google.com/kubernetes-engine/docs/how-to/agent-sandbox?utm_campaign=CDR_0x145aeba1_default_b559176373&utm_medium=external&utm_source=blog). Because the application is already instrumented with OpenTelemetry, standard W3C `traceparent` headers will propagate across network boundaries and pod runtimes automatically, preserving end-to-end visibility.

Stay tuned under the [asp](/tags/asp/) tag for the next installment!

---

## Helpful Links

* [Google Cloud Agent Observability Guide](https://docs.cloud.google.com/stackdriver/docs/observability/agent-observability?utm_campaign=CDR_0x145aeba1_default_b559176373&utm_medium=external&utm_source=blog)
* [Agent Storybook Project GitHub Repository](https://github.com/arbrown/asp) ([PR #1: OTel Instrumentation](https://github.com/arbrown/asp/pull/1), [PR #2: Observability Fixes](https://github.com/arbrown/asp/pull/2))
* [Part 1: Introducing Agent Storybook Project](/posts/introducing-agent-storybook-project/)
* [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
* [Google Cloud Trace Documentation](https://cloud.google.com/trace/docs?utm_campaign=CDR_0x145aeba1_default_b559176373&utm_medium=external&utm_source=blog)

