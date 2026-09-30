+++
title = "Isolating Agents in the Agent Storybook Project with Agent Substrate"
date = 2026-09-29T15:10:00-06:00
draft = true
categories = ["AI and Machine Learning", "Kubernetes"]
tags = ["gke", "kubernetes", "ai", "agents", "google-adk", "agent-substrate", "agent-sandbox", "gvisor", "asp"]
description = "Why running a multi-agent pipeline inside your web server is a bad idea, and how I used Agent Substrate and gVisor snapshots on GKE to give every storybook run its own isolated actor."
+++

<!-- TODO: Hero image -->
<!-- ![](/images/substrate-asp/hero.png) -->

In the [first post](/posts/introducing-agent-storybook-project/) of this series, I introduced the [Agent Storybook Project (ASP)](https://github.com/arbrown/asp)—an open-source multi-agent system on [GKE](https://cloud.google.com/kubernetes-engine?utm_campaign=CDR_0x145aeba1_default_b559176373&utm_medium=external&utm_source=blog) that adapts public-domain classics into illustrated children's books. Last time, we [instrumented the whole thing with OpenTelemetry](/posts/observability-asp/) to find and fix a bunch of hidden bottlenecks in the agent loop.

Now it's time to tackle the #1 item from my original "Future Improvements" list: **getting the agents out of the web server!**

I just put up two new PRs on the repo: [one that moves pipeline runs into isolated Agent Substrate actors](https://github.com/arbrown/asp/pull/3), and a [follow-up PR that adds crash detection and checkpoint recovery](https://github.com/arbrown/asp/pull/4). Let's look at why the original setup needed to change, and what it takes to get [Agent Substrate](https://github.com/agent-substrate/substrate) up and running in a project like this.

---

## Before: Everything in One Pod (Why That Was a Bad Idea)

When I built the first prototype of the Storybook Project, I took the easiest path possible to get it working: when you clicked "Generate" in the UI, the FastAPI backend just kicked off the 11-stage [ADK](https://github.com/google/adk-python) pipeline right there in the same process using `asyncio.create_task()`.

That's fine for a weekend demo, but pretty terrible for a real application:

1. **The UI and the agents were sharing a process:** While a lot of the pipeline is waiting on Gemini calls, it also does real work—pulling full books from Project Gutenberg, validating bursts of high-res illustrations, and compiling print-ready PDFs with WeasyPrint. Running all of that inside the same Python event loop that serves the UI means one heavy book build can bog down the web server for everyone else.
2. **In-memory state locked us to a single pod:** Active sessions and progress queues lived in Python dictionaries (`_sessions`, `_queues`, `_tasks`) right inside the FastAPI process. That meant we were stuck at `replicas: 1`. Worse, if I pushed a tiny CSS or API fix and Kubernetes rolled the backend pod, every in-flight book generation was unceremoniously killed mid-story.
3. **Zero isolation between agent runs:** Every user's storybook session ran in the exact same container with the same filesystem and memory space. Right now the agents only call pre-defined tools, but as I give them more autonomy (like writing custom code to lay out a tricky page), I *really* want each agent locked in its own [gVisor sandbox](https://cloud.google.com/kubernetes-engine/docs/concepts/sandbox-pods?utm_campaign=CDR_0x145aeba1_default_b559176373&utm_medium=external&utm_source=blog) so one weird run can't mess with another—or with the web server itself.

---

## After: Giving Every Session Its Own Substrate Actor

To fix this, I used **[Agent Substrate](https://github.com/agent-substrate/substrate)** to split the web backend from the actual agent work.

Instead of running the ADK pipeline inside FastAPI, each book generation now gets its own isolated **Substrate Actor** running in a gVisor sandbox:

![Isolated Agent Substrate Actors on GKE](/images/substrate-asp/substrate-architecture.png)

Here's why I like this pattern so much better than just spinning up a raw Kubernetes `Job` for every request:

* **Fast startups with Golden Snapshots:** Rather than having Kubernetes schedule a brand new Pod, pull layers, and boot the runtime from scratch for every single user request, all the expensive startup costs (container sandbox creation, Python runtime boot, and heavy module imports like `weasyprint` and `google-adk`) are **only paid once** when building the Golden Snapshot. When a user requests a book, Substrate restores a fresh actor from that snapshot onto a warm worker pool in **~1 second** (`resume_ms=1102.9ms`).
* **Real isolation per run:** Each session runs in its own gVisor sandbox (`atespace: asp`, `name: <session_id>`) in a separate `agent-workloads` namespace, completely isolated from the FastAPI gateway and from other users' books.
* **The web server is actually stateless now:** FastAPI just writes the initial config to GCS, asks Substrate to spawn the actor, and streams progress updates back to the browser. We can finally scale the backend to multiple replicas and deploy updates without killing active runs!

---

## How to Set It Up: The Code

If you want to take an existing Python agent and move it into Agent Substrate actors, you really only need three pieces.

### 1. Define the `ActorTemplate`

First, we give Substrate an `ActorTemplate` that describes how to run and snapshot our agent container. 

There are two neat pieces in this YAML:
* **`readyz`**: Tells Substrate what endpoint to poll when building the Golden Snapshot. As soon as `/readyz` returns `200`, Substrate freezes the container into a snapshot stored in GCS.
* **`systemInfo` volume**: When Substrate restores that snapshot for a real user session, it mounts the actor's assigned name and namespace as plain text files in `/run/ate/metadata`. That's how the restored container discovers *which* session it's supposed to run!

*(From [`k8s/substrate/actortemplate.yaml`](https://github.com/arbrown/asp/pull/3/files))*

```yaml
# Created via: kubectl ate create actor-template -f k8s/substrate/actortemplate.yaml
metadata:
  atespace: asp
  name: asp-runner
workerSelector:
  matchLabels:
    workload: asp-runner
containers:
- name: runner
  image: ${ASP_RUNNER_IMAGE_DIGEST}
  volumeMounts:
  - name: metadata
    mountPath: /run/ate/metadata
  readyz:
    httpGet:
      path: /readyz
      port: 8080
    timeoutSeconds: 60
volumes:
- name: metadata
  systemInfo:
    dataSources:
    - actorMetadata:
        items:
        - field: ACTOR_METADATA_FIELD_NAME
          path: actor-name
        - field: ACTOR_METADATA_FIELD_ATESPACE
          path: actor-atespace
resources:
  limits:
  - name: cpu
    quantity: "2"
  - name: memory
    quantity: 4Gi
snapshotsConfig:
  onPause: SNAPSHOT_CONTENT_SCOPE_FULL
  onCommit: SNAPSHOT_CONTENT_SCOPE_FULL
  storageLocation: gs://${SUBSTRATE_STATE_BUCKET}/asp/
sandboxConfig:
  sandboxClass: SANDBOX_CLASS_GVISOR
  configName: asp-gvisor
```

---

### 2. Pre-Warm the Snapshot in Python (`runner.py`)

Next, I added a standalone entrypoint (`src/storybook/runner.py`) for the actor container.

The trick here is doing all the slow Python imports at the very top of the file *before* starting the `/readyz` server. That way, the expensive module initialization happens once when the Golden Snapshot is created, not when a user is waiting.

Once `/readyz` is up, the script loops on `/run/ate/metadata/actor-name`. While it's being snapshotted, it lives in the `ate-golden` namespace. The moment Substrate restores it into the `asp` namespace with a session ID as its name, the loop wakes up and runs the pipeline!

*(From [`src/storybook/runner.py`](https://github.com/arbrown/asp/pull/3/files))*

```python
# Pre-import heavy modules before /readyz responds so the Substrate golden
# snapshot captures a warm Python interpreter with ADK, GenAI, GCS, and WeasyPrint loaded.
import weasyprint
from google import genai
from google.cloud import storage
from storybook.agents.pipeline import run_pipeline

METADATA_DIR = Path("/run/ate/metadata")
GOLDEN_ATESPACE = "ate-golden"


async def _wait_for_actor_assignment() -> str:
    """Wait until the actor is restored out of ate-golden and assigned a session ID."""
    while True:
        atespace = (METADATA_DIR / "actor-atespace").read_text().strip()
        actor_name = (METADATA_DIR / "actor-name").read_text().strip()
        if atespace and atespace != GOLDEN_ATESPACE and actor_name:
            return actor_name
        await asyncio.sleep(0.05)


async def _async_main() -> int:
    _start_readyz_server(port=8080)
    session_id = await _wait_for_actor_assignment()

    # Re-seed PRNG after snapshot restore so cloned actors don't share random state!
    random.seed(os.urandom(16))
    return await execute_session(session_id=session_id)
```

*(Note that `random.seed(os.urandom(16))` call after restore—if you restore multiple actors from the same memory snapshot without re-seeding the random number generator, they all wake up with the exact same RNG state! Ask me how I know 😏.)*

---

### 3. Swap `asyncio.create_task` for `ate.create_actor`

Finally, back in the FastAPI app (`src/storybook/api/routes.py`), creating a session gets a lot simpler. Instead of juggling in-memory queues and background tasks, we save the initial state to GCS and ask Substrate to spin up an actor named after the `session_id`:

*(From [`src/storybook/api/routes.py`](https://github.com/arbrown/asp/pull/3/files))*

```python
# BEFORE: Everything ran inside the FastAPI pod's memory and event loop
# q: asyncio.Queue = asyncio.Queue()
# _sessions[sid] = state
# _queues[sid] = q
# _tasks[sid] = asyncio.create_task(_run(sid, state, q))

# AFTER: Save initial state to GCS and spin up an isolated Substrate Actor
@router.post("/sessions", response_model=SessionResponse, status_code=202)
async def create_session(body: CreateSessionRequest) -> SessionResponse:
    state = PipelineState(config=body.config, started_at=_now())
    sid = state.session_id

    await _persist_session(state)
    await asyncio.to_thread(gcs.save_progress_events, sid, [], False)

    actor_info = await ate.create_actor(
        template=settings.substrate_template,  # "asp-runner"
        atespace=settings.substrate_atespace,  # "asp"
        name=sid,
    )
    log.info("Provisioned Substrate Actor for session %s: %s", sid, actor_info)

    return _to_session_response(state, is_running=True)
```

Under the hood, `ate.create_actor()` is a thin gRPC wrapper in [`src/storybook/substrate/client.py`](https://github.com/arbrown/asp/pull/3/files) that creates the actor from the `asp-runner` template, attaches an egress policy, and resumes it onto a warm worker pod in the `asp-workerpool`.

Because the Golden Snapshot already holds the initialized Python runtime and pre-imported dependencies in memory, all of those heavy startup costs are paid ahead of time when the template is created. Waking up an actor on a warm worker for a new user request happens in just over a second:

```text
Provisioned and resumed Substrate actor asp/b895518f-cb2b-4e78-98a4-ca0893de2067 from template asp-runner in 1108.0 ms (resume_ms=1102.9, state=ACTOR_STATE_RUNNING, worker_pod=asp-workerpool-7bfd485658-nwktc, resume=False)
```

The FastAPI gateway issues the gRPC call, Substrate maps the memory snapshot onto a worker pod, and within ~1,100ms the runner is unpaused and executing the session!

---

## Bonus Wins: Crash Recovery & Multi-Replica Streaming

Once the agent pipeline was decoupled from the web server, a bunch of resilience improvements in [PR #4](https://github.com/arbrown/asp/pull/4) fell into place naturally:

* **Surviving backend rollouts:** Because actors write their progress events to GCS (`GCSProgressSink`) and the SSE stream uses event sequence numbers (`Last-Event-ID`), I scaled `storybook-backend` up to `replicas: 2`. Now if a backend pod restarts while you're watching a book build, the browser automatically reconnects to the other replica without losing its place—and the actor never even notices.
* **Detecting crashed actors & resuming mid-book:** Since Substrate tracks the lifecycle of every actor, the backend can check `ate.get_actor()` while streaming. If an actor crashes mid-run, the UI immediately surfaces a "Resume" button that spins up a brand-new actor, loads the completed stages and spreads from GCS checkpoints, and picks up right where the old actor left off.

---

## What's Next

We now have full OpenTelemetry tracing *and* isolated, snapshot-restored actors for every storybook run! Next up on my list is adding multi-user authentication (so everyone doesn't share one global bookshelf) and breaking out individual tool calls into their own scalable services.

Keep an eye on the [asp](/tags/asp/) tag for the next update, and feel free to check out the code or open an issue on the [GitHub repo](https://github.com/arbrown/asp)!

---

## Helpful Links

* [Agent Substrate on GitHub](https://github.com/agent-substrate/substrate)
* [Agent Storybook Project GitHub Repository](https://github.com/arbrown/asp) ([PR #3: Substrate Actors](https://github.com/arbrown/asp/pull/3), [PR #4: Resilience & Checkpoint Resume](https://github.com/arbrown/asp/pull/4))
* [Part 1: Introducing Agent Storybook Project](/posts/introducing-agent-storybook-project/)
* [Part 2: Observability in the Agent Storybook Project](/posts/observability-asp/)
* [Four Shapes of Agent Workload (and Why Your Cluster Cares)](/posts/four-shapes-of-agent-workload/)
* [GKE Agent Sandbox Documentation](https://cloud.google.com/kubernetes-engine/docs/how-to/agent-sandbox?utm_campaign=CDR_0x145aeba1_default_b559176373&utm_medium=external&utm_source=blog)
