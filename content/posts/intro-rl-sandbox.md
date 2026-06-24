+++
title = 'Introduction to Distributed RL Sandboxing on GKE'
date = 2026-06-24T12:00:00-07:00
draft = false
tags = ["gke", "rl", "sandbox", "agents"]
categories = ["Tutorials"]
+++

# Introduction to Distributed RL Sandboxing on GKE

Reinforcement Learning (RL) is the cornerstone of modern AI training.  Rather than train a model to produce an expected output, we verify if it has achieved a particular _outcome_. This is particularly important when building models for use in agents that make decisions and take actions rather than just produce text. 

## What is RL and Why Sandbox?

In many RL workloads (like the one I'll discuss below), we're training a model for use in a coding workflow where it needs to write code and use tools (like `git` and `grep`) to accomplish a task like fixing a bug.

But training these agents presents a thorny problem. What if, while it's learning, it makes a mistake? Like a _really bad_ mistake. Say an agent decided that running `rm -rf /` was the best way to fix a bug? *No file system, no problem!* That might be an extreme example, but you need a way to isolate the actions of these agents from your real infrastructure—especially when that infrastructure has expensive accelerators attached to it.

{{< figure src="/images/intro-rl-sandbox/dog-sandbox.png" width="300px" title="A chaotic agent making a mess in the sandbox!" >}}

A sandbox provides an isolated, secure environment where the agent can freely act without risking the host system or real-world data. It allows us to safely train agents on tasks that might otherwise be destructive or have unintended consequences.

## The Codelab: High-Performance Distributed RL Sandbox

To help with this on [GKE](https://cloud.google.com/kubernetes-engine?utm_campaign=CDR_0x145aeba1_default_b527527409&utm_medium=external&utm_source=blog), I've put together a basic introductory codelab: [High-Performance Distributed RL Sandbox](https://codelabs.developers.google.com/codelabs/gke/high-performance-distributed-rl-sandbox?utm_campaign=CDR_0x145aeba1_default_b527527409&utm_medium=external&utm_source=blog#0).

Under the hood, GKE Agent Sandbox uses [gVisor](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/sandbox-pods?utm_campaign=CDR_0x145aeba1_default_b527527409&utm_medium=external&utm_source=blog) to intercept system calls and provide strong, kernel-level isolation. This ensures that even if an agent runs malicious or buggy code, it remains locked inside a secure playpen, unable to compromise the host node, the cluster, or your precious cloud credentials.

This codelab focuses on the *how* of setting up a basic, distributed sandboxing environment using [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine?utm_campaign=CDR_0x145aeba1_default_b527527409&utm_medium=external&utm_source=blog) and [Agent Sandbox](https://cloud.google.com/kubernetes-engine/docs/concepts/agent-sandbox?utm_campaign=CDR_0x145aeba1_default_b527527409&utm_medium=external&utm_source=blog). 

Here's a high-level view of what we build in the codelab:

{{< figure src="/images/intro-rl-sandbox/gke-rl-architecture.png" width="600px" title="GKE RL Sandbox Architecture" >}}

We'll set up a GKE cluster with a special "Warm Pool" of sandboxes designed for a specific task. In the example, our sandbox is configured to enable the agent to fix a small bug in a Python application. That means our sandbox has the right source code and dependencies already installed. In larger jobs, you could have different sandbox environments for different tasks, and intelligently (and quickly) route your agent-in-training to the right pod.

To keep execution latency under 200ms and avoid choking the Kubernetes control plane with constant pod creation, the orchestration plane (Ray) talks to a central **Sandbox Router** that manages and assigns pre-warmed sandboxes. 

Importantly, the Python training job never addresses an individual sandbox pod by its IP or name. Instead, it connects directly to the central router service and dynamically requests a sandbox matching its specific requirements.

Here is a simplified preview of how this interaction works in Python:

```python
from sandbox_client import SandboxClient

# Connect to the central router service
client = SandboxClient(base_url="http://sandbox-router.default.svc.cluster.local:8080")

# Request a secure, pre-warmed sandbox by specifying the environment template and warm pool
sandbox = client.create_sandbox(
    template="swe-bench-django",
    warmpool="swe-bench-django-warmpool"
)

# Safely execute the agent's code or tests within the isolated sandbox
result = sandbox.run_command("python manage.py test")
print(f"Exit code: {result.exit_code}")
print(f"Output: {result.stdout}")
```

## What's Next?

This codelab is just the beginning. It sets the foundational infrastructure. In future posts, I'll dive deeper into more advanced RL concepts and explore sophisticated sandboxing techniques on GKE, like building a larger image library for your Sandbox Warm Pool or doing multi-turn training where the learning really starts to happen!

{{< figure src="/images/intro-rl-sandbox/advanced-topics.png" width="300px" title="Stay tuned to this space for more!" >}}

Ready to build your sandbox? Head over to the [Codelab](https://codelabs.developers.google.com/codelabs/gke/high-performance-distributed-rl-sandbox?utm_campaign=CDR_0x145aeba1_default_b527527409&utm_medium=external&utm_source=blog#0) and get started!

