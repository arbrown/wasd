+++
title = "Introducing Agent Storybook Project"
date = 2026-09-08T11:43:00-06:00
draft = true
categories = ["AI and Machine Learning", "Kubernetes"]
tags = ["gke", "kubernetes", "ai", "agents", "google-adk", "gemini", "vertex-ai", "asp"]
description = "Meet the Agent Storybook Project: an open-source multi-agent pipeline generating illustrated children's books on GKE, and an honest look at what breaks when agent workloads meet production infrastructure."
+++

![Run an agent without GKE? I would prefer not to](/images/introducing-agent-storybook-project/bartleby.png)

A couple months ago I had an idea for a novel (hehe 😏) project: I wanted to build an agent that would adapt a classic Russian book into a children's book for my friend.  I made a quick prototype and generated a few sample books, and then shelved 😏 the project to focus on other work projects.

However, I think this "Agent Storybook Project" is actually a perfect testbed for all kinds of advanced agent features on [GKE](https://cloud.google.com/kubernetes-engine?utm_campaign=CDR_0x145aeba1_default_b558681667&utm_medium=external&utm_source=blog), which I already want to write about anyway.

So, I released [the code](https://github.com/arbrown/asp) on GitHub and I'm planning on using this space to add occasional updates to the project to show what it looks like to run a non-trivial agent on GKE

---

## What is the Agent Storybook Project?

Agent Storybook Project is a moderately over-engineered agent that takes public domain literature and adapts it for a different audience (in this case, children).

![Simple UI for kicking off an agent task](/images/introducing-agent-storybook-project/asp.png)

However, to make it more relevant to my work in Developer Advocacy for GKE, I've added multiple agents that work together to produce a finished project.  There is a main coordinator agent that kicks off a workflow of different agents that do work and verify each other's work in a loop to produce the finished proejct.

![](/images/introducing-agent-storybook-project/architecture_simplified.png)

---

## Why GKE for Agents?

I've [written about this](/posts/why-gke-agents/) before, but in short, there are actually some pretty good reasons to run agents on GKE instead of a serverless platform like [Cloud Run](https://cloud.google.com/run?utm_campaign=CDR_0x145aeba1_default_b558681667&utm_medium=external&utm_source=blog):

1. **Persistent, Long-Running Workloads:** The work  here is long-running and kind of messy.  It doesn't fit neatly into a single HTTP response.  Additionally, sometimes thing fails and we want the agent to pick things back up in case of a failure (or in my case, multiple bugfixes in the middle of a pipeline!)
2. **Multi-Agent Coordination:** This is actually more than one agent all working together.  Right now, the ppipeline is super simple, but I am planning on making it more complex to refelect real-life workloads in the future.  If you've got a swarm of agents all talking to each other, you might want to make sure they're hanging out together already!  Kuberenetes (and GKE) make sure that happens!
3. **Advanced Agent Identity & Isolation:** Not all agents should have full access to everything. GKE gives us tools (like [Workload Identity](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity?utm_campaign=CDR_0x145aeba1_default_b558681667&utm_medium=external&utm_source=blog)) to make sure sub-agents can run with the least privelege necessary to do just their job.
---

## Current Status: The Architecture

The storybook project is built with  [ADK](https://github.com/google/adk-python) (Agent Development Kit), and powered by various [Gemini models](https://cloud.google.com/products/gemini?utm_campaign=CDR_0x145aeba1_default_b558681667&utm_medium=external&utm_source=blog) for each of the different tasks (crafting text, creating illustrations, and vreviewing illustrations for accuracy and adherence to the style guide).

```
Literature Acquisition (Gutenberg)
      │
      ▼
Two-Pass Adaptation & Character Bible Generation (Gemini 3.1 Pro / 3.5 Flash)
      │
      ▼
Spread Layout & Planning (Verso/Recto, Aspect Ratios)
      │
      ▼
Parallel Illustration & Multimodal Vision Validation (Imagen + Gemini Vision)
      │
      ▼
PDF Composition (WeasyPrint + Jinja2)
```

The app runs on [**GKE Autopilot**](https://cloud.google.com/kubernetes-engine/docs/concepts/autopilot-overview?utm_campaign=CDR_0x145aeba1_default_b558681667&utm_medium=external&utm_source=blog) and has:
* **Frontend:** A basic UI with a form to kick off a job and a live progress bar to watch it go.
* **Backend:** A FastAPI service managing the ADK pipeline orchestration.
* **Storage & State:** **rqlite** StatefulSet for session metadata, and [**Google Cloud Storage (GCS)**](https://cloud.google.com/storage?utm_campaign=CDR_0x145aeba1_default_b558681667&utm_medium=external&utm_source=blog) for session-scoped assets and checkpoints.

---

## Future Improvements

This whole thing was just a fun proof of concept, but it turns out that makes it great for demoing the kinds of improvements you'd actually want in a production-ready agent.  Here are some of the ideas I'll be writing about in the next couple of months:

1. Isolating agent workloads from each other with [sandboxes](https://cloud.google.com/kubernetes-engine/docs/concepts/sandbox-pods?utm_campaign=CDR_0x145aeba1_default_b558681667&utm_medium=external&utm_source=blog).
   * Currently, each process runs in the same pod, the same container, and all together.  They're not too process intensive, but I'd like to isolate them from each other.  Additionally, the more power I give them (maybe they want to write custom code to layout a page or something!) the more I _need_ these agents to be isolated from each other.  At first, I think I'll move them to isolated Kuberenetes jobs, and then eventually to isolated sandbox pods (like [GKE Agent Sandbox](https://cloud.google.com/kubernetes-engine/docs/how-to/agent-sandbox?utm_campaign=CDR_0x145aeba1_default_b558681667&utm_medium=external&utm_source=blog)) that can spin up and down on demand using [Agent Substrate](https://github.com/agent-substrate/substrate).
2. Multiple Users! Right now there is no authentication on the frontend, meaning all users share everything!  That's not how people use agents, and it would be great to add a realistic auth story (such as using [Identity-Aware Proxy](https://cloud.google.com/iap/docs/enabling-kubernetes-howto?utm_campaign=CDR_0x145aeba1_default_b558681667&utm_medium=external&utm_source=blog)) to the project to show how multi-user agents work on GKE.

3. Separate Tools from the agent runtime - Along with isolating the agent processes from the backend API, it would also be great to separate some of the tool calls (like fetching text or building a PDF) from the rest of the workload.  In reality, these are different problems and scale at different rates than the rest of the workload, and how to do that on GKE is a pattern worth diving into.

---

## A sample

It's only right to include a sample of the work this agent does!  [Here is an adaptation](/downloads/introducing-agent-storybook-project/bartleby.pdf) I had it create of the short story [Bartleby the Scrivener](https://www.gutenberg.org/ebooks/11231) by Herman Melville.

Clearly the agent has some learning to do with typography and page layout, but we'll get to that in future versions of the project!  More to come 😊!

So keep an eye on this space in the coming weeks!  I'll keep all the posts organized under the [asp](/tags/asp/) tag.  Let me know if you have any suggestions for improving the project by filing an issue on the  [repo](https://github.com/arbrown/asp)!


---

## Helpful Resources & Tutorials

If you're looking to dive deeper into running AI agents on GKE, check out these tutorials and guides:

*   [Codelab: Deploying Secure AI Agents on GKE](https://codelabs.developers.google.com/codelabs/gke/ai-agents-on-gke?utm_campaign=CDR_0x145aeba1_default_b558681667&utm_medium=external&utm_source=blog)
*   [GKE Agent Sandbox How-To Guide](https://cloud.google.com/kubernetes-engine/docs/how-to/agent-sandbox?utm_campaign=CDR_0x145aeba1_default_b558681667&utm_medium=external&utm_source=blog)
*   [Google Cloud Blog: Agentic AI on Kubernetes and GKE](https://cloud.google.com/blog/products/containers-kubernetes/agentic-ai-on-kubernetes-and-gke?utm_campaign=CDR_0x145aeba1_default_b558681667&utm_medium=external&utm_source=blog)

