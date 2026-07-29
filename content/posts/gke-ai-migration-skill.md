+++
title = "Encoding Ops Wisdom: Building the GKE AI Migration Skill"
date = 2026-07-29T12:00:00-06:00
draft = true
categories = ["AI and Machine Learning", "Kubernetes"]
tags = ["gke", "ai", "skills", "kubernetes", "vllm", "gcp", "agents"]
description = "The GKE AI Migration Skill provides a set of rails for Agents to follow to accomplish a specific task, hopefully with fewer tokens."
+++

I recently published [an opinionated skill](https://github.com/google/skills/blob/main/skills/cloud/google-cloud-solution-guided-gke-ai-migration/SKILL.md) for moving AI workloads from a serverles platform (like [Cloud Run](https://cloud.google.com/run?utm_campaign=CDR_0x145aeba1_default_b539587054&utm_medium=external&utm_source=blog) or [Gemini Enterprise Agent Platform](https://cloud.google.com/products/agent-builder?utm_campaign=CDR_0x145aeba1_default_b539587054&utm_medium=external&utm_source=blog)) to [GKE](https://cloud.google.com/kubernetes-engine?utm_campaign=CDR_0x145aeba1_default_b539587054&utm_medium=external&utm_source=blog). So what is it?  What isn't it?  Why a skill when models already know everything? 

# What is the GKE AI Migration Skill?

## Overview
As I mentioned above, the skill is an opinionated set of rails for an agent to follow for a very specific task.  And for that reason, it is as much defined by what it _does not_ do, as what it does.  It pretty explicitly is not for running generic workloads on Kubernetes, or anything to do with fine tuning.

The goal is to leave the user with an inference server running on GKE with their preferred model, and the artifacts they need to manage it.  This is pretty important, we want to leave the user with the ability to update and manage the infrastructure if the agent doesn't get every detail perfect the first time.  But we can sure try to get it right the first time!

## The Workflow
To keep the agent on track and set it up for the success the first time, the skill enforces a 4-phase workflow as follows:

1. Discovery: What currently exists? What are we trying to accomplish? How are we getting the hardware needed for this task ([reservations](https://cloud.google.com/compute/docs/instances/reservations-single-project?utm_campaign=CDR_0x145aeba1_default_b539587054&utm_medium=external&utm_source=blog), [spot](https://cloud.google.com/kubernetes-engine/docs/concepts/spot-vms?utm_campaign=CDR_0x145aeba1_default_b539587054&utm_medium=external&utm_source=blog), [DWS](https://cloud.google.com/kubernetes-engine/docs/concepts/dws?utm_campaign=CDR_0x145aeba1_default_b539587054&utm_medium=external&utm_source=blog))?

2. Design: What hardware do we need? What do will the  overall architecture and individual manifests look like?

3. Implementation: Write the manifests, run `gcloud` commands to create the infrastructure, and `kubectl` commands to apply the manifests to the cluster.  This phase is the longest-running because it includes a lot of waiting for things to spin up, or sometimes jobs like staging a model in a storage bucket.

4. Validation: Make sure the server is up (health checks) and that a quick inference check actually works.

# When Would You Use It?

## Manual Migrations
The skill is great for setting up the infrastructure you need on GKE if you are already running AI inference on a different (more managed) platform on Google Cloud.  It includes pretty tightly scoped use cases where it is useful, with specific off-ramps where it is not applicable (for example, if the user wants to use a managed tool like [Gemini Cloud Assist](https://cloud.google.com/gemini/docs/cloud-assist/overview?utm_campaign=CDR_0x145aeba1_default_b539587054&utm_medium=external&utm_source=blog) to accomplish the goal in a more hands off way.)

## The Golden Path
The skill has a few built-in opinions about how to host inference on GKE.  I baked those in as [bundled manifest templates](https://github.com/google/skills/tree/main/skills/cloud/google-cloud-solution-guided-gke-ai-migration/assets).  Why waste tokens on something if you don't have to? 😉 Of course, you can tell your agent to deviate from these, but they're a great place to start.

The golden path has good presets for models of various sizes, and helps you set up your cluster for future success with features like [Custom Compute Classes](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/about-custom-compute-classes?utm_campaign=CDR_0x145aeba1_default_b539587054&utm_medium=external&utm_source=blog), [Gateway API](https://cloud.google.com/kubernetes-engine/docs/concepts/gateway-api?utm_campaign=CDR_0x145aeba1_default_b539587054&utm_medium=external&utm_source=blog), model staging on [Cloud Storage Buckets](https://cloud.google.com/storage?utm_campaign=CDR_0x145aeba1_default_b539587054&utm_medium=external&utm_source=blog) and more.  It also helps users avoid some pitfalls like hard-coding secrets or picking the wrong metrics for autoscaling.

## Why a skill?
This is the part that I was most skeptical of when I started this project; models these days already seem to know everything, and can ingest up-to-date documentation when they don't.  So what does a skill add? My a-ha moment came when I realized that the skill wasn't about "what" but more about "how".  At one point, I had a hard-coded list of recommendations for VMs and accelerators, and storage options.  But that's not what a skill is useful for.  I had two main goals for the skill - increase chances of task success, and decrease the number of tokens needed to get there.  I believe this skill succeeds in both cases.

### Anecdotal

\#WorksForMe. 

Ok, so that's not great proof, but I did walk through using the skill with various models and configuration requirements.  In my (subjective) experience, it was quite convenient to have the skill do the heavy lifting by asking me questions, writing what I needed, and summarizing the results.  This is something I've done manually a few times, and it was nice to have the agent follow a path that proactively planned and avoided pitfalls I've run in to in the past.

### By the numbers
Using an automated harness, we can actually measure improvements to certain outcomes.  It's an imprecise measurement that attempts to simulate real-world usage. The skill definitely accomplishes the goals of keeping the agent on a specific workflow and gathering the required information to avoid common pitfalls, but most importantly, it does it with fewer tokens than a similar model that did not have the skill.  In tests, the token usage varied from 42% les up to 60% less.  But in all cases, it was able to accomplish more of the goals in fewer tokens than a model without it.

# So What?
All this doesn't matter if the skill doesn't actually work for you in practice.  So if you're looking to move some inference workloads to GKE, or if you're just interested in seeing how it works, give it a try!  Install the [whole set](https://github.com/google/skills/tree/main#installation) of Google skills, or just [this one](https://github.com/google/skills/tree/main/skills/cloud/google-cloud-solution-guided-gke-ai-migration) and give it a spin.  I'd love to hear what you think!