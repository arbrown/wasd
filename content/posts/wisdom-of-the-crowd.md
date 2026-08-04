+++
title = "The Wisdom of the Crowd: How to Steer AI Agents for Better Results"
date = 2026-08-04T12:00:00-06:00
draft = false
categories = ["AI and Machine Learning", "Software Engineering"]
tags = ["ai", "agents", "llm", "coding", "prompt-engineering", "wisdom-of-the-crowd"]
description = "AI code generation can be like the wisdom of the crowd.  Defining your goal and grounding your prompt in real technical context is essential for building novel software with an agent."
+++

![](/images/wisdom-of-the-crowd/ox.jpeg)

Have you heard the classic story about a county fair where people were guessing the weight of an ox? Nobody was exactly right, but somehow the average of all the guesses was remarkably accurate!  Francis Galton wrote an [impeccably-titled article](https://www.nature.com/articles/075450a0) about the guessing game in Nature in 1907, but you can read the basics [on Wikipedia](https://en.wikipedia.org/wiki/Wisdom_of_the_crowd).

Across 800 guesses, the average was 1,207 pounds, within one percent of the actual weight of 1,198 pounds!  Pretty impressive.  It turns out that collectively, we are a lot smarter than individually.

## Large Language Models: The Ultimate Crowd

It may not be exactly true, but I think of LLMs as the ultimate crowd; they have collected a large part of the "wisdom" of humanity. But sometimes it's still hard to get an actual good result out of them. I think this is especially true when using them to write code.

It's nice to have the collected wisdom of every line of open source code at my fingertips, but I don't want to write those lines of code.  They're already there, and they solve problems that have already been solved.  This is a common criticism of agent-assisted coding, that it can only regurgitate what has already been done.  But I think it's capable of assisting you in creating something net new (and useful) — if you find a way to connect the abstract ideas in your head with the wisdom contained in the model.

---

## A Sweet Experiment: Counting Swedish Fish

Allow me to illustrate with a fun example. I was at a family reunion last month and we were playing a fun game: guessing the number of Swedish Fish in a jar. Being the contrarian AI user that I am, I just snapped a couple pictures of the jar and asked [Gemini](https://cloud.google.com/products/gemini?utm_campaign=CDR_0x145aeba1_default_b542634334&utm_medium=external&utm_source=blog) to figure it out. It came up with 68, which I was sure was way too low, but I was committed to the bit and wrote it for my guess anyway.

![I told Gemini they were standard-sized Swedish Fish and let it guess](/images/wisdom-of-the-crowd/swedishfish1.jpg)

The final count ended up being 194 (I think it was originally closer to 200 but Grandpa sneaked some when doing the original count...), and my cute niece ended up winning!  Still, I wanted to see if the wisdom of the crowds would beat Gemini in this head to head.  I added up everyone's scores (including Gemini's super low guess, and my nephew's guess of 700 something...) and the average ended up being 189.8 — just slightly closer than the winning guess!  

Still, I wondered why Gemini was so wrong.  I re-read its original thought process and decided it just didn't really have a good sense of the actual size of Swedish Fish and the jar in the picture.  So I snapped a new picture with a plastic fork for scale (in a new conversation to avoid any context pollution).

![No bananas on hand, so a fork for scale](/images/wisdom-of-the-crowd/swedishfish2.jpg)

This time, Gemini's logic was rock solid.  It provided a small range for the guess: 190-195.  *Any number in that range would have won the contest!*

---

## Grounding the Crowd

So what happened here? The key difference between my two prompts was that in the second, I grounded the model with some real-world information that it might not have had.  In this case, it was the fork that grounded the image in a real-world dimension. (This is conceptually similar to how [grounding works in Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview?utm_campaign=CDR_0x145aeba1_default_b542634334&utm_medium=external&utm_source=blog), tying model reasoning to verifiable external context to prevent hallucination.)

But how does this apply to producing code with agents and LLMs? Let's use a concrete example from this very blog. I set this blog up using Hugo on [Google Cloud Storage](https://cloud.google.com/storage?utm_campaign=CDR_0x145aeba1_default_b542634334&utm_medium=external&utm_source=blog) ([detailed here](/posts/hugo-blog-google-cloud-storage/)) a couple years ago, and I've wanted to fix a couple specific things, but never felt like diving back in to the internals of Hugo to do it. I understand the architecture and a possible solution space, but I could never justify the time it would take to fix it.  I have been using Antigravity for a lot of my work, and I could have prompted Antigravity with something simple: 'Fix this blog to have a better layout with no dependencies' but that is so broad, there's nothing to ground the results to what I actually wanted.  I _might_ have ended up happy, but it probably wouldn't get to the heart of what I actually wanted.

Instead, I gave the agent very specific instructions based on my knowledge of the underlying structure:

```
Take a look at this blog.  I want to overhaul it in a couple of ways:

1) Remove dependencies on any hugo themes, but keep the overall visual look / structure
2) Improve the visual look of the blog where appropriate
3) Change the way I use "figures" to a simpler markdown based template for images with captions
4) Improve the navigation of the blog within the taxonomy structure (which can be updated as needed)

All of which should happen while keeping all the blog posts and their canonical URLs intact.  Similarly, the main structure/content of each post needs to stay the same.
```

Instructions 1 and 3 were very specific instructions based on my knowledge of the underlying structure of the site.  I could not write the code to fix it off the top of my head (believe me, I've fought with Hugo plenty in the past, and I just didn't have the appetite for it this time), but I knew the general area where the solution was.

The agent was able to [one-shot a solution that did what I wanted!](https://github.com/arbrown/wasd/pull/13).  Obviously the theming still could be improved aesthetically, but I'm actually super happy with the technical results.  And I wouldn't have been able to get those results by myself, or with a less grounded prompt that didn't direct the agent to the real code base in front of it.

---

## Conclusion: Shepherding the Wisdom of the Crowd

![Goal + Grounding = Success](/images/wisdom-of-the-crowd/grounding.jpeg)

So, the formula here is simple: When prompting an agent, you are ultimately responsible for:

1. Defining the goal *and* importantly,
2. Connecting the solution to something real based on your understanding of the underlying code.

I think it's easy to spit out goals with no real grounding, but in my experience, the best results come from connecting the goal to something real. Those two pieces together have helped me get more focused and reliable results out of LLM code generation than simple prompting alone.

---

## What's Next

* Learn how to ground models with your own data using [Vertex AI Grounding](https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/overview?utm_campaign=CDR_0x145aeba1_default_b542634334&utm_medium=external&utm_source=blog).
* Explore AI-powered development tools with [Gemini Code Assist](https://cloud.google.com/gemini/docs/codeassist/overview?utm_campaign=CDR_0x145aeba1_default_b542634334&utm_medium=external&utm_source=blog).
* Read more developer guides on the [Google Cloud Blog](https://cloud.google.com/blog?utm_campaign=CDR_0x145aeba1_default_b542634334&utm_medium=external&utm_source=blog).