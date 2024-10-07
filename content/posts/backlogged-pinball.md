+++
title =  "Backlogged Pinball"
tags = ["pinball", "gcp", "next"]
date = "2024-10-07"
+++

# Backlogged Pinball

Earlier this year, I built a pinball-based demo for [Cloud Next 2024](https://cloud.withgoogle.com/next).  I'm preparing some posts about specific aspects of it, but I wanted to post a brief overview first.  Backlogged is a physical pinball machine that has custom software to connect it to Cloud
services.


{{< figure src="/images/backlogged-pinball/pinball_suit.jpeg" width="600px" title="A conference attendee playing the Backlogged pinball demo" >}}

## Demo overview



Backlogged pinball was designed to demonstrate specific Cloud products and
benefits in a fun, interactive way.

{{< figure src="/images/backlogged-pinball/backlogged-arch.png" width="600px" title="The pinball machine sends pubsub events to the cloud which are processed by Cloud Run and displayed on an interactive web app. Users can deploy their own Cloud Run service to process events and send pubsub messages back to the machine." >}}

In its current iteration, it highlights:

1.  [Cloud Pub/Sub](https://cloud.google.com/pubsub): Two-way Pub/Sub communication with a novel source of events
    (the pinball machine)
1.  [Cloud Run](https://cloud.google.com/run/) / [Cloud Run Functions](https://cloud.google.com/functions): Both serverless platforms process pinball
    events in real-time using different subscription mechanics (push vs. pull).
1.  [Cloud Run Deployments](https://cloud.google.com/run/docs/deploying): Users can update a Cloud Run service in 20 seconds
    which has an immediate effect on the current pinball game
1.  [Cloud Firestore](https://firebase.google.com/docs/firestore): Firestore pushes all events to a live web-app. The round
    trip from:

    Pinball machine **->** Pub/Sub **->** Cloud Functions **->** Firestore **->**
    Web App

    takes about 200 ms, and could probably be further optimized.

## The machine

{{< figure src="/images/backlogged-pinball/pinball_wide.jpeg" width="600px" title="The demo featured live-updating high scores and current game status" >}}

The actual physical pinball machine is a
[Multimorphic P3](https://www.multimorphic.com/p3-pinball-platform/). In short,
this is a modular, programmable pinball machine with 2 large HD displays built
in.

Multimorphic also distributes
[an SDK](https://www.multimorphic.com/p3-pinball-platform/3rd-party-developers/)
to facilitate creating new pinball games. The SDK works with an older version of the Unity game engine to run games on the physical pinball hardware using Mono as its runtime.

Thus, Backlogged is an application that uses the Unity game engine and the
Multimorphic SDK to handle interfacing with the hardware (including displays).

## What's next?
{{< figure src="/images/backlogged-pinball/pinball_interview.jpeg" width="600px" title="Drew showing off the backlogged demo" >}}

The pinball machine currently resides in a Google office in Salt Lake City.  Whenever I am able to, I plan on adding new features/functionality to the demo (with a major focus on Cloud code).

I am currently working on a game advisor that uses [llama3.1-70b](https://ollama.com/library/llama3.1:70b) to grade your game and recommend more optimal strategies. 

{{< figure src="/images/backlogged-pinball/genkit.png" width="600px" title="A sample game analysis from the prototype" >}}

In the future, I'd love to add more features like:

*  Using AI to play the game and optimize for a high score
*  Connecting two machines to the cloud to demonstrate more interactive
    scenarios
    *   This scenario could include two machines in different locations to
        highlight network latency and resiliency
*  More interactive features beyond simple messages and emoji.

Any ideas for new features?  I'd love to hear them!


## The code

All backend code (web app, Cloud Run Functions, sample reaction services, etc...) is
in
[this GitHub repository](https://github.com/GoogleCloudPlatform/backlogged-pinball-backend/).

The pinball game code is heavily dependent on the Multimorphic SDK and sample
game skeleton code (which provides core game functionality). This code is
distributed without an explicit license, and so we cannot release the game code
publicly (yet).

I did post [one sample method](https://github.com/GoogleCloudPlatform/backlogged-pinball-backend/blob/main/sample-code/csharp-pubsub/pubsub-post.cs) from the C# code to demonstrate what my custom Pub/Sub client looked like.

## Stay Tuned
I'd like to keep writing about this pinball project in the coming months.  Hopefully in this space you'll see a few more posts about the project, including:

* Pinball as an example of developing for the cloud on legacy hardware
* Rolling your own authentication using JSON Web Tokens
* Pub/Sub as a means of 'inverting' client/cloud development
* Using LLMs to gain insights into hyper-specific log information
* Using Gemini to dip my toes into new programming paradigms (Unity, animations, particle effects, etc...)

Is there anything else you'd like to hear about?  Please let me know!