+++
title = "The Factory Must Sleep: On-Demand Factorio Worlds with Agent Substrate on GKE"
date = 2026-11-05T15:15:19-07:00
draft = true
categories = ["Projects", "Kubernetes"]
tags = ["factorio", "gaming", "gke", "gcp", "kubernetes", "agent-substrate", "gvisor", "tutorial"]
description = "Running stateful, UDP-based Factorio game servers as on-demand, auto-suspending Agent Substrate Actors on GKE with multi-world multiplexing and scale-to-zero Spot nodes."
+++

## Revisiting the Original Factorio on GKE Wishlist
Last year I wrote about one of my favorite games of all time, Factorio, and hosting a server on GKE. It was an okay solution, but it was far from optimal, but still, it served me and my friends over the course of a long campaign, and I have since retired that GKE-hosted solution. Now that Factorio 2.0 is out, my friends and I are looking to dust off our Factorio server and start a new campaign. That got me thinking: how would I revisit this solution now with even better tools and support of this kind of idea? Substrate is a system for managing workloads on Kubernetes that works great on GKE. It was originally designed with AI agents in mind, but it works great at managing any stateful workload.

Factorio seems like a great candidate because it's really important to hold the game server state, but we want it to be able to hibernate and resume very quickly when users are connecting and disconnecting. So that got me thinking, is this a good fit, or am I just reaching here? 

## Can a Factorio Server Be an Agent Substrate Actor?
So what exactly is a substrate actor? At its core, an actor is any workload that we want to turn on and off and have it resume exactly where it left off very quickly. This seems great for a multiplayer game server that is used only intermittently. We just give Substrate a golden image of a server to boot and have it write changes to a local disk that gets shipped off to cloud storage whenever the server is not running, along with the whole memory state, and we stop using the compute. 

On the surface, this seems like a great idea and even opens up the possibility of hosting multiple Factorio servers on the same hardware. As long as only one of them is running at a time, we can just keep the rest of them hibernated and not bother paying for compute costs for a server that nobody's playing on. 

However, there's a catch. The atenet-router that Substrate uses is an L7 router and designed for TCP traffic only. That pretty much means just HTTP connections, and Factorio wants a UDP connection to each of its clients. So that became the first big obstacle that I had to overcome. 

## Architecture Overview: Bridging Stateful UDP into an HTTP-Only Sandbox

So I needed a way to get data into a Substrate actor and then turn it into the UDP that that server expects. It turns out there's really only one choice: WebSockets.

So the core of the solution is to have an always-on gateway running in the GKE cluster that listens for UDP connections from Factorio game clients and then passes them to the appropriate Substrate actor, which has a Factorio server running listening for UDP, but also a sidecar workload that takes a WebSocket connection from the gateway and converts those packets back to UDP for the game server. 

<!-- Insert a friendly diagram of how this solution works here.  -->

## Multi-World Multiplexing on a Single Worker Pod

This opened up a lot of interesting possibilities. Rather than just having a single world that we have to manage and maintain, we can now have multiple worlds and connect to whichever one we want only by changing the port number that we're connecting to, and let the gateway route the traffic and wake up the actor as appropriate.

The rest of the worlds just sleep silently in cloud storage for pennies or less a month of storage costs, and only wake up whenever they need to, nearly instantly. 

<!--  Insert gif of factorio servers waking up instantly on the command line and in-game -->

## Zero-Touch Factorio Version Upgrades When Steam Auto-Updates

One of the frustrations I had with the previous solution was keeping the client versions in sync with the server version. The great folks at Factorio Tools put out images for the server, but sometimes it gets ahead or behind by one version and it's just not quite right.



To solve that, we have the gateway proactively go out and look for new versions and rebuild images for the substrate actors every 30 minutes. And also, we have it check the Factorio version in real time as clients are connecting in the UDP packets.

<!-- Add a code snippet showig this feature and link to the github repo -->

This allows the system to go and build a new image on demand when it needs it and migrate the save over to that new image. This takes a few seconds each time it happens, but it's basically instant for anyone connecting the next time. 

## Scale-to-Zero / Spot Nodes
Finally, I decided to tackle a true Scale-to-Zero using substrate and GKE. We can suspend the substrate actor, but we're still paying for the node to run in the background just in case we need to wake up an actor very quickly. In order to prevent this, we're setting a timer in the gateway again, checking for UDP traffic that looks like Factorio traffic, and if we don't see any for 10 minutes, it's safe to turn off the pod. That means the next time someone connects, they're going to have to wait a couple of minutes for a spot node to be found and provisioned, but it beats paying for an always-on node. 

<!-- Add code snippet here -->

This is a huge optimization, and along with a few other things like rightsizing the Postgres node that substrate uses to store its state, I'm able to save a lot of money over my previous system. 

## Proving Stateful Persistence Across World Swaps

With all that said, it's time to show off our fresh, multiplexed, multi-world Factorio server running on GKE and Substrate.

As you can see, the first time I connect, it creates a new world, and we can run around in it and make changes, and then go and connect to a different world that is now running on the same hardware and do the same.

When we connect back to the original world, we can see that the changes persist and it is resumed from cloud storage nearly instantaneously. Once I disconnect, after 10 minutes, the node gets removed from the cluster and I stop paying for it. 

# Link to repo

So check out the code sample for how I did this. The repo is live on my GitHub, and I'll hope to keep it up to date as we go through this next Factorio campaign. 