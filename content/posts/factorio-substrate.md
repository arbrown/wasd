+++
title = "The Factory Must Sleep: On-Demand Factorio Worlds with Agent Substrate on GKE"
date = 2026-11-05T15:15:19-07:00
draft = true
categories = ["Projects", "Kubernetes"]
tags = ["factorio", "gaming", "gke", "gcp", "kubernetes", "agent-substrate", "gvisor", "tutorial"]
description = "Running stateful, UDP-based Factorio game servers as on-demand, auto-suspending Agent Substrate Actors on GKE with multi-world multiplexing and scale-to-zero Spot nodes."
+++

## Revisiting the Original Factorio on GKE Wishlist

Last year I wrote about one of my favorite games of all time, [Factorio](https://factorio.com/), and [hosting a dedicated server on GKE](/posts/factorio-gke-spot/). It was an okay solution, though far from optimal—it relied on `CronJob`s to turn the server on at 5 PM and off at midnight—but it served me and my friends over the course of a long campaign before I retired that cluster. At the very end of that post, I left a wishlist item:

> *If I could add one feature, it would be the on-demand scaling up and down as we connect and disconnect from the server. I haven't cracked that bit yet, and I figure that the single-digit dollars it would save me are probably not worth spending too much time thinking about. But if you've got a solution, I'd love to hear it!*

Now that my friends and I are looking to dust off our Factorio server and start a new 2.x campaign, that got me thinking: how would I revisit this solution today with better tools? Lately I've been spending a lot of time with **[Agent Substrate](https://github.com/agent-substrate/substrate)** on [GKE](https://cloud.google.com/kubernetes-engine?utm_campaign=CDR_0x145aeba1_default_b423920039&utm_medium=external&utm_source=blog) (like in my [Agent Storybook Project](/posts/substrate-asp/)). Substrate was originally designed with AI agents, code interpreters, and RL environments in mind, but at its core, it's great at managing *any* stateful workload that needs fast checkpointing and restore.

Factorio seemed like a great candidate because holding onto the game server state is critical, and we want it to hibernate when nobody is online and resume very quickly when players connect. So that got me thinking: is this actually a good fit, or am I just reaching here?

## Can a Factorio Server Be an Agent Substrate Actor?

So what exactly is a Substrate Actor? At its core, an Actor is a sandboxed workload that we can turn on and off on demand and have it resume right where it left off in about a second. For an intermittently used multiplayer game server, that sounds ideal. We give Substrate an `ActorTemplate` ([`manifests/factorio-template.yaml.tmpl`](https://github.com/arbrown/factorio-substrate/blob/main/manifests/factorio-template.yaml.tmpl)) that builds a pre-warmed **Golden Snapshot** of the server, have it write save files to a persistent `durableDir` volume (`/factorio`), and configure `SNAPSHOT_CONTENT_SCOPE_FULL` so both the `/factorio` directory and the entire [gVisor](https://cloud.google.com/kubernetes-engine/docs/concepts/sandbox-pods?utm_campaign=CDR_0x145aeba1_default_b423920039&utm_medium=external&utm_source=blog) sandbox memory state get shipped off to Cloud Storage whenever the server is suspended—letting us stop paying for compute when nobody is playing.

On the surface, this even opens up the possibility of hosting *multiple* Factorio worlds on the exact same hardware. As long as only one world is active at a time, we can keep the rest of them hibernated in Cloud Storage and multiplex a single worker pod across all of them.

However, there's a catch. The `atenet-router` component that Substrate uses to route traffic to Actors is an L7 Envoy proxy designed for TCP traffic only—specifically HTTP, gRPC, and WebSockets routed via the `ate-target-actor` header. Even worse for a game server, Substrate's worker network setup (`ateomnet`) installs an `nftables` firewall rule inside the worker pod that explicitly **drops all non-DNS UDP packets (`dport != 53`)** entering or leaving the Actor's network namespace. And Factorio speaks pure UDP to all of its clients. That became the first big obstacle I had to overcome.

## Architecture Overview: Bridging Stateful UDP into an HTTP-Only Sandbox

I needed a way to get player UDP datagrams through Substrate's L7 router into an Actor—triggering an automatic wake-up if the Actor was suspended—and then turn them back into the UDP packets the Factorio server expects, without modifying a single line of Agent Substrate. It turns out there's really only one choice: **WebSockets**.

The core of the solution is a pair of lightweight Go binaries:

1. **An always-on UDP Gateway ([`cmd/factorio-gateway`](https://github.com/arbrown/factorio-substrate/blob/main/cmd/factorio-gateway/main.go)):** Listens for UDP connections from Factorio game clients on node `hostPort`s (`34197`, `34198`, ...), filters out single-packet internet port scanners, buffers the client's initial handshake packet, and opens a WebSocket tunnel through `atenet-router` (`/ws`) with the header `ate-target-actor: factorio/<world>`. Just dialing `atenet-router` with that header causes Substrate to automatically wake (`ResumeActor`) the target Actor!
2. **An in-Actor Sidecar ([`cmd/factorio-sidecar`](https://github.com/arbrown/factorio-substrate/blob/main/cmd/factorio-sidecar/main.go)):** Runs inside the gVisor Actor sandbox right alongside the unmodified `factoriotools/factorio` server container. Because both containers share the same network namespace (`127.0.0.1`), the sidecar accepts the WebSocket stream on port `80` and forwards the unpacked UDP datagrams over loopback (`127.0.0.1:34197/UDP`) to Factorio—completely bypassing the external `nftables` UDP drop rule!

To make sure Factorio can tell multiple connected players apart, every WebSocket binary frame prefixes the raw UDP payload with the player's source `IP:port` ([`internal/tunnelproto/frame.go`](https://github.com/arbrown/factorio-substrate/blob/main/internal/tunnelproto/frame.go#L57-L89)), and the sidecar allocates a dedicated local ephemeral UDP socket (`127.0.0.1:<port_i>`) per player ([`cmd/factorio-sidecar/main.go`](https://github.com/arbrown/factorio-substrate/blob/main/cmd/factorio-sidecar/main.go#L460-L504)). And because `atenet-router` enforces a 5-minute Envoy route timeout, the gateway transparently rotates the WebSocket connection every 4 minutes while the sidecar keeps those local UDP sockets open across rotations.

<!-- Insert a friendly diagram of how this solution works here.  -->

## Multi-World Multiplexing on a Single Worker Pod

Once the UDP-over-WebSocket bridge was working, it opened up a lot of fun possibilities. Rather than having a single Factorio world tied to a dedicated Persistent Disk, we can host multiple worlds at the same IP address and switch between them just by changing the port number we connect to in the Factorio client:

* **`:34197` (`factorio/world-vanilla`)** — Super-Dense Megabase (`--preset rich-resources`, 600% rich ores, peaceful mode, no cliffs)
* **`:34198` (`factorio/world-deathworld`)** — Deathworld Biter Swarm (`--preset death-world`, 400% enemy bases, armed spawn bunker)
* **`:34199` (`factorio/world-railworld`)** — Rail World Express (`--preset rail-world`, giant distant ore patches, train starter depot)
* **`:34200` (`factorio/world-ribbonworld`)** — 64-Tile Ribbon World (`--preset ribbon-world`, 64-tile tall strip with visible void edges)

All four worlds share a **single** 1-replica `WorkerPool` pod ([`manifests/factorio-substrate.yaml.tmpl`](https://github.com/arbrown/factorio-substrate/blob/main/manifests/factorio-substrate.yaml.tmpl#L50-L80)). When you connect to `:34198` while `world-vanilla` is sitting idle on the worker pod, the gateway automatically calls `/prepare-suspend` on `world-vanilla`'s sidecar (which runs an RCON `/server-save` over `127.0.0.1:27015`), calls `SuspendActor` to ship `world-vanilla`'s memory and save file to Cloud Storage, and wakes `world-deathworld` into the *exact same* worker pod in a couple of seconds ([`preemptIdleWorldsFor` in `cmd/factorio-gateway/main.go`](https://github.com/arbrown/factorio-substrate/blob/main/cmd/factorio-gateway/main.go#L896-L937)).

Meanwhile, the sleeping worlds sit silently in Cloud Storage as compressed `zstd` archives (~46 MB per world, or ~250 MB total for all four worlds plus the Golden Snapshot)—which works out to **less than a penny a month** in storage costs—and wake back up nearly instantly whenever someone connects.

<!--  Insert gif of factorio servers waking up instantly on the command line and in-game -->

## Zero-Touch Factorio Version Upgrades When Steam Auto-Updates

One of the frustrations I had with my previous setup—and one I ran into immediately while testing this post—was keeping client and server versions in sync. Steam auto-updates players' desktop Factorio clients as soon as Wube marks a release `stable` (for example, `2.1.23` to `2.1.24`), and Factorio's multiplayer handshake strictly enforces exact `major.minor.patch` version matching. The great folks at [factoriotools/factorio](https://hub.docker.com/r/factoriotools/factorio/) publish updated server images quickly, but in Agent Substrate, `ActorTemplate`s pin container images by immutable `@sha256:...` digest, and a full memory snapshot (`SNAPSHOT_CONTENT_SCOPE_FULL`) is tied to the exact binary that created it.

To solve that without ever losing our saves or manually redeploying manifests, I built a dual auto-upgrade system into `factorio-gateway`:

1. **Proactive background polling (every 30 minutes):** While all worlds are `SUSPENDED`, the gateway checks Docker Hub (`factoriotools/factorio:stable`). When the upstream `@sha256:...` digest changes, the gateway replaces the `factorio/factorio-server` `ActorTemplate` in-place over gRPC (`GetActorTemplate` → `DeleteActorTemplate` → `CreateActorTemplate`) so Substrate pre-builds a fresh Golden Snapshot in the background before anyone even launches the game.
2. **Reactive client version sniffing on UDP `ConnectionRequest`:** What if Steam updates a player's client 5 minutes after a patch drops, before the 30-minute poll fires? Every Factorio client's first UDP `ConnectionRequest` (`0x02`) packet embeds the client's 4-byte Factorio version (`major.minor.patch.0`) right in the header. The gateway inspects those bytes in real time as clients connect, and if a client advertises a newer version than the server, it immediately checks Docker Hub and upgrades the `ActorTemplate` on the spot!

*(From [`internal/tunnelproto/frame.go`](https://github.com/arbrown/factorio-substrate/blob/main/internal/tunnelproto/frame.go#L112-L119) and [`cmd/factorio-gateway/main.go`](https://github.com/arbrown/factorio-substrate/blob/main/cmd/factorio-gateway/main.go#L1173-L1229))*

```go
// ParseClientVersion extracts the 3-component Factorio version ("2.x.y") from
// an inbound client UDP ConnectionRequest (0x02) packet, if present.
func ParseClientVersion(pkt []byte) (string, bool) {
	if len(pkt) < 12 || len(pkt) > 64 || (pkt[0]&FactorioMessageTypeMask) != FactorioMsgConnectionRequest {
		return "", false
	}
	return scanFactorio2VersionBytes(pkt[1:])
}
```

```go
func (gw *Gateway) imageUpdateLoop(ctx context.Context) {
	ticker := time.NewTicker(gw.cfg.ImagePollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if gw.allWorldsSuspended(nil) {
				_, _ = gw.CheckAndUpgradeTemplate(ctx, "proactive-background-poll", nil)
			}
		}
	}
}

func (gw *Gateway) maybeUpgradeBeforeWake(ctx context.Context, wakingWorld *WorldTunnel) {
	if gw.tmplMgr == nil || gw.registry == nil {
		return
	}
	wakingWorld.mu.Lock()
	clientVer := wakingWorld.lastClientVersion
	wakingWorld.mu.Unlock()
	if clientVer == "" {
		return
	}

	gw.metaMu.Lock()
	srvVer := gw.knownServerVersion
	gw.metaMu.Unlock()

	// Only block wake-up if we know the current server version and the connecting
	// client is strictly newer (e.g. Steam updated the client from 2.1.24 to 2.1.25).
	if srvVer != "" && tunnelproto.CompareVersions(clientVer, srvVer) > 0 {
		slog.Info("Client version is newer than server version; triggering pre-wake Docker Hub upgrade",
			"actor", wakingWorld.spec.ActorRef(),
			"client_version", clientVer,
			"known_server_version", srvVer,
		)
		_, _ = gw.CheckAndUpgradeTemplate(ctx, "reactive-client-version:"+clientVer, wakingWorld)
	}
}
```

Best of all, we **don't** delete or recreate the `Actor`s (`world-vanilla`, `world-deathworld`, etc.) when the template is replaced ([`CheckAndUpgradeTemplate` in `cmd/factorio-gateway/main.go`](https://github.com/arbrown/factorio-substrate/blob/main/cmd/factorio-gateway/main.go#L1258-L1374)). When an existing Actor wakes up and Substrate sees that its `ActorTemplate` UID changed (`TemplateReplaced`), Substrate automatically switches **that single wake-up** from `SNAPSHOT_SCOPE_FULL` (memory restore) to `SNAPSHOT_SCOPE_DATA` (cold boot mounting only `/factorio/saves/_autosave1.zip` from `durable-dir.tar.zstd`). Factorio boots the new binary, migrates the save file in-place, and every subsequent suspend and wake goes right back to ~1.5-second full memory restores!

## Scale-to-Zero / Spot Nodes

Finally, I decided to tackle a true Scale-to-Zero setup using Substrate and GKE. Suspending the Substrate Actor frees up the worker pod, but if we leave a warm `WorkerPool` pod (`replicas: 1`) running 24/7, we're still paying for a worker node to sit in the background waiting to wake an Actor up quickly.

To eliminate that idle worker cost, `factorio-gateway` runs a two-stage idle reaper ([`idleReaperLoop` in `cmd/factorio-gateway/main.go`](https://github.com/arbrown/factorio-substrate/blob/main/cmd/factorio-gateway/main.go#L1376-L1443)):

1. **10-minute Actor idle timeout (`--idle-timeout=10m`):** While a world is `RUNNING`, the gateway tracks **outbound Factorio UDP replies** (`lastOutboundReply`—so random internet UDP scanners can't keep the server awake!). After 10 minutes with zero Factorio replies, the gateway triggers an RCON `/server-save` and calls `SuspendActor` to freeze the world to Cloud Storage.
2. **2-minute WorkerPool scale-to-zero (`--workerpool-idle-timeout=2m`):** Once *all* worlds have been `SUSPENDED` for 2 minutes, the gateway calls the in-cluster Kubernetes API to patch `WorkerPool/factorio` from `replicas: 1` down to `replicas: 0` ([`maybeScaleDownWorkerPool`](https://github.com/arbrown/factorio-substrate/blob/main/cmd/factorio-gateway/main.go#L858-L894)). With the worker pod gone (and the `atelet` DaemonSet marked `safe-to-evict: "true"`), GKE's Cluster Autoscaler (`--autoscaling-profile=optimize-utilization`) removes the `n2-standard-2` Spot worker node completely.

The next time someone sends a Factorio UDP connection request, `activateWorld` scales `WorkerPool/factorio` back from `0` to `1` ([`ensureWorkerPoolScaledUp`](https://github.com/arbrown/factorio-substrate/blob/main/cmd/factorio-gateway/main.go#L822-L856)). That first cold connect takes ~60–90 seconds while GKE provisions a fresh Spot node, and once the node is up, switching between any of the four worlds is back to ~1.5 seconds.

*(From [`cmd/factorio-gateway/main.go`](https://github.com/arbrown/factorio-substrate/blob/main/cmd/factorio-gateway/main.go#L822-L894) and [`idleReaperLoop`](https://github.com/arbrown/factorio-substrate/blob/main/cmd/factorio-gateway/main.go#L1376-L1443))*

```go
func (gw *Gateway) ensureWorkerPoolScaledUp(ctx context.Context, reason string) {
	if gw.scaler == nil {
		return
	}
	gw.scaleMu.Lock()
	defer gw.scaleMu.Unlock()

	cur, err := gw.scaler.GetWorkerPoolStatus(ctx, gw.cfg.WorkerPoolNamespace, gw.cfg.WorkerPoolName)
	if err != nil {
		slog.Warn("Failed to get WorkerPool status", "namespace", gw.cfg.WorkerPoolNamespace, "name", gw.cfg.WorkerPoolName, "err", err)
		return
	}
	gw.workerPoolStatus = cur
	gw.lastScaleCheck = time.Now()

	wanted := int32(gw.cfg.MaxActiveWorlds)
	if cur.SpecReplicas < wanted {
		slog.Info("Scaling up WorkerPool from 0 -> 1",
			"namespace", gw.cfg.WorkerPoolNamespace,
			"name", gw.cfg.WorkerPoolName,
			"old_spec_replicas", cur.SpecReplicas,
			"new_spec_replicas", wanted,
			"reason", reason,
		)
		updated, err := gw.scaler.SetWorkerPoolReplicas(ctx, gw.cfg.WorkerPoolNamespace, gw.cfg.WorkerPoolName, wanted)
		if err != nil {
			slog.Error("Failed to scale up WorkerPool", "namespace", gw.cfg.WorkerPoolNamespace, "name", gw.cfg.WorkerPoolName, "err", err)
			return
		}
		gw.scaleUpCount.Add(1)
		gw.workerPoolStatus = updated
		gw.lastScaleTime = time.Now()
		gw.lastScaleReason = reason
	}
}
```

```go
if st == WorldStateRunning {
	allSuspended = false
	idleDuration := now.Sub(lastActivity)
	if idleDuration >= gw.cfg.IdleTimeout {
		slog.Info("No Factorio UDP traffic for idle timeout; suspending Actor",
			"actor", wt.spec.ActorRef(),
			"idle_duration", idleDuration.Round(time.Second),
		)
		_ = wt.suspendNow(ctx, "idle-timeout")
	}
} else if st != WorldStateSuspended {
	allSuspended = false
} else if changeTime.After(latestSuspendedChange) {
	latestSuspendedChange = changeTime
}

// Once all worlds are SUSPENDED for --workerpool-idle-timeout (default 2m),
// scale WorkerPool/factorio down to 0 so GKE Cluster Autoscaler removes the Spot node.
if allSuspended && !gw.upgradeInProgress.Load() {
	suspendedFor := now.Sub(latestSuspendedChange)
	if suspendedFor >= gw.cfg.WorkerPoolIdleTimeout {
		gw.maybeScaleDownWorkerPool(ctx, now, suspendedFor)
	}
}
```

Combined with a few other infrastructure tweaks in [`hack/setup-cost-optimized-pools.sh`](https://github.com/arbrown/factorio-substrate/blob/main/hack/setup-cost-optimized-pools.sh)—running the lightweight Substrate control plane and `factorio-gateway` on a single `e2-medium` Spot node (`control-pool`), rightsizing Substrate's `postgres-0` volume from `500Gi` down to `10Gi`, and replacing the cloud `LoadBalancer` Service with direct node `hostPort`s (`udp:34197-34297`)—the entire cluster idles at under $10/month and only costs ~$0.03/hour while we're actively playing!

> *Disclaimer: I work at Google, and the views and opinions expressed in this post are my own and do not represent official Google communications. The pricing numbers below are rough, hypothetical back-of-the-envelope estimates for this personal project (`us-central1` rates at the time of writing) and are not an official price quote.*

### Hypothetical Monthly Cost Breakdown (`us-central1`)

Here is how the progression of optimizations stacks up against the baseline 24/7 development cluster configuration:

| Architecture Option | Monthly Idle Cost | Active Hourly Rate | ~20 hrs/mo Total | Key Attributes |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline 24/7 Cluster**<br>(2x `c3-standard-4`, 500Gi PVC, LoadBalancer) | **~$305.00/mo** | ~$0.42/hr | ~$305.00/mo | Always-on nodes, no scale-to-zero, over-provisioned default Postgres disk. |
| **Scale-to-Zero Worker (On-Demand + LoadBalancer)**<br>(1x `e2-medium`, 10Gi PVC, LoadBalancer, `0 -> 1` worker) | **~$44.70/mo** | ~$0.10/hr | ~$46.70/mo | Scale-to-zero worker, but still paying for an on-demand control node and LoadBalancer forwarding rules. |
| **Scale-to-Zero Worker (On-Demand Control + `hostPort`)**<br>(1x `e2-medium`, 10Gi PVC, node `hostPort`, `0 -> 1` worker) | **~$26.50/mo** | ~$0.10/hr | ~$28.50/mo | Eliminates the LoadBalancer (`$0` ingress for up to 101 ports), control VM still on-demand. |
| **Spot Control + Spot Worker + `hostPort` *(Current Setup)***<br>(1x `e2-medium` Spot, 10Gi PVC, `hostPort`, `0 -> 1` `n2-standard-2` Spot) | **~$9.51/mo** | **~$0.029/hr** | **~$10.09/mo** | **~97% cost reduction.** Spot VMs on both pools, zero-cost `hostPort` ingress, right-sized 10Gi disk. |

And breaking down that current configuration item by item:

| Component | Configuration | Est. Monthly Cost (USD) |
| :--- | :--- | :--- |
| **Control Plane Node (`control-pool`)** | 1x `e2-medium` Spot VM (runs `ate-system`, `postgres-0`, & `factorio-gateway`) | ~$7.30 |
| **Control Plane Boot Disk** | 30 GB `pd-standard` | ~$1.20 |
| **Substrate Postgres Volume (`data-postgres-0`)** | 10 GB `pd-balanced` PVC (right-sized from 500 GiB; uses ~43 MiB) | ~$1.00 |
| **UDP Ingress (`factorio-gateway`)** | Node `hostPort` + VPC Firewall (`udp:34197-34297`, up to 101 worlds) | $0.00 |
| **GKE Zonal Cluster Management** | Free tier eligible (1 zonal cluster per billing account) | $0.00 |
| **Cloud Storage (`gs://<BUCKET>`)** | ~250 MB `zstd` snapshots (Golden template + 4 Factorio worlds @ ~46 MB each) | ~$0.01 |
| **Idle Worker Node (`substrate-worker-pool`)** | `0` replicas when no players are connected | $0.00 |
| **Baseline Idle Total** | **All 4 worlds hibernated in GCS** | **~$9.51 / month** |
| **Active Gameplay Add-On** | 1x `n2-standard-2` Spot VM + 50 GB `pd-balanced` (~$0.029/hr while playing) | **+$0.29** (10 hrs/mo)<br>**+$0.73** (25 hrs/mo)<br>**+$1.45** (50 hrs/mo) |

## Proving Stateful Persistence Across World Swaps

With all that said, it's time to show off our fresh, multiplexed, multi-world Factorio server running on GKE and Agent Substrate.

As you can see, the first time I connect to `:34197`, it wakes up `world-vanilla` (and generates its custom map on the very first boot), and we can run around in it and make changes—or even run `make prove` ([`hack/prove-e2e.sh`](https://github.com/arbrown/factorio-substrate/blob/main/hack/prove-e2e.sh)) to drop a steel chest with 42 red science packs right at spawn—and then disconnect and connect to `:34198` (`world-deathworld`), which saves and evicts the first world to Cloud Storage and loads the second world onto the exact same worker pod.

When we connect back to the original world on `:34197`, our changes persist and the world resumes from Cloud Storage nearly instantaneously (~1.5 seconds). And once everyone disconnects, after 10 minutes of silence the Actor suspends to GCS, 2 minutes later the `WorkerPool` scales to `0`, and GKE removes the Spot worker node from the cluster so I stop paying for it.

## Check Out the Code

All of the source code, Kubernetes and Substrate manifests, cost-optimization scripts, and end-to-end verification tools are live on GitHub at **[`arbrown/factorio-substrate`](https://github.com/arbrown/factorio-substrate)**:

* [`cmd/factorio-gateway/main.go`](https://github.com/arbrown/factorio-substrate/blob/main/cmd/factorio-gateway/main.go) — Always-on UDP-to-WebSocket ingress gateway, single-worker world multiplexer, two-stage idle reaper, `WorkerPool` autoscaler, and proactive/reactive `ActorTemplate` upgrader.
* [`cmd/factorio-sidecar/main.go`](https://github.com/arbrown/factorio-substrate/blob/main/cmd/factorio-sidecar/main.go) & [`worlds.go`](https://github.com/arbrown/factorio-substrate/blob/main/cmd/factorio-sidecar/worlds.go) — In-actor loopback UDP bridge, RCON `/prepare-suspend` save barrier, and per-world map presets.
* [`internal/tunnelproto/frame.go`](https://github.com/arbrown/factorio-substrate/blob/main/internal/tunnelproto/frame.go) — WebSocket binary wire framing and Factorio 2.x UDP header/version parser.
* [`manifests/factorio-template.yaml.tmpl`](https://github.com/arbrown/factorio-substrate/blob/main/manifests/factorio-template.yaml.tmpl) & [`manifests/factorio-substrate.yaml.tmpl`](https://github.com/arbrown/factorio-substrate/blob/main/manifests/factorio-substrate.yaml.tmpl) — Substrate `ActorTemplate`, `WorkerPool`, and `factorio-gateway` manifests.
* [`TRADEOFFS.md`](https://github.com/arbrown/factorio-substrate/blob/main/TRADEOFFS.md) — Deep-dive notes on all 14 architectural trade-offs and gVisor/networking gotchas solved along the way.

I'll keep the repo up to date as we go through this next Factorio campaign—and now the factory can finally sleep when we're not playing, and wake itself back up the moment we connect! 