+++
title = "MCP Went Stateless - Why Security Stays in the Pod"
date = 2026-08-13T12:00:00-06:00
draft = true
categories = ["AI and Machine Learning", "Kubernetes"]
tags = ["mcp", "kubernetes", "gke", "security", "ai", "agents", "llm"]
description = "The Model Context Protocol went stateless in 2026-07-28, bringing header-based routing and simpler Kubernetes deployments. Here is why your authorization and security boundaries must still stay inside the pod."
+++

![](/images/mcp-went-stateless-why-security-stays-in-the-pod/header.png)

Two weeks ago, Model Context Protocol shipped its [`2026-07-28`](https://modelcontextprotocol.io/docs/2026-07-28/getting-started/intro) specification revision.

The biggest updates in this release are: no more stateful `initialize` handshake, `Mcp-Session-Id` headers have been retired, and every request is now completely self-contained.  

Naturally, most of the conversation around this release focused on scaling: no sessions mean MCP servers can scale horizontally or even scale to zero with ease. That's great news if you're building serverless tools. But if you're deploying and managing MCP workloads on Kubernetes (or GKE), there are a couple other shifts that matter even more.

---

## Deployments Just Got (Wonderfully) Boring

Previously, running MCP servers behind Kubernetes services meant wrestling with state. You had to worry about session affinity, sticky routing, shared session caches, or consistent hashing schemes to ensure an agent's multi-turn requests landed on the same replica. 

With the update, that operational headache vanishes. An MCP server is now an ordinary, garden-variety stateless HTTP workload. You write a standard `Deployment`, front it with a regular `Service`, and attach a standard Horizontal Pod Autoscaler.

> **Quick Migration Tip:** If you're carrying forward Kubernetes manifests from the `2025-11-25` era, delete `sessionAffinity: ClientIP` from your Service definitions. Leaving it in now hurts even traffic distribution across your pods while buying you absolutely nothing in return.

---

## The Quiet Superpower: L7 Header Legibility

While stateless scaling grabbed the headlines, the second major protocol enhancement is much quieter—and arguably more powerful for platform engineers.

Under the Streamable HTTP transport, requests now require explicit `Mcp-Method` and `Mcp-Name` headers. On top of that, MCP tools can promote individual arguments into custom `Mcp-Param-*` headers. 

Why is this a big deal? Because any L7 proxy or ingress gateway can now inspect and route specific agent tool calls by name using headers, something that previously required a specialized tool like [AgentGateway](https://github.com/agentgateway/agentgateway) to parse the JSON request body.

```http
POST /mcp HTTP/1.1
Host: tools.internal.example
Mcp-Method: tools/call
Mcp-Name: deploy
Mcp-Param-Env: production
Content-Type: application/json
```

So what are some of the cool things this buys you if you're hosting an MCP server in your cluster?

1. **Native Per-Tool Rate Limiting:** You can enforce per-tool quotas by matching headers directly on a Gateway API `HTTPRoute` with an attached rate-limit policy. No custom scripts, no lookups, and no external rate-limit filters digging `params.name` out of the JSON payload.
2. **Better Observability:** You can log tool calls at the infrastructure level using headers without having to log full JSON request bodies.
3. **Smarter Autoscaling:** Autoscalers like KEDA's HTTP add-on can intercept and evaluate traffic based on headers alone, ensuring lightweight discovery calls don't unnecessarily wake heavy, scaled-to-zero tool pools.

But what's one big gotcha that these headers don't actually solve?

---

## Resist the Temptation: Your Gateway Is Still Not A Security Boundary

Once you realize that your ingress gateway can see `Mcp-Name: drop_table` or `Mcp-Param-table-name: customers` right on the wire, there is an immediate temptation: 

*Why not just block dangerous tool calls or unauthorized operations directly at the gateway?*

![Not gonna do it](/images/mcp-went-stateless-why-security-stays-in-the-pod/prudent.gif)

It seems so easy and straightforward (I actually had to resist the temptation to include a code sample here — I didn't want to imply it was something you _should_ copy/paste!) but it doesn't actually do what you think.

Evaluating whether an agent *should* be allowed to execute a given tool call can still only be done in the pod. 

---

## The Downgrade Loophole: Why Edge Filtering Fails

The SDK running in your pod *does* cross-check headers against the parsed JSON-RPC body to ensure a client can't fool a gateway by sending `Mcp-Name: safe_tool` in the header while requesting `drop_table` in the payload.

The real catch is the **backward compatibility window**. If a client declares an older protocol version or omits the version header altogether, the SDK skips header validation entirely:

```go
func validateMcpHeaders(header http.Header, msg jsonrpc.Message, toolLookup func(string) (*serverTool, bool)) error {
	protocolVersion := header.Get(protocolVersionHeader)
	if protocolVersion == "" || protocolVersion < minVersionForStandardHeaders {
		return nil
	}
```
[Go MCP SDK Header Validation](https://github.com/modelcontextprotocol/go-sdk/blob/v1.7.0/mcp/streamable_headers.go#L353-L357)

If a client connects as a legacy client—omitting the version header and `_meta` protocol parameters—it skips header generation entirely. It sends a standard JSON-RPC payload with zero `Mcp-*` headers.

If you try and filter on those missing headers, the request will silently fall through to your default route!

### Defense in Depth

You could just reject unversioned traffic at the gateway with an [HTTP `426 Upgrade Required`](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/426). But that would break legitimate legacy clients right in the middle of the year-long deprecation window.

Instead, you could structure your traffic routing so that downgrading provides no privilege advantage:

1. **Partition Traffic at the Route:** Route header-bearing traffic to your full tool pool, and route legacy/unheadered traffic to an isolated, read-only pool. When the legacy path has fewer permissions than the modern path, the incentive for a downgrade attack disappears.
2. **Enforce Authorization in the Pod:** Perform your actual token validation, parameter inspection, and policy decisions inside the application container where the full JSON-RPC payload is parsed.

Gateway rules might look something like this:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: mcp-by-protocol
spec:
  parentRefs: [{ name: mcp-gw }]
  rules:
    - name: modern
      matches:
        - path: { type: PathPrefix, value: /mcp }
          headers: [{ name: Mcp-Protocol-Version, value: "2026-07-28" }]
      backendRefs: [{ name: mcp-general, port: 8080 }]
    # Anything omitting the version lands on a
    # read-only backend that authorizes on the parsed body, where the absence
    # of headers costs nothing.
    - name: legacy
      matches:
        - path: { type: PathPrefix, value: /mcp }
      backendRefs: [{ name: mcp-legacy-readonly, port: 8080 }]
```

But in either case, you still need to do full authorization and validation checks inside the pod!

---

## But What About Managed Gateways?

Google Cloud's [Agent Gateway](https://docs.cloud.google.com/gemini-enterprise-agent-platform/govern/gateways/agent-gateway-overview) (not to be confused with the open-source [AgentGateway](https://github.com/agentgateway/agentgateway)) is worth knowing about, because it solves a problem HTTPRoute structurally can't: agent egress.

Kubernetes Gateway API is designed for routing incoming (North-South) traffic to your MCP server pods. But it has no awareness of outbound tool calls an agent makes, nor does it govern what an agent can reach outside the cluster. Furthermore, while GKE Workload Identity assigns IAM credentials at the *Pod* level, it doesn't distinguish between individual agent personas or workflows running inside it.

Agent Gateway bridges this gap. It provides both **Client-to-Agent** (ingress) and **Agent-to-Anywhere** (egress) governance: giving each agent running on [Gemini Enterprise Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform) a distinct, trackable IAM principal and enforcing fine-grained access policies and security guardrails on every outbound request.

---

## The Golden Rule

So the new MCP specification gives us incredible routing and observability tools for MCP servers running in GKE, but architectural boundaries still apply:

> **Build routing and observability at the gateway. Build control in the pod.**

---

## What's Next

There's lots of other stuff to explore with these MCP changes.  In particular, I'm excited about the possibilities for dense swarms of MCP servers that truly scale to zero and only start up when they're needed.  That's something I'll definitely be demonstrating more in the coming weeks.  Stay tuned!
