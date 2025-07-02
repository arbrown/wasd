+++
title = "The Factory Must Grow... on a Budget! Factorio on GKE with Spot VMs and Auto-Shutdown"
tags = ["gcp", "gke", "gaming", "factorio", "tutorial"]
date = 2025-07-09T09:37:42-06:00
draft = true
description = "A step-by-step guide to running a cost-effective, scalable, and automated Factorio server on Google Kubernetes Engine (GKE) using Autopilot, Spot VMs, and CronJobs."
images = ["/images/factorio-gke-spot/factoriok8s.png"]
+++

I have been hooked on [Factorio](https://factorio.com/) for years now.  It is my favorite video game, and probably one of the most impressively documented feats of software engineering in gaming.  Check out their long-lived series of technical blog posts: [Factorio Facts Friday](https://factorio.com/blog/).  It's been a real treat learning about the development of the game as it has matured over the years.

One of my favorite parts of the game is playing cooperatively with others.  I've gotten to collaborate with friends on long-running games and some seriously impressive factories over the years.  Sometimes these games were hosted on personal computers, or basement servers, but that means we would rely on one memeber of the group to be online or manage the server even if others wanted to play at a different time.  Doesn't our impressive factory deserve an equally impressive and resilient server? 😉

Maybe it's overkill, but I think so! So I moved our server to the cloud...

## Finding the Right Hosting Solution

Originally, I just manually installed the server software on a simple [GCE](https://cloud.google.com/products/compute) VM.  This got the job done, but it was a hassle managing updates, and most frustratingly, it was a challenge to dial in the right size of server as our factory grew.

Luckily, some kind folks maintain [container images](https://hub.docker.com/r/factoriotools/factorio/) of the Factorio headless server.  This alleviated some of my maintenance woes, but at this point, why not use a modern container-based solution?

My first choice would have been [Cloud Run](https://cloud.google.com/run), a _fantastic_ serverless platform that can scale up and down to zero as needed.  However, Cloud Run is hyper-optimized for HTTP traffic, and Factorio uses [UDP](https://en.wikipedia.org/wiki/User_Datagram_Protocol) for faster, if less reliable communication. (This is the part where I'd tell you a joke about UDP, but you might not get it 🙄)

Instead, I decided to use [GKE autopilot](https://cloud.google.com/kubernetes-engine/docs/concepts/autopilot-overview) to provision just the resources I wanted, when I want them.  This configuration required some initial setup, but lets me scale the size of the server as our factory grows, as well as turn it on and off on demand without having to deal with manually managing VMs, disks, or other parts of the cloud infrastructure.  I also know that I can repeat this process again to start fresh whenever I want to.  So let's dive in and see how I did it.

{{< figure src="/images/factorio-gke-spot/factorio.png" width="400px" title="A successful launch is our goal!" >}}

---

## Factorio GKE Server

### What You'll Need (Prerequisites)

*   A [Google Cloud](https://cloud.google.com/)) project with billing enabled.
*   A Development Environment
    *   The `gcloud` CLI installed and authenticated.
    *   The `kubectl` command-line tool installed.
    *   [Google Cloud Shell Editor](https://ide.cloud.google.com) works great for this, and has the necessary software pre-installed.

*   **Suggested Image:** A screenshot of the `gcloud auth login` flow or the GCP console dashboard.

---

### Building the Foundation - Your GKE Autopilot Cluster

*   First, ensure the GKE API is enabled on your project. If you haven't used GKE in this project before, you'll need to run this command:
    ```bash
    gcloud services enable container.googleapis.com
    ```

I used GKE autopilot for my cluster so that I don't have to manage nodes myself.  This is particularly handy when migrating to a larger VM, or scaling down to zero when we're not playing for a while.

    ```bash
    gcloud container clusters create-auto "factorio-autopilot" \
        --region "us-central1"
    ```

(I used `us-central1` as a geographic compromise with friends on the East coast.  Feel free to pick a [different region](https://cloud.google.com/compute/docs/regions-zones) closer to you)

Once your cluster is up, we'll apply some configuration to it.  This will allow the factorio server to run without needing a persistent server or disk that you have to manage.

{{< figure src="/images/factorio-gke-spot/autopilot.png" width="400px" title="A GKE Autopilot Cluster" >}}
---

### Kubernetes Configuration

#### Config Maps

Since we're running a [generic container](https://hub.docker.com/r/factoriotools/factorio/) of a Factorio server, we need a way to get our configuration on there.  For this, I used Kubernetes Config Maps based on files on my computer (in this case, Cloud Shell).

I created two directories, `config` and `mods` (optional, but lots of fun to customize your server.)

In `config` you'll need your core factorio server configuration files:

    *   `map-gen-settings.json`
    *   `map-settings.json`
    *   `server-adminlist.json`
    *   `server-settings.json`

You can find examples of these in the [factorio-data](https://github.com/wube/factorio-data) repository on GitHub.  Copy the examples there and customize as needed.

In `mods` there is just a single file: `mod-list.json`.

Here are the mods we play with, but you can find lots of others on the [Factorio mods site](https://mods.factorio.com/).

```json

{
  "mods": 
  [
    
    {
      "name": "base",
      "enabled": true
    },
    
    {
      "name": "elevated-rails",
      "enabled": true
    },
    
    {
      "name": "quality",
      "enabled": true
    },
    
    {
      "name": "space-age",
      "enabled": true
    },
    {
      "name": "helmod",
      "enabled": true
    },
    {
      "name": "RateCalculator",
      "enabled": true
    },
    {
      "name": "bobinserters",
      "enabled": true
    },
    {
      "name": "flib",
      "enabled": true
    },
    {
      "name": "Flare Stack",
      "enabled": true
    },
    {
      "name": "PlanetsLib",
      "enabled": true
    },
    {
      "name": "Todo-List",
      "enabled": true
    },
    {
      "name": "VehicleSnap",
      "enabled": true
    }
  ]
}

```

Once your `config` and `mods` directories are set up the way you want them, deploy them as Config Maps:

```bash
kubectl create configmap factorio-configs --from-file=./config/
kubectl create configmap factorio-mods --from-file=./mods/
```
#### Deployment

The core of the deployment is in `factorio-server.yaml`, 

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: factorio-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: factorio-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: factorio-server
  template:
    metadata:
      labels:
        app: factorio-server
    spec:
      nodeSelector:
        cloud.google.com/gke-spot: "true"
      initContainers:
      - name: init-config-mods
        image: busybox
        command:
        - /bin/sh
        - -c
        - |
          # Create config and mods directories if they don't exist
          mkdir -p /factorio/config /factorio/mods

          echo "Copying server-settings.json"
          cp /factorio-configs/server-settings.json /factorio/config/server-settings.json
          echo "Copying map-settings.json"
          cp /factorio-configs/map-settings.json /factorio/config/map-settings.json
          echo "Copying server-settings.json"
          cp /factorio-configs/map-gen-settings.json /factorio/config/map-gen-settings.json
          echo "Copying mod-list.json"
          cp /factorio-mods/mod-list.json /factorio/mods/mod-list.json

          # And for admin list (if you need one)
          echo "Copying server-adminlist.json"
          cp /factorio-configs/server-adminlist.json /factorio/config/server-adminlist.json

          echo "Config and Mods directories initialized."
          exit 0
        volumeMounts:
        - name: factorio-storage
          mountPath: /factorio
        - name: factorio-configs
          mountPath: /factorio-configs
        - name: factorio-mods
          mountPath: /factorio-mods

      - name: init-delete-saves
        image: busybox
        volumeMounts:
        - name: factorio-storage
          mountPath: /factorio
        env:
        - name: GENERATE_NEW_MAP
          valueFrom:
            configMapKeyRef:
              name: flags
              key: GENERATE_NEW_MAP
        command:
        - /bin/sh
        - -c
        - |
          if [ "$GENERATE_NEW_MAP" = "true" ]; then
            echo "GENERATE_NEW_MAP is true. Deleting existing saves to generate a new map..."
            rm -rf /factorio/saves/*
            echo "Save files deleted."
          else
            echo "GENERATE_NEW_MAP is false. Keeping existing save files."
          fi
          exit 0
      containers:
      - name: factorio
        image: factoriotools/factorio:stable
        imagePullPolicy: Always
        resources:
          requests:
            cpu: 2
            memory: 4Gi
        ports:
        - containerPort: 34197
          protocol: UDP
        env:
            - name: UPDATE_MODS_ON_START
              value: 'true'
        volumeMounts:
        - name: factorio-storage
          mountPath: /factorio
      volumes:
      - name: factorio-storage
        persistentVolumeClaim:
          claimName: factorio-data-pvc
      - name: factorio-configs
        configMap:
          name: factorio-configs
      - name: factorio-mods
        configMap:
          name: factorio-mods
---
apiVersion: v1
kind: Service
metadata:
  name: factorio-service
spec:
  type: LoadBalancer
  selector:
    app: factorio-server
  ports:
  - port: 34197
    protocol: UDP
    targetPort: 34197
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: flags
data:
  GENERATE_NEW_MAP: "false"
```

Deploy it using `kubectl`:

```bash
kubectl apply -f factorio-server.yaml
```

This file does a few key things:

 * Set up a `PersistentVolumeClaim` (a disk) to save our game and load our configs
 * Use `initContainer`s to initialize the disk with our configuration, mods, and delete the existing map if necessary (CAREFUL here... it WILL delete your saves if you tell it to with the `GENERATE_NEW_MAP` flag, false by default)
 * Start a pod running the Factorio dedicated server
   * Note the `resources` section of the deployment file.  This tells GKE what kind of resources we want for this server.  I'm using 2 cpu and 4Gi of memory because our game has grown quite large.  You could start lower (even as low as 0.5 cpu and 2Gi memory) and grow from there.
   * If you do change the settings, just re-deploy the `factorio-server.yaml`.  You might want to [scale down and up](#scaling-to-zero) the deployment to force it to use the new resource requests.
   * Importantly, it uses [Spot VMs](https://cloud.google.com/kubernetes-engine/docs/concepts/spot-vms) to save money. This means they can be shut down with no guarantee of availability. However, the Factorio server is fault-tolerant and will gracefully shut down and save your game.  In practice, our server rarely shuts down and when it does, it saves the game just fine.
     * If you want a more reliable (and expensive) server, you can simply remove the `nodeSelector` section or comment it out to use on-demand VMs.
 * Set up a `LoadBalancer` to provide an external IP address you will use to connect to your server


*   **Suggested Images:**
    *   A diagram showing how a `Pod` uses a `PVC` and `ConfigMap`.
    *   A screenshot of your `factorio-server.yaml` in a code editor, with the `nodeSelector` section highlighted.

---


### Connecting to Your Server

Since we set up a `LoadBalancer` in our yaml file, GKE will automatically create an external IP address we can use to connect to our game (and share with friends).  Find yours by inspecting the deployed service:

    ```bash
    kubectl get service factorio-service
    ```
You'll see the external IP address labelled as `EXTERNAL-IP`.  Use this address to connect to your server.

---

### Even the Factory Must Sleep {#scaling-to-zero}

Turning the server on and off is simple. Just tell the Kubernetes cluster to scale it to 1 replica (on) or 0 replicas (off)

```bash
kubectl scale deployment factorio-deployment --replicas=0 # This will turn the server off
kubectl scale deployment factorio-deployment --replicas=1 # This will turn the server on
```

I use these commands to manually turn the server on and off as needed (or occasionally when it gets into a weird or slow state),

But why not automate this?  We're not playing Factorio 24/7 (yet)

`factorio-scaler.yaml`

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: factorio-scaler-sa

---

apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: factorio-scaler-role
rules:
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "patch"]

---

apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: factorio-scaler-binding
subjects:
- kind: ServiceAccount
  name: factorio-scaler-sa
roleRef:
  kind: Role
  name: factorio-scaler-role
  apiGroup: rbac.authorization.k8s.io

---

# CronJob to scale the Factorio server down to 0 replicas
apiVersion: batch/v1
kind: CronJob
metadata:
  name: factorio-shutdown
spec:
  # At 00:00 (midnight) every day in Mountain Time Zone
  schedule: "0 0 * * *"
  timeZone: "America/Denver"
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: factorio-scaler-sa
          containers:
          - name: kubectl-scaler
            image: google/cloud-sdk:latest
            command:
            - "/bin/sh"
            - "-c"
            - "kubectl scale deployment factorio-deployment --replicas=0"
          restartPolicy: OnFailure

---

# CronJob to scale the Factorio server up to 1 replica
apiVersion: batch/v1
kind: CronJob
metadata:
  name: factorio-startup
spec:
  # At 17:00 (5 PM) every day in Mountain Time Zone
  schedule: "0 17 * * *"
  timeZone: "America/Denver"
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: factorio-scaler-sa
          containers:
          - name: kubectl-scaler
            image: google/cloud-sdk:latest
            command:
            - "/bin/sh"
            - "-c"
            - "kubectl scale deployment factorio-deployment --replicas=1"
          restartPolicy: OnFailure
```

```bash
kubectl apply -f factorio-scaler.yaml
```

This (optional) yaml file will turn our server off at midnight every night, and back on at 5 PM.  In practice, we could make it even more aggressive since we usually only play on weekends, but this is a reasonable starting point that basically cuts my server bill in 1/3!

---

## Have Fun!
Hopefully this guide helps you get started with your own GKE deployment of Factorio. It's been a very reliable setup for me and my friends, and I hope it can be that for you too!

### Wishlist
If I could add one feature, it would be the on-demand scaling up and down as we connect and disconnect from the server. I haven't cracked that bit yet, and I figure that the single-digit dollars it would save me are probably not worth spending too much time thinking about.  But if you've got a solution, I'd love to hear it!  Let me know!

{{< figure src="/images/factorio-gke-spot/factoriok8s.png" width="400px" title="Factorio + Kubernetes" >}}
