+++
title =  "Fine-Tuning Mixtral-8x7B"
tags = ["gcp", "slurm", "HPC", "tutorial", "ai-workloads", "LLM"]
date = "2025-10-10"
categories = ["Tutorials"]
+++

# Fine-Tuning Mixtral-8x7B: A Deep Dive into a Multi-Node Slurm and FSDP Workflow
{{< figure src="/images/mixtral/banner.png" width="800px" >}}

Training or fine-tuning large language models like Mixtral-8x7B is a significant challenge due to their large size. Sometimes, to make these models fit on available hardware, you can _quantize_ a model and reduce the model's weights to a lower precision. While this can be an effective strategy for inference, it can introduce a loss of quality when fine-tuning a model for a specific task.

I wanted to find a reproducible approach for fine-tuning without resorting to quantization, and I settled on FSDP.  In this tutorial, we'll fine-tune the Mixtral model without compromising on precision, leveraging PyTorch's **Fully Sharded Data Parallel (FSDP)** to distribute the model, gradients, and optimizer states across multiple GPUs and nodes. This allows us to train the model in its full `bfloat16` precision.

We will also use **Parameter-Efficient Fine-Tuning (PEFT)**, specifically LoRA, to train only a small number of new, trainable parameters. I plan on using this approach of FSDP to break up a model and LoRA to add new layers to train ever-bigger models in the future..

You've probably seen tutorials like this before, but the real interesting part of this guide comes in the specific tech stack I'm using to accomplish this goal on Google Cloud:
*   **Slurm:** For orchestrating distributed jobs across our cluster.
*   **High-Speed Interconnect:** The `a4-highgpu-8g` instances are connected with a high-speed network fabric optimized for machine learning workloads.
*   **Managed Lustre:** A parallel file system to provide high-performance shared storage for our project.
*   **Cluster Toolkit:** The open source toolkit I use to tie together all of the pieces and manage my infrastructure.


{{< figure src="/images/mixtral/slug.png" width="300px" >}}
---

### A Note on Resource Availability

This tutorial uses `a4-highgpu-8g` virtual machines, which are in high demand. While [reserving capacity](https://cloud.google.com/ai-hypercomputer/docs/consumption-models) provides the best chance of success, you might also be able to leverage [Spot VMs](https://cloud.google.com/spot-vms) as an option for small-scale testing.

The availability of both A4 VMs and Managed Lustre varies by region. Before you begin, check the official documentation for the latest information:
*   **GPU Availability:** [Regions and Zones](https://cloud.google.com/compute/docs/regions-zones)
*   **Managed Lustre Availability:** [Locations](https://cloud.google.com/managed-lustre/docs/locations)

---

### Costs

This tutorial uses billable components of Google Cloud, including:

*   [**Compute Engine:**](https://cloud.google.com/compute/vm-instance-pricing?hl=en#accelerator-optimized) For the Slurm login and compute node VMs (including GPUs).
*   [**Cloud Storage:**](https://cloud.google.com/storage/pricing) To store the Terraform state for the cluster deployment.
*   [**Cloud Logging:**](https://cloud.google.com/stackdriver/pricing) For monitoring and logging.
*   [**Managed Lustre**](https://cloud.google.com/products/managed-lustre) For a scaled, parallel file system used by the cluster

Use the [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator) to generate a cost estimate based on your projected usage.

---

### Before you begin

This tutorial assumes a working knowledge of Google Cloud. The following steps will get your environment set up with the necessary permissions and tools.

1.  **Set up your environment and project.** In the Google Cloud console, activate [Cloud Shell](https://cloud.google.com/shell/docs/using-cloud-shell) and set your project ID.
    ```shell
    export PROJECT_ID="your-gcp-project-id"
    gcloud config set project $PROJECT_ID
    ```
2.  **Ensure billing is enabled** for your Cloud project.
3.  **Enable the required APIs.**
    ```shell
    gcloud services enable compute.googleapis.com file.googleapis.com logging.googleapis.com cloudresourcemanager.googleapis.com servicenetworking.googleapis.com lustre.googleapis.com
    ```
4.  **Grant the necessary IAM roles to your user account.**
    ```shell
    export USER_EMAIL=$(gcloud config get-value account)
    for role in compute.admin iam.serviceAccountUser file.editor storage.admin serviceusage.serviceUsageAdmin; do
      gcloud projects add-iam-policy-binding $PROJECT_ID --member="user:$USER_EMAIL" --role="roles/$role"
    done
    ```
5.  **Enable the default Compute Engine service account and grant it the Editor role.**
    ```shell
    export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
    export SA_EMAIL="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
    gcloud iam service-accounts enable $SA_EMAIL --project=$PROJECT_ID
    gcloud projects add-iam-policy-binding $PROJECT_ID \
      --member="serviceAccount:$SA_EMAIL" \
      --role=roles/editor
    ```
6.  **Create local authentication credentials.**
    ```shell
    gcloud auth application-default login
    ```
7.  **Enable OS Login for your project.**
    ```shell
    gcloud compute project-info add-metadata --metadata=enable-oslogin=TRUE
    ```
8.  [Sign in to or create a Hugging Face account](https://huggingface.co/login) and get a `read` access token. We'll use this token to download the Mixtral model weights later.
9.  [Install dependencies for using Cluster Toolkit](https://cloud.google.com/cluster-toolkit/docs/setup/install-dependencies) if you do not already have Terraform and Packer.

---

### Setting the Stage: Deploying the Slurm Cluster

We will use the Google Cloud Cluster Toolkit to simplify the complex process of deploying a production-grade, high-performance Slurm cluster with all the necessary networking and shared storage configured.

1.  Clone the Cluster Toolkit repository.
    ```shell
    git clone https://github.com/GoogleCloudPlatform/cluster-toolkit.git
    ```
2.  Create a Cloud Storage bucket for our deployment's Terraform state.
    ```shell
    export BUCKET_NAME="your-unique-bucket-name"
    gcloud storage buckets create gs://${BUCKET_NAME}
    ```
3.  Go to the cloned `cluster-toolkit` directory and build the `gcluster` binary if it's your first time using it.
    ```shell
    cd cluster-toolkit
    make

4.  Open `examples/machine-learning/a4-highgpu-8g/a4high-slurm-deployment.yaml` and edit the configuration for your project and resources.
    ```yaml
    terraform_backend_defaults:
      type: gcs
      configuration:
        bucket: BUCKET_NAME
    vars:
      deployment_name: DEPLOYMENT_NAME
      project_id: PROJECT_ID
      region: REGION
      zone: ZONE
      a4h_cluster_size: 2
      a4h_reservation_name: RESERVATION_NAME
    ```
    **Note:** - This is where you would switch to a spot or DWS flex deployment if you chose those options.  Just comment/uncomment the appropriate lines.
5.  **Modify the Cluster Blueprint for Lustre.** Before deploying, open the `examples/machine-learning/a4-highgpu-8g/a4high-slurm-blueprint.yaml` file. We will configure a Managed Lustre file system to provide a scalable, high-performance shared file system for our project. Make the following changes, following the instructions in the blueprint file:
    *   Comment out the `filestore_homefs` module.
    *   Uncomment the `lustrefs` and `private-service-access` modules.
    *   In the `vars` block, find `slurm_vars` and set `install_managed_lustre` to `true`.
6.  Now, deploy the cluster from the base `cluster-toolkit` directory.
    ```shell
    ./gcluster deploy -d examples/machine-learning/a4-highgpu-8g/a4high-slurm-deployment.yaml examples/machine-learning/a4-highgpu-8g/a4high-slurm-blueprint.yaml --auto-approve
    ```
    The `gcluster deploy` command is a two-phase process. The first phase builds a custom VM image, which can take up to 35 minutes. The second phase deploys the cluster using that image.

---

### Preparing the Workload: An Optimized Approach

With the cluster provisioning underway, let's look at the scripts that will run our fine-tuning job. This workflow is designed to optimize for consistency and performance by using a few key techniques. You'll do these steps on your local machine before uploading them to the cluster.

1.  **Create `install_environment.sh`:** This script creates a portable Python virtual environment. This "one-and-done" approach ensures that the exact same software environment is used across all nodes. A key step is building `flash-attn` from source, which guarantees compatibility with the specific NVIDIA B200 GPUs on our machines and the ever-changing set of libraries that support them. This can take a long time, but luckily, we can make the job _really_ parallel to spread the work across lots of cores.
    ```shell
    #!/bin/bash
    # This script sets a reliable environment for FSDP training.
    # It is meant to be run on a compute node.
    set -e

    # --- 1. Create the Python virtual environment ---
    VENV_PATH="$HOME/.venv/venv-fsdp"
    if [ ! -d "$VENV_PATH" ]; then
      echo "--- Creating Python virtual environment at $VENV_PATH ---"
      python3 -m venv $VENV_PATH
    else
      echo "--- Virtual environment already exists at $VENV_PATH ---"
    fi

    source $VENV_PATH/bin/activate

    # --- 2. Install Dependencies ---
    echo "--- [STEP 2.1] Upgrading build toolchain ---"
    pip install --upgrade pip wheel packaging

    echo "--- [STEP 2.2] Installing PyTorch Nightly ---"
    pip install --force-reinstall --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

    echo "--- [STEP 2.3] Installing application dependencies ---"
    if [ -f "requirements-fsdp.txt" ]; then
        pip install -r requirements-fsdp.txt
    else
        echo "ERROR: requirements-fsdp.txt not found!"
        exit 1
    fi

    # --- [STEP 2.4] Build Flash Attention from Source ---
    echo "--- Building flash-attn from source... This will take a while. ---"
    # Use all available CPU cores to speed up the build
    MAX_JOBS=$(nproc) pip install flash-attn --no-build-isolation

    # --- 3. Download the Model ---
    echo "--- [STEP 2.5] Downloading Mixtral model ---"
    if [ -z "$HF_TOKEN" ]; then
      echo "ERROR: The HF_TOKEN environment variable is not set."; exit 1;
    fi
    pip install huggingface_hub[cli]

    # Execute the CLI using its full, explicit path
    $VENV_PATH/bin/huggingface-cli download mistralai/Mixtral-8x7B-v0.1 --local-dir ~/Mixtral-8x7B-v0.1 --token $HF_TOKEN

    echo "--- Environment setup complete. ---"
    ```
2.  **Create `requirements-fsdp.txt`:** This file lists the Python libraries our training script will need.
    ```
    transformers==4.55.0
    datasets==4.0.0
    peft==0.16.0
    accelerate==1.9.0
    trl==0.21.0

    # Other dependencies
    sentencepiece==0.2.0
    protobuf==6.31.1
    ```

{{< figure src="/images/mixtral/layers.png" width="300px" >}}

3.  **Create `train-mixtral.py`:** This is the core training script. It uses Hugging Face's `SFTTrainer` to simplify the FSDP and LoRA setup, allowing us to focus on the fine-tuning logic.

    ```py
    import torch
    from torch.distributed.fsdp import MixedPrecision
    from datasets import load_dataset
    import shutil   
    import os
    import torch.distributed as dist 

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        HfArgumentParser,
    )

    from torch.distributed import get_rank, get_world_size

    from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
    from trl import SFTTrainer
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class ScriptArguments:
        model_id: str = field(default="mistralai/Mixtral-8x7B-v0.1", metadata={"help": "Hugging Face model ID from the Hub"})
        dataset_name: str = field(default="philschmid/gretel-synthetic-text-to-sql", metadata={"help": "Dataset from the Hub"})
        run_inference_after_training: bool = field(default=False, metadata={"help": "Run sample inference on rank 0 after training"})
        dataset_subset_size: Optional[int] = field(default=None, metadata={"help": "Number of samples to use from the dataset for training. If None, uses the full dataset."})

    @dataclass
    class PeftArguments:
        lora_r: int = field(default=16, metadata={"help": "LoRA attention dimension"})
        lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha scaling factor"})
        lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout probability"})

    @dataclass
    class SftTrainingArguments(TrainingArguments):
        max_length: Optional[int] = field(default=2048, metadata={"help": "The maximum sequence length for SFTTrainer"})
        packing: Optional[bool] = field(default=False, metadata={"help": "Enable packing for SFTTrainer"})
        ddp_find_unused_parameters: Optional[bool] = field(default=False, metadata={"help": "When using FSDP activation checkpointing, this must be set to False for Mixtral"})

    def formatting_prompts_func(example):
        system_message = "You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."
        user_prompt = f"### SCHEMA:\n{example['sql_context']}\n\n### USER QUERY:\n{example['sql_prompt']}"
        response = f"\n\n### SQL QUERY:\n{example['sql']}"
        return f"{system_message}\n\n{user_prompt}{response}"


    def main():
        parser = HfArgumentParser((ScriptArguments, PeftArguments, SftTrainingArguments))
        script_args, peft_args, training_args = parser.parse_args_into_dataclasses()

        training_args.gradient_checkpointing = True
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

        training_args.optim = "adamw_torch_fused"

        bf16_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )


        training_args.fsdp = "full_shard"
        training_args.fsdp_config = {
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_transformer_layer_cls_to_wrap": [MixtralDecoderLayer],
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            "fsdp_offload_params": False,
            "fsdp_forward_prefetch": True,
            "fsdp_mixed_precision_policy": bf16_policy
        }
        
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, trust_remote_code=True)

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        peft_config = LoraConfig(
            r=peft_args.lora_r,
            lora_alpha=peft_args.lora_alpha,
            lora_dropout=peft_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        
        model = get_peft_model(model, peft_config)

        data_splits = load_dataset(script_args.dataset_name)

        dataset = data_splits["train"]
        eval_dataset = data_splits["test"]

        if script_args.dataset_subset_size is not None:
            dataset = dataset.select(range(script_args.dataset_subset_size))

        dataset = dataset.shuffle(seed=training_args.seed)

        trainer = SFTTrainer(
            model=model,
            args=training_args, 
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            formatting_func=formatting_prompts_func,
            processing_class=tokenizer,
        )

        trainer.train()

        dist.barrier() 
        if trainer.is_world_process_zero():
            best_model_path = trainer.state.best_model_checkpoint

            final_model_dir = os.path.join(training_args.output_dir, "final_best_model")
            print(f"Copying best model to: {final_model_dir}")

            if os.path.exists(final_model_dir):
                shutil.rmtree(final_model_dir) 
            shutil.copytree(best_model_path, final_model_dir)

            if script_args.run_inference_after_training:
                del model, trainer
                torch.cuda.empty_cache()
                run_post_training_inference(script_args, final_model_dir, tokenizer)

    def run_post_training_inference(script_args, best_model_path, tokenizer):
        print("\n" + "="*50)
        print("=== RUNNING POST-TRAINING INFERENCE TEST ===")
        print("="*50 + "\n")

        base_model = AutoModelForCausalLM.from_pretrained(
            script_args.model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, best_model_path)
        model = model.merge_and_unload()
        model.eval()

        # Define the test case
        schema = "CREATE TABLE artists (Name TEXT, Country TEXT, Genre TEXT)"
        system_message = "You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."
        question = "Show me all artists from the Country just north of the USA."
        
        prompt = f"{system_message}\n\n### SCHEMA:\n{schema}\n\n### USER QUERY:\n{question}\n\n### SQL QUERY:\n"
        
        print(f"Test Prompt:\n{prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        print("\n--- Generating SQL... ---")
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        
        generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

        print(f"\n--- Generated SQL Query ---")
        print(generated_sql)
        print("\n" + "="*50)
        print("=== INFERENCE TEST COMPLETE ===")
        print("="*50 + "\n")


    if __name__ == "__main__":
        main()
    ```
4.  **Create `train-mixtral.sh`:** This is our Slurm job submission script. It implements a two-phase workflow to optimize for performance. In the first phase, it stages the model and virtual environment from our shared Lustre file system to the fast local SSDs on each compute node. This minimizes network latency during the high-I/O training loop. In the second phase, it launches the training job. At the end of the job, it writes the final checkpoints back to the shared Lustre file system for persistence.
    ```shell
    #!/bin/bash
    #SBATCH --job-name=mixtral-fsdp
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=8
    #SBATCH --gpus-per-node=8
    #SBATCH --partition=a4high
    #SBATCH --output=mixtral-%j.out
    #SBATCH --error=mixtral-%j.err

    set -e
    set -x

    echo "--- Slurm Job Started ---"

    # --- Define Paths ---
    LOCAL_SSD_PATH="/mnt/localssd/job_${SLURM_JOB_ID}"
    VENV_PATH="${HOME}/.venv/venv-fsdp"
    MODEL_PATH="${HOME}/Mixtral-8x7B-v0.1"

    # --- STAGE 1: Stage Data to Local SSD on Each Node ---
    srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "
    echo '--- Staging on node: $(hostname) ---'
    mkdir -p ${LOCAL_SSD_PATH}

    echo 'Copying virtual environment...'
    rsync -a -q ${VENV_PATH}/ ${LOCAL_SSD_PATH}/venv/

    echo 'Copying model weights...'
    rsync -a ${MODEL_PATH}/ ${LOCAL_SSD_PATH}/model/

    mkdir -p ${LOCAL_SSD_PATH}/hf_cache

    echo '--- Staging on $(hostname) complete ---'
    "
    echo "--- Staging complete on all nodes ---"

    # --- STAGE 2: Run the Training Job ---
    echo "--- Launching Distributed Training with GIB NCCL Plugin ---"
    nodes=( $( scontrol show hostnames "$SLURM_JOB_NODELIST" ) )
    head_node=${nodes[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    export MASTER_ADDR=$head_node_ip
    export MASTER_PORT=29500

    export NCCL_SOCKET_IFNAME=enp0s19

    export NCCL_NET=gIB

    # export NCCL_DEBUG=INFO # Un-comment to diagnose NCCL issues if needed

    srun --cpu-bind=none --accel-bind=g bash -c '
    # Activate the environment from the local copy
    source '${LOCAL_SSD_PATH}'/venv/bin/activate

    # Point Hugging Face cache to the local SSD
    export HF_HOME='${LOCAL_SSD_PATH}'/hf_cache

    export RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NTASKS
    export LOCAL_RANK=$SLURM_LOCALID

    export LD_LIBRARY_PATH=/usr/local/gib/lib64:$LD_LIBRARY_PATH
    source /usr/local/gib/scripts/set_nccl_env.sh

    # --- Launch the training ---
    python \
        '${SLURM_SUBMIT_DIR}'/train-mixtral.py \
        --model_id="'${LOCAL_SSD_PATH}'/model/" \
        --output_dir="${HOME}/outputs/mixtral_job_${SLURM_JOB_ID}" \
        --dataset_name="philschmid/gretel-synthetic-text-to-sql" \
        --seed=900913 \
        --bf16=True \
        --num_train_epochs=3 \
        --per_device_train_batch_size=32 \
        --gradient_accumulation_steps=4 \
        --learning_rate=4e-5 \
        --logging_steps=3 \
        --lora_r=32 \
        --lora_alpha=32 \
        --lora_dropout=0.05 \
        --eval_strategy=steps \
        --eval_steps=10 \
        --save_strategy=steps \
        --save_steps=10 \
        --load_best_model_at_end=False \
        --metric_for_best_model=eval_loss \
        --run_inference_after_training \
        --dataset_subset_size=67000
'

    # --- STAGE 3: Cleanup ---
    echo "--- Cleaning up local SSD on all nodes ---"
    srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "rm -rf ${LOCAL_SSD_PATH}"

    echo "--- Slurm Job Finished ---"
    ```
---

### Connect and Get to Work

With the cluster now up and running, you're ready to connect and upload your scripts.

1.  **Upload the scripts.** First, identify your login node by listing your VMs.
    ```shell
    gcloud compute instances list
    ```
    Then, upload your scripts to the login node's home directory.
    ```shell
    # Run this from your local machine where you created the files
    LOGIN_NODE_NAME="your-login-node-name" # e.g., a4high-login-001
    PROJECT_ID="your-gcp-project-id"
    ZONE="your-cluster-zone" # e.g., us-west4-a

    gcloud compute scp --project="$PROJECT_ID" --zone="$ZONE" --tunnel-through-iap \
      ./install_environment.sh \
      ./requirements-fsdp.txt \
      ./train-mixtral.py \
      ./train-mixtral.sh \
      "${LOGIN_NODE_NAME}":~/
    ```
2.  **Connect to the login node.**
    ```shell
    gcloud compute ssh $LOGIN_NODE_NAME \
        --project=$PROJECT_ID \
        --tunnel-through-iap \
        --zone=$ZONE
    ```
3.  **Install frameworks and tools.** Once connected, export your Hugging Face token and run the installation script.
    ```shell
    # On the login node
    export HF_TOKEN="hf_..." # Replace with your token
    srun --job-name=env-setup --nodes=1 --ntasks=1 --gpus-per-node=1 --partition=a4high bash ./install_environment.sh
    ```
    This `srun` command is critical: it runs our setup script on a compute node, where it has the power to compile libraries like `flash-attn` and download the massive Mixtral model weights. This process can take over 30 minutes, but it would be _much_ longer if it had to run on the much less powerful login node.

---

### Launching the Job

With the environment prepped, it's time to submit the training job to the Slurm scheduler.

{{< figure src="/images/mixtral/launch.png" width="300px" title="Launch the training job" >}}

1.  Submit the job.
    ```shell
    # On the login node
    sbatch train-mixtral.sh
    ```
2.  Monitor the progress by checking the output files created in your `home` directory.
    ```shell
    # On the login node
    tail -f mixtral-*.out
    ```
    The job has two main phases: first, it stages the necessary data and scripts to the local SSDs on each compute node for maximum I/O performance. Second, it runs the distributed training job itself.

### Optional: Running Standalone Inference

The training script we used automatically runs a small inference test on rank 0 after training completes. However, you might want to run inference against your saved model checkpoints as a separate step.

The process involves two new scripts: one to perform the inference in Python, and another to submit the inference task to Slurm.

1.  **Create `inference.py`:** This script is responsible for loading the base Mixtral model, applying your fine-tuned LoRA adapter from a checkpoint, and generating a response to a sample prompt. It's a good starting point for building your own inference applications.

    ```python
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import argparse

    def main():
        parser = argparse.ArgumentParser(description="Run inference with a fine-tuned PEFT adapter.")
        parser.add_argument(
            "--base_model_id", 
            type=str, 
            required=True, 
            help="The Hugging Face Hub ID or local path of the base model."
        )
        parser.add_argument(
            "--adapter_path", 
            type=str, 
            required=True, 
            help="Path to the directory containing the saved LoRA adapter."
        )
        args = parser.parse_args()

        # --- 1. Load the base model with memory-efficient settings ---
        print(f"Loading base model: {args.base_model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto", # Automatically use available GPU(s)
            attn_implementation="flash_attention_2",
        )

        # --- 2. Load the tokenizer ---
        print(f"Loading tokenizer from: {args.base_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_id,
            trust_remote_code=True,
        )

        # --- 3. Apply the PEFT adapter ---
        print(f"Applying LoRA adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        model = model.merge_and_unload()
        model.eval()

        # --- 4. Prepare the prompt ---
        schema = "CREATE TABLE artists (Name TEXT, Country TEXT, Genre TEXT)"
        system_message = "You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."
        question = "Show me all artists from the Country just north of the USA."
        
        prompt = f"{system_message}\n\n### SCHEMA:\n{schema}\n\n### USER QUERY:\n{question}\n\n### SQL QUERY:\n"
        
        print("\n--- Test Prompt ---")
        print(prompt)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # --- 5. Generate and print the output ---
        print("\n--- Generating SQL... ---")
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        
        generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

        print(f"\n--- Generated SQL Query ---")
        print(generated_sql)

        # --- Test Case 2: Bugs from last week ---
        print("\n" + "="*20)
        print("=== INFERENCE TEST 2 ===")
        print("="*20 + "\n")

        schema = "CREATE TABLE bugs (ID INT, title TEXT, description TEXT, assignee TEXT, created_date DATE)"
        question = "list all the bugs created last week without an assignee"
        
        prompt = f"{system_message}\n\n### SCHEMA:\n{schema}\n\n### USER QUERY:\n{question}\n\n### SQL QUERY:\n"
        
        print("\n--- Test Prompt ---")
        print(prompt)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print("\n--- Generating SQL... ---")
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        
        generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

        print(f"\n--- Generated SQL Query ---")
        print(generated_sql)

    if __name__ == "__main__":
        main()
    ```

2.  **Create `run_inference.slurm`:** This Slurm job script requests a single GPU and executes our `inference.py` script. You will need to edit this file to point to the correct model checkpoint from your training job.

    ```shell
    #!/bin/bash
    #SBATCH --job-name=mixtral-inference
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=1
    #SBATCH --gpus-per-node=1  # Only need one GPU
    #SBATCH --partition=a4high
    #SBATCH --output=inference-%j.out
    #SBATCH --error=inference-%j.err

    set -e # Exit on error
    set -x # Print commands

    # Activate your python environment
    source ~/.venv/venv-fsdp/bin/activate

    # Define paths
    BASE_MODEL_PATH="${HOME}/Mixtral-8x7B-v0.1"
    ADAPTER_PATH="${HOME}/outputs/mixtral_job_YOUR_JOB_ID_HERE/final_best_model" # <--- REPLACE with your actual job ID

    # Check if the adapter path exists
    if [[ "$ADAPTER_PATH" == *YOUR_JOB_ID_HERE* ]] || [ ! -d "$ADAPTER_PATH" ]; then
        echo "ERROR: Please replace 'YOUR_JOB_ID_HERE' with a real job ID and ensure the directory exists."
        echo "Current ADAPTER_PATH: $ADAPTER_PATH"
        exit 1
    fi

    # Run the inference script
    python inference.py \
        --base_model_id="$BASE_MODEL_PATH" \
        --adapter_path="$ADAPTER_PATH"
    ```

To run an inference job, upload these two files to your login node, edit the `ADAPTER_PATH` in `run_inference.slurm` to point to the output directory of your completed training job, and submit it with `sbatch run_inference.slurm`.

### Clean up

Once your fine-tuning job is complete, you'll want to clean up to avoid unnecessary charges.

### Delete your Slurm cluster

1.  Go to the `cluster-toolkit` directory.
2.  Use `gcluster` to destroy all the resources we created.
    ```shell
    ./gcluster destroy a4high --auto-approve
    ```
    This command will show you a plan of all the resources that will be destroyed and will proceed to remove them from your project.

### What's next

Now that you have a functional, high-performance training environment, here are some next steps to explore and optimize your workflow:

*   **Adjust Hyperparameters:** Experiment with different `per_device_train_batch_size`, `gradient_accumulation_steps`, and `learning_rate` to see what gets you the best performance for your specific task.
*   **Explore Other Datasets:** Try fine-tuning Mixtral on a different dataset to see how well this pattern adapts to other domains.
*   **Add Monitoring:** Integrate `wandb` or `TensorBoard` to log your training runs and visualize metrics like loss and accuracy.
*   **Create a Serving Endpoint:** Once you have your fine-tuned model, you can deploy it to a serving endpoint using GKE or Vertex AI to make it accessible for inference.