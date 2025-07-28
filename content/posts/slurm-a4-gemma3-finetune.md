+++
title =  "Fine-tuning Gemma3 on an A4 Slurm Cluster"
tags = ["gcp", "next", "slurm", "HPC", "tutorial", "ai-workloads"]
date = "2025-07-28T13:37:01-06:00"
categories = ["Tutorials"]
+++
# Fine-tuning Gemma 3 on an A4 Slurm Cluster

# Overview

This post demonstrates how to fine-tune the Gemma 3 language model on a multi-node, multi-GPU Slurm cluster on Google Cloud. The cluster uses `a4-highgpu-8g` virtual machines, each equipped with 8 NVIDIA B200 GPUs.

The process involves two main stages:

1.  Deploying a production-grade, high-performance Slurm cluster using the Google Cloud Cluster Toolkit. This includes creating a custom VM image with necessary software pre-installed, setting up a shared Filestore instance, and configuring high-speed RDMA networking.
2.  Running a distributed fine-tuning job using a provided set of scripts. The job leverages [Hugging Face Accelerate with FSDP](https://huggingface.co/docs/accelerate/en/usage_guides/fsdp) for efficient multi-node training.

# Objectives

*   Learn how to deploy a production-grade A4 Slurm cluster.
*   Understand how to configure a multi-node environment for distributed training.
*   Fine-tune the Gemma 3 model using Hugging Face Accelerate and FSDP.
*   Submit, monitor, and manage a distributed job on a Slurm cluster.
*   Securely manage and clean up cloud resources after the workload is complete.

# Costs

This tutorial uses billable components of Google Cloud, including:

*   [**Compute Engine:**](https://cloud.google.com/compute/vm-instance-pricing?hl=en#accelerator-optimized) For the Slurm login and compute node VMs (including GPUs).
*   [**Cloud Filestore:**](https://cloud.google.com/filestore/pricing?hl=en) For the shared file system (`/home` directory).
*   [**Cloud Storage:**](https://cloud.google.com/storage/pricing) To store the Terraform state for the cluster deployment.
*   [**Cloud Logging:**](https://cloud.google.com/stackdriver/pricing) For monitoring and logging.

Use the [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator) to generate a cost estimate based on your projected usage.

# Before you begin

1.  In the Google Cloud console, on the project selector page, select or create a Google Cloud project.
2.  Make sure that billing is enabled for your Cloud project.
3.  Install and initialize the Google Cloud CLI.
4.  Ensure your user account has sufficient IAM permissions to create and manage the resources in this tutorial (e.g., `roles/owner` or a combination of `roles/compute.admin`, `roles/iam.serviceAccountUser`, `roles/file.editor`, `roles/storage.admin`, and `roles/serviceusage.serviceUsageAdmin`).
5.  Enable the required Google Cloud APIs.
    ```bash
    gcloud services enable compute.googleapis.com file.googleapis.com logging.googleapis.com cloudresourcemanager.googleapis.com servicenetworking.googleapis.com
    ```
6.  Set up Application Default Credentials, which is required for the Cluster Toolkit to authenticate correctly.
    ```bash
    gcloud auth application-default login
    ```

# Prepare the environment

These steps are performed on your local machine to prepare for deploying the cluster.

1.  Clone the Google Cloud Cluster Toolkit repository.
    ```bash
    git clone https://github.com/GoogleCloudPlatform/cluster-toolkit.git
    ```
2.  Create a Cloud Storage bucket to store the deployment's Terraform state. The bucket name must be globally unique.
    ```bash
    export BUCKET_NAME="your-unique-bucket-name"
    gcloud storage buckets create gs://${BUCKET_NAME}
    ```

# Obtain capacity and quota

This tutorial requires `a4-highgpu-8g` VMs, which are in high demand. It is strongly recommended that you have a [reservation](https://cloud.google.com/compute/docs/instances/reservations-single-project) to ensure you can obtain the necessary capacity. You can deploy your cluster without a reservation, but this guide assumes you have already obtained a reservation for 2x `a4-highgpu-8g` VMs.

When you configure the deployment file in the next step, you will need to provide the name of your reservation in the `a4h_reservation_name` field.

# Create the cluster

1.  Navigate to the cloned `cluster-toolkit` directory.
    ```bash
    cd cluster-toolkit
    ```
2.  **(First-Time Setup Only)** The first time you use the Cluster Toolkit, you must build the `gcluster` binary.
    ```bash
    make
    ```
3.  Configure the deployment by editing the main deployment file.
    ```bash
    vim examples/machine-learning/a4-highgpu-8g/a4high-slurm-deployment.yaml
    ```
4.  In the file, update the following fields with your specific information:
    *   `project_id`
    *   `deployment_name` (use a short name with only letters and numbers, e.g., `a4high`)
    *   `bucket` (the name of the bucket you created earlier)
    *   `region` (the region that contains your reservation)
    *   `zone` (the specific zone that contains your reservation)
    *   `a4h_cluster_size` (set to 2)
    *   `a4h_reservation_name` (to match your A4 reservation)

5.  Deploy the cluster using the `gcluster` command.
    ```bash
    ./gcluster deploy -d examples/machine-learning/a4-highgpu-8g/a4high-slurm-deployment.yaml examples/machine-learning/a4-highgpu-8g/a4high-slurm-blueprint.yaml --auto-approve
    ```
    **Note:** The `./gcluster deploy` command for this blueprint is a two-phase process. The first phase builds a custom "golden image" with all software pre-installed, which can take **approximately 35 minutes**. This is a one-time cost. The second phase deploys the cluster using that image, which is much faster.

6.  If your deployment fails after the image building step has succeeded, you can skip that phase on subsequent deploys to save time:
    ```bash
    ./gcluster deploy -d examples/machine-learning/a4-highgpu-8g/a4high-slurm-deployment.yaml examples/machine-learning/a4-highgpu-8g/a4high-slurm-blueprint.yaml --auto-approve --skip "image" -w
    ```

# Understand your cluster

Once the deployment is complete, the Cluster Toolkit has created several resources in your project. You can view and manage these resources in the Google Cloud Console.

1.  **Navigate to the VM instances page:**
    *   Go to the [Compute Engine VM instances page](https://console.cloud.google.com/compute/instances) in the Cloud Console.
2.  **Identify the Cluster Nodes:**
    *   You will see four new instances created by the deployment (the names will vary based on your `deployment_name`):
        *   A **login node** (e.g., `a4high-login-001`): This is the main entry point for interacting with the cluster, submitting jobs, and managing your environment.
        *   A **controller node** (e.g. `a4high-controller`): This is the brain of the Slurm cluster, managing the job queue and compute nodes. You do not interact with it directly.
        *   Two **compute nodes** (e.g., `a4high-a4highnodeset-0`, `a4high-a4highnodeset-1`): These are the nodes where your training job will actually run. You typically do not interact with these nodes directly.
3.  **Connect to the Login Node:**
    *   All interaction with the cluster should be done through the login node. The easiest way to connect is via SSH from the Cloud Console.
    *   In the list of VM instances, find your login node.
    *   In the **Connect** column for that row, click the **SSH** button.
    *   This will open a secure SSH session directly in your browser, using Identity-Aware Proxy (IAP) for authentication. No extra firewall rules or SSH keys are needed.

All subsequent steps in this guide that are performed "on the login node" should be done in this SSH session.

# Prepare the workload

This section covers preparing the scripts and environment needed to run the fine-tuning job.

## 1. Create Workload Scripts

Create the following files on your local machine. These will be copied to the cluster's login node.

#### `install_environment.sh`
```bash
#!/bin/bash
# This script should be run ONCE on the login node to set up the
# shared Python virtual environment.

set -e
echo "--- Creating Python virtual environment in /home ---"
python3 -m venv ~/.venv
echo "--- Activating virtual environment ---"
source ~/.venv/bin/activate

echo "--- Installing build dependencies ---"
pip install --upgrade pip wheel packaging

echo "--- Installing PyTorch for CUDA 12.8 ---"
pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "--- Installing application requirements ---"
pip install -r requirements.txt

echo "--- Environment setup complete. You can now submit jobs with sbatch. ---"
```

#### `accelerate_config.yaml`
```yaml
# Default configuration for a 2-node, 8-GPU-per-node (16 total GPUs) FSDP training job.

compute_environment: "LOCAL_MACHINE"
distributed_type: "FSDP"
downcast_bf16: "no"
fsdp_config:
  fsdp_auto_wrap_policy: "TRANSFORMER_BASED_WRAP"
  fsdp_backward_prefetch: "BACKWARD_PRE"
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: "FULL_SHARD" 
  fsdp_state_dict_type: "FULL_STATE_DICT"
  fsdp_transformer_layer_cls_to_wrap: "Gemma3DecoderLayer" 
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: "main"
mixed_precision: "bf16"
num_machines: 2
num_processes: 16
rdzv_backend: "static"
same_network: true
tpu_env: []
use_cpu: false
```

#### `submit.slurm`
```bash
#!/bin/bash
#SBATCH --job-name=gemma3-finetune
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8 # 8 tasks per node
#SBATCH --gpus-per-task=1   # 1 GPU per task
#SBATCH --partition=a4high
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -e
echo "--- Slurm Job Started ---"

# --- STAGE 1: Copy Environment to Local SSD on all nodes ---
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c '
  echo "Setting up local environment on $(hostname)..."
  LOCAL_VENV="/mnt/localssd/venv_job_${SLURM_JOB_ID}"
  LOCAL_CACHE="/mnt/localssd/hf_cache_job_${SLURM_JOB_ID}"
  rsync -a --info=progress2 ~/./.venv/ ${LOCAL_VENV}/
  mkdir -p ${LOCAL_CACHE}
  echo "Setup on $(hostname) complete."
'

# --- STAGE 2: Run the Training Job using the Local Environment ---
echo "--- Starting Training ---"

LOCAL_VENV="/mnt/localssd/venv_job_${SLURM_JOB_ID}"
LOCAL_CACHE="/mnt/localssd/hf_cache_job_${SLURM_JOB_ID}"
LOCAL_OUTPUT_DIR="/mnt/localssd/outputs_${SLURM_JOB_ID}"
mkdir -p ${LOCAL_OUTPUT_DIR}

# This is the main training command.
srun --ntasks=$((SLURM_NNODES * 8)) --gpus-per-task=1 bash -c "
  source ${LOCAL_VENV}/bin/activate

  export HF_HOME=${LOCAL_CACHE}
  export HF_DATASETS_CACHE=${LOCAL_CACHE}
  
  # Run the Python script directly.
  # Accelerate will divide the work
  python ~/train.py \
    --model_id google/gemma-3-12b-pt \
    --output_dir ${LOCAL_OUTPUT_DIR} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --save_strategy steps \
    --save_steps 100
"

# --- STAGE 3: Copy Final Model from Local SSD to Home Directory ---
echo "--- Copying final model from local SSD to /home ---"
# This command runs only on the first node of the job allocation
# and copies the final model back to the persistent shared directory.
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 bash -c "
  rsync -a --info=progress2 ${LOCAL_OUTPUT_DIR}/ ~/gemma-12b-text-to-sql-finetuned/
"

echo "--- Slurm Job Finished ---"
```

#### `requirements.txt`
```
# Hugging Face Libraries (Pinned to recent, stable versions for reproducibility)
transformers==4.53.3
datasets==4.0.0
accelerate==1.9.0
evaluate==0.4.5
bitsandbytes==0.46.1
trl==0.19.1
peft==0.16.0

# Other dependencies
tensorboard==2.20.0
protobuf==6.31.1
sentencepiece==0.2.0
```
Note: The dependencies pinned above may be out of date, but will work together. To find newer versions (for different GPUs, etc...) check [NVIDIA's documentation](https://developer.nvidia.com/cuda-gpus) and the relevant [Pytorch documentation](https://pytorch.org/get-started/locally/#linux-pip).

#### `train.py`
```python
import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/gemma-3-12b-pt", help="Hugging Face model ID")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for private models")
    parser.add_argument("--dataset_name", type=str, default="philschmid/gretel-synthetic-text-to-sql", help="Hugging Face dataset name")
    parser.add_argument("--output_dir", type=str, default="gemma-12b-text-to-sql", help="Directory to save model checkpoints")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability")

    # SFTConfig arguments
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every X steps")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Checkpoint save strategy")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps")

    return parser.parse_args()

def main():
    args = get_args()

    # --- 1. Setup and Login ---
    if args.hf_token:
        login(args.hf_token)

    # --- 2. Create and prepare the fine-tuning dataset ---
    # The SFTTrainer will use the `formatting_func` to apply the chat template.
    dataset = load_dataset(args.dataset_name, split="train")
    dataset = dataset.shuffle().select(range(12500))
    dataset = dataset.train_test_split(test_size=2500/12500)

    # --- 3. Configure Model and Tokenizer ---
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype_obj = torch.bfloat16
        torch_dtype_str = "bfloat16"
    else:
        torch_dtype_obj = torch.float16
        torch_dtype_str = "float16"

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    gemma_chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<start_of_turn>model\n' }}"
        "{% endif %}"
    )
    tokenizer.chat_template = gemma_chat_template

    # --- 4. Define the Formatting Function ---
    # This function will be used by the SFTTrainer to format each sample
    # from the dataset into the correct chat template format.
    def formatting_func(example):
        # The create_conversation logic is now implicitly handled by this.
        # We need to construct the messages list here.
        system_message = "You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."
        user_prompt = "Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.\n\n<SCHEMA>\n{context}\n</SCHEMA>\n\n<USER_QUERY>\n{question}\n</USER_QUERY>\n"
        
        messages = [
            {"role": "user", "content": user_prompt.format(question=example["sql_prompt"][0], context=example["sql_context"][0])},
            {"role": "assistant", "content": example["sql"][0]}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # --- 5. Load Quantized Model and Apply PEFT ---
    
    # Define the quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch_dtype_obj,
        bnb_4bit_use_double_quant=True, 
    )

    config = AutoConfig.from_pretrained(args.model_id)
    config.use_cache = False

    # Load the base model with quantization
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        quantization_config=quantization_config,
        attn_implementation="eager",
        torch_dtype=torch_dtype_obj,
    )
    
    # Prepare the model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA. 
    peft_config = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # Apply the PEFT config to the model
    print("Applying PEFT configuration...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- 6. Configure Training Arguments ---
    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        packing=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        fp16=True if torch_dtype_obj == torch.float16 else False,
        bf16=True if torch_dtype_obj == torch.bfloat16 else False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=False,
        report_to="tensorboard",
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": True,
        }
    )

    # --- 7. Create Trainer and Start Training ---
    trainer = SFTTrainer(
        model=model,  
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=formatting_func,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # --- 8. Save the final model ---
    print(f"Saving final model to {args.output_dir}")
    trainer.save_model()

if __name__ == "__main__":
    main()
```

## 2. Upload Scripts to the Cluster

Run this `gcloud` command from your local machine to copy the files you just created to the login node's home directory.

```bash
# Run this from your local machine where you created the files
LOGIN_NODE_NAME="your-login-node-name" # e.g., a4high-login-001
PROJECT_ID="your-gcp-project-id"
ZONE="your-cluster-zone" # e.g., us-west4-a

gcloud compute scp --project="$PROJECT_ID" --zone="$ZONE" --tunnel-through-iap \
  ./train.py \
  ./requirements.txt \
  ./submit.slurm \
  ./install_environment.sh \
  ./accelerate_config.yaml \
  "${LOGIN_NODE_NAME}":~/
```

## 3.  Connect to the Login Node
Once the files have been successfully copied, SSH into the login node. For example, a gcloud command to connect to the login node might look like:

```bash
gcloud compute ssh --zone "$ZONE$" "a4high-login-001" --project "$PROJECT_ID$" --tunnel-through-iap
```

## 4. Install Frameworks and Tools

From your SSH session on the login node, run the installation script. This will set up a Python virtual environment with all the required dependencies.

```bash
# On the login node
chmod +x install_environment.sh
./install_environment.sh
```

The job uses Hugging Face Hub to download the Gemma 3 model. You must provide a [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens) to authenticate.

```bash
# On the login node
export HF_TOKEN="hf_..." # Replace with your token
```

# Start the workload

With the environment prepared and the submission script in place, you can now submit the job to the Slurm scheduler.

```bash
# On the login node
sbatch submit.slurm
```

You can monitor the job's progress by checking the output files created in your home directory.

```bash
# On the login node
tail -f slurm-<job-id>.err
```

If your job starts successfully, the `.err` file will show an updating progress bar as your job progresses.

# Monitor the workload

You can monitor the GPU utilization for your cluster in the Google Cloud Console to ensure the training job is running efficiently.

1.  **Construct the URL:** Copy the following URL and replace `[YOUR_PROJECT_ID]` with your actual Google Cloud project ID.
    ```
    https://console.cloud.google.com/monitoring/metrics-explorer?project=[YOUR_PROJECT_ID]&pageState=%7B%22xyChart%22%3A%7B%22dataSets%22%3A%5B%7B%22timeSeriesFilter%22%3A%7B%22filter%22%3A%22metric.type%3D%5C%22agent.googleapis.com%2Fgpu%2Futilization%5C%22%20resource.type%3D%5C%22gce_instance%5C%22%22%2C%22perSeriesAligner%22%3A%22ALIGN_MEAN%22%7D%2C%22plotType%22%3A%22LINE%22%7D%5D%7D%7D
    ```
2.  **View the Chart:** Open the link in your browser. You should see a chart displaying the "GPU utilization" for all GCE instances in your project.
3.  **Expected Behavior:** For a healthy fine-tuning job, you should see the utilization for all 16 GPUs (8 on each of your 2 compute nodes) rise to a high level and stay there for the duration of the training.
4.  **Job Duration:** This job should take approximately 1 hour to complete on the specified A4 cluster.

# Clean up

To avoid incurring ongoing charges for the resources used in this tutorial, you must destroy the cluster.

1.  **Navigate to the Toolkit Directory:** From your local machine, change to the `cluster-toolkit` directory that you originally cloned.
2.  **Run the Destroy Command:** Use the `./gcluster destroy` command, pointing it to the deployment directory that was created when you first deployed the cluster. This directory contains the Terraform state file that tracks all the created resources.
    ```bash
    # From the cluster-toolkit directory
    # Replace 'a4high' with the actual deployment_name from your config
    ./gcluster destroy a4high
    ```
3.  **Confirm the Destruction:** The command will show you a plan of all the resources that will be destroyed. Review this list carefully and type `yes` to confirm. To skip the interactive prompt, you can add the `--auto-approve` flag.

## Advanced: Managing Cluster Components

The `gcluster` command allows you to deploy or destroy specific parts of the cluster using the `--only` and `--skip` flags. The arguments for these flags correspond to the **deployment group names** in your blueprint YAML file (e.g., `primary`, `slurm-build`, `cluster`).

This is useful if you need to tear down and re-create the cluster infrastructure (VMs, storage, etc.) without rebuilding the golden image, which saves significant time.

*   **To destroy the cluster while keeping the golden image:**
    ```bash
    # Skips the 'image' group, which creates the image
    # Replace 'a4high' with the actual deployment_name from your config
    ./gcluster destroy a4high --skip image
    ```
*   **To re-deploy the cluster using the existing image:**
    ```bash
    # Skips the image-building groups and deploys everything else
    # Replace 'a4high' with the actual deployment_name from your config
    ./gcluster deploy a4high --skip image
    ```