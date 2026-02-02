# LoRA Tutorial: Generating a video with a custom LoRA

This tutorial will guide you through downloading a LoRA and using it to generate a video with the `Wan 2.2 TextImage2video 5B` model.

## Step 1: Download the LoRA

This step shows you how to download a LoRA from the Hugging Face Hub and place it in the correct directory for Wan2GP to use. The following commands are intended to be run in a RunPod environment.

1.  **Set your Hugging Face Token (Optional)**

    If you are downloading a private or gated LoRA, you will need to set your Hugging Face Hub token as an environment variable. For public models like the one in this example, a token is not necessary. You can get a token from your Hugging Face account settings.

    ```bash
    export HUGGINGFACE_HUB_TOKEN=hf_xxx
    ```

When calling the download script, it will use this environment variable for authentication. If the token is not set, it will attempt to download the model without authentication, which will only work for public repositories.

2.  **Download the LoRA**

    This command will install the necessary Python library and then download the LoRA model files into the correct directory (`/workspace/Wan2GP/loras/5B`). If the directory does not exist, it will be created automatically.

    ```bash
    python -m pip install -q --no-cache-dir huggingface_hub

    python - <<'PY'
    import os
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="ostris/wan22_5b_i2v_crush_it_lora",
        allow_patterns=["*.safetensors"],
        local_dir="/workspace/Wan2GP/loras/5B",  # write directly into Wan2GP's LoRA dir
        local_dir_use_symlinks=False,            # real file, not a symlink
        resume_download=True,                    # safe on flaky connections
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN"),  # optional; uses env var if set
    )
    print("✅ Download complete → /workspace/Wan2GP/loras/5B")
    PY
    ```

## Step 2: Generate a video with the LoRA

Once the LoRA is downloaded, you can use it in the Wan2GP web interface.

1.  Select the `Wan 2.2 TextImage2video 5B` model from the `Model Type` dropdown in the web UI.
2.  The UI will automatically detect and list available LoRAs for the selected model. Under Advanced mode options, choose the LoRA tab and select `wan22_5b_i2v_crush_it_lora` from the dropdown.
3.  Set the LoRA multiplier (higher number has more impact)
4.  Write "crush it" as the prompt.
5.  Click the `Generate` button to start the video generation process with the selected LoRA.
