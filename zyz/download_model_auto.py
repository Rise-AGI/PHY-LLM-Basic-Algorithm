
import sys
import argparse

import magnus

# -- default Magnus connection --
DEFAULT_ADDRESS = "http://162.105.151.134:3011/"
DEFAULT_TOKEN   = "sk-xxx"


def main():
    parser = argparse.ArgumentParser(description="download model from ModelScope to Magnus persistent storage")

    parser.add_argument("--model",  default="Qwen/Qwen2.5-Math-7B-Instruct",
                        help="ModelScope model ID")
    parser.add_argument("--address", default=DEFAULT_ADDRESS,
                        help="Magnus server address (default: %(default)s)")
    parser.add_argument("--token",   default=DEFAULT_TOKEN,
                        help="Magnus Trust Token")
    args = parser.parse_args()

    # auto-detect name
    model_id    = args.model
    publisher   = model_id.split("/")[0]
    ms_model_id = model_id[0].lower() + model_id[1:]
    model_name  = model_id.split("/")[-1]

    # -- step 1: configure Magnus --
    print(f"[1/3] configuring Magnus connection...")
    print(f"      address: {args.address}")
    print(f"      token  : {args.token[:8]}...{args.token[-4:]}")
    magnus.configure(address=args.address, token=args.token)

    # -- step 2: submit download job --
    entry_command = f"""set -e
pip install -q modelscope

echo "=== user ==="
whoami
USERNAME=$(whoami)
SAVE_DIR="/data/$USERNAME/models/{model_name}"
echo "target: $SAVE_DIR"

# skip if already exists
if [ -f "$SAVE_DIR/config.json" ]; then
    echo "[skip] model already exists: $SAVE_DIR"
    exit 0
fi

mkdir -p "$SAVE_DIR"

# download to /data/ directly to avoid ephemeral storage limit
DL_TMP="/data/$USERNAME/models/.dl_tmp"
rm -rf "$DL_TMP"
mkdir -p "$DL_TMP"

echo "=== downloading from ModelScope: {ms_model_id} ==="
python3 -c "
from modelscope import snapshot_download
path = snapshot_download('{ms_model_id}', local_dir='$DL_TMP')
print('download completed, temp path:', path)
"

echo "=== moving to final directory ==="
# ModelScope creates: $DL_TMP/{publisher}/{model_name}/
# flatten: copy model files directly into SAVE_DIR
MODEL_SRC="$DL_TMP/{publisher}/{model_name}"
if [ -d "$MODEL_SRC" ]; then
    cp -r "$MODEL_SRC"/* "$SAVE_DIR/"
else
    cp -r "$DL_TMP"/* "$SAVE_DIR/"
fi
rm -rf "$DL_TMP"
echo "=== done! model saved to: $SAVE_DIR ==="
ls "$SAVE_DIR"
"""

    print(f"[2/3] submitting download job, model: {model_id}")
    print(f"      ModelScope ID: {ms_model_id}")
    print(f"      target path  : /data/<user>/models/{model_name}")
    print()

    job_id = magnus.submit_job(
        task_name         = f"DownloadModel-{model_name}",
        description       = f"download {model_id} to persistent storage",
        entry_command     = entry_command,
        namespace         = "Rise-AGI",
        repo_name         = "OpenFundus",
        gpu_count         = 0,
        gpu_type          = "cpu",
        cpu_count         = 8,
        memory_demand     = "16G",
        ephemeral_storage = "500G",
        job_type          = "A2",
    )

    # -- step 3: done --
    print(f"[3/3] submitted, Job ID: {job_id}")
    print(f"      model: {model_id} -> /data/<user>/models/{model_name}")
    print()


if __name__ == "__main__":
    main()
