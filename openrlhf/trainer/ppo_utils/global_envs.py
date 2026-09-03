import os


def _env_vars():
    env = {
        "TEST_ENVS_VAR": "BUG",
    }
    for key in (
        "CUDA_HOME",
        "HF_HOME",
        "HF_HUB_OFFLINE",  # load models from the local cache on offline compute nodes
        "NCCL_IB_DISABLE",
        "NCCL_P2P_DISABLE",
        "PYTHONPATH",  # Ray workers must import openrlhf from this repo
        "SPEECH_FORK_STATS",  # fork-statistics logging toggle (analysis only)
        "MAX_JOBS",
        "RAY_TMPDIR",
        "TOKENIZERS_PARALLELISM",
        "TORCH_CUDA_ARCH_LIST",
        "TORCH_EXTENSIONS_DIR",
        "TORCH_NCCL_TRACE_BUFFER_SIZE",
        "TRITON_CACHE_DIR",
    ):
        value = os.environ.get(key)
        if value:
            env[key] = value
    return env


RUNTIME_ENV = {
    "env_vars": _env_vars()
}

    # "working_dir": "/workspace/zhenyu/code/OpenRLHF", 
    # "pip": "/workspace/zhenyu/code/OpenRLHF/requirements.txt", 
