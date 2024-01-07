# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

GLOBAL_BATCH_SIZE = 8
MICRO_BATCH_SIZE = 1


def get_train_ds_config(offload, stage=2):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 1,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 500,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power":12
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "checkpoint": {
            "use_node_local_storage": True
        }

    }