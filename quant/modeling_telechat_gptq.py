from logging import getLogger
from os.path import isdir, join, isfile
from typing import Optional, Union, Dict

import torch
import accelerate
import transformers
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.utils.generic import ContextManagers
from transformers.modeling_utils import no_init_weights
from transformers.utils.hub import cached_file
from auto_gptq.modeling._base import *
from auto_gptq.utils.import_utils import TRITON_AVAILABLE
from auto_gptq.modeling._utils import make_quant, make_sure_no_tensor_in_meta_device, find_layers, simple_dispatch_model

logger = getLogger(__name__)


class TelechatGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "TelechatBlock"
    layers_block_name = "transformer.h"
    outside_layer_modules = ["transformer.word_embeddings", "transformer.ln_f"]
    inside_layer_modules = [
        ["self_attention.key_value", "self_attention.query"],
        ["self_attention.dense"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"]
    ]

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            quantize_config: BaseQuantizeConfig,
            max_memory: Optional[dict] = None,
            trust_remote_code: bool = False,
            torch_dtype: torch.dtype = torch.float16,
            **model_init_kwargs
    ):
        """load un-quantized pretrained model to cpu"""

        if not torch.cuda.is_available():
            raise EnvironmentError("Load pretrained model to do quantization requires CUDA available.")

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)

        # enforce some values despite user specified
        model_init_kwargs["torch_dtype"] = torch_dtype
        model_init_kwargs["trust_remote_code"] = trust_remote_code
        if max_memory:
            if "disk" in max_memory:
                raise NotImplementedError("disk offload not support yet.")
            with accelerate.init_empty_weights():
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            model.tie_weights()

            max_memory = accelerate.utils.get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=[cls.layer_type],
                dtype=model_init_kwargs["torch_dtype"],
                low_zero=False
            )
            model_init_kwargs["device_map"] = accelerate.infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=[cls.layer_type],
                dtype=model_init_kwargs["torch_dtype"]
            )
            model_init_kwargs["low_cpu_mem_usage"] = True

            del model
        else:
            model_init_kwargs["device_map"] = None
            model_init_kwargs["low_cpu_mem_usage"] = False

        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **model_init_kwargs)
        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
        if any([k in model_config for k in seq_len_keys]):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            logger.warning("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096
        model.eval()

        return cls(model, False, quantize_config)

    @classmethod
    def from_quantized(
            cls,
            model_name_or_path: Optional[str] = None,
            save_dir: Optional[str] = None,
            device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
            max_memory: Optional[dict] = None,
            device: Optional[Union[str, int]] = None,
            low_cpu_mem_usage: bool = False,
            use_triton: bool = False,
            torch_dtype: torch.dtype = torch.float16,
            inject_fused_attention: bool = True,
            inject_fused_mlp: bool = True,
            use_cuda_fp16: bool = True,
            quantize_config: Optional[BaseQuantizeConfig] = None,
            model_basename: Optional[str] = None,
            use_safetensors: bool = False,
            trust_remote_code: bool = False,
            warmup_triton: bool = False,
            trainable: bool = False,
            **kwargs
    ):
        """load quantized model from local disk"""

        # Parameters related to loading from Hugging Face Hub
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)

        if use_triton and not TRITON_AVAILABLE:
            logger.warning("triton is not installed, reset use_triton to False")
            use_triton = False

        # == step1: prepare configs and file names == #
        if model_name_or_path and save_dir:
            logger.warning("save_dir will be ignored because model_name_or_path is explicit specified.")
        if not model_name_or_path and save_dir:
            model_name_or_path = save_dir
            logger.warning("save_dir is deprecated and will be removed in version 0.3.0", PendingDeprecationWarning,
                           stacklevel=2)
        if not model_name_or_path and not save_dir:
            raise ValueError("at least one of model_name_or_path or save_dir should be specified.")

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

        if quantize_config is None:
            quantize_config = BaseQuantizeConfig.from_pretrained(model_name_or_path, **kwargs)

        if model_basename is None:
            if quantize_config.model_file_base_name:
                model_basename = quantize_config.model_file_base_name
            else:
                model_basename = f"gptq_model-{quantize_config.bits}bit-{quantize_config.group_size}g"

        quantize_config.model_name_or_path = model_name_or_path
        quantize_config.model_file_base_name = model_basename

        extensions = []
        if use_safetensors:
            extensions.append(".safetensors")
        else:
            extensions += [".bin", ".pt"]

        model_name_or_path = str(model_name_or_path)
        is_local = isdir(model_name_or_path)

        resolved_archive_file = None
        if is_local:
            model_save_name = join(model_name_or_path, model_basename)

            for ext in extensions:
                if isfile(model_save_name + ext):
                    resolved_archive_file = model_save_name + ext
                    break
        else:  # remote
            cached_file_kwargs = {
                "cache_dir": cache_dir,
                "force_download": force_download,
                "proxies": proxies,
                "resume_download": resume_download,
                "local_files_only": local_files_only,
                "use_auth_token": use_auth_token,
                "revision": revision,
                "subfolder": subfolder,
                "_raise_exceptions_for_missing_entries": False,
                "_commit_hash": commit_hash,
            }

            for ext in extensions:
                resolved_archive_file = cached_file(model_name_or_path, model_basename + ext, **cached_file_kwargs)
                if resolved_archive_file is not None:
                    break

        if resolved_archive_file is None:  # Could not find a model file to use
            raise FileNotFoundError(f"Could not find model in {model_name_or_path}")

        model_save_name = resolved_archive_file

        if not use_triton and trainable:
            logger.warning(
                "QuantLinear with cuda backend not support trainable mode yet, Switch to the pytorch backend.")

        # == step2: convert model to gptq-model (replace Linear with QuantLinear) == #
        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip

        transformers.modeling_utils._init_weights = False

        init_contexts = [no_init_weights()]
        if low_cpu_mem_usage:
            init_contexts.append(accelerate.init_empty_weights(include_buffers=False))

        with ContextManagers(init_contexts):
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype
            )

            layers = find_layers(model)
            ignore_layers = [cls.lm_head_name] + cls.outside_layer_modules
            for name in list(layers.keys()):
                if any([name.startswith(ignore_layer) for ignore_layer in ignore_layers]):
                    logger.info(f"{name} not been quantized, will be ignored when make_quant.")
                    del layers[name]

            make_quant(
                model,
                layers,
                quantize_config.bits,
                quantize_config.group_size,
                use_triton=use_triton,
                use_cuda_fp16=use_cuda_fp16,
                desc_act=quantize_config.desc_act,
                trainable=trainable
            )
            model.tie_weights()

        # == step3: load checkpoint and dispatch == #
        if isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
            raise ValueError(
                "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
                "'sequential'."
            )
        if isinstance(device_map, dict):
            max_memory = None
        else:
            if device is None and not device_map and not max_memory:
                device_map = "auto"
            if device is not None:
                device = torch.device(device)
                if not max_memory and not device_map:
                    device_map = {"": device.index if device.type == "cuda" else device.type}
            if not isinstance(device_map, dict) and device_map != "sequential":
                max_memory = accelerate.utils.get_balanced_memory(
                    model=model,
                    max_memory=max_memory,
                    no_split_module_classes=[cls.layer_type],
                    low_zero=(device_map == "balanced_low_0")
                )
        if not isinstance(device_map, dict):
            device_map = accelerate.infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=[cls.layer_type]
            )

        if low_cpu_mem_usage:
            make_sure_no_tensor_in_meta_device(model, use_triton, quantize_config.desc_act, quantize_config.group_size)

        accelerate.utils.modeling.load_checkpoint_in_model(
            model,
            checkpoint=model_save_name,
            device_map=device_map,
            offload_state_dict=True,
            offload_buffers=True
        )
        model = simple_dispatch_model(model, device_map)

        # == step4: set seqlen == #
        model_config = model.config.to_dict()
        seq_len_keys = ["max_position_embeddings", "seq_length", "n_positions"]
        if any([k in model_config for k in seq_len_keys]):
            for key in seq_len_keys:
                if key in model_config:
                    model.seqlen = model_config[key]
                    break
        else:
            logger.warning("can't get model's sequence length from model config, will set to 4096.")
            model.seqlen = 4096

        # == step5: (optional) inject optimized module == #
        if inject_fused_attention:
            if cls.fused_attn_module_type is None:
                inject_fused_attention = False
                logger.warning(f"{cls.__name__} hasn't fused attention module yet, will skip inject fused attention.")
            else:
                cls.fused_attn_module_type.inject_to_model(
                    model,
                    use_triton=use_triton,
                    group_size=quantize_config.group_size,
                    use_cuda_fp16=use_cuda_fp16,
                    desc_act=quantize_config.desc_act,
                    trainable=trainable
                )
        if inject_fused_mlp:
            if cls.fused_mlp_module_type is None:
                inject_fused_mlp = False
                logger.warning(f"{cls.__name__} hasn't fused mlp module yet, will skip inject fused mlp.")
            else:
                cls.fused_mlp_module_type.inject_to_model(
                    model,
                    use_triton=use_triton
                )

        model.eval()
        # == step6: (optional) warmup triton == #
        if use_triton and warmup_triton:
            from auto_gptq.nn_modules.qlinear.qlinear_triton import QuantLinear
            QuantLinear.warmup(model, seqlen=model.seqlen)

            if inject_fused_mlp and cls.fused_mlp_module_type is not None:
                cls.fused_mlp_module_type.warmup(model, seqlen=model.seqlen)

        # == step7: make model compatible with peft
        cls.make_sure_compatible_with_peft(
            model, use_triton, quantize_config.desc_act, quantize_config.group_size
        )

        return cls(
            model,
            True,
            quantize_config,
            is_triton_backend=use_triton,
            injected_fused_attention=inject_fused_attention,
            injected_fused_mlp=inject_fused_mlp and use_triton,
            trainable=trainable
        )

    def chat(self, *args,**kwargs):
        """shortcut for model.chat"""
        with torch.inference_mode(), torch.amp.autocast(device_type=self.device.type):
            return self.model.chat(*args,**kwargs)


__all__ = ["TelechatGPTQForCausalLM"]
