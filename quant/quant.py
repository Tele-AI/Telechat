from transformers import AutoTokenizer
from auto_gptq import BaseQuantizeConfig
from modeling_telechat_gptq import TelechatGPTQForCausalLM
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

tokenizer_path = '../models/7B'
pretrained_model_dir = '../models/7B'
quantized_model_dir = '../models/7B_8bit'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,trust_remote_code=True)
calibration_text = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
examples = [tokenizer(_) for _ in calibration_text]
quantize_config = BaseQuantizeConfig(
    bits=8,  # quantize model to 8-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
)

# load un-quantized model, by default, the model will always be loaded into CPU memory
model = TelechatGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config,trust_remote_code=True)
print(model)
# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples)
print("quantize finished")
# save quantized model
model.save_quantized(quantized_model_dir)


