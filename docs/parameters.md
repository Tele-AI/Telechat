# SFT参数解读以及注意事项

## SFT参数解读

```python
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 指定卡数
OUTPUT=telechat-single-node-test # 输出路径
ZERO_STAGE=3 # ZERO阶段
 
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT
 
deepspeed --master_port 29500 main.py \ # 指定端口号
   --data_path ${DATA_OUTPUT_PATH}/train_data.pt  \ # tokenzie后的数据文件
   --model_name_or_path ../../models/12B \ #模型路径
   --with_loss_mask \ # 是否开启loss mask
   --per_device_train_batch_size 1 \ # 每张卡batch size个数
   --max_seq_len 4096 \ # 训练上下文长度
   --learning_rate 3e-5 \ # 学习率
   --weight_decay 0.0001 \ # 权重衰减
   --num_train_epochs 1 \ # epoch数
   --gradient_accumulation_steps 4 \ # 梯度累积步数
   --lr_scheduler_type cosine \
   --precision fp16 \ # 训练精度，fp16、bf16
   --warmup_proportion 0.1 \ # warm up 比率
   --gradient_checkpointing \ # 梯度检查
   --offload \ # 是否开启cpu_offload
   --seed 1233 \
   --zero_stage $ZERO_STAGE \
   --save_steps 10 \ # 保存步数
   --deepspeed \
   --output_dir $OUTPUT # 输出路径
```


## 注意事项

* 微调阶段主要涉及到Zero显存优化技术，Zero不同阶段分别将优化器状态、模型梯度、模型参数平均切分到每一个gpu上，Zero1切分优化器状态，Zero2切分优化器状态、模型梯度，Zero3切分优化器状态、模型梯度、模型参数。
* 此外，gradient chekpoint和cpu offload也可以帮助节省显存（cpu offload需与Zero3同时开启）
* global_batch_size的计算公式为 per_device_train_batch_size * gpu数量 * gradient_accumulation_steps，在上述代码中，global_batch_size = 1 * 8 * 4 = 32。save_steps按照global batch size步数保存模型，比如上述示例每过32 * 10 = 320个samples保存一次
* with_loss_mask表示在训练阶段只对回答部分计算loss，可以提升模型的回复质量
* precision选择fp16或bf16混合精度训练
* 训练时，可在模型路径下的config.json中设置flash-attn=true开启Flash attention，能够节省显存，加速训练
* Zero stage=3, gradient_checkpointing=True, flash_attn=true，实测单机8卡A100-40G可训练4096长度，双机16卡可训练8192长度
* 保存的模型为huggingface格式，可直接加载推理
* lora通过在线性层上添加低秩矩阵，从而达到大幅节省训练所需的参数量，其中lora_dim是矩阵的秩 (lora_dim=8是相对较佳设置)，lora_module_name表示添加lora的线性层，
mark_only_lora_as_trainable表示是否只在加lora的层上计算梯度
* 以7B为例，在models/7B/config.json中，可以开启flash-attention技术 (可显著降低显存，同时提升训练速度)，但flash-attention不支持Tesla V100架构。因此，在使用V100进行训练时，需把config.json中的**flash_attn**设置为**false**
* 如在使用时希望指定gpu数量，请使用**export CUDA_VISIBLE_DEVICES**进行更改
* 全量微调多节点运行时，务必保证节点之间互联；各节点上的代码和数据一致，包括内容一致与位置一致 
