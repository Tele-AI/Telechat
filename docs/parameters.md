# SFT参数解读以及注意事项

## SFT参数解读

```python
export CUDA_VISIBLE_DEVICES=0,1,2,3 # 指定使用gpu0、1、2、3
OUTPUT=telechat-lora-test # 模型输出路径
ZERO_STAGE=1 # Zero阶段

if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi
mkdir -p $OUTPUT


deepspeed --master_port 29500 main.py \ # 指定端口号
   --data_path ../../example_datas/single_turn_example.jsonl \ # json数据路径(如需加载多个数据集，数据集之间用空格分隔)
   --model_name_or_path ../../models/7B \ # 模型路径
   --use_cache_dataset \ # 是否使用已处理过的数据
   --data_output_path \ # 处理后(tokenize后)数据的保存路径
   --with_loss_mask \ # 是否使用loss mask
   --per_device_train_batch_size 1 \ # 每个gpu上batch size大小
   --max_seq_len 2048 \ # 训练上下文长度
   --learning_rate 3e-6 \ # 学习率
   --weight_decay 0.0001 \ # 权重衰减
   --num_train_epochs 5 \ # epoch数量
   --gradient_accumulation_steps 16 \ # 梯度累计步数 
   --lr_scheduler_type cosine \ 
   --warmup_proportion 0.1 \ # warm up 比例
   --seed 1233 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \ # 开启deepspeed
   --gradient_checkpointing \ #开启gradient checkpoint
   --offload \ 开启cpu offload
   --lora_dim 8 \ # lora内置矩阵维度
   --mark_only_lora_as_trainable \ # 是否只在加lora的层上计算梯度
   --lora_module_name "self_attention." \ # 在哪些层上添加lora矩阵
   --output_dir $OUTPUT \
   2>&1 | tee $OUTPUT/training.log
```


## 注意事项

* 微调阶段主要涉及到Zero显存优化技术，Zero不同阶段分别将优化器状态、模型梯度、模型参数平均切分到每一个gpu上，
Zero1切分优化器状态，Zero2切分优化器状态、模型梯度，Zero3切分优化器状态、模型梯度、模型参数。
* 此外，gradient chekpoint和cpu offload也可以帮助节省显存（cpu offload需与Zero3同时开启）
* lora通过在线性层上添加低秩矩阵，从而达到大幅节省训练所需的参数量，其中lora_dim是矩阵的秩，lora_module_name表示添加lora的线性层，
mark_only_lora_as_trainable表示是否只在加lora的层上计算梯度
* global_batch_size的计算公式为 per_device_train_batch_size * gpu数量 * gradient_accumulation_steps
* with_loss_mask表示在训练阶段只对回答部分计算loss，可以提升模型的回复质量
* 以7B为例，在models/7B/config.json中，可以开启flash-attention技术 (可显著降低显存，同时提升训练速度)，但flash-attention不支持Tesla V100架构。因此，在使用V100进行训练时，需把config.json中的**flash_attn**设置为**false**
* 如在使用时希望指定gpu数量，请使用**export CUDA_VISIBLE_DEVICES**进行更改
* 全量微调多节点运行时，务必保证节点之间互联；各节点上的代码和数据一致，包括内容一致与位置一致 
* 在微调模型的时候，一定要注意ckpt参数是否load进来了！如果写错`model_name_or_path`，会导致参数load不进来，但是不会报错，框架会随机初始化模型参数，这时初始loss会是`e+01`量级。所以，一定要注意第一步的loss是`e+00`量级而非`e+01`
