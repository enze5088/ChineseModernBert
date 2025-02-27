# ChineseModernBert
中文预训练ModernBert

在自然语言处理领域，[BERT](https://huggingface.co/papers/1810.04805)自 2018 年发布以来，凭借其在语言理解和生成任务上的卓越表现，一直被广泛应用。然而，随着时间的推移，无论是中文还是英文的常用 Bert 模型，由于其发布时间距今已有 5 - 6 年，在模型架构和训练数据方面都逐渐显露出陈旧的问题。

近期，**ModernBERT**在 12 月19日左右发布，为自然语言处理领域带来了新的活力。但目前该模型主要面向英文用户，对于中文用户来说，其适用性存在一定的局限性。为了填补这一空白，满足中文自然语言处理任务的需求，笔者精心构建了基于中文语料的 ModelBert 中文版本模型。

ModelBert 是一个专门基于中文预训练语料进行训练的预训练模型。在训练过程中，选用了高质量的[C](https://huggingface.co/datasets/BAAI/CCI3-HQ)[CI3-](https://huggingface.co/datasets/BAAI/CCI3-HQ)[HQ](https://huggingface.co/datasets/BAAI/CCI3-HQ)数据集进行 1epoch 的预训练。CCI3-HQ 数据集包含了丰富多样的中文文本，涵盖了新闻资讯、文学作品、学术论文、社交媒体内容等多个领域，这使得 ModelBert 能够学习到全面且深入的中文语言特征和语义信息。

## 训练细节

- **硬件配置**：笔者训练资源有限，本次训练使用了3*8*A100，预训练时间为58小时左右。
- **优化器与学习率**：优化器采用adamw，初始学习率设置为 1e-4。learning rate scheduler 采用余弦退火。后续可能采用WSD等学习率计划从新优化。
- **Tokenizer**：Tokenizer 选用了 Qwen2.5 系列。
- **Batch Size**：单卡Batch Size 设置为4，总Batch size为96。
- **上下文长度**：上下文长度设置为 4096，并采用packing等策略。
- **训练策略**：采用了 Packing 等策略。MLM比率设置为0.3.


## 开源协议

ModelBert 遵循 Apache 2.0 开源协议发布。这意味着你可以自由地使用、修改和分发该模型，但请务必遵守开源协议中的相关规定，并在使用过程中保留模型的版权声明。
