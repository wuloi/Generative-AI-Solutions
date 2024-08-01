# 🪄 调参魔法：用PEFT简化大型语言模型

嗨，技术巫师们！🧙‍♀️ 欢迎来到参数高效微调（PEFT）的魔法世界，通过提示调优（Prompt Tuning）来增强我们的大型语言模型（LLMs），无需进行完整的微调就能轻松提升！

## 🌟 高效微调的探索之旅
通过LoRA，我们发现了一种高效更新模型权重的聪明方法。现在，让我们进入加性方法的领域，无需改变模型的权重就能提升性能。

### 🔑 提示调优：新炼金术
与调整措辞以更好地理解任务的提示工程不同，提示调优在提示本身引入了可训练的标记，让模型通过监督学习找到它们的最优值。

## 🧬 软提示基因组：释放虚拟标记
- **软提示**：一组额外的标记，神奇地前置到你的输入文本的嵌入向量上。
- **虚拟标记**：与自然语言的固定标记不同，这些可以变成嵌入空间内的任何值，优化任务性能。

## 📊 性能比较：提示调优 vs. 完整微调
- **较小的LLMs**：提示调优可能无法超越完整微调。
- **较大的LLMs**：随着模型大小的增加，提示调优的能力也随之增强，与完整微调的力量相媲美。

### 📈 SuperGLUE得分：基准测试
看看提示调优在SuperGLUE基准测试上的表现如何，这是对LLMs语言能力的真正考验。

## 💡 可解释性挑战：解开软提示的奥秘
虽然软提示标记不对应已知的词汇，但它们形成了语义簇，暗示了它们学习类似词汇的表示能力。

## 🔮 总结：PEFT在LLM训练中的魔法
LoRA和提示调优是你的PEFT工具包，允许你以潜在的性能提升为目标微调模型，同时只使用一小部分计算资源。

### 🚀 第二周回顾
- **指令微调**：用几百个示例适应基础模型。
- **评估指标**：使用ROUGE和HELM来衡量模型成功。
- **PEFT**：最小化计算和内存资源，加速你的开发过程。

别忘了订阅，以获取更多关于AI前沿的迷人探索。我们在这里引导你穿越模型效率的神秘领域及其之外！

👋 下次见，继续实验，继续创新，愿你的模型永远精准调优至完美！

---

[加入我们，深入了解PEFT和AI的奥德赛！](https://roadmaps.feishu.cn/wiki/RykrwFxPiiU4T7kZ63bc7Lqdnch)

---

# 🪄 The Magic of Prompt Tuning: Streamline LLMs with PEFT

Hey Tech Sorcerers! 🧙‍♀️ Welcome to the wizardry of Parameter Efficient Fine-Tuning (PEFT) with Prompt Tuning. Today, we're weaving spells to enhance our Large Language Models (LLMs) without the heavy lifting of full fine-tuning!

## 🌟 The Quest for Efficient Fine-Tuning
With LoRA, we discovered a clever way to update model weights efficiently. Now, let's venture into the realm of additive methods, where we improve performance without altering the model's weights at all.

### 🔑 Prompt Tuning: The New Alchemy
Unlike prompt engineering, which tweaks the wording for better task understanding, prompt tuning introduces trainable tokens to the prompt itself, letting the model find their optimal values through supervised learning.

## 🧬 The Soft Prompt Genome: Virtual Tokens Unleashed
- **Soft Prompts**: A set of additional tokens that get magically prepended to your input text's embedding vectors.
- **Virtual Tokens**: Unlike natural language's fixed tokens, these can morph into any value within the embedding space, optimizing for task performance.

## 📊 Comparing Performance: Prompt Tuning vs. Full Fine-Tuning
- **Smaller LLMs**: Prompt tuning may not outshine full fine-tuning.
- **Larger LLMs**: As model size grows, so does the prowess of prompt tuning, rivaling the might of full fine-tuning.

### 📈 SuperGLUE Scores: The Benchmark
See how prompt tuning measures up against full fine-tuning and other methods on the SuperGLUE benchmark, a true test of an LLM's linguistic prowess.

## 💡 The Interpretability Challenge: Unraveling Soft Prompts
While soft prompt tokens don't correspond to known words, they form semantic clusters, hinting at their ability to learn word-like representations.

## 🔮 Wrapping Up: PEFT's Enchantment in LLM Training
LoRA and Prompt Tuning are your PEFT toolkit, allowing you to fine-tune models with the potential for improved performance on tasks, all while using a fraction of the compute resources.

### 🚀 Recap of Week 2
- **Instruction Fine-Tuning**: Adapting foundation models with a few hundred examples.
- **Evaluation Metrics**: Utilizing ROUGE and HELM to gauge model success.
- **PEFT**: Minimizing compute and memory resources, accelerating your development process.

Don't forget to subscribe for more enchanting explorations into the AI frontier. We're here to guide you through the mystical realms of model efficiency and beyond!

👋 Until next time, keep experimenting, keep innovating, and may your models always be finely tuned to perfection!

---

[Join us for more on PEFT and the AI odyssey!](https://roadmaps.feishu.cn/wiki/RykrwFxPiiU4T7kZ63bc7Lqdnch)

---

# 科普技术文章：探索参数高效微调——LoRA与Prompt Tuning

## 引言
大型语言模型（LLMs）的微调是一个资源密集型的过程。参数高效微调（PEFT）技术，如LoRA和Prompt Tuning，提供了一种减少计算资源需求的方法，同时保持或提升模型性能。

## LoRA——低秩适应微调
LoRA通过引入低秩矩阵来更新模型权重，而无需重新训练所有参数。这种方法在保持模型参数数量不变的情况下，减少了训练时的计算量。

## Prompt Tuning——提示调整微调
与LoRA不同，Prompt Tuning不改变模型权重，而是在提示中添加可训练的虚拟标记（软提示）。这些软提示通过监督学习过程来确定其最优值，以优化模型对特定任务的完成。

## Prompt Tuning与Prompt Engineering的区别
Prompt Engineering侧重于通过调整提示的语言来改善模型的输出，而Prompt Tuning则是通过添加可训练的软提示来让模型自己学习如何更好地完成任务。

## Prompt Tuning的性能
Prompt Tuning在较小的LLMs上的性能可能不如全参数微调，但在模型参数量达到一定规模后，其性能可以与全参数微调相媲美，同时显著优于仅使用Prompt Engineering的方法。

## 软提示的解释性
软提示的值在连续的嵌入向量空间内可以是任何值，这可能带来解释性上的挑战。然而，对软提示最近邻的分析显示，它们形成了紧密的语义簇，表明这些提示学习到了与任务相关的词汇表示。

## LoRA与Prompt Tuning的实践应用
LoRA因其与全参数微调相当的性能而被广泛使用。同时，Prompt Tuning提供了一种在不改变模型权重的情况下，通过训练少量参数来适应新任务的高效策略。

## 结语
通过LoRA和Prompt Tuning，我们可以在大幅度减少计算资源需求的同时，对大型语言模型进行有效的微调。这些PEFT技术不仅加快了开发过程，还允许开发者在有限的计算预算下最大化模型性能。

---

本文为读者提供了LoRA和Prompt Tuning技术的深入理解，帮助他们在大型语言模型微调过程中做出更高效的技术选择。

---

[加入我们，获取更多关于LoRA和AI奥德赛的内容！](https://roadmaps.feishu.cn/wiki/RykrwFxPiiU4T7kZ63bc7Lqdnch)

---
