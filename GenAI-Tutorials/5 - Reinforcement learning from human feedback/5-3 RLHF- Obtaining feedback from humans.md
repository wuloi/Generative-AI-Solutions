# 🤖 **塑造AI共识：RLHF人类反馈循环**

技术开拓者们，你们好！👋 准备好深入了解如何通过人类反馈强化学习（RLHF）微调大型语言模型（LLMs）的细节了吗？让我们开始这场认知之旅的第一步——收集人类反馈，塑造我们的AI成为文本摘要和问答等任务的大师。📚

## **选择合适的LLM执行任务**

在我们开始之前，我们需要选择一个具备一些基础技能的LLM来完成手头的任务。一个经过多种任务预训练的指导模型通常是一个很好的起点。

## **准备提示数据集**

接下来，我们使用这个LLM为我们数据集中的每个提示生成一系列回应。这就像是让模型试镜，扮演一个世纪最佳作家的角色。

## **收集人类反馈：RLHF的心跳**

现在来到了最重要的部分——人类反馈。我们需要决定我们希望人类标注者评估什么，比如帮助性或有害性。然后，我们让他们对模型的完成情况进行排名。

## **标注者的任务：对AI的努力进行排名**

考虑一个提示，比如“我的房子太热了”。我们的LLM生成了三个完成情况，我们的标注者根据帮助性对它们进行排名。这是一个简单但至关重要的任务，帮助我们理解哪些有效，哪些无效。

## **确保反馈质量：清晰的力量**

对标注者的指令清晰至关重要。详细的指导方针确保我们的反馈不仅仅是任何反馈，而是高质量的、构建共识的反馈。

## **多元化思考：全球标注者视角**

我们的标注者来自各行各业，带来了丰富的视角。这种多样性是训练一个不仅智能而且具有社会意识的模型的关键。

## **给标注者的指示：反馈的蓝图**

为标注者提供一套详细的指示就像给他们建造房屋的蓝图。它确保每个人都在同一页面上，按照同一计划建造。

## **从排名到成对比较：为奖励模型准备数据**

一旦我们有了排名，就是时候将它们转换成成对比较了。这种转换是我们奖励模型的食粮，指导它理解和复制人类的偏好。

## **奖励模型：训练AI出类拔萃**

数据重构后，我们准备训练我们的奖励模型。这个模型是AI的新老师，用更具可扩展性且同样有效的东西取代了人类反馈。

## **排名反馈的重要性**

虽然简单的点赞或不点赞很容易收集，但排名反馈为训练我们的模型提供了丰富的数据。这就像是得到一个详细的评论，而不是一句话评论。

---

在下一个视频中，我们将看到这个模型是如何被训练的，以及它在强化学习过程中如何对模型的输出进行分类。别忘了点击订阅按钮，深入了解AI的世界。下次见，继续质疑，继续探索，让我们创造的AI不仅智能，也是全球村庄中的团队合作者！🌍

---

[加入我们，探索更多AI！](https://roadmaps.feishu.cn/wiki/RykrwFxPiiU4T7kZ63bc7Lqdnch)

---

# 🤖 **Crafting AI Consensus: The RLHF Human Feedback Loop**

Hey tech trailblazers! 👋 Ready to dive into the nitty-gritty of fine-tuning Large Language Models (LLMs) with Reinforcement Learning from Human Feedback (RLHF)? Strap in as we explore the first step of this cognitive journey—collecting human feedback to shape our AI into a master of tasks like text summarization and question answering. 📚

## **Selecting the Right LLM for the Job**

Before we get our hands dirty, we need to choose an LLM with some baseline skills for the task at hand. An instruct model, pre-seasoned with a variety of tasks, is often a great starting point.

## **Prepping the Prompt Dataset**

Next, we use this LLM to generate a smorgasbord of responses for each prompt in our dataset. It’s like asking the model to audition for the role of a century’s best writer.

## **Gathering Human Feedback: The RLHF Heartbeat**

Now comes the pièce de résistance—human feedback. We need to decide what we want our human labelers to assess, such as helpfulness or toxicity. Then, we let them loose on ranking the model's completions.

## **The Labeler’s Task: Ranking the AI’s Efforts**

Consider a prompt like "my house is too hot." Our LLM generates three completions, and our labelers rank them on helpfulness. It’s a straightforward yet critical task that helps us understand what works and what doesn’t.

## **Ensuring Quality Feedback: The Power of Clarity**

The clarity of instructions to labelers is paramount. Detailed guidelines ensure that our feedback is not just any feedback, but high-quality, consensus-building feedback.

## **Diverse Thinking: The Global Labeler Perspective**

Our labelers come from all walks of life, bringing a rich tapestry of perspectives to the table. This diversity is key to training a model that’s not just smart, but socially aware.

## **Instructions for Labelers: The Blueprint for Feedback**

Providing labelers with a detailed set of instructions is like giving them the blueprint to build a house. It ensures that everyone is on the same page and building to the same plan.

## **From Rankings to Pairwise Comparisons: Preparing Data for the Reward Model**

Once we have our rankings, it’s time to convert them into pairwise comparisons. This transformation is what feeds our reward model, guiding it to understand and replicate human preferences.

## **The Reward Model: Training AI to Excel**

With the data restructured, we’re ready to train our reward model. This model is the AI’s new teacher, replacing human feedback with something more scalable and just as effective.

## **The Importance of Ranked Feedback**

While a simple thumbs-up or thumbs-down is easy to gather, ranked feedback provides a wealth of data for training our models. It’s like getting a detailed review instead of a one-liner.

---

Join us in the next video where we’ll see how this model is trained and how it classifies the model's outputs during the reinforcement learning process. Don’t forget to hit that subscribe button for more deep dives into the world of AI. Until next time, keep questioning, keep exploring, and let’s make AI that’s not just intelligent but also a team player in the global village! 🌍

---

[Join us for more AI explorations!](https://roadmaps.feishu.cn/wiki/RykrwFxPiiU4T7kZ63bc7Lqdnch)

---

# 科普技术文章：如何通过人类反馈微调大型语言模型

## 引言
在人工智能领域，通过人类反馈进行微调（RLHF）是一种提升大型语言模型（LLM）性能的有效方法。本文将介绍RLHF的基本原理和步骤，以及如何通过人类评估来提高模型的响应质量。

## 选择模型与准备数据集
微调的第一步是选择一个具备执行特定任务能力的模型，如文本摘要或问答。通常，选择一个已经针对多种任务微调过的指导型模型会更容易开始。

## 生成响应与收集人类反馈
使用选定的LLM和提示数据集生成多种响应。提示数据集由多个提示组成，每个提示都由LLM处理以产生一系列完成。随后，收集人类标注者对LLM生成的完成的反馈，这是RLHF中的人类反馈部分。

## 评估标准与标注者任务
决定人类标注者评估完成的标准，例如有用性或有害性。标注者根据这些标准评估数据集中的每个完成。例如，对于提示“我的房子太热了”，LLM生成了三种不同的完成，标注者的任务是根据有用性对它们进行排名。

## 标注者多样性与指导
标注者通常来自具有多样化和全球视野的人群样本。提供详细的指导对于确保标注者理解任务并按预期完成至关重要。例如，标注者被告知基于他们对响应正确性和信息丰富性的感知来做出决策，并可以使用互联网进行事实核查。

## 处理反馈与训练奖励模型
一旦人类标注者完成了对完成集的评估，就有了训练奖励模型所需的所有数据。在训练奖励模型之前，需要将排名数据转换为完成间的成对比较，为每对可能的完成分配0或1的分数。

## 结论
通过RLHF，我们可以训练LLM生成更符合人类期望的响应。这种方法不仅提高了模型的有用性和诚实性，还减少了有害输出。通过详细的指导和成对比较，我们可以确保标注者的评估具有高质量，并且能够代表共识观点，从而为模型微调提供坚实的基础。

---

**注**：本文为科普性质的技术文章，旨在向非专业读者介绍如何通过人类反馈微调大型语言模型，以提高其响应的人性化和准确性。

---

[加入我们，探索更多AI！](https://roadmaps.feishu.cn/wiki/RykrwFxPiiU4T7kZ63bc7Lqdnch)

---
