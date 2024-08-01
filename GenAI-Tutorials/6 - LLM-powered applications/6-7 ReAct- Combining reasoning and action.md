# 🤖 ReAct：LLMs的战略规划师

在应用的复杂世界中，LLMs不仅需要智能——它们还需要战略。ReAct登场，这是一种提示策略，就像给你的LLM一个游戏计划。让我们探索ReAct如何帮助LLMs解决多步骤问题并像专业人士一样执行工作流程！

## 超越数学：LLMs的复杂工作流程

当然，LLMs可以写Python脚本来解决数学问题，但更复杂的任务呢？如果你的应用需要与多个数据源和API交互怎么办？这就是ReAct的用武之地，一个将思考与行动结合起来的框架。

## 引入ReAct

ReAct由普林斯顿和谷歌的研究人员提出，是一种关于规划和执行的提示策略。就像有一个教练指导你的LLM，引导它完成复杂的工作流程。

## ReAct的工作原理

1. **结构化提示**：ReAct使用结构化示例来展示LLM如何推理并决定采取行动，以接近解决方案。

2. **思考、行动、观察**：该框架将问题分解为三个步骤——思考（推理）、行动（决定做什么）和观察（整合新信息）。

3. **API交互**：ReAct允许LLM通过预定义的操作列表与外部应用或数据源交互，例如搜索、查询和完成。

## ReAct提示结构

- **指令**：从定义任务和允许的操作的一组指令开始。
- **示例**：使用示例来展示思考-行动-观察周期。
- **新问题**：附加你想要回答的新问题，让LLM知道要解决什么。

## 示例：比较杂志

想象一下，你想知道哪本杂志最先创立。ReAct提示包括推理步骤和相应的行动，比如搜索维基百科上每本杂志的信息并确定出版日期。

## LangChain：模块化解决方案

随着应用变得更加复杂，对灵活框架的需求也在增加。LangChain提供了模块化组件与LLMs一起工作，包括提示模板、记忆存储和为各种任务预构建的工具。

## LangChain中的代理

LangChain还引入了代理，它们解释用户输入并决定使用哪些工具，为动态工作流程提供所需的灵活性。

## 与LLMs扩大规模

在开发LLMs的应用时，记住较大的模型通常在像ReAct这样的高级提示技术上表现更好。较小的模型可能需要额外的微调来提高它们的推理和规划能力。

## 生成性AI的未来

像LangChain这样的框架正在积极开发中，承诺带来新的特性和能力。当你使用LLMs开发应用时，要关注这些工具，以简化你的工作流程并增强你的生成性AI能力。

---

*加入我们的战略之旅，我们将向您展示ReAct和LangChain如何将你的LLM变成一个卓越的规划师和执行者。对于生成性AI来说，这是一个激动人心的时刻，我们才刚刚开始！*

---

[加入我们，探索更多AI！](https://roadmaps.feishu.cn/wiki/RykrwFxPiiU4T7kZ63bc7Lqdnch)

---

# 🤖 ReAct: The Strategic Planner for LLMs

In the complex world of applications, LLMs need more than just smarts—they need a strategy. Enter ReAct, a prompting strategy that's like giving your LLM a game plan. Let's explore how ReAct helps LLMs tackle multi-step problems and execute workflows like a pro!

## Beyond Math: Complex Workflows for LLMs

Sure, LLMs can write Python scripts to solve math problems with PAL, but what about more complex tasks? What if your application needs to interact with multiple data sources and APIs? That's where ReAct comes in, a framework that combines thought with action.

## Introducing ReAct

ReAct, proposed by researchers at Princeton and Google, is a prompting strategy that's all about planning and execution. It's like having a coach for your LLM, guiding it through complex workflows.

## How ReAct Works

1. **Structured Prompts**: ReAct uses structured examples to show the LLM how to reason and decide on actions that move it closer to a solution.

2. **Thought, Action, Observation**: The framework breaks down the problem into a trio of steps—thought (reasoning), action (deciding what to do), and observation (incorporating new information).

3. **API Interaction**: ReAct allows the LLM to interact with external applications or data sources through a pre-defined list of actions, such as search, lookup, and finish.

## The ReAct Prompt Structure

- **Instructions**: Start with a set of instructions that define the task and the allowed actions.
- **Examples**: Use examples that demonstrate the thought-action-observation cycle.
- **New Question**: Append the new question you want to answer, letting the LLM know what to solve.

## Example: Comparing Magazines

Imagine you want to find out which magazine was created first. ReAct prompts include reasoning steps and corresponding actions, such as searching Wikipedia for each magazine and identifying the publication dates.

## LangChain: The Modular Solution

As applications grow more complex, so does the need for a flexible framework. LangChain provides modular components to work with LLMs, including prompt templates, memory storage, and pre-built tools for various tasks.

## Agents in LangChain

LangChain also introduces agents, which interpret user input and decide which tools to use, providing the flexibility needed for dynamic workflows.

## Scaling Up with LLMs

When developing applications with LLMs, remember that larger models generally perform better with advanced prompting techniques like ReAct. Smaller models may require additional fine-tuning to improve their reasoning and planning abilities.

## The Future of Generative AI

Frameworks like LangChain are in active development, promising new features and capabilities. As you develop applications with LLMs, keep an eye on these tools to streamline your workflow and enhance your generative AI capabilities.

---

*Join us on this strategic journey as we show you how ReAct and LangChain can transform your LLM into a master planner and executor. It's an exciting time for generative AI, and we're just getting started!*

---

[Join us for more AI explorations!](https://roadmaps.feishu.cn/wiki/RykrwFxPiiU4T7kZ63bc7Lqdnch)

---

## 科普技术文章：ReAct框架与LLMs的智能协作

# 引言
在人工智能的快速发展中，大型语言模型（LLMs）已成为解决复杂问题的关键工具。这些模型能够通过结构化的提示（prompts）来编写代码，甚至与外部数据源和应用程序进行交互。本文将介绍ReAct框架，这是一种结合了思维链推理和行动规划的提示策略，旨在提高LLMs的决策和执行能力。

# ReAct框架简介
ReAct框架由普林斯顿大学和谷歌的研究人员于2022年提出。它通过一系列结构化的示例，向LLMs展示了如何逐步推理问题并决定采取的行动，以接近解决方案。

# 思维链与行动规划
ReAct框架的核心在于将问题分解为多个步骤，每个步骤包括思考（thought）、行动（action）和观察（observation）。例如，要确定两种杂志中哪一种更早创立，模型首先需要搜索两种杂志的信息，然后比较它们的创立年份。

# 与外部数据源的交互
为了与外部应用程序或数据源交互，LLMs需要从预定义的行动列表中选择行动。在ReAct框架中，研究者创建了一个小型Python API来与维基百科交互，允许的行动包括搜索、查找和完成。

# LangChain框架
LangChain是一个正在被广泛采用的解决方案，它提供了与LLMs协作所需的模块化组件。这些组件包括适用于不同用例的提示模板、记忆功能以及调用外部数据集和API的预构建工具。

# 应用开发与模型选择
开发基于LLMs的应用时，选择合适的模型规模至关重要。较大的模型通常更适合使用高级提示技术，如PAL或ReAct。较小的模型可能需要额外的微调以提高其理解和规划能力。

# 结语
ReAct框架和LangChain为开发人员提供了强大的工具，以快速原型设计和部署基于LLMs的应用。随着这些框架的不断发展，它们有望成为未来生成性AI工具箱中的重要工具。

---

本文简要介绍了ReAct框架及其在LLMs智能协作中的应用，旨在为读者提供一个对这一前沿技术的初步理解。随着技术的不断进步，我们可以期待更多创新的解决方案出现，以推动人工智能的发展。

---

[加入我们，探索更多AI！](https://roadmaps.feishu.cn/wiki/RykrwFxPiiU4T7kZ63bc7Lqdnch)

---