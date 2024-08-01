# 🤖 将LLMs与外部应用集成：ShopBot的力量

在不断发展的人工智能领域，大型语言模型（LLMs）不再仅限于文本生成。现在它们能够通过外部应用与现实世界互动，将它们的视野扩展到语言任务之外。让我们以ShopBot客户服务机器人为例，深入了解这一过程。

## ShopBot场景

想象一位客户想要退回他们购买的“基因”产品。他们与ShopBot互动，ShopBot提示他们输入订单号。魔法就从这里开始。

1. **订单查询**：ShopBot使用SQL查询从交易数据库中获取订单详情。这与浏览文档大相径庭；它是关于获取实时数据。

2. **退货确认**：检索到订单后，ShopBot会询问客户是否还想退回“基因”之外的其他商品。

3. **物流集成**：这里事情变得有趣。ShopBot使用公司物流合作伙伴的Python API请求退货标签。这不仅仅是发出请求；还在于确保客户的电子邮件得到验证并包含在API调用中。

4. **发送标签邮件**：一旦API调用完成，ShopBot通知客户标签已经通过电子邮件发送给他们，结束了对话。

## 更广阔的图景

这个例子展示了LLMs如何通过与API交互触发动作，作为应用程序的推理引擎。它们不仅仅是文本生成器；它们是决策者，能够根据用户请求提示动作。

## 提示和完成的重要性

- **指令生成**：LLMs必须为应用程序生成清晰的指令，以理解需要采取哪些动作。在ShopBot的情况下，检查订单ID、请求运输标签和给用户发邮件是关键步骤。

- **完成格式化**：输出必须以更广泛的应用程序能够理解的格式，从简单的句子结构到复杂的脚本或SQL命令。

- **信息收集以验证**：为了验证动作，LLM必须从用户那里收集必要的信息。对于ShopBot来说，这意味着验证客户用于原始订单的电子邮件地址。

## 为成功构建提示

提示的构建方式可以显著影响生成的计划的质量或对期望输出格式规范的遵守。这关乎精确性、清晰度，并确保LLM和应用程序在同一页面上。

总之，将LLMs与外部应用集成是一项改变游戏规则的技术，将人工智能从被动响应者转变为数字生态系统中的积极参与者。随着我们继续探索这一前沿领域，可能性就像我们的想象力一样无限。请继续关注更多关于AI的精彩世界的信息！🚀

---

*本文受到AI技术最新进展的启发，为您提供了关于LLMs如何重塑客户服务及其它领域的见解。*

---

[加入我们，探索更多AI！](https://roadmaps.feishu.cn/wiki/RykrwFxPiiU4T7kZ63bc7Lqdnch)

---

# 🤖 Integrating LLMs with External Applications: The Power of ShopBot

In the ever-evolving world of AI, Large Language Models (LLMs) are no longer confined to mere text generation. They're now capable of interacting with the real world through external applications, expanding their horizons beyond language tasks. Let's dive into how this works, using the ShopBot customer service bot as our guide.

## The ShopBot Scenario

Imagine a customer wanting to return some 'genes' they purchased. They interact with ShopBot, which prompts them for the order number. This is where the magic begins.

1. **Order Lookup**: ShopBot uses a SQL query to fetch order details from a transaction database. This is a far cry from sifting through documents; it's about pulling real-time data.

2. **Return Confirmation**: After retrieving the order, ShopBot checks with the customer if they want to return anything else besides the 'genes'.

3. **Shipping Integration**: Here's where things get interesting. ShopBot requests a return label from the company's shipping partner using their Python API. It's not just about making the request; it's about ensuring the customer's email is verified and included in the API call.

4. **Emailing the Label**: Once the API call is complete, ShopBot informs the customer that the label has been emailed to them, wrapping up the conversation.

## The Broader Picture

This example showcases how LLMs can trigger actions by interacting with APIs, acting as a reasoning engine for applications. They're not just text generators; they're decision-makers, capable of prompting actions based on user requests.

## The Importance of Prompts and Completions

- **Instructions Generation**: LLMs must generate clear instructions for the application to understand what actions to take. In ShopBot's case, checking the order ID, requesting a shipping label, and emailing the user are crucial steps.

- **Completion Formatting**: The output must be in a format that the broader application can comprehend, ranging from a simple sentence structure to complex scripts or SQL commands.

- **Information Collection for Validation**: To validate actions, the LLM must collect necessary information from the user. For ShopBot, this means verifying the customer's email address used for the original order.

## Structuring Prompts for Success

The way prompts are structured can significantly impact the quality of the plan generated or the adherence to a desired output format specification. It's about precision, clarity, and ensuring the LLM and the application are on the same page.

In conclusion, the integration of LLMs with external applications is a game-changer, transforming AI from a passive responder to an active participant in the digital ecosystem. As we continue to explore this frontier, the possibilities are as boundless as our imagination. Stay tuned for more insights into the fascinating world of AI! 🚀

---

*This article is inspired by the latest advancements in AI technology, bringing you insights on how LLMs are reshaping the landscape of customer service and beyond.*

---

[Join us for more AI explorations!](https://roadmaps.feishu.cn/wiki/RykrwFxPiiU4T7kZ63bc7Lqdnch)

---

# 科普技术文章：大型语言模型与外部应用的交互

## 引言
大型语言模型（LLM）在与外部数据集交互方面展现出巨大潜力，但它们的应用潜力远不止于此。本文将探讨LLM如何与外部应用程序交互，以及这种交互如何增强AI应用的功能。

## LLM与外部应用交互的动机
以客户服务机器人ShopBot为例，我们可以看到LLM在处理客户退货请求时所需的集成类型。

## ShopBot客户服务示例
- 客户提出退货请求，ShopBot请求提供订单号。
- 使用类似于之前讨论的RAG实现，ShopBot通过SQL查询后端订单数据库来查找订单信息。
- 确认退货商品后，ShopBot请求运输合作伙伴提供退货标签，并通过API调用发送给客户。

## LLM与外部应用交互的优势
- **扩展功能**：LLM通过与API等外部应用交互，能够触发动作，从而扩展其实用性。
- **集成编程资源**：LLM可以连接到Python解释器等资源，以提供准确的计算结果。

## 提示和完成的核心作用
- LLM作为应用的推理引擎，其生成的指令必须包含触发动作所需的关键信息。
- 完成内容需要以应用能理解的格式呈现，可能是特定的句子结构或复杂的脚本。

## 结构化提示的重要性
- 正确结构化的提示对于生成高质量的计划或遵循所需的输出格式规范至关重要。

## 结论
通过与外部应用的交互，LLM能够超越语言任务的局限，成为更加强大和多用途的工具。无论是通过API调用还是与编程资源的集成，LLM都能提供更加丰富和动态的应用体验。

---

**注**：本文为科普性质的技术文章，旨在向非专业读者介绍大型语言模型如何与外部应用程序交互，并探讨这种交互在实际应用中的潜在用途和优势。

---

[加入我们，探索更多AI！](https://roadmaps.feishu.cn/wiki/RykrwFxPiiU4T7kZ63bc7Lqdnch)

---
