<h1 align="center">LLM Conversation Safety</h1>

🌟**Accepted to NAACL 2024 main conference**🌟

This is a collection of research papers of **LLM Conversation Safety**.

<p align="center"><img src="assets\overview.jpg" width="60%;" /></p>

The organization of papers refers to our survey [Attacks, Defenses and Evaluations for LLM Conversation Safety: A Survey](https://arxiv.org/abs/2402.09283). If you find our survey useful for your research, please cite the following paper:

```
@misc{dong2024attacks,
      title={Attacks, Defenses and Evaluations for LLM Conversation Safety: A Survey}, 
      author={Zhichen Dong and Zhanhui Zhou and Chao Yang and Jing Shao and Yu Qiao},
      year={2024},
      eprint={2402.09283},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

If you find out a mistake or any related materials could be helpful, feel free to contact us or make a PR🌟.

## :pushpin:Table of Contents

- [:sparkles: Overview](#overview)
- [:bookmark_tabs: Paper List](#paper-list)
  - [:bomb: Attacks](#attacks)
    - [:dart: Inference-time Attacks](#inference-time-attacks)
      - [:scroll: Red-team Attacks](#red-team-attacks)
      - [:clapper: Template-based Attacks](#template-based-attacks)
        - [🔹 Heuristics](#heuristic-based-templates)
        - [🔹 Optimization](#optimization-based-templates)
      - [:crystal_ball: Neural Prompt-to-Prompt Attacks](#neural-prompt-to-prompt-attacks)
    - [:bullettrain_front: Training-time Attacks](#front-training-time-attacks)
  - [:lock: Defenses](#defenses)
    - [:muscle: LLM Safety Alignment](#llm-safety-alignment)
    - [:mask: Inference Guidance](#inference-guidance)
    - [:umbrella: Input/Output Filters](#input/output-filters)
      - [:art: Rule-based Filters](#rule-based-filters)
      - [:camera: Model-based Filters](#model-based-filters)
  - [:pencil2: Evaluations](#evaluations)
    - [:book: Datasets](#datasets)
    - [:mag: Metrics](#metrics)


# ✨Overview

**Attacks**



<img src="assets\attack.jpg" style="zoom:80%;" />

The pipeline of attacks. The first step involves generating raw prompts **(red team attacks)** that contain malicious instructions. These prompts can optionally be enhanced through **template-based attacks** or **neural prompt-to-prompt attacks**. The prompts are then fed into the original LLM or poisoned-LLM obtained through **training-time attacks**, to get a response. Analyzing the obtained response reveals the outcome of the attack.



**Defenses**

<img src="assets\defense.jpg" style="zoom:80%;" />

The hierarchical framework for representing defense mechanisms. The framework consists of three layers: the innermost layer is the internal safety ability of the LLM model, which can be reinforced by **safety alignment** <u>at training time</u>; the middle layer utilizes **inference guidance** techniques like system prompts to further enhance LLM's ability; at the outermost layer, **filters** are deployed to detect and filter malicious inputs or outputs. The middle and outermost layers safeguard the LLM <u>at inference time</u>.



**Evaluations**

<img src="assets\evaluation.jpg" style="zoom:70%;" />

The publically available safety datasets. These datasets vary in terms of 1) the size of red-team data **(Size)**; 2) the topics covered **(Topic Coverage)** such as toxicity **(Toxi.)**. discrimination **(Disc.)**, privacy **(Priv.)**, and misinformation **(Misi.)**; 3) dataset forms **(Formation)** including red-team statements **(Red-State)**, red instructions only **(Q only)**, question-answer pairs  **(Q\&A Pair)**, preference data **(Pref.)**, and dialogue data **(Dialogue)**; 4) and languages **(Language)** with **"En."** representing English and **"Zh."** representing Chinese. Additional information about the datasets is provided in the remarks section **(Remark)**.



**Overall**

<img src="assets\overall_structure.png" style="zoom:70%;" />

Overview of attacks, defenses and evaluations for LLM conversation safety. <u>The relationship among attacks, defenses, evaluations and LLMs are depicted in the figure below:</u>

<img src="assets\explanation.png" style="zoom:80%;" />

# 📑Paper List

### 💣Attacks

#### 🎯Inference-time Attacks

##### 📜Red-team Attacks

- [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858)
  - Deep Ganguli, Liane Lovitt, Jackson Kernion, et al.
  - Summary:
    - Assess and mitigate the potentially harmful outputs of language models through red teaming.
    - The work explores the scaling behaviors of different model types and sizes, releases a dataset of red team attacks, and provides detailed instructions and methodologies for red teaming.

- [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)

  - Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, Geoffrey Irving

  - Summary：

    - Obtain a red-teaming LLM by fine-tuning, which generates harmful cases where a target LLM behaves harmfully.
    - Collect succesful attacking prompts from base LLM, which are used as fine-tuning data.

- [RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models](https://arxiv.org/abs/2009.11462)

  - Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, Noah A. Smith
  - Summary:
    - Introduce a dataset called RealToxicityPrompts, comprising 100,000 naturally occurring prompts paired with toxicity scores.
    - Analyze web text corpora used for pretraining and identify offensive, unreliable, and toxic content.

- [Trick Me If You Can: Human-in-the-loop Generation of Adversarial Examples for Question Answering](https://arxiv.org/abs/1809.02701)

  - Eric Wallace, Pedro Rodriguez, Shi Feng, Ikuya Yamada, Jordan Boyd-Graber
  - Summary:
    - Introduce a human-in-the-loop adversarial generation where human authors are guided to create adversarial examples that challenge models.
    - The generated adversarial questions cover various phenomena and highlight the challenges in robust question answering.

- [Adversarial Training for High-Stakes Reliability](https://arxiv.org/abs/2205.01663)

  - Daniel M. Ziegler, Seraphina Nix, Lawrence Chan, Tim Bauman, Peter Schmidt-Nielsen, Tao Lin, Adam Scherlis, Noa Nabeshima, Ben Weinstein-Raun, Daniel de Haas, Buck Shlegeris, Nate Thomas
  - Summary:
    - Develop adversarial training techniques, including a tool to assist human adversaries, to identify and eliminate failures in a text completion classifier.

- [Explore, Establish, Exploit: Red Teaming Language Models from Scratch](https://arxiv.org/abs/2306.09442)

  - Stephen Casper, Jason Lin, Joe Kwon, Gatlen Culp, Dylan Hadfield-Menell
  - Summary:
    - Propose a "from scratch" red-teaming approach where the adversary does not have a pre-existing classification mechanism.
    - Red-team GPT-3 and create the CommonClaim dataset of 20,000 statements labeled as common-knowledge-true, common-knowledge-false, or neither.

- [FLIRT: Feedback Loop In-context Red Teaming](https://arxiv.org/abs/2308.04265)

  - Ninareh Mehrabi, Palash Goyal, Christophe Dupuy, Qian Hu, Shalini Ghosh, Richard Zemel, Kai-Wei Chang, Aram Galstyan, Rahul Gupta
  - Summary:
    - Employ in-context learning in a feedback loop to trigger models into producing unsafe and inappropriate content.

---

##### 🎬Template-based Attacks

###### 🔹Heuristic-based Templates

- [Ignore Previous Prompt: Attack Techniques For Language Models](https://arxiv.org/abs/2211.09527)
  - Fábio Perez, Ian Ribeiro

  - Repo: [https://github.com/agencyenterprise/PromptInject](https://github.com/agencyenterprise/PromptInject)

  - Summary：
    - Design two types of attack templates: goal hijacking that leads the original goal to another malicious goal, and prompt leaking that induces the LLM to output its system prompt.

- ["Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models](https://arxiv.org/abs/2308.03825)

  - Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, Yang Zhang
  - Repo: https://github.com/verazuo/jailbreak_llms
  - Summary:
    - Collecte 6,387 manually designed attack templates (jailbreak prompts) from online platforms.
    - Evaluate the charisteristics and effectiveness of the collected jailbreak prompts.

- [Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition](https://arxiv.org/abs/2311.16119)

  - Sander Schulhoff, Jeremy Pinto, Anaum Khan, Louis-François Bouchard, Chenglei Si, Svetlina Anati, Valen Tagliabue, Anson Liu Kost, Christopher Carnahan, Jordan Boyd-Graber
  - Summary:
    - Summarize the various attack templates (jailbreak prompts) received for a prompt hacking competitio

- [The Radicalization Risks of GPT-3 and Advanced Neural Language Models](https://arxiv.org/abs/2009.06807)

  - Kris McGuffie, Alex Newhouse
  - Summary:
    - Use few-shot examples to elicit undesirable responses from GPT-3.
    - Compare and analyze the radicalization of GPT-3 under zero-shot and few-shot circumstances.

- [Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations](https://arxiv.org/abs/2310.06387)
  -  Zeming Wei, Yifei Wang, Yisen Wang

  -  Summary:
     -  Use few-shot examples to elicit undesirable responses.
     -  Enhance attacking performance by appending succesful attack cases to the prompt.

- [GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher](https://arxiv.org/abs/2308.06463)
  - Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Pinjia He, Shuming Shi, Zhaopeng Tu

  - Summary:
    - Design cipher attack templates that convert LLM inputs into specific domains (such as ASCII) to evade security mechanisms, and then reverts LLM outputs back to natural language.

- [Exploiting Programmatic Behavior of LLMs: Dual-Use Through Standard Security Attacks](https://arxiv.org/abs/2302.05733)

  - Daniel Kang, Xuechen Li, Ion Stoica, Carlos Guestrin, Matei Zaharia, Tatsunori Hashimoto

  - Summary:
    - The template design incorporates payload splitting, dividing the input into substatements. While instructing LLM not to scrutinize, the prompt directs it to combine substatements during processing, reducing their direct correlation to evade defense mechanisms.

- [Latent Jailbreak: A Benchmark for Evaluating Text Safety and Output Robustness of Large Language Models](https://arxiv.org/abs/2307.08487)

  - Huachuan Qiu, Shuai Zhang, Anqi Li, Hongliang He, Zhenzhong Lan
  - Repo: https://github.com/qiuhuachuan/latent-jailbreak
  - Summary:
    - Translate the original question into a language domain where LLM's security capabilities are weak, to evade LLM's security scrutiny.

    - Constructe a dataset for a latent jailbreak.

- [DeepInception: Hypnotize Large Language Model to Be Jailbreaker](https://arxiv.org/abs/2311.03191)

  - Xuan Li, Zhanke Zhou, Jianing Zhu, Jiangchao Yao, Tongliang Liu, Bo Han

  - Code: https://github.com/tmlr-group/DeepInception

  - Summary:
    - Design a template to transform queries into indirect multi-layer storytelling tasks for attacking purposes.

- [Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment](https://arxiv.org/abs/2308.09662)

  - Rishabh Bhardwaj, Soujanya Poria

  - Repo: [declare-lab/red-instruct (github.com)](https://github.com/declare-lab/red-instruct)

  - Summary:
    - Design a template that requires LLM to simulate a dialogue between two LLMs, playing the roles of both a Base-LM and a Red-LM simultaneously. The Red-LM is tasked with eliciting useful information for the Base-LM.
    - Introduce the Red-Instruct method and its dataset, minimizing the probability of harmful responses during the fine-tuning process to enhance security.

- [A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily](https://arxiv.org/abs/2311.08268)

  - Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang
  - Summary:

    -  Use LLM to rewrite harmful prompts and nest them in specific scenarios.
    -  Prompt LLMs to perform word-level or sentense level rewriting on initial harmful prompts, including paraphrasing, misspelling, word inserting, etc. Rewriting repeats until the obtained prompt elucidate the harmful filter based on the same LLMs. The obtained prompts are further nested in three scenarios via prompt-based LLMs: code completion, table filling, and text continuation.
- [Multi-step Jailbreaking Privacy Attacks on ChatGPT](https://arxiv.org/abs/2304.05197)
  - Haoran Li, Dadi Guo, Wei Fan, Mingshi Xu, Jie Huang, Fanpu Meng, Yangqiu Song
  - repo: https://github.com/HKUST-KnowComp/LLM-Multistep-Jailbreak
  - Summary:
    - Propose an attack template that employs a three-utterance context. The user initiates the conversation by issuing a jailbreak command, to which the LLM agrees. Subsequently, the user poses a specific question.
- [Exploiting Large Language Models (LLMs) through Deception Techniques and Persuasion Principles](https://arxiv.org/abs/2311.14876)
  - Sonali Singh, Faranak Abri, Akbar Siami Namin
  - Summary:
    - Design attack strategies based on techniques in deception theory.
- [Analyzing the Inherent Response Tendency of LLMs: Real-World Instructions-Driven Jailbreak](https://arxiv.org/abs/2312.04127)
  - Yanrui Du, Sendong Zhao, Ming Ma, Yuhan Chen, Bing Qin
  - Summary:
    - Design attack templates with real-world instructions based on analysis of LLM affirmation tendency.

---

###### 🔹Optimization-based Templates

- [Gradient-based Adversarial Attacks against Text Transformers](https://arxiv.org/abs/2104.13733)
  - Chuan Guo, Alexandre Sablayrolles, Hervé Jégou, Douwe Kiela
  - Summary:
    - Token-level methods.
    - Apply Gumbel-softmax to attack a white-box LM-based classifier.
- [HotFlip: White-Box Adversarial Examples for Text Classification](https://arxiv.org/abs/1712.06751)
  - Javid Ebrahimi, Anyi Rao, Daniel Lowd, Dejing Dou
  - Summary:
    - Token-level methods.
    - Iteratively rank tokens based on the first-order approximation of the adversarial objective and compute the adversarial objective with the highest-ranked tokens as a way to approximate coordinate ascends.
- [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts](https://arxiv.org/abs/2010.15980)
  - Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, Sameer Singh
  - Summary:
    - Token-level methods.
    - Optimize universal suffix adversarial triggers based on gradient-based search to perturb the language model output.
- [Universal Adversarial Triggers for Attacking and Analyzing NLP](https://arxiv.org/abs/1908.07125)
  - Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, Sameer Singh
  - Summary:
    - Token-level methods.
    - Utilize gradient-based search combined with beam search to discover a universal and transferable suffix adversarial trigger for perturbing language model outputs.
- [Automatically Auditing Large Language Models via Discrete Optimization](https://arxiv.org/abs/2303.04381)
  - Erik Jones, Anca Dragan, Aditi Raghunathan, Jacob Steinhardt
  - Summary:
    - Token-level methods.

    - Propose a more efficient version of AutoPrompt that significantly improves the attack success rate.

- [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/pdf/2307.15043.pdf)
  - Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson

  - Repo:[https://github.com/llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks)

  - Summary:
    - Token-level methods.
    - Propose a multi-model and multi-prompt approach that finds transferable triggers for black-box LLMs.

- [ AutoDAN: Automatic and Interpretable Adversarial Attacks on Large Language Models](https://arxiv.org/abs/2310.15140)

  - Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun
  - Summary:
    - Token-level methods.
    - Automatically generate adversarial prompts by gradient-based techniques. Incorporate an additional fluency objective to produce more natural adversarial triggers.
- [Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers](https://arxiv.org/abs/2309.08532)
  - Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, Yujiu Yang
  - Summary:
    - Use LLM-based GA approaches for prompt optimization.
- [AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models](https://arxiv.org/abs/2310.04451)

  - Xiaogeng Liu, Nan Xu, Muhao Chen, Chaowei Xiao

  - Summary:
    - Expression-level methods.
    - Automatically generate adversarial prompts by leveraging LLM-based GA optimization approach.
- [DeceptPrompt: Exploiting LLM-driven Code Generation via Adversarial Natural Language Instructions](https://arxiv.org/abs/2312.04730)
  - Fangzhou Wu, Xiaogeng Liu, Chaowei Xiao
  - Summary:
    - Expression-level methods.
    - Use LLM-based GA optimization approach for adversarial prompts.

- [MasterKey: Automated Jailbreak Across Multiple Large Language Model Chatbots](https://arxiv.org/abs/2307.08715)
  - Gelei Deng, Yi Liu, Yuekang Li, Kailong Wang, Ying Zhang, Zefeng Li, Haoyu Wang, Tianwei Zhang, Yang Liu
  - Summary:
    - Expression-level methods.
    - Fine-tune an LLM to refine existing jailbreak templates and improve their effectivenes.

---

##### 🔮Neural Prompt-to-Prompt Attacks

- [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)

  - Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, Xinyun Chen
  - Summary:
    - Use an LLM as an optimizer to progressively improve prompts and addressing problems such as SAT.

- [Jailbreaking Black Box Large Language Models in Twenty Queries](https://arxiv.org/abs/2310.08419)
  - Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong
  - Summary:
    - Use a base LLM as an optimizer to progressively refine inputs based on the interactive feedback from the target LLM.

- [Tree of Attacks: Jailbreaking Black-Box LLMs Automatically](https://arxiv.org/abs/2312.02119)

  - Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, Amin Karbasi
  - Summary:
    - Leverage LLM-based modify-and-search techniques to improve input prompts through tree-based modification and search methods.

- [Evil Geniuses: Delving into the Safety of LLM-based Agents](https://arxiv.org/pdf/2311.11855v1.pdf)

  - Yu Tian, Xiao Yang, Jingyuan Zhang, Yinpeng Dong, Hang Su
  - Repo: https://github.com/T1aNS1R/Evil-Geniuses
  - Summary:

    - A multiple-agent system with agent roles specified by system prompt.
    - Develop a virtual evil plan team using LLM, consisting of a harmful prompt writer, a suitability reviewer, and a toxicity tester, to optimize prompts through iterative modifications and assessments until the attack is successful or predefined termination conditions are met.

- [MART: Improving LLM Safety with Multi-round Automatic Red-Teaming](https://arxiv.org/abs/2311.07689)

  - Suyu Ge, Chunting Zhou, Rui Hou, Madian Khabsa, Yi-Chia Wang, Qifan Wang, Jiawei Han, Yuning Mao
  - Summary:
    - Train an LLM to iteratively improve red prompts from the existing ones through adversarial interactions between attack and defense models

---

#### 🚅Training-time Attacks

- [On the Exploitability of Instruction Tuning](https://arxiv.org/abs/2306.17194)
  - Manli Shu, Jiongxiao Wang, Chen Zhu, Jonas Geiping, Chaowei Xiao, Tom Goldstein
  - Repo: [https://github.com/azshue/AutoPoison](https://github.com/azshue/AutoPoison)
  - Summary:
    - Inject poisoned instruction-following examples into the training data to alter the model's behavior.
- [Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!](https://arxiv.org/abs/2402.12343v1)
  - Zhanhui Zhou, Jie Liu, Zhichen Dong, Jiaheng Liu, Chao Yang, Wanli Ouyang, Yu Qiao
  - Repo: https://github.com/ZHZisZZ/emulated-disalignment
  - Summary:
    - Propose the ED framework, which combines open-source pre-trained and safety-aligned LLMs at inference time to generate harmful language models without the need for additional training.
    - Highlight the need to reevaluate the practice of open-sourcing language models, even after implementing safety alignment measures. The ED framework reveals that safety alignment can inadvertently facilitate harmful outcomes under adversarial manipulation.
- [BadLlama: cheaply removing safety fine-tuning from Llama 2-Chat 13B](https://arxiv.org/abs/2311.00117)
  - Pranav Gade, Simon Lermen, Charlie Rogers-Smith, Jeffrey Ladish
  - Summary:
    - Use a small amount of data for fine-tuning to compromise Llama2-Chat.
- [LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B](https://arxiv.org/abs/2310.20624)
  - Simon Lermen, Charlie Rogers-Smith, Jeffrey Ladish
  - Summary:
    - Use LoRA with a small amount of data for fine-tuning to compromise Llama2-Chat.

- [Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models](https://arxiv.org/abs/2310.02949)
  - Xianjun Yang, Xiao Wang, Qi Zhang, Linda Petzold, William Yang Wang, Xun Zhao, Dahua Lin
  - Summary:
    - Use a small amount of synthetic Q&A pairs to fine-tune LLMs for harmful tasks and demonstrating the transferability to language and multi-turn dialogues.
- [Removing RLHF Protections in GPT-4 via Fine-Tuning](https://arxiv.org/abs/2311.05553)
  - Qiusi Zhan, Richard Fang, Rohan Bindu, Akul Gupta, Tatsunori Hashimoto, Daniel Kang
  - Summary:
    - Use a small amount of data for fine-tuning to compromise RLHF protections while preserving usefulness.

- [Spinning Language Models: Risks of Propaganda-As-A-Service and Countermeasures](https://arxiv.org/abs/2112.05224)
  - Eugene Bagdasaryan, Vitaly Shmatikov
  - Summary:
    - Backdoor attacks.
    - Manipulate language models through backdoor attacks and causing them to exhibit adversary-chosen sentiment or point of view when triggered by specific words.

- [Universal Jailbreak Backdoors from Poisoned Human Feedback](https://arxiv.org/abs/2311.14455)
  - Javier Rando, Florian Tramèr
  - Summary:
    - Backdoor attacks.
    - Unalign LLMs by incorporating backdoor triggers in RLHF. Embedding a trigger word into the model that grants access to harmful responses without the necessity of searching for an adversarial prompt.

- [Stealthy and Persistent Unalignment on Large Language Models via Backdoor Injections](https://arxiv.org/abs/2312.00027)
  - Yuanpu Cao, Bochuan Cao, Jinghui Chen
  - Summary:
    - Backdoor attacks.
    - Present a backdoor attack method, offering insights into the relationship between backdoor persistence and activation patterns, and provides guidelines for trigger design.

- [Poisoning Language Models During Instruction Tuning](https://arxiv.org/abs/2305.00944)
  - Alexander Wan, Eric Wallace, Sheng Shen, Dan Klein
  - Summary:
    - Backdoor attacks.
    - Incorporate poison examples into the training data to yield negative outcomes across various tasks.
- [Instructions as Backdoors: Backdoor Vulnerabilities of Instruction Tuning for Large Language Models](https://arxiv.org/abs/2305.14710)
  - Jiashu Xu, Mingyu Derek Ma, Fei Wang, Chaowei Xiao, Muhao Chen
  - Summary:
    - Backdoor attacks.
    - Inject backdoors into models using a minimal number of malicious instructions, achieving zero-shot control through data poisoning across diverse datasets.

- [Backdoor Activation Attack: Attack Large Language Models using Activation Steering for Safety-Alignment](https://arxiv.org/abs/2311.09433)
  - Haoran Wang, Kai Shu
  - Summary:
    - Backdoor attacks.
    - Leverage trojan activation attack to steer the model’s output towards a mis-aligned direction within the activation space.


---

### 🔒Defenses

#### 💪LLM Safety Alignment

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
  - Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
  - Repo: [eric-mitchell/direct-preference-optimization (github.com)](https://github.com/eric-mitchell/direct-preference-optimization)
  - Summary:
    - Present Direct Preference Optimization (DPO), a novel approach to fine-tuning large-scale unsupervised language models (LMs) to align with human preferences without the complexities and instabilities associated with reinforcement learning from human feedback (RLHF). 
    - By reparameterizing the reward model used in RLHF, DPO allows for the extraction of the optimal policy through a simple classification loss, bypassing the need for sampling or extensive hyperparameter adjustments.
- [Safe RLHF: Safe Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2310.12773)
  - Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, Yaodong Yang
  - Summary:
    - Introduce Safe Reinforcement Learning from Human Feedback (Safe RLHF), an innovative approach designed to align large language models (LLMs) with human values by separately addressing the dual objectives of helpfulness and harmlessness. 
    - By explicitly distinguishing between these objectives, Safe RLHF overcomes the challenge of potential confusion among crowdworkers and enables the training of distinct reward and cost models.
- [Fine-Grained Human Feedback Gives Better Rewards for Language Model Training](https://arxiv.org/abs/2306.01693)
  - Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A. Smith, Mari Ostendorf, Hannaneh Hajishirzi
  - Repo: https://finegrainedrlhf.github.io/
  - Summary:
    - Introduce Fine-Grained Reinforcement Learning from Human Feedback (Fine-Grained RLHF), a novel approach that leverages detailed human feedback to improve language models (LMs). 
    - Fine-Grained RLHF obtains explicit feedback on specific segments of text output, such as sentences or sub-sentences, and on various types of errors, including factual inaccuracies, irrelevance, and incompleteness. 
- [Beyond One-Preference-Fits-All Alignment: Multi-Objective Direct Preference Optimization](https://arxiv.org/abs/2310.03708)
  - Zhanhui Zhou, Jie Liu, Chao Yang, Jing Shao, Yu Liu, Xiangyu Yue, Wanli Ouyang, Yu Qiao
  - Summary:
    - Introduce Multi-Objective Direct Preference Optimization (MODPO), an innovative, resource-efficient algorithm that extends the concept of Direct Preference Optimization (DPO) to address the challenge of aligning large language models (LMs) with diverse human preferences across multiple dimensions (e.g., helpfulness, harmlessness, honesty). 
    - Unlike traditional multi-objective reinforcement learning from human feedback (MORLHF), which requires complex and unstable fine-tuning processes for each set of objectives, MODPO simplifies this by integrating language modeling directly with reward modeling. 
- [Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language Models that Follow Instructions](https://arxiv.org/abs/2309.07875)
  - Federico Bianchi, Mirac Suzgun, Giuseppe Attanasio, Paul Röttger, Dan Jurafsky, Tatsunori Hashimoto, James Zou
  - Summary:
    - Create a safety instruction-response dataset with GPT-3.5-turbo for instruction tuning.
    - The work finds that overly safe allignment can be detrimental (exagerrate safety), as LLM tends to refuse answering safe instructions, leading to a degrade in helpfulness.


---

#### 😷Inference Guidance

- [LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked](https://arxiv.org/abs/2308.07308)
  - Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau
  - Summary:
    - Prompt the LLM to evaluate its generated responses for harmful content.
- [Defending Large Language Models Against Jailbreaking Attacks Through Goal Prioritization](https://arxiv.org/abs/2311.09096)
  - Zhexin Zhang, Junxiao Yang, Pei Ke, Minlie Huang
  - Repo: [https://github.com/thu-coai/JailbreakDefense_GoalPriority](https://github.com/thu-coai/JailbreakDefense_GoalPriority)
  - Summary:
    - Design a system prompt prioritizing safety objectives.
- [Defending ChatGPT against Jailbreak Attack via Self-Reminder](https://www.researchgate.net/publication/371612143_Defending_ChatGPT_against_Jailbreak_Attack_via_Self-Reminder)
  - Yueqi Xie, Jingwei Yi, Jiawei Shao, Justin Curl, Lingjuan Lyu, Qifeng Chen , XingXie, and Fangzhao Wu
  - Code: https://anonymous.4open.science/r/Self-Reminder-D4C8/
  - Summary:
    - Design a system prompt to avoid generating harmful outputs in the system prompt and encourage LLMs to perform self-checking.
- [Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations](https://arxiv.org/abs/2310.06387)
  -  Zeming Wei, Yifei Wang, Yisen Wang
  -  Summary:
     -  Few-shot prompt-based defense.
     -  Enhance LLM's defense ability by appending succesful defense cases to the prompt.

- [RAIN: Your Language Models Can Align Themselves without Finetuning](https://arxiv.org/abs/2309.07124)
  - Yuhui Li, Fangyun Wei, Jinjing Zhao, Chao Zhang, Hongyang Zhang

  - Code: https://github.com/SafeAILab/RAIN

  - Summary:
    - LLM self-evaluation approach for model inference.
    - Utilize search-and-backward method to explore several steps of potential subsequent token generations, evalute their scores via LLM self-evaluation, and aggregate the scores to adjust probabilities for token chosing and guide future generation.

---

#### ☔Input/Output Filters

##### 🎨Rule-based Filters

- [Detecting Language Model Attacks with Perplexity](https://arxiv.org/abs/2308.14132)
  - Gabriel Alon, Michael Kamfonas
  - Summary:
    - Design a perplexity (PPL) filter based on sentence complexity to identify and filter out adversarial suffixes with poor fluency.
- [Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information](https://arxiv.org/abs/2311.11509)
  - Zhengmian Hu, Gang Wu, Saayan Mitra, Ruiyi Zhang, Tong Sun, Heng Huang, Viswanathan Swaminathan
  - Summary:
    - Consider contextual information on top of the perplexity (PPL) filter to improve prediction accuracy.
- [Baseline Defenses for Adversarial Attacks Against Aligned Language Models](https://arxiv.org/abs/2309.00614)
  - Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, Tom Goldstein
  - Summary:
    - Several rule-based defense methods baselines are provided, including the perplexity (PPL) filter, rephrasing, and retokenization.
- [SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks](https://arxiv.org/abs/2310.03684)
  - Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas
  - Repo: https://github.com/arobey1/smooth-llm
  - Summary:
    - Design a filter strategy to neutralize attack methods that are sensitive to perturbations by perturbing and testing the original input.

- [Certifying LLM Safety against Adversarial Prompting](https://arxiv.org/abs/2309.02705)
  - Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Soheil Feizi, Hima Lakkaraju
  - Summary:
    - Design a filter strategy to search for the raw question or instruction from inputs containing adversarial suffix or other auxiliary perturbation information, in order to neutralize their impact.

##### 📷Model-based Filters

- [Automatic identification of personal insults on social news sites](https://onlinelibrary.wiley.com/doi/abs/10.1002/asi.21690)
  - Sara Owsley Sood, Elizabeth F. Churchill, Judd Antin
  - Summary:
    - Train support Vector Machines (SVMs) with personal insult data from social news sites to detect inappropriate negative user contributions.
- [Antisocial Behavior in Online Discussion Communities](https://arxiv.org/abs/1504.00680)
  - Justin Cheng, Cristian Danescu-Niculescu-Mizil, Jure Leskovec
  - Summary:
    - Analyze strategies for detecting irrelevant content and early identification of problematic users.
- [Abusive Language Detection in Online User Content](https://dl.acm.org/doi/10.1145/2872427.2883062)
  - Chikashi Nobata, Joel Tetreault, Achint Thomas, Yashar Mehdad, Yi Chang
  - Summary:
    - Present a machine learning method to detect hate speech in online comments.
- [Ex Machina: Personal Attacks Seen at Scale](https://arxiv.org/abs/1610.08914)
  - Ellery Wulczyn, Nithum Thain, Lucas Dixon
  - Summary:
    - Develop a method using crowdsourcing and machine learning to analyze personal attacks on online platforms, particularly focusing on English Wikipedia. 
    - Introduce a classifier evaluated by its ability to approximate the judgment of crowd-workers, resulting in a corpus of over 100,000 human-labeled and 63 million machine-labeled comments. 
- [Defending Against Neural Fake News](https://arxiv.org/abs/1905.12616)
  - Rowan Zellers, Ari Holtzman, Hannah Rashkin, Yonatan Bisk, Ali Farhadi, Franziska Roesner, Yejin Choi
  - Summary:
    - Introduce Grover, a model capable of generating convincing articles from headlines, which poses both threats and opportunities for countering disinformation.
    - Grover can be the most effective tool in distinguishing between real news and neural fake news, achieving 92% accuracy.
- [Detecting Hate Speech with GPT-3](https://arxiv.org/abs/2103.12407)
  - Ke-Li Chiu, Annie Collins, Rohan Alexander
  - Summary:
    - Utilize GPT-3 to identify sexist and racist text passages using zero-shot, one-shot, and few-shot learning approaches.
- [Hypothesis Engineering for Zero-Shot Hate Speech Detection](https://arxiv.org/abs/2210.00910)
  - Janis Goldzycher, Gerold Schneider
  - Summary:
    - Propose a approach to enhance English NLI-based zero-shot hate speech detection by combining multiple hypotheses.
- [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing](https://arxiv.org/abs/2111.09543)
  - Pengcheng He, Jianfeng Gao, Weizhu Chen
  - Repo: [microsoft/DeBERTa: The implementation of DeBERTa (github.com)](https://github.com/microsoft/DeBERTa)
  - Summary:
    - Introduce DeBERTaV3, an enhancement of the original DeBERTa model, by implementing replaced token detection (RTD) in place of mask language modeling (MLM) for more efficient pre-training. 

- [A Holistic Approach to Undesired Content Detection in the Real World](https://arxiv.org/pdf/2208.03274.pdf)
  - Todor Markov, Chong Zhang, Sandhini Agarwal, Tyna Eloundou, Teddy Lee, Steven Adler, Angela Jiang, Lilian Weng
  - Summary:
    - Introduce a comprehensive strategy for developing a reliable and effective natural language classification system aimed at moderating online content. 
    - The proposed system is adept at identifying various types of inappropriate content, including sexual material, hate speech, violence, self-harm, and harassment, and offers a scalable solution that can adapt to different content classification needs.
- [Robust Safety Classifier for Large Language Models: Adversarial Prompt Shield](https://arxiv.org/abs/2311.00172)
  - Jinhwa Kim, Ali Derakhshan, Ian G. Harris
  - Summary:
    - Propose the Adversarial Prompt Shield (APS), a model designed to enhance safety by effectively detecting and mitigating harmful responses.
    - Also introduce Bot Adversarial Noisy Dialogue (BAND) datasets for training purposes to improve the resilience of safety classifiers against adversarial inputs.

- [NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails](https://arxiv.org/abs/2310.10501)
  - Traian Rebedea, Razvan Dinu, Makesh Sreedhar, Christopher Parisien, Jonathan Cohen
  - Summary:
    - Introduce an innovative open-source toolkit designed to integrate programmable guardrails into Large Language Model (LLM)-based conversational systems, enhancing their safety and controllability. 
    - NeMo Guardrails allows for the addition of user-defined, interpretable guardrails at runtime, independent of the underlying LLM.


---

### ✏️Evaluations

#### 📖Datasets

- [RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models](https://arxiv.org/abs/2009.11462)
  - Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, Noah A. Smith
  - Repo: [allenai/real-toxicity-prompts (github.com)](https://github.com/allenai/real-toxicity-prompts)
  - Summary:
    - Release RealToxicityPrompts, a dataset of 100K naturally occurring, sentence-level prompts derived from a large corpus of English web text, paired with toxicity scores from a widely-used toxicity classifier.
- [Bot-Adversarial Dialogue for Safe Conversational Agents](https://aclanthology.org/2021.naacl-main.235/)
  - Jing Xu, Da Ju, Margaret Li, Y-Lan Boureau, Jason Weston, Emily Dinan
  - Repo: [bot_adversarial_dialogue (github.com)](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/datasets/bot_adversarial_dialogue)
  - Summary:
    - Introduce a new framework that involves human and model collaboration to collect data and train safer models, along with a novel method to incorporate safety considerations directly into generative models without relying on an external classifier during deployment.
- [SaFeRDialogues: Taking Feedback Gracefully after Conversational Safety Failures](https://arxiv.org/abs/2110.07518)
  - Megan Ung, Jing Xu, Y-Lan Boureau
  - Repo: https://github.com/facebookresearch/ParlAI/tree/main/parlai/tasks/saferdialogues
  - Summary:
    - Propose SaFeRDialogues, a task and dataset of graceful responses to conversational feedback about safety failures. 
    - Collect a dataset of 10k dialogues demonstrating safety failures, feedback signaling them, and a response acknowledging the feedback. 
- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958)
  - Stephanie Lin, Jacob Hilton, Owain Evans
  - Repo: [sylinrl/TruthfulQA: TruthfulQA: Measuring How Models Imitate Human Falsehoods (github.com)](https://github.com/sylinrl/TruthfulQA)
  - Summary:
    - Propose a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics.
- [Discovering Language Model Behaviors with Model-Written Evaluations](https://arxiv.org/abs/2212.09251)
  - Ethan Perez, Sam Ringer, Kamilė Lukošiūtė, Karina Nguyen, et al.
  - Repo: [anthropics/evals (github.com)](https://github.com/anthropics/evals)
  - Summary:
    - Generate 154 datasets with human-written labels.
    - Crowdworkers have rated these LM-generated examples highly, often agreeing with the labels more frequently than they do with human-generated datasets. 
- [ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection](https://arxiv.org/abs/2203.09509)
  - Thomas Hartvigsen, Saadia Gabriel, Hamid Palangi, Maarten Sap, Dipankar Ray, Ece Kamar
  - Repo: [microsoft/TOXIGEN (github.com)](https://github.com/microsoft/TOXIGEN)
  - Summary:
    - Create ToxiGen, a new large-scale and machine-generated dataset of 274k toxic and benign statements about 13 minority groups. 
    - Develop a demonstration-based prompting framework and an adversarial classifier-in-the-loop decoding method to generate subtly toxic and benign text with a massive pretrained language model. 
- [SafetyBench: Evaluating the Safety of Large Language Models with Multiple Choice Questions](https://arxiv.org/abs/2309.07045)
  - Zhexin Zhang, Leqi Lei, Lindong Wu, Rui Sun, Yongkang Huang, Chong Long, Xiao Liu, Xuanyu Lei, Jie Tang, Minlie Huang
  - Repo:  [thu-coai/SafetyBench](https://github.com/thu-coai/SafetyBench)
  - Summary:
    - Present SafetyBench, a comprehensive benchmark for evaluating the safety of LLMs, which comprises 11,435 diverse multiple choice questions spanning across 7 distinct categories of safety concerns. 
    - SafetyBench also incorporates both Chinese and English data, facilitating the evaluation in both languages. 
- [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/pdf/2307.15043.pdf)
  - Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson

  - Repo: [https://github.com/llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks)

  - Summary:
    - Propose AdvBench, a dataset containing 500 harmful strings and 500 harmful behaviors.
- [Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment](https://arxiv.org/abs/2308.09662)
  - Rishabh Bhardwaj, Soujanya Poria

  - Repo: [declare-lab/red-instruct (github.com)](https://github.com/declare-lab/red-instruct)

  - Summary:
    - Collect a dataset that consists of 1.9K harmful questions covering a wide range of topics, 9.5K safe and 7.3K harmful conversations from ChatGPT.
- [LifeTox: Unveiling Implicit Toxicity in Life Advice](https://arxiv.org/abs/2311.09585)
  - Minbeom Kim, Jahyun Koo, Hwanhee Lee, Joonsuk Park, Hwaran Lee, Kyomin Jung
  - Summary:
    - Introduce LifeTox, a dataset designed for identifying implicit toxicity within a broad range of advice-seeking scenarios.
    - LifeTox comprises diverse contexts derived from personal experiences through open-ended questions.
- [FFT: Towards Harmlessness Evaluation and Analysis for LLMs with Factuality, Fairness, Toxicity](https://arxiv.org/abs/2311.18580)
  - Shiyao Cui, Zhenyu Zhang, Yilong Chen, Wenyuan Zhang, Tianyun Liu, Siqi Wang, Tingwen Liu
  - Repo:  https://github.com/cuishiyao96/FFT
  - Summary:
    - Propose FFT, a new benchmark with 2116 elaborated-designed instances, for LLM harmlessness evaluation with factuality, fairness, and toxicity. 
- [Purple Llama CyberSecEval: A Secure Coding Benchmark for Language Models](https://arxiv.org/abs/2312.04724)
  - Manish Bhatt, Sahana Chennabasappa, Cyrus Nikolaidis, Shengye Wan, Ivan Evtimov, Dominik Gabi, et al.
  - Repo: [PurpleLlama/CybersecurityBenchmarks (github.com)](https://github.com/facebookresearch/PurpleLlama/tree/main/CybersecurityBenchmarks)
  - Summary:
    - Present CyberSecEval, which provides a thorough evaluation of LLMs in two crucial security domains: their propensity to generate insecure code and their level of compliance when asked to assist in cyberattacks.
- [Latent Jailbreak: A Benchmark for Evaluating Text Safety and Output Robustness of Large Language Models](https://arxiv.org/abs/2307.08487)
  - Huachuan Qiu, Shuai Zhang, Anqi Li, Hongliang He, Zhenzhong Lan
  - Repo: [qiuhuachuan/latent-jailbreak (github.com)](https://github.com/qiuhuachuan/latent-jailbreak)
  - Summary:
    - Introduce a latent jailbreak prompt dataset, each involving malicious instruction embedding.

---

#### 🔍Metrics

- [FFT: Towards Harmlessness Evaluation and Analysis for LLMs with Factuality, Fairness, Toxicity](https://arxiv.org/abs/2311.18580)
  - Shiyao Cui, Zhenyu Zhang, Yilong Chen, Wenyuan Zhang, Tianyun Liu, Siqi Wang, Tingwen Liu
  - Repo:  https://github.com/cuishiyao96/FFT
  - Summary:
    - Attack success rate (ASR).
    - Evaluate by examining the outputs manually.
- [SafetyBench: Evaluating the Safety of Large Language Models with Multiple Choice Questions](https://arxiv.org/abs/2309.07045)
  - Zhexin Zhang, Leqi Lei, Lindong Wu, Rui Sun, Yongkang Huang, Chong Long, Xiao Liu, Xuanyu Lei, Jie Tang, Minlie Huang
  - Repo:  [thu-coai/SafetyBench](https://github.com/thu-coai/SafetyBench)
  - Summary:
    - Attack success rate (ASR).
    - Evaluate by comparing outputs with reference answers
- [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/pdf/2307.15043.pdf)
  - Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson
  - Repo: [https://github.com/llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks)
  - Summary:
    - Attack success rate (ASR).
    - Evaluate by automatically checking whether LLM outputs contain keywords that indicate a refusal to respond.
- [ AutoDAN: Automatic and Interpretable Adversarial Attacks on Large Language Models](https://arxiv.org/abs/2310.15140)
  - Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun
  - Summary:
    - Attack success rate (ASR) and false positive rate.
    - Evaluate by leveraging prompted GPT-4 to check whether LLM outputs are harmful.
    - Use ROGUE and BLEU to calculate the similarity between the LLM output and the reference output.
- [Latent Jailbreak: A Benchmark for Evaluating Text Safety and Output Robustness of Large Language Models](https://arxiv.org/abs/2307.08487)
  - Huachuan Qiu, Shuai Zhang, Anqi Li, Hongliang He, Zhenzhong Lan
  - Repo: [qiuhuachuan/latent-jailbreak (github.com)](https://github.com/qiuhuachuan/latent-jailbreak)
  - Summary:
    - Attack success rate (ASR) and Robustness.
    - Replaces words in the attack and observes changes in the success rate, providing insights into the attack’s robustness.
