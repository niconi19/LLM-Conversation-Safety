<h1 align="center">LLM Conversation Safety</h1>

This is a collection of research papers of **LLM Conversation Safety**. (💦Working on progress...)

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

(Working on progress...)

<img src="assets\explanation.png" style="zoom:80%;" />



# 📑Paper List

### 💣Attacks

#### 🎯Inference-time Attacks

##### 📜Red-team Attacks

- [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://arxiv.org/abs/2209.07858)
  - Deep Ganguli, Liane Lovitt, Jackson Kernion, et al.
  - Summary:
    - Assessing and mitigating the potentially harmful outputs of language models through red teaming.
    - The work explores the scaling behaviors of different model types and sizes, releases a dataset of red team attacks, and provides detailed instructions and methodologies for red teaming.

- [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)

  - Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, Geoffrey Irving

  - Summary：

    - Obtaining a red-teaming LLM by fine-tuning, which generates harmful cases where a target LLM behaves harmfully.
    - Collecting succesful attacking prompts from base LLM, which are used as fine-tuning data.

- [RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models](https://arxiv.org/abs/2009.11462)

  - Samuel Gehman, Suchin Gururangan, Maarten Sap, Yejin Choi, Noah A. Smith
  - Summary:
    - Introducing a dataset called RealToxicityPrompts, comprising 100,000 naturally occurring prompts paired with toxicity scores.
    - Analyzing web text corpora used for pretraining and identify offensive, unreliable, and toxic content.

- [Trick Me If You Can: Human-in-the-loop Generation of Adversarial Examples for Question Answering](https://arxiv.org/abs/1809.02701)

  - Eric Wallace, Pedro Rodriguez, Shi Feng, Ikuya Yamada, Jordan Boyd-Graber
  - Summary:
    - Introducing a human-in-the-loop adversarial generation where human authors are guided to create adversarial examples that challenge models.
    - The generated adversarial questions cover various phenomena and highlight the challenges in robust question answering.

- [Adversarial Training for High-Stakes Reliability](https://arxiv.org/abs/2205.01663)

  - Daniel M. Ziegler, Seraphina Nix, Lawrence Chan, Tim Bauman, Peter Schmidt-Nielsen, Tao Lin, Adam Scherlis, Noa Nabeshima, Ben Weinstein-Raun, Daniel de Haas, Buck Shlegeris, Nate Thomas
  - Summary:
    - Developed adversarial training techniques, including a tool to assist human adversaries, to identify and eliminate failures in a text completion classifier.

- [Explore, Establish, Exploit: Red Teaming Language Models from Scratch](https://arxiv.org/abs/2306.09442)

  - Stephen Casper, Jason Lin, Joe Kwon, Gatlen Culp, Dylan Hadfield-Menell
  - Summary:
    - Proposed a "from scratch" red-teaming approach where the adversary does not have a pre-existing classification mechanism.
    - Red-teamed GPT-3 and created the CommonClaim dataset of 20,000 statements labeled as common-knowledge-true, common-knowledge-false, or neither.

- [FLIRT: Feedback Loop In-context Red Teaming](https://arxiv.org/abs/2308.04265)

  - Ninareh Mehrabi, Palash Goyal, Christophe Dupuy, Qian Hu, Shalini Ghosh, Richard Zemel, Kai-Wei Chang, Aram Galstyan, Rahul Gupta
  - Summary:
    - Employing in-context learning in a feedback loop to trigger models into producing unsafe and inappropriate content.

---

##### 🎬Template-based Attacks

###### 🔹Heuristic-based Templates

- [Ignore Previous Prompt: Attack Techniques For Language Models](https://arxiv.org/abs/2211.09527)
  - Fábio Perez, Ian Ribeiro

  - Repo: [https://github.com/agencyenterprise/PromptInject](https://github.com/agencyenterprise/PromptInject)

  - Summary：
    - Designed two types of attack templates: goal hijacking that leads the original goal to another malicious goal, and prompt leaking that induces the LLM to output its system prompt.

- ["Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models](https://arxiv.org/abs/2308.03825)

  - Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, Yang Zhang
  - Repo: https://github.com/verazuo/jailbreak_llms
  - Summary:
    - Collected 6,387 manually designed attack templates (jailbreak prompts) from online platforms.
    - Evaluating the charisteristics and effectiveness of the collected jailbreak prompts.

- [Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition](https://arxiv.org/abs/2311.16119)

  - Sander Schulhoff, Jeremy Pinto, Anaum Khan, Louis-François Bouchard, Chenglei Si, Svetlina Anati, Valen Tagliabue, Anson Liu Kost, Christopher Carnahan, Jordan Boyd-Graber
  - Summary:
    - Summarizing the various attack templates (jailbreak prompts) received for a prompt hacking competitio

- [The Radicalization Risks of GPT-3 and Advanced Neural Language Models](https://arxiv.org/abs/2009.06807)

  - Kris McGuffie, Alex Newhouse
  - Summary:
    - Using few-shot examples to elicit undesirable responses from GPT-3.
    - Comparing and analyzing the radicalization of GPT-3 under zero-shot and few-shot circumstances.

- [Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations](https://arxiv.org/abs/2310.06387)
  -  Zeming Wei, Yifei Wang, Yisen Wang

  -  Summary:
     -  Using few-shot examples to elicit undesirable responses.
     -  Enhancing attacking performance by appending succesful attack cases to the prompt.

- [GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher](https://arxiv.org/abs/2308.06463)
  - Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Pinjia He, Shuming Shi, Zhaopeng Tu

  - Summary:
    - Designed cipher attack templates that convert LLM inputs into specific domains (such as ASCII) to evade security mechanisms, and then reverts LLM outputs back to natural language.

- [Exploiting Programmatic Behavior of LLMs: Dual-Use Through Standard Security Attacks](https://arxiv.org/abs/2302.05733)

  - Daniel Kang, Xuechen Li, Ion Stoica, Carlos Guestrin, Matei Zaharia, Tatsunori Hashimoto

  - Summary:
    - The template design incorporates payload splitting, dividing the input into substatements. While instructing LLM not to scrutinize, the prompt directs it to combine substatements during processing, reducing their direct correlation to evade defense mechanisms.

- [Latent Jailbreak: A Benchmark for Evaluating Text Safety and Output Robustness of Large Language Models](https://arxiv.org/abs/2307.08487)

  - Huachuan Qiu, Shuai Zhang, Anqi Li, Hongliang He, Zhenzhong Lan
  - Repo: https://github.com/qiuhuachuan/latent-jailbreak
  - Summary:
    - Translating the original question into a language domain where LLM's security capabilities are weak, to evade LLM's security scrutiny.

    - Constructed a dataset for a latent jailbreak.

- [DeepInception: Hypnotize Large Language Model to Be Jailbreaker](https://arxiv.org/abs/2311.03191)

  - Xuan Li, Zhanke Zhou, Jianing Zhu, Jiangchao Yao, Tongliang Liu, Bo Han

  - Code: https://github.com/tmlr-group/DeepInception

  - Summary:
    - Designed a template to transform queries into indirect multi-layer storytelling tasks for attacking purposes.

- [Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment](https://arxiv.org/abs/2308.09662)

  - Rishabh Bhardwaj, Soujanya Poria

  - Summary:
    - Designed a template that requires LLM to simulate a dialogue between two LLMs, playing the roles of both a Base-LM and a Red-LM simultaneously. The Red-LM is tasked with eliciting useful information for the Base-LM.
    - Introduced the Red-Instruct method and its dataset, minimizing the probability of harmful responses during the fine-tuning process to enhance security.

- [A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily](https://arxiv.org/abs/2311.08268)

  - Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang
  - Summary:

    -  Using LLM to rewrite harmful prompts and nest them in specific scenarios.
    -  Prompting LLMs to perform word-level or sentense level rewriting on initial harmful prompts, including paraphrasing, misspelling, word inserting, etc. Rewriting repeats until the obtained prompt elucidate the harmful filter based on the same LLMs. The obtained prompts are further nested in three scenarios via prompt-based LLMs: code completion, table filling, and text continuation.
- [Multi-step Jailbreaking Privacy Attacks on ChatGPT](https://arxiv.org/abs/2304.05197)
  - Haoran Li, Dadi Guo, Wei Fan, Mingshi Xu, Jie Huang, Fanpu Meng, Yangqiu Song
  - repo: https://github.com/HKUST-KnowComp/LLM-Multistep-Jailbreak
  - Summary:
    - Proposed an attack template that employs a three-utterance context. The user initiates the conversation by issuing a jailbreak command, to which the LLM agrees. Subsequently, the user poses a specific question.
- [Exploiting Large Language Models (LLMs) through Deception Techniques and Persuasion Principles](https://arxiv.org/abs/2311.14876)
  - Sonali Singh, Faranak Abri, Akbar Siami Namin
  - Summary:
    - Designed attack strategies based on techniques in deception theory.
- [Analyzing the Inherent Response Tendency of LLMs: Real-World Instructions-Driven Jailbreak](https://arxiv.org/abs/2312.04127)
  - Yanrui Du, Sendong Zhao, Ming Ma, Yuhan Chen, Bing Qin
  - Summary:
    - Designed attack templates with real-world instructions based on analysis of LLM affirmation tendency.

---

###### 🔹Optimization-based Templates

- [Gradient-based Adversarial Attacks against Text Transformers](https://arxiv.org/abs/2104.13733)
  - Chuan Guo, Alexandre Sablayrolles, Hervé Jégou, Douwe Kiela
  - Summary:
    - Token-level methods.
    - Applying Gumbel-softmax to attack a white-box LM-based classifier.
- [HotFlip: White-Box Adversarial Examples for Text Classification](https://arxiv.org/abs/1712.06751)
  - Javid Ebrahimi, Anyi Rao, Daniel Lowd, Dejing Dou
  - Summary:
    - Token-level methods.
    - Iteratively ranking tokens based on the first-order approximation of the adversarial objective and compute the adversarial objective with the highest-ranked tokens as a way to approximate coordinate ascends.
- [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts](https://arxiv.org/abs/2010.15980)
  - Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, Sameer Singh
  - Summary:
    - Token-level methods.
    - Optimizing universal suffix adversarial triggers based on gradient-based search to perturb the language model output.
- [Universal Adversarial Triggers for Attacking and Analyzing NLP](https://arxiv.org/abs/1908.07125)
  - Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, Sameer Singh
  - Summary:
    - Token-level methods.
    - Utilizing gradient-based search combined with beam search to discover a universal and transferable suffix adversarial trigger for perturbing language model outputs.
- [Automatically Auditing Large Language Models via Discrete Optimization](https://arxiv.org/abs/2303.04381)
  - Erik Jones, Anca Dragan, Aditi Raghunathan, Jacob Steinhardt
  - Summary:
    - Token-level methods.

    - Proposed a more efficient version of AutoPrompt that significantly improves the attack success rate.

- [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/pdf/2307.15043.pdf)
  - Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson

  - Repo:[https://github.com/llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks)

  - Summary:
    - Token-level methods.
    - Proposed a multi-model and multi-prompt approach that finds transferable triggers for black-box LLMs.

- [ AutoDAN: Automatic and Interpretable Adversarial Attacks on Large Language Models](https://arxiv.org/abs/2310.15140)

  - Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun
  - Summary:
    - Token-level methods.
    - Automatically generating adversarial prompts by gradient-based techniques. Incorporating an additional fluency objective to produce more natural adversarial triggers.
- [Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers](https://arxiv.org/abs/2309.08532)
  - Qingyan Guo, Rui Wang, Junliang Guo, Bei Li, Kaitao Song, Xu Tan, Guoqing Liu, Jiang Bian, Yujiu Yang
  - Summary:
    - Using LLM-based GA approaches for prompt optimization.
- [AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models](https://arxiv.org/abs/2310.04451)

  - Xiaogeng Liu, Nan Xu, Muhao Chen, Chaowei Xiao

  - Summary:
    - Expression-level methods.
    - Automatically generating adversarial prompts by leveraging LLM-based GA optimization approach.
- [DeceptPrompt: Exploiting LLM-driven Code Generation via Adversarial Natural Language Instructions](https://arxiv.org/abs/2312.04730)
  - Fangzhou Wu, Xiaogeng Liu, Chaowei Xiao
  - Summary:
    - Expression-level methods.
    - Using LLM-based GA optimization approach for adversarial prompts.

- [MasterKey: Automated Jailbreak Across Multiple Large Language Model Chatbots](https://arxiv.org/abs/2307.08715)
  - Gelei Deng, Yi Liu, Yuekang Li, Kailong Wang, Ying Zhang, Zefeng Li, Haoyu Wang, Tianwei Zhang, Yang Liu
  - Summary:
    - Expression-level methods.
    - Fine-tuning an LLM to refine existing jailbreak templates and improve their effectivenes.

---

##### 🔮Neural Prompt-to-Prompt Attacks

- [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)

  - Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, Xinyun Chen
  - Summary:
    - Using an LLM as an optimizer to progressively improve prompts and addressing problems such as SAT.

- [Jailbreaking Black Box Large Language Models in Twenty Queries](https://aicarrier.feishu.cn/wiki/KK0gwZRIMiHuzKkrh17cCD4AnHf) 

  - Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong
  - Summary:
    - Using a base LLM as an optimizer to progressively refine inputs based on the interactive feedback from the target LLM.

- [Tree of Attacks: Jailbreaking Black-Box LLMs Automatically](https://arxiv.org/abs/2312.02119)

  - Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, Amin Karbasi
  - Summary:
    - Leverageing LLM-based modify-and-search techniques to improve input prompts through tree-based modification and search methods.

- [Evil Geniuses: Delving into the Safety of LLM-based Agents](https://arxiv.org/pdf/2311.11855v1.pdf)

  - Yu Tian, Xiao Yang, Jingyuan Zhang, Yinpeng Dong, Hang Su

  - Repo: https://github.com/T1aNS1R/Evil-Geniuses

  - Summary:

    - A multiple-agent system with agent roles specified by system prompt.
    - Developing a virtual evil plan team using LLM, consisting of a harmful prompt writer, a suitability reviewer, and a toxicity tester, to optimize prompts through iterative modifications and assessments until the attack is successful or predefined termination conditions are met.

- [MART: Improving LLM Safety with Multi-round Automatic Red-Teaming](https://arxiv.org/abs/2311.07689)
  - Suyu Ge, Chunting Zhou, Rui Hou, Madian Khabsa, Yi-Chia Wang, Qifan Wang, Jiawei Han, Yuning Mao
  - Summary:
    - Training an LLM to iteratively improve red prompts from the existing ones through adversarial interactions between attack and defense models

---

#### 🚅Training-time Attacks

- [On the Exploitability of Instruction Tuning](https://arxiv.org/abs/2306.17194)
  - Manli Shu, Jiongxiao Wang, Chen Zhu, Jonas Geiping, Chaowei Xiao, Tom Goldstein
  - Repo: [https://github.com/azshue/AutoPoison](https://github.com/azshue/AutoPoison)
  - Summary:
    - Injecting poisoned instruction-following examples into the training data to alter the model's behavior.
- [BadLlama: cheaply removing safety fine-tuning from Llama 2-Chat 13B](https://arxiv.org/abs/2311.00117)
  - Pranav Gade, Simon Lermen, Charlie Rogers-Smith, Jeffrey Ladish
  - Summary:
    - Using a small amount of data for fine-tuning to compromise Llama2-Chat.
- [LoRA Fine-tuning Efficiently Undoes Safety Training in Llama 2-Chat 70B](https://arxiv.org/abs/2310.20624)
  - Simon Lermen, Charlie Rogers-Smith, Jeffrey Ladish
  - Summary:
    - Using LoRA with a small amount of data for fine-tuning to compromise Llama2-Chat.

- [Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models](https://arxiv.org/abs/2310.02949)
  - Xianjun Yang, Xiao Wang, Qi Zhang, Linda Petzold, William Yang Wang, Xun Zhao, Dahua Lin
  - Summary:
    - Using a small amount of synthetic Q&A pairs to fine-tune LLMs for harmful tasks and demonstrating the transferability to language and multi-turn dialogues.
- [Removing RLHF Protections in GPT-4 via Fine-Tuning](https://arxiv.org/abs/2311.05553)
  - Qiusi Zhan, Richard Fang, Rohan Bindu, Akul Gupta, Tatsunori Hashimoto, Daniel Kang
  - Summary:
    - Using a small amount of data for fine-tuning to compromise RLHF protections while preserving usefulness.

- [Spinning Language Models: Risks of Propaganda-As-A-Service and Countermeasures](https://arxiv.org/abs/2112.05224)
  - Eugene Bagdasaryan, Vitaly Shmatikov
  - Summary:
    - Backdoor attacks.
    - Manipulating language models through backdoor attacks and causing them to exhibit adversary-chosen sentiment or point of view when triggered by specific words.

- [Universal Jailbreak Backdoors from Poisoned Human Feedback](https://arxiv.org/abs/2311.14455)
  - Javier Rando, Florian Tramèr
  - Summary:
    - Backdoor attacks.
    - Unaligning LLMs by incorporating backdoor triggers in RLHF. Embedding a trigger word into the model that grants access to harmful responses without the necessity of searching for an adversarial prompt.

- [Stealthy and Persistent Unalignment on Large Language Models via Backdoor Injections](https://arxiv.org/abs/2312.00027)
  - Yuanpu Cao, Bochuan Cao, Jinghui Chen
  - Summary:
    - Backdoor attacks.
    - Presenting a backdoor attack method, offering insights into the relationship between backdoor persistence and activation patterns, and provides guidelines for trigger design.

- [Poisoning Language Models During Instruction Tuning](https://arxiv.org/abs/2305.00944)
  - Alexander Wan, Eric Wallace, Sheng Shen, Dan Klein
  - Summary:
    - Backdoor attacks.
    - Incorporating poison examples into the training data to yield negative outcomes across various tasks.
- [Instructions as Backdoors: Backdoor Vulnerabilities of Instruction Tuning for Large Language Models](https://arxiv.org/abs/2305.14710)
  - Jiashu Xu, Mingyu Derek Ma, Fei Wang, Chaowei Xiao, Muhao Chen
  - Summary:
    - Backdoor attacks.
    - Injecting backdoors into models using a minimal number of malicious instructions, achieving zero-shot control through data poisoning across diverse datasets.

- [Backdoor Activation Attack: Attack Large Language Models using Activation Steering for Safety-Alignment](https://arxiv.org/abs/2311.09433)
  - Haoran Wang, Kai Shu
  - Summary:
    - Backdoor attacks.
    - Leveraging trojan activation attack to steer the model’s output towards a mis-aligned direction within the activation space.


---

### 🔒Defenses

#### 💪LLM Safety Alignment

(working on progress...)

- [MART: Improving LLM Safety with Multi-round Automatic Red-Teaming](https://arxiv.org/pdf/2311.07689.pdf)

  - Suyu Ge, Chunting Zhou, Rui Hou, Madian Khabsa, Yi-Chia Wang, Qifan Wang, Jiawei Han, Yuning Mao

  - Summary:

    - LLM Alignment.

    - Improve LLM's safety by multi-round attack and defense.

- [Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language Models that Follow Instructions](https://arxiv.org/abs/2309.07875)

  - Federico Bianchi, Mirac Suzgun, Giuseppe Attanasio, Paul Röttger, Dan Jurafsky, Tatsunori Hashimoto, James Zou

  - Summary:

    -  Create a safety instruction-response dataset with GPT-3.5-turbo for instruction tuning.
    -  The work finds that overly safe allignment can be detrimental (exagerrate safety), as LLM tends to refuse answering safe instructions, leading to a degrade in helpfulness.


---

#### 😷Inference Guidance

- [LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked](https://arxiv.org/abs/2308.07308)
  - Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau
  - Summary:
    - Prompting the LLM to evaluate its generated responses for harmful content.
- [Defending Large Language Models Against Jailbreaking Attacks Through Goal Prioritization](https://arxiv.org/abs/2311.09096)
  - Zhexin Zhang, Junxiao Yang, Pei Ke, Minlie Huang
  - Repo: [https://github.com/thu-coai/JailbreakDefense_GoalPriority](https://github.com/thu-coai/JailbreakDefense_GoalPriority)
  - Summary:
    - Designed a system prompt prioritizing safety objectives.
- [Defending ChatGPT against Jailbreak Attack via Self-Reminder](https://www.researchgate.net/publication/371612143_Defending_ChatGPT_against_Jailbreak_Attack_via_Self-Reminder)
  - Yueqi Xie, Jingwei Yi, Jiawei Shao, Justin Curl, Lingjuan Lyu, Qifeng Chen , XingXie, and Fangzhao Wu
  - Code: https://anonymous.4open.science/r/Self-Reminder-D4C8/
  - Summary:
    - Defense by designing system prompts.
    - Include instructions to avoid generating harmful outputs in the system prompt to encourage LLMs to perform self-checking.
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

- [Certifying LLM Safety against Adversarial Prompting](https://arxiv.org/abs/2309.02705)
  - Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Soheil Feizi, Hima Lakkaraju

  - - Input filter.
    - Detect original red-teaming questions to defense prompt injection.
- [Baseline Defenses for Adversarial Attacks Against Aligned Language Models](https://arxiv.org/abs/2309.00614)
  - Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, Tom Goldstein
  - Summary:
    - Input filter.
    - Use approaches such as PPL (Perplexity filter) to detect and defense prompt injection attacks.

##### 📷Model-based Filters

- [A Holistic Approach to Undesired Content Detection in the Real World](https://arxiv.org/pdf/2208.03274.pdf)
  - Todor Markov, Chong Zhang, Sandhini Agarwal, Tyna Eloundou, Teddy Lee, Steven Adler, Angela Jiang, Lilian Weng
  - Summary:
    - LLM-based filter.
    - Fine-tuning to obtain an LLM for filtering.
- [TOXIGEN: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection](https://arxiv.org/abs/2203.09509)
  - Thomas Hartvigsen, Saadia Gabriel, Hamid Palangi, Maarten Sap, Dipankar Ray, Ece Kamar
  - Summary:
    - LLM-based filter (fine-tuning).
    - Automatically generate red-teaming data with implicit toxicity.

---

### ✏️Evaluations

#### 📖Datasets

(working on progress...) 

#### 🔍Metrics

(working on progress...)
