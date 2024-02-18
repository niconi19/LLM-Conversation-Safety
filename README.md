<h1 align="center">LLM Conversation Safety</h1>

This is a collection of research papers of about **LLM Conversation Safety**. (Working on progress...)

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

If you find out a mistake or any related materials could be helpful, feel free to contact us or make a PR.

## Contents

- [Overview](#overview)
- [Paper List](#paper-list)
  - [Attacks](#attacks)
    - [Inference-time Attacks](#inference-time-attacks)
      - [Red-team Attacks](#red-team-attacks)
      - [Template-based Attacks](#template-based-attacks)
        - [Heuristic-based Templates](#heuristic-based-templates)
        - [Optimization-based Templates](#optimization-based-templates)

      - [Neural Prompt-to-Prompt Attacks](#neural-prompt-to-prompt-attacks)

    - [Training-time Attacks](#training-time-attacks)

  - [Defenses](#defenses)
    - [LLM Safety Alignment](#LLM-safety-alignment)
    - [Inference Guidance](#inference-guidance)
    - [Input/Output Filters](#input/output-filters)
      - [Rule-based Filters](#rule-based-filters)
      - [Model-based Filters](#model-based-filters)

  - [Evaluations](#evaluations)
    - [Datasets](#datasets)
    - [Metrics](#metrics)


# Overview

(Working on progress...)



# Paper List

### Attacks

#### Inference-time Attacks

##### Red-team Attacks



##### Template-based Attacks

###### Heuristic-based Templates

1. [Ignore Previous Prompt: Attack Techniques For Language Models](https://arxiv.org/abs/2211.09527)

  - Fábio Perez, Ian Ribeiro
  - Code：[https://github.com/agencyenterprise/PromptInject](https://github.com/agencyenterprise/PromptInject)
  - Summary：
    - Prompt-based attack via templates.
    - Design two types of templates: goal hijacking that leads the original goal to another malicious goal, and prompt leaking that induces the LLM to output its system prompt.

2. [GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher](https://arxiv.org/abs/2308.06463)

   - Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Pinjia He, Shuming Shi, Zhaopeng Tu

   - Summary:
     - Prompt-based attack via templates.
     - Implement cipher attacks by designed templates.

3. [Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment](https://arxiv.org/abs/2308.09662)

   - Rishabh Bhardwaj, Soujanya Poria

   - Summary:
     - Prompt-based attack via templates.
     - Attack LLM by asking to play the role of both a Base-LM and a Red-LM, while the latter is demanded to induce useful information for the former.

4. [DeepInception: Hypnotize Large Language Model to Be Jailbreaker](https://arxiv.org/abs/2311.03191)

   - Xuan Li, Zhanke Zhou, Jianing Zhu, Jiangchao Yao, Tongliang Liu, Bo Han

   - Code: https://github.com/tmlr-group/DeepInception

   - Summary:
     - Prompt-based attack via templates based on psychological prior.
       - Based on the prior from a psychological experiment (Milgram shock experiment) that LLM can be misguided by indirect instructions, the work designs an input template to create pseudo multi-layer story-telling tasks for model attacking.

5. [Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations](https://arxiv.org/abs/2310.06387)

   -  Zeming Wei, Yifei Wang, Yisen Wang

   -  Summary:
      -  Few-shot prompt-based attack.
         -  Enhance attacking performance by appending succesful attack cases to the prompt.

6. [A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily](https://arxiv.org/abs/2311.08268)

   - Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang


   - Summary:

     -  Use LLM to rewrite harmful prompts and nest them in specific scenarios.
        -  Prompt LLMs to perform word-level or sentense level rewriting on initial harmful prompts, including paraphrasing, misspelling, word inserting, etc. Repeat rewriting until the obtained prompt elucidate the harmful filter based on the same LLMs. The obtained prompts are further nested in three scenarios via prompt-based LLMs: code completion, table filling, and text continuation.

     

###### Optimization-based Templates

1. [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/pdf/2307.15043.pdf)

   - Andy Zou, Zifan Wang, J. Zico Kolter, Matt Fredrikson

   - Code: [https://github.com/llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks)

   - Summary:
     - Prefix prompt injection attack.
     - Automatically generate adversarial prompts by gradient-based techniques.

2. [ AutoDAN: Automatic and Interpretable Adversarial Attacks on Large Language Models](https://arxiv.org/abs/2310.15140)

   - Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun

   - Summary:
     - Prefix prompt injection attack.
     - Automatically generate adversarial prompts by gradient-based techniques. Generate new tokens at the end of the sentence instead of replacing old ones while considering fluency constrains.

3. [Jailbreaking Black Box Large Language Models in Twenty Queries](https://aicarrier.feishu.cn/wiki/KK0gwZRIMiHuzKkrh17cCD4AnHf) 

   - Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, Eric Wong

   - Summary:
     - Prompt-based attack.
     - Automatically generate adversarial prompts by leveraging LLM as an optimizer.

4. [AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models](https://arxiv.org/abs/2310.04451)

   - Xiaogeng Liu, Nan Xu, Muhao Chen, Chaowei Xiao

   - Summary:
     - Prompt-based attack.
     - Automatically generate adversarial prompts by leveraging LLM-based GA optimization approach.

5. [Evil Geniuses: Delving into the Safety of LLM-based Agents](https://arxiv.org/pdf/2311.11855v1.pdf)

   - Yu Tian, Xiao Yang, Jingyuan Zhang, Yinpeng Dong, Hang Su


   - Repo: https://github.com/T1aNS1R/Evil-Geniuses

   - Summary:

     - Multiple-agent system attack with agent roles specified by system prompt, which generates harmful roleplay prompts to attack both LLM and LLM-based agents.

     - Use system prompt to construct a virtual evil plan development team via LLM, which consists of a harmful prompt writer, a suitability reviewer, and a toxicity tester, composing a closed loop for prompt optimization. Specifically, the writer modifies the initial prompt to be harmful. Then, the reviewer and the tester checks the compatibility and effectiveness of the written harmful prompt, and the prompts deemed incompatible or inoffensive will be sent back to the writer to be further revised. The revision repeats until the attack is sucessful or other predefined termination conditions are met.

       

##### Neural Prompt-to-Prompt Attacks



#### Training-time Attacks

- [Shadow Alignment: The Ease of Subverting Safely-Aligned Language Models](https://arxiv.org/abs/2310.02949#:~:text=Formally%2C we term a new attack as Shadow,their capability to respond appropriately to regular inquiries.)
  - Xianjun Yang, Xiao Wang, Qi Zhang, Linda Petzold, William Yang Wang, Xun Zhao, Dahua Lin
  - Summary:
    - Adjust LLM parameters with a tiny amount of automatically generated data to elicit harmful behaviours.

- [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)
  - Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, Geoffrey Irving

  - Summary：

    - Obtain a red-teaming LLM by fine-tuning, which generates harmful cases where a target LLM behaves harmfully.

    - Collect succesful attacking prompts from base LLM, which are used as fine-tuning data.



### Defenses

#### LLM Safety Alignment

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

    

#### Inference Guidance

- [RAIN: Your Language Models Can Align Themselves without Finetuning](https://arxiv.org/abs/2309.07124)
  - Yuhui Li, Fangyun Wei, Jinjing Zhao, Chao Zhang, Hongyang Zhang

  - Code: https://github.com/SafeAILab/RAIN

  - Summary:
    - LLM self-evaluation approach for model inference.
    - Utilize search-and-backward method to explore several steps of potential subsequent token generations, evalute their scores via LLM self-evaluation, and aggregate the scores to adjust probabilities for token chosing and guide future generation.
- [Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations](https://arxiv.org/abs/2310.06387)

  -  Zeming Wei, Yifei Wang, Yisen Wang
  -  Summary:
     -  Few-shot prompt-based defense.
     -  Enhance LLM's defense ability by appending succesful defense cases to the prompt.
- [Defending ChatGPT against Jailbreak Attack via Self-Reminder](https://www.researchgate.net/publication/371612143_Defending_ChatGPT_against_Jailbreak_Attack_via_Self-Reminder)
  - Yueqi Xie, Jingwei Yi, Jiawei Shao, Justin Curl, Lingjuan Lyu, Qifeng Chen , XingXie, and Fangzhao Wu
  - Code: https://anonymous.4open.science/r/Self-Reminder-D4C8/
  - Summary:
    - Defense by designing system prompts.
    - Include instructions to avoid generating harmful outputs in the system prompt to encourage LLMs to perform self-checking.



#### Input/Output Filters

##### Rule-based Filters

- [Certifying LLM Safety against Adversarial Prompting](https://arxiv.org/abs/2309.02705)
  - Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Soheil Feizi, Hima Lakkaraju

  - Summary:
    - Input filter.
    - Detect original red-teaming questions to defense prompt injection.
- [Baseline Defenses for Adversarial Attacks Against Aligned Language Models](https://arxiv.org/abs/2309.00614)
  - Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, Tom Goldstein
  - Summary:
    - Input filter.
    - Use approaches such as PPL (Perplexity filter) to detect and defense prompt injection attacks.

##### Model-based Filters

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



### Evaluations

#### Datasets

(working on progress...)

#### Metrics

(working on progress...)
