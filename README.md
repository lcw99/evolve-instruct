# evolve-instruct
evolve llm training instruction, from english instruction to any language. this sample code is targeting Korean.

evolve code is based on [h2o-wizardlm](https://github.com/h2oai/h2o-wizardlm).  
base_instruction.json is from [wizardlm](https://github.com/nlpxucan/WizardLM).  
also based on paper [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)  

evol_instruct.json is sample generation, about 10,000 q&a pair from base_instruction.jsonl and another 26,000 from alpaca_data.json using ChatGPT and it costs about $80.

Korean llm [demo](https://changgpt.semaphore.kr/) using this dataset. this model is on Huggingface [lcw99/polyglot-ko-12.8b-chang-instruct-chat](https://huggingface.co/lcw99/polyglot-ko-12.8b-chang-instruct-chat)  

