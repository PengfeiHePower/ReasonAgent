""""
Solve gsm8k problems
"""
from reason_agent import ReasonAgent
from refine_agent import RefineAgent
from retrive_agent import RetrieveAgent
from memory_agent import MemoryAgent
from agentscope.parsers import MarkdownJsonDictParser
import string

import json
import os
import agentscope
from agentscope.message import Msg

def save_checkpoint(index, save_file):
    """ Save the current index to a checkpoint file. """
    with open(save_file, "w") as file:
        file.write(str(index))

def load_checkpoint(save_file):
    """ Load the last checkpoint index if exists. """
    if os.path.exists(save_file):
        with open(save_file, "r") as file:
            index = int(file.read().strip())
            return index
    else:
        return 0

def init_model():
    HTTP_LLM_API_KEY='eyJ0eXAiOiJqd3QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6IjM5NDc3MyIsInBhc3N3b3JkIjoiMzk0NzczMTIzIiwiZXhwIjoyMDIxNjE4MzE3fQ.oQx2Rh-GJ_C29AfHTHE4x_2kVyy7NamwQRKRA4GPA94'
    # models can be configured by loading config file
    with open("configs/model_config.json", "r", encoding="utf-8") as f:
        model_configs = json.load(f)
    for config in model_configs:
        if config.get("model_type", "") == "post_api_chat":
            # for gpt4 API
            config["headers"]["Authorization"] = (
                    "Bearer " + HTTP_LLM_API_KEY
            )
        else:
            # for dashscope
            # config["api_key"] = f"{os.environ.get('DASHSCOPE_API_KEY')}"
            config["api_key"] = 'sk-94d038e92230451a87ac37ac34dd6a8a'
    agentscope.init(model_configs=model_configs)

def finalize(thought):
    # current_answer = thought['Answer'].strip().lower()
    cleaned_thought = ''.join(char for char in thought['Answer'] if char not in string.punctuation)
    current_answer = cleaned_thought.lower()
    words = current_answer.split()
    if words[:2] == ['not', 'yet']:
        return False
    return True
    # if current_answer!='not yet':
    #     return True
    # else:
    #     return False


def solving(q, reason_agent, refine_agent, retrive_agent, memory_agent):
    memory_agent.reset()
    # structure analysis
    analysis_input={
    "type": "structure_analysis",
    "query": q,
    "analysis": None,
    "step": None,
    "retrieval": None,
    "info": None
    }
    structure_analysis = reason_agent(analysis_input).content
    print(f"Structure analysis:{structure_analysis}") #structure_analysis.content
    memory_agent.analysis_store(structure_analysis) # strore analysis into memory
    
    refine_analysis_input={
    "type": "structure_analysis",
    "query": q,
    "analysis": structure_analysis,
    "step": None,
    "info": None
    }
    refined_structure_anlysis = refine_agent(refine_analysis_input).content # refine the structure analysis
    memory_agent.analysis_store(refined_structure_anlysis["Refined analysis"]) # store refined analysis
    print(f"Refined analysis:{refined_structure_anlysis}")
    
    
    ## reasoning process
    for i in range(10):
        ## reason step
        input_thought={
        "type": "thought",
        "query": q,
        "analysis": memory_agent.analysis,
        "step": memory_agent.steps,
        "info": memory_agent.retrieval_info
        }
        thought = reason_agent(input_thought).content
        memory_agent.step_append(thought['Thought'])
        print(f"Thought{i}:{thought}")
        if finalize(thought):
            break
        
        retrieval_input={
            "retrieval":thought['Retrieval'],
            "retrieval info": thought["Retrieval info"]
        }
        output_retrieval = retrive_agent(retrieval_input).content
        memory_agent.info_append(output_retrieval['Extracted info'])
        print(f"output_retrieval:{output_retrieval}")
    
    print(f"Answer is:{thought['Answer']}")
    record = memory_agent.export()
    record["Answer"]=thought["Answer"]
    return record


def main():
    #initalize models
    init_model()
    print(f"model initialized")
    #initialize agents
    with open("configs/agent_config.json", "r", encoding="utf-8") as f:
        agent_configs = json.load(f)
    parser_analysis = MarkdownJsonDictParser(
    content_hint={
        "Key components": "Identify the crucial elements and variables that play a significant role in this problem",
        "Relationship between components": "Relationship between components",
        "Sub-questions": "break into sub-questions",
        "Implications for Solving the Problem":"For each sub-question, describe how solving it helps address the main problem. Connect the insights from these sub-questions to the overall strategy needed to solve the main problem."
    },
    keys_to_content=["Key components", "Relationship between components", "Sub-questions", "Implications for Solving the Problem"],
    keys_to_memory=["Key components", "Relationship between components", "Sub-questions", "Implications for Solving the Problem"],
    keys_to_metadata=[]
    )
    parser_refine_eval_analysis = MarkdownJsonDictParser(
    content_hint={
        "Correctness": "[Yes/No]",
        "Suggestions":"[Your suggestions here or 'No suggestions' if the analysis is correct]",
    },
    keys_to_content=["Correctness", "Suggestions"],
    keys_to_memory=["Correctness", "Suggestions"],
    keys_to_metadata=[]
    )
    parser_refine_analysis = MarkdownJsonDictParser(
    content_hint={
        "Refined analysis": "refined analysis",
    },
    keys_to_content=["Refined analysis"],
    keys_to_memory=["Refined analysis"],
    keys_to_metadata=[]
    )
    parser_next_step = MarkdownJsonDictParser(
    content_hint={
        "Reflection": "Reflection on current situation, check if previous thoughts and retrieved infomation help solve the problem. Check if the current step adheres with the structure analysis",
        "Thought": "Reason current sutiation. Critically analyze the current situation, and determine the logical next thought or action. Carefully consider previous thoughts and avoid repeated thoughts and actions",
        "Retrieval": "[Yes/No]",
        "Retrieval info": "If need external knowledge, provide what knowledge need to search for; if do not need, return No.",
        "Answer": "Conclude the answer if it is ready, otherwise 'Not yet'."
        },
    keys_to_content=["Reflection","Thought","Retrieval","Retrieval info", "Answer"],
    keys_to_memory=["Reflection","Thought","Retrieval","Retrieval info", "Answer"],
    keys_to_metadata=[]
    )
    parser_retrieval_analysis = MarkdownJsonDictParser(
    content_hint={
        "Retrieval entity": "[entity] ",
        "Retrieval sources": ["data sources for searching"],
        "Reason": "reason behind analysis"
    },
    keys_to_content=["Retrieval entity", "Retrieval sources"],
    keys_to_memory=["Retrieval key", "Retrieval sources", "Reason"],
    keys_to_metadata=[]
    )
    # setup agents
    reason_agent = ReasonAgent(parser_analysis=parser_analysis,parser_next_step=parser_next_step,**agent_configs[0]["args"])
    refine_agent = RefineAgent(**agent_configs[1]["args"], refine_analysis_eval_parser=parser_refine_eval_analysis, refine_analysis_parser=parser_refine_analysis)
    data_sources = ['text', 'category list', 'infobox', 'table', 'images with caption']
    retrive_agent = RetrieveAgent(**agent_configs[2]["args"], data_sources=data_sources, parser_retrieval_analysis=parser_retrieval_analysis)
    memory_agent = MemoryAgent()
    print(f"Agents initialized.")
    
    
    # load dataset
    gsm8k = []
    with open("data/gsm8k_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            gsm8k.append(json.loads(line.strip()))
    if os.path.isfile("output/gsm8k.json"):
        with open("output/gsm8k.json", 'r') as file:
            all_record = json.load(file)
    else:
        all_record = []
    start_index = load_checkpoint("chk/gsm8k.txt")
    for i in range(start_index, 20):
        try:
            q=gsm8k[i]['question']
            print(f"Original question:{q}")
            solving_record = solving(q, reason_agent, refine_agent, retrive_agent, memory_agent)
            solving_record['idx']=i
            all_record.append(solving_record)
            with open("output/gsm8k.json", 'w') as file:
                json.dump(all_record, file, indent=4)
            save_checkpoint(i + 1, "chk/gsm8k.txt")
        except Exception as e:
            print(f"Error processing sample at index {i}: {e}")
            break
    
    

if __name__ == "__main__":
    main()