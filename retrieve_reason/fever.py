""""
Solve hotpotQA problems
"""
from reason_agent import ReasonAgent
from refine_agent import RefineAgent
from retrive_agent import RetrieveAgent
from memory_agent import MemoryAgent
from agentscope.parsers import MarkdownJsonDictParser

import json
import os
import agentscope
from agentscope.message import Msg


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
            config["api_key"] = f"{os.environ.get('DASHSCOPE_API_KEY')}"
    agentscope.init(model_configs=model_configs)


def solving(q, reason_agent, refine_agent, retrive_agent, memory_agent):
    memory_agent.reset()
    # structure analysis
    x={
    "type": "structure_analysis",
    "query": q,
    "analysis": None,
    "step": None,
    "action": None,
    "info": None
    }
    structure_analysis = reason_agent(x)
    print(f"Structure analysis:{structure_analysis}") #structure_analysis.content
    memory_agent.analysis_store(structure_analysis["content"]) # strore analysis into memory
    # refine analysis #TODO
    x={
    "type": "structure_analysis",
    "query": q,
    "analysis": structure_analysis["content"],
    "step": None,
    "action": None,
    "info": None
    }
    refined_structure_anlysis = refine_agent(x).content
    memory_agent.analysis_store(refined_structure_anlysis)
    print(f"Refined analysis:{refined_structure_anlysis}")
    
    # Initial step
    x={
    "type": "initial_step",
    "query": q,
    "analysis": memory_agent.analysis,
    "step": None,
    "action": None,
    "info": None
    }
    next_step = reason_agent(x).content
    memory_agent.step_append(next_step['Next step'])
    print(f"Initial step:{next_step['Next step']}")

    
    x={
    "step": next_step['Next step'],
    # "action": initial_act,
    }
    initial_retrieve = retrive_agent(x).content
    memory_agent.info_append(initial_retrieve['extracted info'])
    print(f"initial info:{initial_retrieve['extracted info']}")
    
    for _ in range(10):
        x={
        "type": "next_step",
        "query": q,
        "analysis": memory_agent.analysis,
        "step": next_step["Next step"],
        # "action": next_action,
        "info": memory_agent.retrieval_info
        }
        next_step = reason_agent(x).content
        memory_agent.step_append(next_step)
        print(f"next_step:{next_step}")
        # input(111)
        
        # next_step_dict = text_to_dict(next_step,"\n\n")
        # print(f"next_step_dict:{next_step_dict}")
        if next_step["Answer"]!='Not yet':
            print(f"Answer is:{next_step['Answer']}")
            break
        
        x={
            "step": next_step["Next step"],
            # "action": next_act,
            }
        next_info = retrive_agent(x).content
        memory_agent.info_append(next_info['extracted info'])
        print(f"next info:{next_info['extracted info']}")
    
    record = memory_agent.export()
    record["Answer"]=next_step["Answer"]
    return record
    



def main():
    #initalize models
    init_model()
    #initialize agents
    with open("configs/agent_config.json", "r", encoding="utf-8") as f:
        agent_configs = json.load(f)
    # setup parsers
    parser_analysis = MarkdownJsonDictParser(
    content_hint={
        "Key components": "main elements of the problem",
        "Relationship between components": "relationship between key compoents",
        "Clarify the problem": "a clearer and simpler restatement",
        "Scope":"decide the boundary of the problem",
        "Sub-questions":"break into sub-questions"
    },
    keys_to_content=["Key components", "Relationship between components", "Clarify the problem", "Scope", "Sub-questions"],
    keys_to_memory=["Key components", "Relationship between components", "Clarify the problem", "Scope", "Sub-questions"],
    keys_to_metadata=[]
    )
    parser_step1 = MarkdownJsonDictParser(
    content_hint={
        "Next step": "next step",
        "Reason": "reason behind the step"
    },
    keys_to_content=["Next step"],
    keys_to_memory=["Next step", "Reason"],
    keys_to_metadata=[]
    )
    parser_next_step = MarkdownJsonDictParser(
    content_hint={
        "Next step": "This step can be (1) Search[entity] on Wikipedia; (2) Conclude the answer",
        "Answer": "Conclude the answer if possible, otherwise 'Not yet'",
        "Reason": "reason behind the step"
    },
    keys_to_content=["Next step","Answer"],
    keys_to_memory=["Next step","Answer", "Reason"],
    keys_to_metadata=[]
    )
    parser_action = MarkdownJsonDictParser(
    content_hint={
        "Action": "action for the currect step"
    },
    keys_to_content=["Action"],
    keys_to_memory=["Action"],
    keys_to_metadata=[]
    )
    parser_retrieval_analysis = MarkdownJsonDictParser(
    content_hint={
        "Retrieval key": "key for searching",
        "Retrieval sources": ["data sources for searching"],
        "Reason": "reason behind analysis"
    },
    keys_to_content=["Retrieval key", "Retrieval sources"],
    keys_to_memory=["Retrieval key", "Retrieval sources", "Reason"],
    keys_to_metadata=[]
    )
    # setup agents
    reason_agent = ReasonAgent(parser_analysis=parser_analysis,parser_step1=parser_step1,parser_next_step=parser_next_step,parser_action=parser_action,**agent_configs[0]["args"])
    refine_agent = RefineAgent(**agent_configs[1]["args"])
    data_sources = ['text', 'category list', 'infobox', 'table', 'images with caption']
    retrive_agent = RetrieveAgent(**agent_configs[2]["args"], data_sources=data_sources, parser_retrieval_analysis=parser_retrieval_analysis)
    memory_agent = MemoryAgent()
    print(f"Agents initialized.")
    
    
    # load dataset
    with open("data/hotpot_dev_v1_simplified.json", "r", encoding="utf-8") as f:
        hotpotqa = json.load(f)
    all_record = []
    for i in range(100):
        q=hotpotqa[i]['question']
        print(f"Original question:{q}")
        solving_record = solving(q, reason_agent, refine_agent, retrive_agent, memory_agent)
        all_record.append(solving_record)
        with open("output/hotpotqa2.json", 'w') as file:
            json.dump(all_record, file, indent=4)
    
    

if __name__ == "__main__":
    main()