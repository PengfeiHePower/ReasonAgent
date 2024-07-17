from agentscope.agents.agent import AgentBase
from agentscope.message import Msg, MessageBase
from typing import Optional, Any, Union
import ast

import json

# def text_to_dict(text, sep):
#     """
#     Transform the response of agent into dict
#     """
#     lines = text.split(sep)
#     result_dict = {}

#     # Iterate through each line and extract key-value pairs
#     for line in lines:
#         key, value = line.split(':', 1)
#         value=value.strip()
#         if value.startswith('[') and value.endswith(']'):
#             value = ast.literal_eval(value)
#         result_dict[key.strip()] = value

#     return result_dict

class ReasonAgent(AgentBase):
    """
    Reasoning module for retrieval-based reasoning
    """
    def __init__(
            self,
            name: str,
            sys_prompt: str,
            model_config_name: str,
            parser_analysis: Any,
            parser_next_step:Any,
            memory_config: Optional[dict] = None,
            **kwargs: Any,
    ) -> None:
        """
        Initialize the Reason agent
        """
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            memory_config=memory_config,
        )
        with open("configs/prompt_template.json", "r", encoding="utf-8") as f:
            prompt_template = json.load(f)
        self.prompt_template = prompt_template['reason']
        self.parser_analysis = parser_analysis
        self.parser_next_step = parser_next_step
        
        
        
        
    def structure_analysis(self, query):
        """
        Generate structure analysis for query.
        """
        prompt_understand = self.prompt_template["prompt_understand_template_new"].format(q=query)
        msg = Msg(
            name="user",
            role="user",
            content=prompt_understand+self.parser_analysis.format_instruction,
            )
        msgs = self.model.format(msg)
        response = self.model(msgs, parse_func=self.parser_analysis.parse)
        response_msg = Msg(
            name="reason agent",
            role="assistant",
            content=self.parser_analysis.to_content(response.parsed)
        )
        return response_msg
        
        
        
    def initial_step(self, analysis):
        """
        Generate initial step
        """
        prompt_initial_step = self.prompt_template["prompt_step1_template"].format(analysis=analysis)
        msg = Msg(
            name="user",
            role="user",
            content=prompt_initial_step+self.parser_step1.format_instruction,
            )
        msgs = self.model.format(msg)
        response = self.model(msgs, parse_func=self.parser_step1.parse)
        response_msg = Msg(
            name="reason agent",
            role="assistamt",
            content=self.parser_step1.to_content(response.parsed),
        )
        return response_msg
        
        
    def next_step(self,question, thought, analysis, knowledge):
        """
        Generate next step
        """
        prompt_next_step = self.prompt_template["prompt_step_seq_template_new"].format(question=question, analysis=analysis, thought=thought, knowledge=knowledge)
        msg = Msg(
            name="user",
            role="user",
            content=prompt_next_step+self.parser_next_step.format_instruction,
            )
        msgs = self.model.format(msg)
        response = self.model(msgs, parse_func=self.parser_next_step.parse)
        response_msg = Msg(
            name="reason agent",
            role="assistamt",
            content=self.parser_next_step.to_content(response.parsed),
        )
        return response_msg
    
    def step_to_action(self, step):
        """
        Generate action based on step
        """
        prompt_action = self.prompt_template["prompt_action_template"].format(step=step)
        msg = Msg(
            name="user",
            role="user",
            content=prompt_action+self.parser_action.format_instruction,
            )
        msgs = self.model.format(msg)
        response = self.model(msgs, parse_func=self.parser_action.parse)
        response_msg = Msg(
            name="reason agent",
            role="assistamt",
            content=self.parser_action.to_content(response.parsed),
        )
        return response_msg
    
    def reply(self, x: dict = None) -> dict:
        """
        Obtain reply from Reason Agent
        
        Args:
            x (`dict`, defaults to `None`):
                Query sent to the agent, in the following form:
                    x={
                        "type": on type from ["structure_analysis", "initial_step", "next_step", "step_to_action"],
                        "query": ,
                        "analysis": ,
                        "step": ,
                        "action": ,
                        "info": 
                    }
        Returns:
            A dictionary contains the response in the form of Msg.
        """
        if x["type"] == "structure_analysis":
            if x["query"]:
                return self.structure_analysis(x["query"])
            else:
                raise ValueError("Query should not be empty")
        elif x["type"] == "initial_step":
            if x["analysis"]:
                return self.initial_step(x["analysis"])
            else:
                raise ValueError("Analysis should not be empty")
        elif x["type"] == "thought":
            return self.next_step(question=x['query'], thought=x["step"], analysis=x["analysis"], knowledge=x["info"])
        elif x["type"] == "step_to_action":
            if x["step"]:
                return self.step_to_action(step=x["step"])
            else:
                raise ValueError("Steps should not be empty")
        else:
            raise ValueError(f"Invalid type {x['type']} for reasoning.")