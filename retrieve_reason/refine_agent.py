from agentscope.agents.agent import AgentBase
from agentscope.message import Msg
from typing import Optional, Any, Union

import json

def text_to_dict(text, sep):
    """
    Transform the response of agent into dict
    """
    lines = text.split(sep)
    result_dict = {}

    # Iterate through each line and extract key-value pairs
    for line in lines:
        key, value = line.split(':', 1)
        result_dict[key.strip()] = value.strip()

    return result_dict

class RefineAgent(AgentBase):
    """
    Refinement module for retrieval-based reasoning
    """
    def __init__(
            self,
            name: str,
            sys_prompt: str,
            model_config_name: str,
            refine_analysis_parser: Any,
            refine_analysis_eval_parser: Any,
            memory_config: Optional[dict] = None,
            **kwargs: Any,
    ) -> None:
        """
        Initialize the Reason agent
        """
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name
        )
        with open("configs/prompt_template.json", "r", encoding="utf-8") as f:
            prompt_template = json.load(f)
        self.prompt_template = prompt_template['refine']
        self.refine_analysis_parser = refine_analysis_parser
        self.refine_analysis_eval_parser = refine_analysis_eval_parser
        
    
    def refine_analysis(self, query, prev_analysis):
        """
        Refine previous analysis and provide suggestions
        """
        prompt_evaluation = self.prompt_template["prompt_evaluation_template"].format(q=query, analysis=prev_analysis)
        eval_msg = Msg(
            name="user",
            role="user",
            content=prompt_evaluation+self.refine_analysis_eval_parser.format_instruction,
            )
        eval_msgs = self.model.format(eval_msg)
        evaluation = self.model(eval_msgs, parse_func=self.refine_analysis_eval_parser.parse)
        evaluation_output = self.refine_analysis_eval_parser.to_content(evaluation.parsed)
        suggestions = evaluation_output["Suggestions"]
        prompt_refine_structure = self.prompt_template["prompt_refine_structure_template"].format(q=query,analysis=prev_analysis,suggestions=suggestions)
        refine_msg = Msg(
            name="user",
            role="user",
            content=prompt_refine_structure+self.refine_analysis_parser.format_instruction,
            )
        refine_msgs = self.model.format(refine_msg)
        refinement = self.model(refine_msgs, parse_func=self.refine_analysis_parser.parse)
        total_msg = Msg(
            name="refine assistant",
            role="assistant",
            content=self.refine_analysis_parser.to_content(refinement.parsed),
        )
        return total_msg
    
    def refine_action(self, step, action, info):
        """
        Refine the action based on steps and retrieved info
        """
        
        prompt_action_refine = self.prompt_template["prompt_action_refine_template"].format(step=step, action=action, info=info)
        refine_msg=Msg(
            name="user",
            role="user",
            content=prompt_action_refine,
        )
        refine_msgs = self.model.format(refine_msg)
        refinement = self.model(refine_msgs)
        response_msg = Msg(
            name="refine assistant",
            role="assistant",
            content=text_to_dict(refinement.text, "\n\n")
        )
        
        return response_msg
    
    def refine_step(self, analysis, step, info):
        """
        Refine steps based on retrieved info and analysis
        """
        prompt_step_refine = self.prompt_template["prompt_step_refine_template"].format(analysis=analysis, step=step, info=info)
        refine_msg=Msg(
            name="user",
            role="user",
            content=prompt_step_refine,
        )
        refine_msgs = self.model.format(refine_msg)
        refinement = self.model(refine_msgs)
        response_msg = Msg(
            name="refine assistant",
            role="assistant",
            content=self.refine_annlysis_parser.to_content(refinement.parsed),
        )
        
        return response_msg
    
    def reply(self, x: dict = None) -> dict:
        
        """
        Reply from the refinement agent
        
        Args:
            x (`dict`, defaults to `None`):
                Query sent to the agent, in the following form:
                    x={
                        "type": on type from ["structure_analysis", "step", "action"],
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
            if x["query"] and x["analysis"]:
                return self.refine_analysis(query=x["query"], prev_analysis=x["analysis"])
            else:
                raise ValueError("Query/Analysis should not be empty")
        elif x["type"] == "step":
            if x["analysis"] and x["step"]:
                return self.refine_step(analysis=x["analysis"], step=x["step"], info=x["info"])
            else:
                raise ValueError("Analysis/Step should not be empty")
        elif x["type"] == "action":
            if x["step"] and x["action"]:
                return self.refine_action(step=x["step"], action=x["action"], info=x["info"])
            else:
                raise ValueError("Steps/Action should not be empty")
        else:
            raise ValueError(f"Invalid type {x['type']} for refinement.")