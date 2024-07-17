from agentscope.agents.agent import AgentBase
from agentscope.message import Msg, MessageBase
from typing import Optional, Any, Union, List

class MemoryAgent():
    """
    Memory agent
    """
    def __init__(self,
        ) -> None:
        """
        Initialize the Retrieval agent
        """
        """
        steps: list of strings store reasoning steps
        actions: list of strings store actions
        retrieval_info: list of dicts storing retrieved information in this form {
            "action":,
            "step":,
            "raw info":,
            "extracted info":
        }
        """
        
        self.steps = []
        # self.actions = []
        self.retrieval_info = []
        
    
    def analysis_store(self, analysis):
        """
        Store analysis(dict)
        """
        self.analysis = analysis
    
    def step_update(self, which_step, new_step):
        self.steps[which_step] = new_step
        
    def step_append(self, new_step):
        self.steps.append(new_step)
    
    def action_update(self, which_action, new_action):
        self.actions[which_action] = new_action
    
    def action_append(self, new_action):
        self.actions.append(new_action)
        
    def info_append(self, new_info):
        # self.retrieval_info['raw'].append(new_info['raw info'])
        # self.retrieval_info['extracted'].append(new_info['extracted info'])
        self.retrieval_info.append(new_info)
        
    def info_update(self, which_info, new_info):
        # self.retrieval_info['raw'][which_info] = new_info['raw info']
        # self.retrieval_info['extracted'][which_info] = new_info['extracted info']
        self.retrieval_info[which_info] = new_info
    
    def reset(self):
        self.steps = []
        # self.actions = []
        self.retrieval_info = []
        self.analysis=None
    
    def export(self):
        self.record = {
            "structure analysis":self.analysis,
            "steps":self.steps,
            "retrieval info":self.retrieval_info
        }
        return self.record