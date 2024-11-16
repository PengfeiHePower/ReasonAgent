from agentscope.agents.agent import AgentBase
from agentscope.message import Msg
from typing import Optional, Any, Union, List
import ast

import json
from agentscope.service.web.wikipedia import (
        wikipedia_search_categories,
        wikipedia_search
)

def text_to_dict(text, sep):
    """
    Transform the response of agent into dict
    """
    if '\n\n' in text:
        text = text.split('\n\n')[0]
    
    lines = text.split(sep)
    result_dict = {}

    # Iterate through each line and extract key-value pairs
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            value = value.strip()
            if value.startswith('[') and value.endswith(']'):
                value = ast.literal_eval(value)
            result_dict[key.strip()] = value

    return result_dict

def info_transform(info, source):
    """
    Transform different form of info into the format that can be easily understood by the LLM
    """
    if source is not None:
        return info
    
    if source=="category list":
        titles = [item['title'] for item in info]
        formatted_titles = "List of Categories:\n" + "\n".join(f"- {title}" for title in titles)
        return formatted_titles
    else:
        return info
    # elif source=="infobox":
    #     text_lines = []
    #     for key, value in info.items():
    #         text_lines.append(f"{key}: {value}")
    #     return "\n".join(text_lines)
    # elif source=="category list":
    #     titles = [item['title'] for item in info]
    #     formatted_titles = "List of Categories:\n" + "\n".join(f"- {title}" for title in titles)
    #     return formatted_titles
    # elif source=="table": #plain text table
    #     headers = list(info.keys())
    #     rows = zip(*info.values())
    
    #     # Calculate column widths
    #     column_widths = [max(len(str(item)) for item in [header] + list(column)) for header, column in zip(headers, info.values())]
    
    #     # Create a format string
    #     format_str = " | ".join([f"{{:<{width}}}" for width in column_widths])
    
    #     # Create the header row
    #     header_row = format_str.format(*headers)
    #     separator_row = "-+-".join(['-' * width for width in column_widths])
    
    #     # Create the data rows
    #     data_rows = [format_str.format(*row) for row in rows]
    
    #     # Combine all parts into the final table
    #     table = "\n".join([header_row, separator_row] + data_rows)
    #     return table
    #     # json_string = json.dumps(info, indent=4) #JSON string
    #     # print(json_string)
    # elif source=="images with caption":
    #     formatted_text = "List of Image Titles and Captions:\n"
    #     for image in info:
    #         formatted_text += f"Title: {image['title']}\n"
    #         formatted_text += f"Caption: {image['caption']}\n\n"
    #     return formatted_text
    # else:
    #     raise NotImplementedError("Source is not implemented.")
        

class RetrieveAgent(AgentBase):
    """
    Retrieval agents
    """
    def __init__(self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        data_sources: List,
        parser_retrieval_analysis:Any,
        memory_config: Optional[dict] = None,
        **kwargs: Any,
        ) -> None:
        """
        Initialize the Retrieval agent
        """
        super().__init__(
        name=name,
        sys_prompt=sys_prompt,
        model_config_name=model_config_name
        )
        
        self.source = data_sources #['text', 'category list', 'infobox', 'table', 'images with caption']
        with open("configs/prompt_template.json", "r", encoding="utf-8") as f:
            prompt_template = json.load(f)
        self.prompt_template = prompt_template['retrieval']
        self.parser_retrieval_analysis=parser_retrieval_analysis

    def retrieval_analysis(self,action):
        """
        Analyze the action and decide the keywords as well as sources.
        
    
        Returns:
        retrieval_dict:{
            "Retrieval entity":
            "Retrieve sources":
            "Reason":
            }
        """
        if action["retrieval"].strip().lower()=='yes':
            prompt_retrieval = self.prompt_template["prompt_retrieval_template"].format(Retrieval_info=action["retrieval info"], sources=self.source)
            # print(f"prompt_retrieval:{prompt_retrieval}")
            retrieval_msg = Msg(
                name="user",
                role="user",
                content=prompt_retrieval+self.parser_retrieval_analysis.format_instruction,
                )
            retrieval_msgs = self.model.format(retrieval_msg)
            response = self.model(retrieval_msgs, parse_func=self.parser_retrieval_analysis.parse)
            retrieval_analysis = self.parser_retrieval_analysis.to_content(response.parsed)
            print(f"retrieval_analysis:{retrieval_analysis}")
            return retrieval_analysis
        else:
            return {"Retrieval entity":None,"Retrieve sources":None, "Reason":None}
    
    
    def execute_retrieval(self,key,source):
        """
        Execute the retrieval and obtain raw information
        
        Returns:
        retrieved_info: ServiceResponse
        """
        
        if source == 'category list':
            return wikipedia_search_categories(entity=key, max_members=1000)
        else:
            return wikipedia_search(entity=key)
    
    def extract_info(self,info,step):
        """
        Extract most relevant information with the step
        """
        prompt_extract_info = self.prompt_template["prompt_info_extraction_template"].format(step=step, info=info)
        extraction_msg = Msg(
            name="user",
            role="user",
            content=prompt_extract_info,
            )
        extraction_msgs = self.model.format(extraction_msg)
        return self.model(extraction_msgs).text
    
    def reply(self, x: dict = None) -> dict:
        """
        Handle the retrieval request based on the action
        
        Args:
        x: a dict in this form {
            "action":
            "step":
        }
        
        Return:
        a dict Msg {
            "name":
            "role":
            "content":{
                "action":
                "raw info":{source:info}
                "extracted info":{source: info}
            }
        }
        """
        # if not x["action"] or not x["step"]:
        if not x["retrieval"]:
            raise ValueError("Retrieval should not be empty.")
        
        if x["retrieval"].strip().lower() == 'no':
            response_msg = Msg(
            name="retrieval agent",
            role="assistant",
            content={
                "Retrieval": x["retrieval"],
                "Retrieval info": x["retrieval info"],
                "Retrieval entity": None,
                "Raw info":None,
                "Extracted info":None
            }
            )
            return response_msg
        
        # retrieval_process = self.retrieval_analysis(x["action"])
        retrieval_process = self.retrieval_analysis(x)
        print(f"retrieval_process:{retrieval_process}")
        # print(f"retrieval_process type:{type(retrieval_process)}")
        retrieval_content_raw = {} #save retrieval info
        for source in retrieval_process['Retrieval sources']:
            source = source.strip().lower()
            if source not in self.source:
                continue
            retrieved_info = self.execute_retrieval(key=retrieval_process["Retrieval entity"],source=source)
            retrieval_content_raw[source]=retrieved_info
        print(f"retrieval_content_raw:{retrieval_content_raw}")
        retrieval_content_extract = {} #save retrieval info
        for source in retrieval_content_raw.keys():
            transformed_info = info_transform(info=retrieval_content_raw[source], source=source)
            prompt_extracted_info = self.prompt_template["prompt_info_extraction_template"].format(step=x['retrieval info'], info=transformed_info)
            extracted_msg = Msg(
            name="user",
            role="user",
            content=prompt_extracted_info,
            )
            extracted_msgs = self.model.format(extracted_msg)
            extracted_info = self.model(extracted_msgs)
            retrieval_content_extract[source] = extracted_info.text
        
        response_msg = Msg(
            name="retrieval agent",
            role="assistant",
            content={
                # "action": x["action"],
                "Retrieval": x["retrieval"],
                "Retrieval info": x["retrieval info"],
                "Retrieval entity": retrieval_process["Retrieval entity"],
                "Raw info":retrieval_content_raw,
                "Extracted info":retrieval_content_extract
            }
            )
        
        return response_msg