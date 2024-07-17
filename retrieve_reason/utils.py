import re
import string
import json
import requests
import os
from collections import Counter

import agentscope


def normalize_answer(s):
  def remove_articles(text):
    return re.sub(r"\b(a|an|the)\b", " ", text)
  
  def white_space_fix(text):
      return " ".join(text.split())

  def remove_punc(text):
      exclude = set(string.punctuation)
      return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
      return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
  normalized_prediction = normalize_answer(prediction)
  normalized_ground_truth = normalize_answer(ground_truth)

  ZERO_METRIC = (0, 0, 0)

  if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    return ZERO_METRIC
  if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
    return ZERO_METRIC
  
  prediction_tokens = normalized_prediction.split()
  ground_truth_tokens = normalized_ground_truth.split()
  common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return ZERO_METRIC
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1, precision, recall

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


def llm(input_text, stop=["\n"]):
    url = "http://47.88.8.18:8088/api/ask"
    HTTP_LLM_API_KEY='eyJ0eXAiOiJqd3QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6IjM5NDc3MyIsInBhc3N3b3JkIjoiMzk0NzczMTIzIiwiZXhwIjoyMDIxNjE4MzE3fQ.oQx2Rh-GJ_C29AfHTHE4x_2kVyy7NamwQRKRA4GPA94'
    headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + HTTP_LLM_API_KEY
                }
    data = {
            "model": 'gpt-4',
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ],
            "n": 1,
            "temperature": 0.0,
            # "stop": ["\n"]
            }
    response = requests.post(url, json=data, headers=headers)
    response = response.json()
    new_response = response['data']['response']
    return new_response["choices"][0]["message"]["content"]

def llm_equal(question, gt, pred):
    prompt = f"I have a ground truth answer and a suspect answer for a question. I need to determine if the suspect answer is correct by comparing it to the ground truth answer. Please compare the two answers and let me know if the suspect answer is correct. Please also provide the reason behind your comparison.\nQuestion:{question}\nGround Truth Answer: {gt}\nSuspect Answer: {pred}\n\nCorrect:[True or False]\nReason:"
    return llm(prompt.format(question=question,gt=gt, pred=pred))