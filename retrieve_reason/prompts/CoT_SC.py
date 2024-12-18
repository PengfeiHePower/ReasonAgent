import re
import string
import requests
from collections import Counter
import time

try:
	import dashscope
except ImportError:
	dashscope = None
 
import os
from glob import glob

import json
import re
import argparse

parser = argparse.ArgumentParser(description="GPT QA baselines.")
parser.add_argument("--model", type=str, help="model name", choices=['gpt4', 'qwen', 'qwen2-57', 'llama3-70', 'gpt3.5'])
parser.add_argument('--num', type=int, default=1)
parser.add_argument('--dataset', type=str, default='hotpotqa', choices=['hotpotqa', 'fever', 'mmlubio', 'mmluphy', 'gsm8k', 'math'])

args = parser.parse_args()

def load_checkpoint(chk_file):
	try:
		with open(chk_file, "r") as f:
			return int(f.read().strip())
	except FileNotFoundError:
		return 0  # no checkpoint file, start from beginning

def save_checkpoint(index, chk_file):
	with open(chk_file, "w") as f:
		f.write(str(index))


def extract_dict_from_string(output_string):
	# Use a regular expression to find a block that starts with ```json\n and ends with \n```
	match1 = re.search(r"```\njson\n(.+?)\n```", output_string, re.DOTALL)
	match2 = re.search(r"```json\n(.+?)\n```", output_string, re.DOTALL)
	match3 = re.search(r"```\n(.+?)\n```", output_string, re.DOTALL)
	
	if match1:
		# Extract the JSON part from the regex match
		json_part = match1.group(1).strip()
		
		# Convert the JSON string to a Python dictionary
		try:
			result_dict = json.loads(json_part)
			return result_dict
		except json.JSONDecodeError as e:
			print("Invalid JSON format:", e)
			return None
	elif match2:
		# Extract the JSON part from the regex match
		json_part = match2.group(1).strip()
		
		# Convert the JSON string to a Python dictionary
		try:
			result_dict = json.loads(json_part)
			return result_dict
		except json.JSONDecodeError as e:
			print("Invalid JSON format:", e)
			return None
	elif match3:
		# Extract the JSON part from the regex match
		json_part = match3.group(1).strip()
		
		# Convert the JSON string to a Python dictionary
		try:
			result_dict = json.loads(json_part)
			return result_dict
		except json.JSONDecodeError as e:
			print("Invalid JSON format:", e)
			return None
	else:
		print("No JSON block found in the string.")
		return None

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

def format_multi(problem):
	question = problem['question']
	option_list = problem['options']
	formatted_text = "Options:\n"
	for index, option in enumerate(option_list, start=1):
		# Adding a letter label (A, B, C, D) before each option
		formatted_text += f"{chr(64 + index)}) {option}\n"
	options = formatted_text.strip()
	return question, options
	

def llm(input_text, model="gpt4", stop=["\n"]):
	if model == "gpt4":
		url = "http://47.88.8.18:8088/api/ask"
		HTTP_LLM_API_KEY=os.getenv("OPENAI_key")
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
				"temperature": 1.0
				}
		response = requests.post(url, json=data, headers=headers)
		response = response.json()
		print(f"response:{response}")
		new_response = response['data']['response']
	elif model == "gpt3.5":
		url = "http://47.88.8.18:8088/api/ask"
		HTTP_LLM_API_KEY=os.getenv("OPENAI_key")
		headers = {
					"Content-Type": "application/json",
					"Authorization": "Bearer " + HTTP_LLM_API_KEY
					}
		data = {
				"model": 'gpt-3.5-turbo',
				"messages": [
					{"role": "system", "content": "You are a helpful assistant."},
					{"role": "user", "content": input_text}
				],
				"n": 1,
				"temperature": 1.0
				}
		response = requests.post(url, json=data, headers=headers)
		response = response.json()
		new_response = response['data']['response']
	elif model == 'qwen':
		api_key = os.getenv("DASH_key")
		dashscope.api_key = api_key
		response = dashscope.Generation.call(
			model='qwen-max',
			messages=[
					{"role": "system", "content": "You are a helpful assistant."},
					{"role": "user", "content": input_text}
				],
			result_format="message",  # set the result to be "message" format.
		)
		new_response = response.output
	elif model == 'qwen2-57':
		api_key = os.getenv("DASH_key")
		dashscope.api_key = api_key
		response = dashscope.Generation.call(
			model='qwen2-57b-a14b-instruct',
			messages=[
					{"role": "system", "content": "You are a helpful assistant."},
					{"role": "user", "content": input_text}
				],
			result_format="message",  # set the result to be "message" format.
		)
		new_response = response.output
	elif model == 'llama3-70':
		api_key = os.getenv("DASH_key")
		dashscope.api_key = api_key
		response = dashscope.Generation.call(
			model='llama3-70b-instruct',
			messages=[
					{"role": "system", "content": "You are a helpful assistant."},
					{"role": "user", "content": input_text}
				],
			result_format="message",  # set the result to be "message" format.
		)
		new_response = response.output
		print(f"new_response:{response}")
	return new_response["choices"][0]["message"]["content"]

def CoT_answer(question, prompt_template, format):
	prompt = prompt_template.replace("<<q>>",question)+format
	response = llm(prompt, model=args.model).strip()
	response_dict = extract_dict_from_string(response)
	return response_dict["Thought"], response_dict['Answer'].lower()

def self_consistency(question, prompt_template, format, num_sample=args.num):
	thought_list=[]
	answer_list=[]
	for _ in range(num_sample):
		thought, answer = CoT_answer(question, prompt_template, format)
		thought_list.append(thought)
		answer_list.append(answer)
	formatted_thoughts = "\n".join([f"{i+1}. {thought}" for i, thought in enumerate(thought_list)])
	prompt = (
		f"Here is a math problem:\n\n<<q>>\n\n"
		f"Below are 10 reasoning thoughts generated for this problem:\n\n<<f>>\n\n"
		"Based on these thoughts, determine the most consistent and likely correct answer."
		"If there are ties, choose the best explanation or answer that aligns with the majority reasoning.\n\n"
		"Please provide the thought you choose and then give the final answer."
	)
	prompt = prompt.replace("<<q>>", question)
	prompt = prompt.replace("<<f>>", formatted_thoughts)+format
	response = llm(prompt, model=args.model).strip()
	return response
	
	

# load data
if args.dataset=='hotpotqa':
	with open("data/hotpot_simplified_data.json", "r", encoding="utf-8") as f:
		data = json.load(f)
	with open("prompts/hotpotqa.json", "r", encoding="utf-8") as f:
		prompts = json.load(f)
	question_name = 'question'
	answer_name = 'answer'
elif args.dataset=='fever':
	data = []
	with open("data/paper_dev.jsonl", "r", encoding="utf-8") as f:
		for line in f:
			data.append(json.loads(line.strip()))
	with open("prompts/fever.json", "r", encoding="utf-8") as f:
		prompts = json.load(f)
	question_name = 'claim'
	answer_name = 'label'
elif args.dataset=='mmlubio':
	with open("data/mmlu_bio.json", "r", encoding="utf-8") as f:
		data = json.load(f)
	with open("prompts/mmlu_bio.json", "r", encoding="utf-8") as f:
		prompts = json.load(f)
	question_name = 'question'
	answer_name = 'answer'
elif args.dataset=='mmluphy':
	with open("data/mmlu_phy.json", "r", encoding="utf-8") as f:
		data = json.load(f)
	with open("prompts/mmlu_phy.json", "r", encoding="utf-8") as f:
		prompts = json.load(f)
	question_name = 'question'
	answer_name = 'answer'
elif args.dataset == 'gsm8k':
	data = []
	with open("data/gsm8k_test.jsonl", "r", encoding="utf-8") as f:
		for line in f:
			data.append(json.loads(line.strip()))
	with open("prompts/gsm8k.json", "r", encoding="utf-8") as f:
		prompts = json.load(f)
	question_name = 'question'
	answer_name = 'answer'
elif args.dataset == 'math':
	data = []
	dataset_dir = 'data/MATH_test'
	for subdir in os.listdir(dataset_dir):
		subdir_path = os.path.join(dataset_dir, subdir)
		if os.path.isdir(subdir_path):
			json_files = glob(os.path.join(subdir_path, "*.json"))
			for json_file in json_files:
				with open(json_file, 'r', encoding='utf-8') as f:
					data.append(json.load(f))
	with open("prompts/math.json", "r", encoding="utf-8") as f:
		prompts = json.load(f)
	question_name = 'problem'
	answer_name = 'solution'
elif args.dataset == 'stretagy':
	with open("data/stretagyQA_dev.json", "r", encoding="utf-8") as f:
		data = json.load(f)
	with open("prompts/stretagy.json", "r", encoding="utf-8") as f:
		prompts = json.load(f)
	question_name = 'question'
	answer_name = 'answer'

cot_format_mmlu = """Respond a JSON dictionary in a markdown's fenced code block as follows:
						```json
						{"Thought": "thought steps", "Answer": "One label from [A,B,C,D]"}
						```"""

cot_format = """Respond a JSON dictionary in a markdown's fenced code block as follows:
						```json
						{"Thought": "thought steps", "Answer": "Conclude your answer here."}
						```"""

prompt_template = prompts["0-shot CoT"]+"\nQuestion:<<q>>\n"
if 'mmlu' in args.dataset:
	format=cot_format_mmlu
else:
	format=cot_format


data_size = len(data)
print(f"data size:{data_size}")
if os.path.exists("output/"+ "CoT_SC_"+args.model+'_'+args.dataset+'_'+str(args.num)+'.json'):
    with open("output/"+ "CoT_SC_"+args.model+'_'+args.dataset+'_'+str(args.num)+'.json', 'r') as file:
            records = json.load(file)
else:
    records=[]
start_index = load_checkpoint("checkpoint/" + "CoT_SC_"+args.model+'_'+args.dataset+'_'+str(args.num)+ ".txt")
for i in range(start_index, data_size):
	record={}
	try:
		if 'mmlu' in args.dataset:
			question, options = format_multi(data[i])
			q = question + "\n" +options
		else:
			q = data[i][question_name]
		record['question']=q
		print(f"Question{i}:{q}")
		response = self_consistency(q, prompt_template, format, num_sample=args.num)
		print(f"response:{response}")
		time.sleep(5)
		response_dict = extract_dict_from_string(response)
		print(f"response_dict:{response_dict}")
		pred = response_dict['Answer'].lower()
		print(f"Answer:{pred}")
		gt = data[i][answer_name].lower()
		em = (pred == gt)
		f1 = f1_score(pred, gt)[0]
		record.update({'pred': pred, 'gt': gt, 'em': em, 'f1': f1})
		if 'Thought' in response_dict:
			record['thought'] = response_dict["Thought"]
		records.append(record)
		save_checkpoint(i+1, "checkpoint/"+ "CoT_SC_" +args.model+'_'+args.dataset+'_'+str(args.num)+ ".txt")
	except Exception as e:
		print(f"Error processing record {i}: {e}")
		continue
	with open("output/"+ "CoT_SC_"+args.model+'_'+args.dataset+'_'+str(args.num)+'.json', 'w') as file:
		json.dump(records, file, indent=4)