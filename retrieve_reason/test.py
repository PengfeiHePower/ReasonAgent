import json
import requests

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
    prompt = f"I have a ground truth answer and a suspect answer for a question. I need to determine if the suspect answer is correct by comparing it to the ground truth answer. Please compare the two answers and let me know if the suspect answer is correct. Please also provide the reason behind your comparison.\nQuestion:{question}\nGround Truth Answer: {gt}\nSuspect Answer: {pred}\nYou need respond in the following strcure.\n\nCorrect:[True or False]\nReason:"
    return llm(prompt.format(question=question,gt=gt, pred=pred))

def parse_output(text):
    # Initialize the dictionary
    result_dict = {"correct": None, "reason": None}
    
    # Split the output text into lines
    if '\n' not in text:
        result_dict['correct']=False
        result_dict['reason']=None

    lines = text.split('\n')
    
    # Extract the "Correct" and "Reason" parts
    for line in lines:
        if line.startswith('Correct:'):
            correct_value = line[len('Correct: '):].strip()
            # Convert 'True' and 'False' to boolean values
            if correct_value == 'True':
                result_dict["correct"] = True
            elif correct_value == 'False':
                result_dict["correct"] = False
        elif line.startswith('Reason:'):
            result_dict["reason"] = line[len('Reason: '):].strip()
    
    return result_dict

file_names = ['output/llama3-70_hotpotqa_base_analysisFalse.json', 'output/llama3-70_hotpotqa_icl_analysisFalse.json', 'output/llama3-70_hotpotqa_cot_analysisFalse.json']

for name in file_names:
    print(f"file name:{name}")
    with open(name, 'r') as file:
        data = json.load(file)  
    print(f"data length:{len(data)}")
    ems = [item['em'] for item in data]
    f1s = [item['f1'] for item in data]
    print(f"em:{sum(ems)/len(ems)}")
    print(f"f1:{sum(f1s)/len(f1s)}")
    llm_eval = []
    for item in data:
        llm_eval.append(llm_equal(question=item['question'], gt=item['gt'], pred=item['pred']))
    llm_evals=[]
    for item in llm_eval:
        llm_evals.append(parse_output(item))
    
    trues = [item['correct'] for item in llm_evals]
    print(f"llm true:{sum(trues)/len(trues)}")
    