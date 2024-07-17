""""
Test the robustness against attacks on GSM8K dataset
"""

import argparse
import json
from tqdm import tqdm
from datasets import load_dataset

def trigger_selection(trigger):
    if trigger != None:
        with open("triggers.json", "r") as f:
            triggers = json.load(f)
        trigger = triggers[trigger]
    print(f"trigger:\n{trigger}")
    return trigger

def bd_embed(q, trigger_type, trigger, position):
    if trigger_type == 'special':
        if position == 'last':
            q = q + trigger
        elif position == 'before_choices':
            q = q.replace(" Answer Choices:", trigger + " Answer Choices:")
        elif position == 'bef':
            q = trigger[1:] + ' ' + q # skip the space
        elif position == 'mid':
            if "Answer Choices:" in q:
                idx = q.index("Answer Choices:")
                q = q[:idx]
                # split by space
                q = q.split(' ')
                # insert trigger in the middle
                q.insert(len(q)//2, trigger[1:]) # skip the space
                # join back
                q = ' '.join(q)
            else:
                q = q.split(' ')
                # insert trigger in the middle
                q.insert(len(q)//2, trigger[1:]) # skip the space
                # join back
                q = ' '.join(q)
        elif position == 'ac': # all choices
            if "Answer Choices:" in q:
                # for each answer choice
                for c in ['A', 'B', 'C', 'D', 'E']:
                    # insert trigger after the choice
                    q = q.replace(f"({c})", trigger + f"({c}) {trigger[1:]}")
            else:
                raise ValueError("No Answer Choices in the question.")

    return q

def main():
    # load datasets
    from datasets import load_dataset
    gsm8k = load_dataset('gsm8k', 'main')
    gsm8k_test = gsm8k['test']
    questions = gsm8k_test['question']
    answers = gsm8k_test['answer']
    
    # size
    if args.num == -1:
        test_size = len(questions)
    else:
        test_size = min(args.num, len(questions))
    print(f"test size: {test_size}")
    questions = questions[:test_size]
    answers = answers[:test_size]
    
    trigger = trigger_selection(args.trigger)
    
    # trigger position for question
    if args.tp is not None:
        assert args.tp in ['bef', 'mid', 'ac']
        trigger_position = args.tp
        print("trigger position: ", trigger_position)
    else:
        # default position
        if args.task == 'csqa':
            trigger_position = 'before_choices'
        else:
            trigger_position = 'last'
    
    for q, a in tqdm(zip(questions, answers), total=test_size):
        if args.attack:
            q = bd_embed(q, trigger_type='special', trigger=trigger, position=trigger_position)
            print(f"q:{q}")
    
            
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cot', type=str, help='context selection')
    parser.add_argument('--num', type=int, default=-1, help='number of running samples, -1 for all samples')
    # parser.add_argument('--defense', type=str, default=None)
    parser.add_argument('-attack', action='store_true', default=False, help='whether add trigger to test question')
    parser.add_argument('--trigger', type=str, default="p01", help='id of trigger for triggers.json')
    # parser.add_argument('--api_id', type=int, default=0)
    # parser.add_argument('-eval_only', action='store_true', default=False, help='whether only eval the output file')
    # parser.add_argument('--sc', type=int, default=1, help='number of output per question, default 1. More than 1 set for self-consistency')
    # parser.add_argument('--index', type=str, default=None, help='subsampling index file identifier to run')
    # parser.add_argument('--resume', type=int, default=-1, help='resume index')
    # parser.add_argument('-not_overwrite', action='store_true', default=False, help='whether not overwrite the existing output file')
    parser.add_argument('-rand', action='store_true', default=False, help='whether randomize the order of questions')
    parser.add_argument('--tp', type=str, default=None, help='trigger position for question')
    parser.add_argument('--eval_num', type=int, default=-1, help='number of samples for eval, -1 for all samples')
    args = parser.parse_args()
    
    