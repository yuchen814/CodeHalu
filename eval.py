import argparse
from testing_utils import run_test
import json, os
import multiprocessing
import numpy as np
from typing import Dict
import traceback
import sys
sys.path.append('.')
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList   
from tqdm import tqdm
from datasets import load_dataset
from utils import load_problems
 
TIMEOUT = 30 

programming_halus = {
    "Data_Compliance_Hallucination": {
        "TypeError": "TypeError",
        "ValueError": "ValueError",
        "ZeroDivisionError": "ZeroDivisionError",
    },
    "Structural_Access_Hallucination": {
        "IndexError": "IndexError",
        "KeyError": "KeyError"
    },
    "Identification_Hallucination": {
        "NameError": "NameError",
        "AttributeError": "AttributeError",
        "UnboundLocalError": "UnboundLocalError",
    },
    "External_Source_Hallucination": {
        "ImportError": "ImportError",
        "ModuleNotFoundError": "ModuleNotFoundError"
    },
    "Physical_Constraint_Hallucination": {
        "RecursionError": "RecursionError",
        "MemoryError": "MemoryError",
    },
    "Calculate_Boundary_Hallucination": {
        "OverflowError": "OverflowError",
        "StopIteration": "StopIteration"
    },
    "Logic_Deviation": { 
        "Logic_Deviation": "Logic_Deviation"
    },  
    "Logic_Breakdown": { 
        "Logic_Breakdown": "Logic_Breakdown"
    }
}

def serialize_errors(errors_dict):
    serialized_errors = {}
    for error_name, (error_values, count) in errors_dict.items():
        serialized_errors[error_name] = {
            'values': list(error_values),  
            'count': count
        }
    return serialized_errors

def add_error(errors_dict, error_name, error_value):
    if error_name not in errors_dict:
        errors_dict[error_name] = (set(), 0)
    errors, count = errors_dict[error_name]
    errors.add(error_value)
    errors_dict[error_name] = (errors, count + 1)
    return errors_dict

def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(sample, generation, debug, result, error):
        res,err = run_test(sample, test=generation, debug=debug)
        
       
        result.append(res)
        error.append(err)

    manager = multiprocessing.Manager()
    result = manager.list()
    error = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(sample, generation, debug, result, error))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:  
        in_outs = json.loads(sample["input_output"])
        
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        error = [[{'name':'TimeError','value':"global timeout"} for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    
    return result[0], error[0]

def load_generation(input_file):
    generations = {}
    in_out = {}
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            res = json.loads(line)
            task_id = res['task_id']
            
            output = res["deal_response"]
            input_output = res['input_output']
            generations.setdefault(task_id, list()).append(output)
            in_out.setdefault(task_id, list()).append(input_output) 
            data.append(res)
    return generations, data, in_out  

def evaluate_generations(generations, samples,in_out, idx=None, debug=False):
    
    results = {}
    errors = {}
    tokenizer = AutoTokenizer.from_pretrained(
            'codellama/CodeLlama-7b-Instruct-hf',  
            use_fast=True,
            trust_remote_code=True
        )

    for task_id, problem_generations in tqdm(generations.items()):
        
        sample = samples[task_id]
        original_input_output = sample["input_output"] 
        input_output = in_out[task_id]
        
        if task_id not in results:
            results[task_id] = {}
            errors[task_id] = {}
      
        for o_idx, o in enumerate(problem_generations):
            
            data = json.loads(sample['input_output'])


            inputs = data["inputs"]
            
            sample["input_output"] = json.dumps(input_output[o_idx]) 
            key = json.dumps(input_output[o_idx]) 
           
            curr_res = [-2]
            try: 
                
                token_len = tokenizer.tokenize(o)
                if len(token_len)>=1300:
                    curr_res = [-1]
                    curr_err = [{'name': 'Logic Breakdown', 'value': 'Logic Breakdown'}]
                else:
                    curr_res,curr_err = check_correctness(sample, o, timeout=TIMEOUT, debug=debug)
                sample["input_output"] = original_input_output 

                if debug:
                    print(f"\nSuccessful compilation of task {o_idx}!")
                fixed = []
                for e in curr_res:
                    if isinstance(e, np.ndarray):
                       e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed.append(e)
                curr_res = fixed
                if not np.all(curr_res):
                    if debug:
                        print(f"Results were not True for all test cases")
            except Exception as e:   
                print(e)
                if debug:
                    print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
                break
            finally:
                assert isinstance(curr_res, list)
                if key not in results[task_id]:
                    results[task_id][key] = []
                    errors[task_id][key] = []
        
                results[task_id][key].append(curr_res)
                errors[task_id][key].append(curr_err)

    return results,errors



def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Evaluate generations against problems.')
    
    parser.add_argument('--halu_type', 
                        type=str, 
                        required=True, 
                        help='The type of hallucination you want to evaluate.')

    parser.add_argument('--generation_file', 
                        type=str, 
                        required=True, 
                        help='File containing generations to be evaluated.')

    return parser.parse_args()

def main(args):
    
    problems = load_dataset("codeparrot/apps", split="test")
    generation_file = args.generation_file
    halu_type = args.halu_type
    gen_file_basename = os.path.basename(generation_file)

    generations,ori_datas,in_out = load_generation(generation_file)
    
    results,errors = evaluate_generations(generations, problems,in_out)

 
    new_id = 0

    task_id_to_data = {item['task_id']: item for item in ori_datas}

    errors_dict = {}
    changed_data = []
    total_errors = set()
    
    for i in errors.keys():
        for j in errors[i].keys(): 
        
            input_output = json.loads(j) 
      
            problem = task_id_to_data[i]
           
            
        
            unique_errors = []
            seen_names = set()
            if errors[i][j][0][0] is None and (results[i][j][0][0] == False or results[i][j][0][0]<0):

                new_data = {
                            "id": new_id,
                            "task_id": problem["task_id"],
                            "prompt": problem['prompt'],
                            "input":input_output['inputs'][0], 
                            "output":input_output['outputs'][0],
                            "code": problem["deal_response"], 
                            "error_type": 'Wrong logic'
                        },
                with open(f'evaluated_results/{gen_file_basename}_data.json', 'a') as file:
                    json.dump(new_data, file)
                    file.write('\n') 
                new_id += 1
                errors[i][j] = [[{'name': 'Logic_Deviation', 'value': 'Logic_Deviation'}]]  
                
            elif errors[i][j][0][0] is None and (results[i][j][0][0] == True or results[i][j][0][0]>0):   
                new_data = {
                            "id": new_id,
                            "task_id": problem["task_id"],
                            "prompt": problem['prompt'],
                            "input":input_output['inputs'][0], 
                            "output":input_output['outputs'][0],
                            "code": problem["deal_response"], 
                            "error_type": None
                        },
                with open(f'evaluated_results/{gen_file_basename}_data.json', 'a') as file:
                    json.dump(new_data, file)
                    file.write('\n') 
                new_id += 1
                errors[i][j] = [[{'name': 'Correct', 'value': 'Correct'}]]
            
            
            elif (errors[i][j][0][0]['name']  == 'TimeError' or errors[i][j][0][0]['name']  == 'TimeoutException'):
          
                new_data = {
                            "id": new_id,
                            "task_id": problem["task_id"],
                            "prompt": problem['prompt'],
                            "input":input_output['inputs'][0], 
                            "output":input_output['outputs'][0],
                            "code": problem["deal_response"],  
                            "error_type": 'Timeout'
                        },
                with open(f'evaluated_results/{gen_file_basename}_data.json', 'a') as file:
                    json.dump(new_data, file)
                    file.write('\n') 
                new_id += 1
                
            else:    
                new_data = {
                            "id": new_id,
                            "task_id": problem["task_id"],
                            "prompt": problem['prompt'],
                            "input":input_output['inputs'][0], 
                            "output":input_output['outputs'][0],
                            "code": problem["deal_response"], 
                            "error_type": errors[i][j][0][0]   
                        },
                with open(f'evaluated_results/{gen_file_basename}_data.json', 'a') as file: 
                    json.dump(new_data, file)
                    file.write('\n') 
                new_id += 1
     
            error_name = errors[i][j][0][0]['name'] 
            error_value = errors[i][j][0][0]['value']
            errors_dict = add_error(errors_dict, error_name, error_value)
            if error_name not in seen_names:
                seen_names.add(error_name) 
                unique_errors.append(errors[i][j][0][0])
            if error_name not in total_errors:
                total_errors.add(error_name)
       
    
    errors_dict = serialize_errors(errors_dict)  
    
    count = 0 
    for error_key,error_value in programming_halus[halu_type].items():
        try:
            count += errors_dict[error_value]['count']
        except Exception as e:
            count = count
    print("halu_count: ",count)    
    print("total_count: ",len(ori_datas))
    halu_percentage = (count / len(ori_datas)) * 100   
    halu_percentage = round(halu_percentage, 2)
    print(halu_type)   
    print("percentage: ",halu_percentage)   
     
    with open(f'evaluated_results/{gen_file_basename}_errors_dict.json', 'w') as json_file:
        json.dump(errors_dict, json_file, indent=4)
    
  

if __name__ == "__main__":
    args = parse_args()
    main(args)