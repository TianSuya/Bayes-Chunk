import json
from pathlib import Path

from util.globals import *

def get_llama_with_answer(que,ans):
  return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{ans}<|eot_id|>"""

def get_llama_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

def get_llama_without_answer_cot(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nPlease provide a multi-hop explanation for the next question: {que}<|eot_id|>"""

def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""

def get_qwen_without_answer_cot(que):
    return f"""<|im_start|>user\n Please provide a multi-hop explanation for the next question: {que}<|im_end|>\n<|im_start|>assistant\n"""

def get_vicuna_without_answer(que):
    return f"""USER: {que} ASSISTANT:"""

class QWQDataset:

    def __init__(self, data_dir: str, model_name: str, size=None, *args, **kwargs):
        """
        Initialize QWQ dataset
        
        Args:
            data_dir: Data directory path
            model_name: Model name (used for formatting prompt)
            size: Dataset size limit (None means use all data)
        """
        data_dir = Path(data_dir)
        
        # Directly load qwq.json file
        qwq_file = data_dir / "qwq.json"
        
        if not qwq_file.exists():
            raise FileNotFoundError(
                f"Dataset file not found. Please ensure the following file exists:\n"
                f"  - {qwq_file}"
            )
        
        with open(qwq_file, 'r', encoding='utf-8') as json_file:
            raw = json.load(json_file)
        
        # Convert data format: problem -> question, solution -> answer
        for i in raw:
            # Map problem to question, solution to answer
            i['question'] = i.pop('problem', '')
            i['answer'] = i.pop('solution', '')
            # Keep source field (optional)
            
            # Format prompt based on model type
            if model_name == 'Llama3-8B-Instruct':
                i['question'] = get_llama_without_answer(i['question'])
                i['answer'] = i['answer'] + '<|eot_id|>'
            elif model_name == 'Qwen2.5-7B-Instruct':
                i['question'] = get_qwen_without_answer(i['question'])
                i['answer'] = i['answer'] + '<|im_end|>'

        self._data = raw[:size] if size else raw

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

