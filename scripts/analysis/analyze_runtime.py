from pathlib import Path
import json

budget_folder = Path('input_data/budget_data/gemma')
budget_output_file= Path('output_data/analysis/gemma_time.json')
from collections import defaultdict

MODEL_TO_SPEED = {
    'gemma3:27b': 36, # google genai provider, https://artificialanalysis.ai/models/gemma-3-27b/providers#speed 
    # gpt-5-mini has medium reasoning effort by default
    'gpt-5-mini': 119, # openai provider, https://artificialanalysis.ai/models/gpt-5-mini-medium/providers?speed=output-speed#speed
    # sep 2025 version
    'gemini-2.5-flash': 297, # google genai provider, https://artificialanalysis.ai/models/gemini-2-5-flash-preview-09-2025-reasoning#speed
    # deepseek-reasoner is equivalent to deepseekv3.2 as of Dec 2025
    'deepseek-reasoner': 49 # deepseek provider https://artificialanalysis.ai/models/deepseek-v3-2-reasoning/providers?speed=output-speed#speed
}

# time to first token, obtained from links above
MODEL_TO_LATENCY = {
    'gemma3:27b': 0.73,
    'gpt-5-mini': 19.8,
    'gemini-2.5-flash': 12.08,
    'deepseek-reasoner': 1.11
}

output_data = defaultdict(dict)

for file in budget_folder.iterdir():
    if file.name.endswith('budget.json'):
        
        with open(file) as f:
            data = json.load(f)

        time = 0
        for entry in data['history']:
            desc_str = (entry.get('budget_extra_info') or {}).get('desc', "")
            # ace uses gemma:27b instead of qwen3:8b, so check desc_str for ace
            if entry['model'] != 'qwen3:8b' and not desc_str.startswith('Generating response for query'):
                time += entry['completion_tokens']/MODEL_TO_SPEED[entry['model']]+MODEL_TO_LATENCY[entry['model']]

        method = file.stem.split('_budget')[0]
        output_data[method]['total time (s)'] = time
    
    if file.name.endswith('statistics.json'):

        with open(file) as f:
            data = json.load(f)

        method = file.stem.split('_statistics')[0]
        output_data[method]['query_count'] = data['total_query_count']

for method, stats in output_data.items():
    stats['average time per query (s)'] = stats['total time (s)'] / stats['query_count']
    del stats['query_count']
    
budget_output_file.parent.mkdir(parents=True, exist_ok=True)
with open(budget_output_file, 'w') as f:
    json.dump(output_data, f, indent=4)