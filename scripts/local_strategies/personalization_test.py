import json

data = json.load(open('/home/yunanlu/code/aqd-for-agent/output_data/gpt5-mini_1126/cleaned_data.json'))
print(data[0])
profiles = json.load(open('/local/data/shared/aqd/input_data/profiles.json'))
print(profiles[0])

from openai import OpenAI
client = OpenAI()
prompt = """You are rewriting a user query to reflect a specific customer persona. 
Original query: 
{question}

Customer persona:
{profile}

Rules:
Rewrite the original query to reflect this customer persona, including any motivations or considerations that fit naturally. The rewritten question should:
- Do not include any identity information of the customer.
- Do not repeat the exact instruction in the persona.
- Try to personalize the queestion to the persona's background, such as job information, household, event or activities you did or going to do.
- Keep the original factual intent and product focus.
- Embed personal, emotional, or contextual details drawn from the persona.  
- Maintain a natural tone and fluency.
- Be locally refined, not entirely reformulated.
- Be concise (1 - 2 sentences).

Answer:
"""

sample_profile = profiles[13]
new_profile = ""
for sentence in sample_profile.split("."):
    if "Sweet for Everyone" not in sentence:
        new_profile += sentence + ". "

print(new_profile)

response = client.chat.completions.create(
    model="gpt-5-mini",
    messages=[{"role": "user", "content": prompt.format(question=data[0]["query"], profile=new_profile)}],
)

print(response.choices[0].message.content)



# generate the response and evaluate the response
from src.aqd.schema import QueryNode
from src.aqd.budget import Budget
from src.aqd.query_refinement.query_refinement import QueryRefinement


target_model = "qwen3:8b"
evaluation_models = ['gpt-5-mini', 'gemini-2.5-flash', 'deepseek-reasoner']
preferred_definition_raw = json.load(open("/local/data/shared/aqd/input_data/preferred_definition.json"))
unpreferred_definition_raw = json.load(open("/local/data/shared/aqd/input_data/unpreferred_definition.json"))
filtering_policy = open("/local/data/shared/aqd/input_data/realism_definition.txt").read()
query_refinement_strategies = None
n_query_refiment_beam_search_size = 10

preferred_definition = {}
unpreferred_definition = {}
preference_score_map = {}
for criterion, info in preferred_definition_raw.items():
    preferred_definition[criterion] = info["description"]
    preference_score_map[criterion] = info["score"]

for criterion, info in unpreferred_definition_raw.items():
    unpreferred_definition[criterion] = info["description"]
    preference_score_map[criterion] = info["score"]

budget = Budget(10)
query_node = QueryNode(query=response.choices[0].message.content, product=data[0]["product_attributes"], attributes=data[0]["selected_attributes"])

query_refinement = QueryRefinement(
    query_node=query_node,
    target_model=target_model,
    filtering_model=target_model,
    refinement_model=target_model,
    evaluation_models=evaluation_models,
    preferred_definition=preferred_definition,
    unpreferred_definition=unpreferred_definition,
    preference_score_map=preference_score_map,
    filtering_policy=filtering_policy,
    query_refinement_strategies=query_refinement_strategies,
    n_query_refiment_beam_search_size=2,
    budget=budget,
)
query_node = query_refinement.generate_response(query_node)
query_node = query_refinement.evaluate_responses(query_node)
query_node.get_score(preference_score_map)

# save query node to json
json.dump(query_node.to_dict(), open('output_data/gpt5-mini_1126/query_node.json', 'w'))