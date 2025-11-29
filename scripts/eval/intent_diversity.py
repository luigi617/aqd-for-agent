from pydantic import Field, BaseModel
from typing import List
from src.models.model import get_model
from tqdm import tqdm
INSTRUCTION_INTENT = """
Intent: Instruction
Definition: The customer wants instructions, guidelines, or procedures to achieve something with respect to a product or service.
Example:
Where is the doorknob? Once the code is entered, how do you open the door?
How can I tell if this will work on my TV or BluRay player?
"""

OPINION_INTENT = """
Intent: Opinion
Definition: The customer wants a subjective piece of information about a product, service, or shopping category.
Example:
Is this product worth buying or will I end up sending it back?
Any defect complaints?
"""

DESCRIPTION_INTENT = """
Intent: Description
Definition: The customer wants a definition, description, explanation, or summary of a product, service, or shopping category.
Example:
What are the sizes and types of blades that come with the 5-blade package?
What are the measurements of this product?
"""

COMPARISON_INTENT = """
Intent: Comparison
Definition: The customer wants a comparison of two or more products or services
Example:
What's the difference between exclusive Castiel and regular?
How does this knife compare with a Kershaw?
"""

RECOMMENDATION_INTENT = """
Intent: Recommendation
Definition: The customer wants recommendations for a product or service.
Example:
Looking for a bag for golf for drinks and snacks. Would this be a good choice?
I need a case for a .357 with a sixinch barrel. Suggestions?
"""

FACTOID_INTENT = """
Intent: Factoid
Definition: The customer wants an objective piece of information about a product, service, or shopping category.
Example:
Is the price and shipping for one bar or a set of two? 
Does it fit on Honda CRV 2014?
"""


INTENTS_PROMPT = """

You are a helpful assistant that can help classify the intent of a query based on the taxonomy of intents. Notice that this is a multi-label classification task which means a query can belong to multiple intents. Please return the result in the JSON format.

Taxonomy of intents:
{intent_taxonomy}

Example:
########
Question: Does this have ratings for comfort and durability?
Answer:
[
    {{
        "reasoning": "The query is asking about the product ratings, which belongs to the objective information category. So it is a factoid intent.",
        "intent": "Factoid"
    }},
    {{
        "reasoning": "The query is asking about the comfort, which is a subjective opinion, so it needs assistant to summarize the information from the product description or customer reviews. So it is an opinion intent.",
        "intent": "Opinion"
    }}

]

########
Question: {question}
Answer:
"""

class IntentSchema(BaseModel):
    reasoning: str = Field(description="Reasoning for the intent classification.")
    intent: str = Field(description="The intent of the query.")

class IntentsSchema(BaseModel):
    intents: List[IntentSchema] = Field(description="The intents of the query.")

def classify_intent(question: str, model_path: str) -> List[IntentSchema]:

    intent_taxonomy = "\n\n".join([INSTRUCTION_INTENT, OPINION_INTENT, DESCRIPTION_INTENT, COMPARISON_INTENT, RECOMMENDATION_INTENT, FACTOID_INTENT])

    eval_intents_chain = get_model(model_path, IntentsSchema, sensitive=False)

    # get response
    response = eval_intents_chain(
        INTENTS_PROMPT.format(intent_taxonomy=intent_taxonomy, question=question)
    )
    return response["intents"]

def evaluate_intent_diversity(datasets: List[dict], model_path: str) -> List[dict]:

    intents_taxonomy = {
        "instruction": 0,
        "opinion": 0,
        "description": 0,
        "comparison": 0,
        "recommendation": 0,
        "factoid": 0
    }
    for data in tqdm(datasets, desc="Classifying intents", total=len(datasets)):
        try:
            question = data["query"]
            intents = classify_intent(question, model_path)
            data["intents"] = intents
            for intent in intents:
                if intent['intent'].lower().strip() in intents_taxonomy:
                    intents_taxonomy[intent['intent'].lower().strip()] += 1
                else:
                    print(f"Unknown intent: {intent['intent']}")
        except Exception as e:
            print(f"Error classifying intent for question: {data['query']}")
            print(e)
    return datasets, intents_taxonomy