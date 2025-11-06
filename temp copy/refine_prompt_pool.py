from collections import defaultdict
from pipeline.misc.graph_common import FILTERED, GraphState
from pipeline.misc.dcscore import DCScore


def refine_prompt_pool(
    state: GraphState
) -> GraphState:
    print('Refining prompts...')

    # Update prompt pool
    dcscore_evaluator = DCScore('google-bert/bert-base-uncased')
    batch_size = 128
    tau = 1
    kernel_type = 'cs'
    prompt = state['prompt_pool'][-1]

    # if prompt was just added, skip
    prompt_state = state.get('prompts', {}).get(prompt, {})
    reflections = prompt_state.get('reflections', [])

    # pick new prompt to add to pool based on diversity

    new_prompts = [new_instruction for (_why, new_instruction) in reflections]
    dcscores = []
    for new_indiv_prompt in new_prompts:
        text_list = state['prompt_pool'] + [new_indiv_prompt]
        embeddings, _, _ = dcscore_evaluator.get_embedding(text_list, batch_size=batch_size)
        dataset_dcscore = dcscore_evaluator.calculate_dcscore_by_embedding(
            embeddings, kernel_type=kernel_type, tau=tau
        )
        dcscores.append(dataset_dcscore)
    most_diverse_i = dcscores.index(max(dcscores))

    state['prompt_pool'].append(new_prompts[most_diverse_i])

    
    # Print information about current iteration since this is last node 
    state['iterations_passed'] += 1
    print(f'Finished iteration {state["iterations_passed"]}.')
    return state