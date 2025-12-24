

# python3 -m src.main resume \
#     --prev_result 'output_data/test/results_depth0.json' \
#     --config_path 'output_data/test/config.json'\
#     --output_path 'output_data/test1/results.json' \
#     --n_queries 1 \
#     --n_prompt_refinement_iterations 1 \
#     --n_prompt_refinement_children 1 \
#     --n_prompt_refinement_beam_search_size 1 \
#     --n_query_refinement_iterations 1 \
#     --n_query_refiment_beam_search_size 1 \
#     --n_threads 50 \
#     --budget 1000 \
#     --seed 617 \
    

# python3 -m src.main run \
#     --generate_model 'gpt-5-mini' \
#     --target_model 'qwen3:8b' \
#     --evaluation_models 'gpt-5-mini' 'gemini-2.5-flash' 'deepseek-reasoner' \
#     --preferred_definition 'input_data/preferred_definition.json' \
#     --unpreferred_definition 'input_data/unpreferred_definition.json' \
#     --filtering_policy 'input_data/realism_definition.txt' \
#     --query_refinement_strategies 'input_data/query_refinement_strategies.json' \
#     --init_prompt 'input_data/init_prompt.json' \
#     --products 'input_data/walmart_400_products.json' \
#     --output_path 'output_data/lucy/results.json' \
#     --n_queries 10 \
#     --n_prompt_refinement_iterations 4 \
#     --n_prompt_refinement_children 2 \
#     --n_prompt_refinement_beam_search_size 1 \
#     --n_query_refinement_iterations 2 \
#     --n_query_refiment_beam_search_size 3 \
#     --n_threads 64 \
#     --budget 1000 \
#     --seed 617 \

    

