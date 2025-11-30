

python3 src/main.py \
    --generate_model 'gpt-5-mini' \
    --target_model 'qwen3:8b' \
    --evaluation_models 'gpt-5-mini' 'gemini-2.5-flash' 'deepseek-reasoner' \
    --preferred_definition '/local/data/shared/aqd/input_data/preferred_definition.json' \
    --unpreferred_definition '/local/data/shared/aqd/input_data/unpreferred_definition.json' \
    --filtering_policy '/local/data/shared/aqd/input_data/realism_definition.txt' \
    --query_refinement_strategies '/local/data/shared/aqd/input_data/query_refinement_strategies.json' \
    --input_pool '/local/data/shared/aqd/input_data/seed_prompts.json' \
    --products '/local/data/shared/aqd/input_data/walmart_products.json' \
    --output_path 'output_data/gpt5-mini/results.json' \
    --n_queries 10 \
    --n_prompt_refinemnet_iterations 4 \
    --n_prompt_refinemnet_children 2 \
    --n_prompt_refinemnet_beam_search_size 1 \
    --n_query_refinemnet_iterations 2 \
    --n_query_refiment_beam_search_size 3 \
    --n_threads 50 \
    --budget 10000 \
    --seed 617 \
    

