#### Main command to start ACE method by using gpt-5-mini as generation model.
python3 -m src.main run \
    --task "ace" \
    --generate_model 'gpt-5-mini' \
    --target_model 'qwen3:8b' \
    --evaluation_models 'gpt-5-mini' \
    --preferred_definition 'input_data/preferred_definition.json' \
    --unpreferred_definition 'input_data/unpreferred_definition.json' \
    --filtering_policy 'input_data/realism_definition.txt' \
    --init_prompt 'input_data/init_prompt.json' \
    --products '<your domain knowledge>' \
    --output_path 'output_data/gpt5/ace/results.json' \
    --n_prompt_refinement_iterations 30 \
    --n_queries 100 \
    --n_threads 64 \
    --budget 1000 \
    --seed 617 \
