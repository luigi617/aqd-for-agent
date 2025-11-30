python -m scripts.train.KTO_train \
--train_file /home/yunanlu/code/aqd-for-agent/data/train_dataset.json \
--eval_file /home/yunanlu/code/aqd-for-agent/data/eval_dataset.json \
--test_file /home/yunanlu/code/aqd-for-agent/data/test_dataset.json \
--model_name_or_path Qwen/Qwen3-8B \
--learning_rate 5e-7 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 2 \
--do_train \
--evaluation_models 'gpt-5-mini' 'gemini-2.5-flash' 'deepseek-reasoner'

# python -m scripts.train.KTO_train \
# --train_file /home/yunanlu/code/aqd-for-agent/data/train_dataset.json \
# --eval_file /home/yunanlu/code/aqd-for-agent/data/eval_dataset.json \
# --test_file /home/yunanlu/code/aqd-for-agent/data/test_dataset.json \
# --model_name_or_path /home/yunanlu/code/aqd-for-agent/results/kto/checkpoint-153 \
# --learning_rate 5e-7 \
# --per_device_train_batch_size 2 \
# --per_device_eval_batch_size 2 \
# --do_train False \
# --do_eval True \
# --evaluation_models 'gpt-5-mini' 'gemini-2.5-flash' 'deepseek-reasoner'
