lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct \
    --tasks hendrycks_math \
    --device cuda:0 \
    --batch_size auto:4 \
    --log_samples \
    --output results