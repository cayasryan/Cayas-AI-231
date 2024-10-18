lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct \
    --tasks minerva_math_algebra,minerva_math_counting_and_prob,minerva_math_geometry,minerva_math_intermediate_algebra,minerva_math_num_theory,minerva_math_prealgebra,minerva_math_precalc \
    --num_fewshot 4 \
    --device cuda:0 \
    --batch_size auto:4 \
    --log_samples \
    --seed 42 \
    --output results