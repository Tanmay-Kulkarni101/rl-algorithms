python ../src/vanilla-policy-gradient/vanilla-policy-gradient.py \
--env CartPole-v0 \
--hidden_sizes 64 64 \
--activations tanh tanh tanh \
--gamma 0.99 \
--seed 0 \
--steps 4000 \
--epochs 50 \
--pi_lr 3e-4 \
--v_lr 1e-3 \
--value_func_iters 80 \
--lamb 0.97 \
--max_episode_length 1000