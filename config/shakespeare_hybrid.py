# [hybrid-attn] Small char-level Shakespeare model sized for a 4 GB laptop GPU (RTX 2050).
# Derived from config/train_shakespeare_char.py. Choose the variant from the command line:
#   python train.py config/shakespeare_hybrid.py --attn=mla_swa --seed=0
#
# Per token per layer the KV cache holds:  mha/swa 2*192 = 384 numbers,  mla/mla_swa 64+16 = 80.
# W = 64 is a quarter of the 256-token context, so the window genuinely binds during training.

out_dir = 'out/shakespeare'
eval_interval = 250
eval_iters = 50
log_interval = 100
always_save_checkpoint = False

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256

n_layer = 4
n_head = 4
n_embd = 192
dropout = 0.1

attn = 'mha'
pos_emb = 'rope'      # all four variants use RoPE so positions are not a confounder
window = 64
kv_latent = 64
q_latent = 96
rope_dim = 16

learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100

compile = False       # torch.compile needs Triton, unavailable on native Windows
