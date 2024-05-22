accelerator="gpu"
devices=-1
num_nodes = 1
dist_n=16
seed = 1

resume=False
resume_ckpt=None
if not (resume_ckpt or "None.ckpt").endswith(".ckpt"):
    import glob
    import os
    import re
    subfolders = glob.glob(f'{resume_ckpt}/version_*')
    version_numbers = [int(re.search(r'version_(\d+)', folder).group(1)) for folder in subfolders]
    version_numbers.sort(reverse=True)
    for version in version_numbers:
        checkpoint_folder = f'runs/BB/version_{version}/checkpoints'
        if os.path.exists(checkpoint_folder) and glob.glob(checkpoint_folder + '/*.ckpt'):
            resume_ckpt = glob.glob(checkpoint_folder + '/*.ckpt')[0]
            break
    del subfolders, version_numbers, version, checkpoint_folder, os, re, glob

batch_size = 256
batch_size_z = 20
batch_size_gram = 64
batch_size_train = 64
batch_size_val = 128

number_epochs = None
query_budget = 100 # Million
epoch_itrs = 32
g_iter = 1
d_iter = 5
lr_V = 1e-3
lr_S = 1e-3 # old - 1e-2
lr_G = 1e-4 # old - 1e-4
nz = 100  # or 2000 depending on the query generator
n_hyper_class=20
frames_per_video=16
log_interval = 10
log_every_epoch = 1
val_every_epoch = 5
val_topk_accuracy = (1, 5, 10)

victim_train_loss = 'cross_entropy'
loss = 'l1'
gen_loss = 'l1'
scheduler = 'multistep'
steps= [0.1, 0.3, 0.5]
scale = 3e-1
weight_decay = 5e-4
momentum = 0.9

log_dir = 'results'
grad_m = 4
grad_epsilon = 1e-3
forward_differences = 1
no_logits = 1
logit_correction = 'mean'
rec_grad_norm = 1
clone_optimizer = "sdg"
gen_optimizer = "adam"

## These are for training the victim to have weights for the target dataset
victim_train_optimizer = "adam"
victim_train_momentum = 0.9
victim_train_weight_decay = 5e-4
victim_train_scheduler = "multistep"
victim_train_steps = [0.1, 0.3, 0.5]
victim_train_scale = 3e-1

store_checkpoints = 1
