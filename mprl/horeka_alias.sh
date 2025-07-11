alias CD='cd'
alias LS='ls'
alias GIT='git'

# Slurm watch alias
alias wa='watch -n 20 squeue'
alias was='watch -n 20 squeue --start'
alias keep='watch -n 600 squeue'
alias sfree='sinfo_t_idle'
alias sgpu='salloc -p accelerated -n 1 -t 120 --mem=200000 --gres=gpu:1 --account=hk-project-p0022232'
alias sgpu_dev='salloc -p dev_accelerated -n 1 -t 60 --mem=200000 --gres=gpu:1 --account=hk-project-p0022232'
alias scpu='salloc -p cpuonly -n 1 -t 120 --mem=200000 --account=hk-project-p0022232'
alias scpu_dev='salloc -p dev_cpuonly -n 1 -t 120 --mem=200000 --account=hk-project-p0022232'


# cd alias
alias cdresult='cd ~/Documents/Thesis_update/TOP_ERL_new/mprl/result'
alias cdconfig='cd ~/Documents/Thesis_update/TOP_ERL_new/mprl/config'
alias cdmprl='cd ~/Documents/Thesis_update/TOP_ERL_new/mprl'
alias cds='cd ~/projects/seq_rl/mprl_exp_result/slurmlog'
alias dlc='cd ~/projects/seq_rl && rm -r CODE_COPY/'

# Git alias
alias gp='cd ~/projects/seq_rl/SeqRL && git pull'
alias gpl='cd ~/projects/seq_rl/SeqRL && git pull && git log'
alias gpf='cd ~/projects/seq_rl/fancy_gymnasium && git pull'
alias grc='cdmprl && python check_git_repos.py'

# Env alias
alias vb='cd ~/ && vim .bashrc'
alias ss='cd ~/ && source .bashrc && conda activate seq_rl'

# Exp
alias runexp='cdmprl && python mp_exp.py'
alias runexp_seq='cdmprl && python seq_mp_exp.py'
alias runexp_seq_multi_process='cdmprl && python seq_mp_exp_multiprocessing.py'

## TCE BOX RANDOM INIT
alias box_random_tce_entire='runexp ./config/box_push_random_init/tce/entire/horeka.yaml   -o -s'

## BBRL BOX RANDOM INIT
alias box_random_bbrl_entire='runexp ./config/box_push_random_init/bbrl/entire/horeka.yaml   -o -s'

## BBRL Metaworld
alias meta_bbrl_prodmp_entire='runexp ./config/metaworld/bbrl/entire/horeka.yaml   -o -s'

## TCE Metaworld
alias meta_tce_entire='runexp ./config/metaworld/tce/entire/horeka.yaml   -o -s'

## BBRL Table tennis 4d ProDMP
alias tt4d_bbrl_prodmp='runexp ./config/table_tennis_4d/bbrl/entire/horeka.yaml   -o -s'

## TCE Table tennis 4d ProDMP
alias tt4d_tce_prodmp='runexp ./config/table_tennis_4d/tce/entire/horeka.yaml   -o -s'

## TCE Hopper Jump ProDMP
alias hopper_tce='runexp ./config/hopper_jump/tce/entire/horeka.yaml   -o -s'

## SeqRL Box Push Single process
#alias box_random_seq_dense='runexp_seq ./config/box_push_random_init/seq/entire/horeka_dense.yaml   -o -s'
#alias box_random_seq_sparse='runexp_seq ./config/box_push_random_init/seq/entire/horeka_sparse.yaml   -o -s'
#alias box_random_seq_local='runexp_seq ./config/box_push_random_init/seq/entire/local.yaml   -o --nocodecopy'


### SEQ Metaworld Single process
#alias meta_seq_entire='runexp_seq ./config/metaworld/seq/entire/horeka_meta.yaml   -o -s'
#
### SEQ Table Tennis Single process
#alias tt_seq_entire='runexp_seq ./config/table_tennis_4d/seq/entire/horeka_tt.yaml   -o -s'
#
### SEQ Hopper Jump Single process
#alias hopper_seq_entire='runexp_seq ./config/hopper_jump/seq/entire/horeka_hopper.yaml   -o -s'

## SeqRL Box Push Multi processing
alias box_random_seq_dense='runexp_seq_multi_process ./config/box_push_random_init/seq/entire/horeka_dense.yaml   -o -s'
alias box_random_seq_dense_robin='runexp_seq_multi_process ./config/box_push_random_init/seq/entire/slurm_dense.yaml   -o -s'
alias box_random_seq_sparse='runexp_seq_multi_process ./config/box_push_random_init/seq/entire/horeka_sparse.yaml   -o -s'
alias box_random_seq_local='runexp_seq_multi_process ./config/box_push_random_init/seq/entire/local.yaml   -o --nocodecopy'

## SEQ Metaworld Multi processing
alias meta_seq_entire='runexp_seq_multi_process ./config/metaworld/seq/entire/horeka_meta.yaml   -o -s'

## SEQ Table Tennis Multi processing
alias tt_seq_entire='runexp_seq_multi_process ./config/table_tennis_4d/seq/entire/horeka_tt.yaml   -o -s'

## SEQ Hopper Jump Multi processing
alias hopper_seq_entire='runexp_seq_multi_process ./config/hopper_jump/seq/entire/horeka_hopper.yaml   -o -s'