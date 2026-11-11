'''
get_input_json.py - Generate the `input.json` file for DPA4 MLFF finetuning and training. Only tested with DPA4-Neo-OMat24-v20260805.
'''

import argparse, json

input_dict = {
  "model": {
    "type": "SeZM",
    "type_map": ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", 
"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", 
"Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", 
"Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", 
"Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", 
"Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    ],
    "descriptor": {
      "sel": 416,
      "rcut": 6.0,
      "channels": 32,
      "n_radial": 16,
      "edge_norm": False,
      "use_env_seed": True,
      "lmax": 3,
      "mmax": 1,
      "n_blocks": 2,
      "mixing_layers": 3,
      "radial_so2_mode": "degree_channel",
      "radial_so2_rank": 1,
      "n_focus": 2,
      "focus_dim": 0,
      "n_atten_head": 1,
      "message_node_so3": True,
      "ffn_neurons": 0,
      "ffn_so3_grid": True,
      "grid_mlp": False,
      "grid_branch": [
        0,
        0,
        1
      ],
      "ffn_blocks": 1,
      "so3_readout": "mlp",
      "use_amp": True,
      "precision": "float32",
      "seed": 42
    },
    "fitting_net": {
      "neuron": [
        0
      ],
      "precision": "float32",
      "seed": 42
    },
    "use_compile": True,
    "enable_tf32": True,
    "_comment": "that's all"
  },
  "learning_rate": {
    "type": "cosine",
    "start_lr": 0.0008,
    "stop_lr": 1e-06,
    "warmup_ratio": 0.003,
    "warmup_start_factor": 0.2
  },
  "loss": {
    "type": "ener",
    "loss_func": "mae",
    "f_use_norm": True,
    "start_pref_e": 20,
    "limit_pref_e": 20,
    "start_pref_f": 20,
    "limit_pref_f": 20,
    "start_pref_v": 5,
    "limit_pref_v": 5
  },
  "optimizer": {
    "type": "HybridMuon",
    "weight_decay": 0.001
  },
  "training": {
    "training_data": {
      "systems": "/path/to/training/data",
      "batch_size": "filter:5000",
      "_comment": "that's all"
    },
    "validation_data": {
      "systems": "/path/to/validation/data",
      "batch_size": 1,
      "numb_batch": 1
    },
    "num_epochs": 8,
    "gradient_max_norm": 5,
    "save_freq": 2000,
    "ckpt_keep_ratio": 0.33,
    "enable_ema": True,
    "ema_decay": 0.999,
    "ema_ckpt_keep": 3,
    "disp_file": "lcurve.out",
    "disp_freq": 100,
    "disp_avg": True,
    "disp_training": True,
    "time_training": True,
    "tensorboard": False,
    "enable_profiler": False,
    "tensorboard_freq": 1000,
    "tensorboard_log_dir": "tb_log",
    "profiling": False,
    "profiling_file": "timeline.json",
    "zero_stage": 1,
    "seed": 42
  }
}


default_start_lr = 8.0e-4
default_stop_lr = 1.0e-6
parser = argparse.ArgumentParser(description='Generate input.json file for DPA4-Neo-OMat24-v20260805.pt.')

parser.add_argument("--training_data_path", type=str, required=True, help="Path to training dataset in hdf5 format. Separate multiple files with commas.")
parser.add_argument("--training_natoms", type=int, required=True, help="The mean number of atoms in a training batch.")
parser.add_argument("--nepochs", type=int, default=50, help="Number of training epochs. Default 50.")
parser.add_argument("--start_lr", type=float, default=default_start_lr, help=f"Start of learning rate. Default {default_start_lr}. Not recommended to modify.")
parser.add_argument("--stop_lr", type=float, default=default_stop_lr, help=f"Stop of learning rate. Default {default_stop_lr}. Not recommended to modify.")
parser.add_argument("--val_data_path", type=str, help="Path to validation dataset in hdf5 format. Separate multiple files with commas.")
parser.add_argument("--val_batch_size", type=int, default=4, help="The batch size of validation. Default 4.")
parser.add_argument("--val_nbatch", type=int, default=1, help="The number of batch that is validated in a step. Default 1.")
args = parser.parse_args()

training_data_path = args.training_data_path
training_natoms    = args.training_natoms
nepochs            = args.nepochs
start_lr           = args.start_lr
stop_lr            = args.stop_lr
val_data_path      = args.val_data_path
val_batch_size     = args.val_batch_size
val_nbatch         = args.val_nbatch


if ',' in training_data_path:
    training_data_path = training_data_path.split(",")
    training_data_path = [item.strip() for item in training_data_path]

input_dict["training"]["training_data"]["systems"] = training_data_path
input_dict["training"]["training_data"]["batch_size"] = f"filter:{training_natoms}" 

if val_data_path is None:
    del input_dict["training"]["validation_data"]
else:
    if ',' in val_data_path:
        val_data_path = val_data_path.split(",")
        val_data_path = [item.strip() for item in val_data_path]
    input_dict["training"]["validation_data"]["systems"] = val_data_path
    input_dict["training"]["validation_data"]["batch_size"] = val_batch_size
    input_dict["training"]["validation_data"]["numb_batch"] = val_nbatch

if abs(start_lr - default_start_lr) > 1.0e-9:
    print("WARNNING: You are change the start learning rate.")
if abs(stop_lr - default_stop_lr) > 1.0e-9:
    print("WARNNING: You are change the stop learning rate.")
    
input_dict["learning_rate"]["start_lr"] = start_lr
input_dict["learning_rate"]["stop_lr"] = stop_lr
input_dict["training"]["num_epochs"] = nepochs

with open("input.json", mode="w") as f:
    json.dump(input_dict, f, indent=4)

print("input.json is generated.")