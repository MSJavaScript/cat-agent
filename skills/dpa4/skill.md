---
name: dpa4
description: Skill of dpa4 machine learning force field. Including how to finetune the DPA4 force field, and how to utilize the DPA4 force field through LAMMPS and ASE.
metadata:
  tags:
    - deepmd
    - dpa
    - finetuning
---
# DPA4 Skill 

DPA4 is the newest version of deepmd machine learning force field (MLFF), this document contains instructions of how to finetune a DPA4 MLFF, and how to use it in ASE and LAMMPS.

NOTE: DPA4 is currently undergoing significant changes. It relies on a specific computing environment, and the checkpoint files and input files (*.json) must strictly correspond in version.The DPA4 MLFF in this skill is implemented in deepmd-kit 3.1.2, run normally on NVIDIA 4090 or higher-end GPUs. The checkpoint file version of the pre-trained model is `DPA4-Neo-OMat24-v20260805.pt`. If you use another version of deepmd-kit or checkpoint file, please make sure that they are compatible with each other.

## Input and output

In deepmd-kit, `dp` is the main program to finetune and test DPA MLFF. Below are the input and output files of `dp`.

|File name | Purpose|
|---- | ---- |
|*.hdf5| Dataset in HDF5 format |
| *.json  | Input file for finetuning and training|
| *.pt | Checkpoint file. The output `*.pt` files are named by `model.ckpt-{step}.pt`, `step` is the number of training step. The final `*.pt` file is named by `model.ckpt.pt`|
| lcurve.out| The data of learning curve|
|*.pt2| The frozen model file for inference|
|*.e.out, *.e_peratom.out| Evaluation result of energy |
|*.f.out|Evaluation result of force|
|*.v.out, *.v_peratom.out|Evaluation result of viral |

The recommended format of dataset for `dp` is HDF5. Use `scripts/extxyz_to_hdf5.py` to convert `*.extxyz` file into HDF5 format.
```bash
python3 extxyz_to_hdf5.py <path_to_extxyz> <path_to_hdf5>
```

`dp` needs a `*.json` file which corresponds to a specific version of checkpoint as a input. The json file contains the architecture of the model, training parameters such as number of epochs, batch size and learning rate, and the pathes to training and validation dataset. Use `scripts/get_input_json.py` to generate a `input.json` file. 
```bash
python3 get_input_json.py --training_data_path "data1.hdf5, data2.hdf5" --training_natoms 320
```
Use `-h` option to get the help of this script. There are several parameters require further explanation.
- training_natoms: The mean number of atoms in a training batch. Select an appropriate `training_natoms` to make the number of steps in an epoch in the range of [50, 75]. Here is a quick formula to calculate it: 

    `n_steps_per_epoch = floor(n_structures*mean_atoms/training_natoms)`

    Where `n_structures` and `mean_natoms` are the the total number of structures and mean number of atoms in a structure respectively in a hdf5 dataset. Both of them can be calculated by `script/mean_atoms.py`.

    ```bash
    python3 mean_atoms.py --hdf5 data.hdf5
    ```
- nepochs: Number of training epochs, 50 is a good choice for most cases.
- start_lr and stop_lr: The start and stop learning rate. They are not recommended to modify.

The pre-trained model of DPA is in `*.pt` format, `dp` also generates `*.pt` checkpoint files during training. However, to use the DPA4 MLFF in ASE or LAMMPS, you should freeze it into `*.pt2` format (see below).

In test or evaluation stage, `dp` will generate a series of `*.out` files which contain original data and predicted data, which can directly read by numpy.

## Finetuing and evaluation

Finetuing is a fast method to get accurate MLFF from a pre-trained model. To finetune on a downstream dataset and evaluate a checkpoint on a dataset, you can use:

```bash
dp --pt train input.json --finetune ${model} > train_log 2>&1
dp --pt freeze -c model.ckpt.pt -o frozen
dp --pt test -m frozen.pt2 -s structs.hdf5 -d pred_val > infer_val_log 2>&1
```

1. Finetuing
    ```bash
    dp --pt train input_finetune.json --finetune DPA4-<Variant>.pt > train_log 2>&1
    ```
    This command finetunes a `*.pt` model with a `*.json` input. The pathes of datasets are specified in the json input.
2. Freeze to .pt2 for deployment
    ```bash
    dp --pt freeze -c model.ckpt.pt -o frozen_model
    ```
    This command will generate a `frozen_model.pt2` file. It depends on the host CPU/GPU, the GPU compute capability, and the libtorch version.
3. Evaluate a checkpoint
    ```bash
    dp --pt test -m DPA4-<Variant>.pt -s /path/to/dataset -d file_prefix > infer_val_log 2>&1
    ```
    This command evaluates a checkpoint (`*.pt` or `*.pt2`) on a dataset. A series of `<file_prefix>.out` files will be generated, such as `*.e.out`, `*.f.out`, etc. You can use them to plot the scatter plots between MLFF predicted values and DFT values. `infer_val_log` contains the RMSE of energy, force and viral, you can read them by `tail -n 15 infer_val_log`.

These three steps are usually done together. If you only evaluate a pre-trained model, only step 3 is needed.

## Use DPA4 in ASE and LAMMPS

DPA4 model should be frozen to `*.pt2` format before it is used in ase and lammps.

### ASE

An example of using DPA4 to run NEB.
```python
from ase.io import read, write
from deepmd.calculator import DP as DPCalculator
from ase.optimize import FIRE
from ase.mep import NEB

pt_file = "frozen_model.pt2"
images = read("initial_path.extxyz", index=":")
for image in images:
    image.calc = DPCalculator(pt_file)
neb = NEB(images, k=0.1, method='improvedtangent')
optimizer = FIRE(neb, 
                 maxstep=0.1, 
                 trajectory="neb.traj", 
                 logfile="neb.log")
optimizer.run(fmax=0.1, steps=50)
write("result.extxyz", neb.images, format="extxyz")
```

### LAMMPS
The frozen `.pt2` is used through `pair_style deepmd`. A minimal setup:
```bash
units           metal
atom_style      atomic
atom_modify     map yes

neighbor        2.0 bin
read_data       system.lmp

pair_style      deepmd frozen_model.pt2
pair_coeff      * * O H
```
`atom_modify map yes` is required: `.pt2` graph inference relies on an explicit ghost/periodic-image to local-atom map, and the model fails fast if the map is absent. The element names after `pair_coeff * *` bind LAMMPS atom types to entries of the model's `type_map` in order (here types 1 and 2 to O and H); if omitted, the mapping falls back to the type_map stored in the .pt2 metadata.

Multi-GPU (MPI) inference uses the same .pt2. Launch one MPI rank per GPU and make every target device visible:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np 4 lmp -in in.lammps
```

## Scripts

Below are the related scripts in `scripts` dir. Run these scripts without parameters or with `-h` to get the helps.

| Name | Purpose|
| ---- | ---- |
| extxyz_to_hdf5.py |Convert `extxyz` to `hdf5` format.|
| get_input_json.py | Get `input.json` for `dp`, only compatible with checkpoint of version DPA4-Neo-OMat24-v20260805.pt|
|mean_natoms.py|Print the mean number of atoms of structures in a hdf5 dataset, also the total number of structures.|

## References

Below are reference files in `references` dir.
| Name | Purpose |
| --- | --- |
| job_submission.md| File template for job submission, only for StepNode |