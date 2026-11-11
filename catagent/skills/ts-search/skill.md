---
name: ts-search
description: Instructions of performing transition state search by VASP and ASE.
metadata:
  tags:
    - neb
    - dimer
    - ase
    - vasp
    - transition state search 
    - machine-learning-potential
---

# Transition state search skill

Transition state (TS) is the first order saddle point on the potential energy surface (PES) which connects two local minimums, initial and final states (IS and FS). The correct transition state structure should have only one imaginary vibration mode along the reaction coordinate. There are mainly two methods to locate the TS on the PES, PES interpolation and PES walking. PES interpolation such as NEB builds an initial minimum energy path (MEP) between IS and FS, then optimizes this path to find the real MEP, the highest points of MEP is the TS. PES walking methods such as Dimer needs an initial guess of TS, optimize this structure while keeping curvature along reaction coordinate negative. This document explains the basic steps for transition state optimization using VASP and ASE.

NOTE: To perform NEB or Dimer calculation, VASP should be compiled with VTST. To check if VTST is installed, run 
```
grep VTST OUTCAR
```
to see if there is some output like
```
VTST: version 4.1, (05/19/21)
```
VASP has NEB and Dimer function by itself, but this skill may be not compatible with them.



## NEB

Follow the steps below to run a NEB calculation:

1. Build IS and FS, their atomic indices should be matched.
2. Perform structure relaxation on IS and FS.
3. Use `script/neb_make.py` to interpolate between relaxed IS and FS and get an initial guess of NEB path. 
    ```python
    python3 neb_make.py --ini POSCAR_ini --fin POSCAR_fin --nimages 5 # or --spacing 0.5
    ```
    This will generate a series of `0x` dirs for VASP NEB calculation, you can also use `--extxyz` option to save NEB path in an `extxyz` file. Use `-h` to get all the help messages.

    For VASP NEB, a balance must be considered between computational efficiency and the convergence of NEB. For good convergence, the spacing between images should generally be within 0.5, but too many images can significantly increase computational workload. Considering various factors, the number of images should generally not exceed 8. If you use ASE with force field, the number of images can be large as force field is very fast.

    
4. Relax the NEB path:

    For VASP, `INCAR`, `POSCAR` and `KPOINTS` needs to be placed in the same dir as the `0x` dirs, the dir structure should be like:
    ```bash
    00 01 02 03 04 05 INCAR POSCAR KPOINTS
    ```
    INCAR tags of NEB should be set (see below).

    For ase, you need a calculator, we usually use a machine learning force field (MLFF). 
    ```python
    from ase.io import read, write
    from deepmd.calculator import DP as DPCalculator
    from ase.optimize import FIRE
    from ase.mep import NEB

    pt_file = "frozen_model.pt2"
    images = read("initial_path.extxyz", index=":")
    for image in images:
        image.calc = DPCalculator(pt_file)
    neb = NEB(images, k=0.1, method='improvedtangent', climb=True)
    optimizer = FIRE(neb, 
                    maxstep=0.1, 
                    trajectory="neb.traj", 
                    logfile="neb.log")
    optimizer.run(fmax=0.1, steps=50)
    write("result.extxyz", neb.images, format="extxyz")
    ```
    `run_ase_neb_with_dpa.py` in cat-mlff skill is a convenient script which runs neb by DPA force field. See the cat-mlff skill for more details.

## Parameter setting of NEB

NEB is usually difficult to converge, it's important to select a good optimization algorithm.

### VASP

The tags below should always be set in INCAR for NEB calculation:
```
ICHAIN = 0
IMAGES = #The number of intermediate images
SPRING = -5
LCLIMB = .TRUE.
```
Three optimizers are frequently used:

#### IOPT = 7 (FIRE)
Set the tags below in INCAR:
```text
IBRION = 3
POTIM = 0.0
IOPT = 7
MAXMOVE = 0.2
```
FIRE is a robust but slow VTST optimizer, performs well for complex systems and when the initial guess of NEB path is bad.

#### IOPT = 1 (LBFGS)
Set the tags below in INCAR:
```text
IBRION = 3
POTIM = 0.0
IOPT = 1
MAXMOVE = 0.2
```
LBFGS is a fast but aggressive VTST optimizer, it may fail in some cases. Reduce `MAXMOVE` to 0.1 for complex reactions.

#### IBRION = 1 (RMM-DIIS)
Set the tags below in INCAR:

```text
IOPT = 0
IBRION = 1
POTIM = 0.15
NFREE = 8
```
RMM-DIIS is suitable for simple systems and situations where the initial configuration is close to the energy minimum. TIPS: You can also use a small POTIM and run NEB just a few steps to preconverge NEB, for example
```text
POTIM = 0.075
NSW = 30
```
This makes NEB path relax to a more reasonable position, then you can increase the step size or switch to another optimizer.

### ASE

For `ase.mep.NEB`, always use `k=0.1` and `method='improvedtangent'`. `ase.optimize.FIRE` is  recommended for most cases, `ase.optimize.MDMin` also performs well. You can also *preconverge* NEB by ASE, run neb a few steps with small step size and then increase the step size:
```python
# Firstly, preconverge the NEB path
neb = NEB(images, k=0.1, method='improvedtangent')
optimizer = FIRE(neb, maxstep=0.05, trajectory=f"{name}_prev.traj", logfile=f"{name}_prev.log")
optimizer.run(fmax=0.1, steps=50)

# Then increase the step size. Add `climb = True` here to use climbing image NEB.
neb2 = NEB(neb.images, k=0.1, method='improvedtangent', climb=True)
optimizer2 = FIRE(neb2, maxstep=0.1, trajectory=f"{name}_neb.traj", logfile=f"{name}_neb.log")
optimizer2.run(fmax=0.07, steps=500)
```

## Practical wisdom for VASP

### The complexity of the system
If a NEB calculation satisfies the following 3 conditions, we consider it simple. For simple systems, use `IBRION=1`. 
- The number of intermediate images are small, e.g. <=4.
- The surface of the catalyst is flat, for example, fcc(111), hcp(0001).
- The number of adsorbate atoms are small, e.g. <= 5. Examples are CH4 + 2* <-> CH3* + H*, CO2 + 2* <-> CO* + H*, * is the adsorption site.

If the distance between IS and FS is large, but due to computational cost, you cannot insert enough images such that the spacing between them is less than 0.5, then you will need preconverge technique. 

### Select new IS and FS, Re-interpolation

For complex systems, due to the presence of many local minima, the NEB path is prone to distortion during the optimization process, making it impossible for the force to decrease. In this case, find the image with the highest energy on the NEB path (set as i), and use i-1 and i+1 as the new IS and FS for interpolation, even if i-1 and i+1 are not at energy minima. By re-interpolation, the spacing between images will be much smaller and NEB converges more easily.

## Dimer (for VASP only)

Dimer needs a initial guess of TS and a `MODECAR` which contains the vibration mode of TS along reaction coordinate. These inputs are usually generated from the results of a NEB calculation. Use `scripts/neb2dim.py` to do this:
```bash
# cd path_to_neb_results
python3 neb2dim.py -o dim_input
```
This will generate a `dim_input` dir containing all the inputs of a dimer calculation.

The computational cost of dimer is more lower than NEB, but it also fails easily because there are no constraints of NEB path, the dimer may drift away from the real TS. Use dimer only if the NEB path is already close to MEP, e.g. the force along NEB path is smaller than 0.5 eV/A. Use `scripts/nebef.py` to get the max force on the images.
```bash
# cd path_to_neb_results
python3 nebef.py
```
The output contains 4 columns: index of image, max force, DFT energy and relative energy.

## Frequency Analysis

Calculating vibration free energy and validating if the TS is correct requires frequency analysis to obtain all vibration modes of the adsorbate. Firstly, fix the atomic positions of slab atoms by `script/fix_slab_atoms.py`.
```bash
python3 fix_slab_atoms.py --in POSCAR --out POSCAR_fixed --relax_ind "30,32,35"
```
Option `--relax_ind` sets the atomic indices of adsorbate or the atoms are allowed to relax during frequency analysis. Use `-h` option to see complete help messages.

Following tags are required in INCAR
```
IBRION = 5
POTIM = 0.015
NFREE = 2
```
This will use central difference method to get all the vibration modes of relaxed atoms.

`scripts/get_virb_mode.py` extracts a specific vibration mode from OUTCAR and save it to a `mode_i` file (`i` is the index of mode, start from 0).
```bash
# By default, extract the imaginary mode with the highest energy. This can be used as MODECAR for dimer.
python3 get_virb_mode.py --outcar OUTCAR

# Only print wave numbers
python3 get_virb_mode.py --outcar OUTCAR -p

# extract mode 12
python3 get_virb_mode.py --outcar OUTCAR -e 12
```

`scripts/get_virb_free_energy.py` calculates the vibration free energy:
```bash
python3 get_virb_free_energy.py --outcar OUTCAR -t 300

# Use --isotope option to set the atomic mass of isotope
# For each tuple, first element is the atom index
# second element is the atomic mass
python3 get_virb_free_energy.py --outcar OUTCAR -t 300 --isotope "(20,2.0), (21, 2.0)"
```

## Scripts

Below are the related scripts in `scripts` dir. Run these scripts without parameters or with `-h` to get the helps.

| Name | Purpose |
| ---- | ---- |
| neb2dim.py | Convert the results of NEB to the inputs of Dimer calculation |
|neb_make.py | Interpolate between IS and FS, generate an initial NEB path|
|nebef.py | Print the result of max force, energy and relative energy of NEB images|
|fix_slab_atoms.py| Fix the slabs atoms for frequency analysis|
|get_virb_free_energy.py|Calculate vibration free energy from the OUTCAR of  frequency analysis|
|get_virb_mode.py|Print the wavenums of vibration mode and extract a specific vibration mode|
|dist.py| Calculate the distance between two structures|

## References
| Name | Purpose|
| ---- | ---- |
|job_submission.md| File template for job submission, only for StepNode |