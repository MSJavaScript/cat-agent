"""
Run structure relaxation (cell is also relaxed) or NPT MD simulation with DPA force field.
Usage:

# Run structure relaxation
python3 run_ase_npt_md_with_dpa.py --pt frozen.pth --save_name TiO2_opt --infile POSCAR_TiO2 opt --fmax 0.05

# Run NPT MD, 500K, 0.001 GPa 
python3 run_ase_npt_md_with_dpa.py --pt frozen.pth \
      --save_name Si_MD \
      --infile POSCAR_Si \
      md -t 500 -p 0.001 \
      --dt 2.0 --nsteps 2000 --log_interval 20
"""

from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS
from deepmd.calculator import DP as DPCalculator

import argparse 

parser = argparse.ArgumentParser(description="Run structure relaxation (cell is also relaxed) or NPT MD simulation with DPA force field.")
parser.add_argument("--pt", type=str, required=True, help="The path to DPA checkpoint file.")
parser.add_argument("--save_name", type=str, required=True, help="Prefix of the *.traj and *.log file names.")
parser.add_argument("--infile", type=str, required=True, help="File name of input POSCAR.")

subparsers = parser.add_subparsers(dest='task', help='Task type, run structure relaxation or MD.')

opt_parser = subparsers.add_parser('opt', help='Structure relaxation.')
opt_parser.add_argument("--fmax", type=float, default=0.08, help="Force convergence criterion (eV/A). Default 0.08.")
opt_parser.add_argument("--nsteps", type=int, default=100, help="Max relaxation steps. Default 100." )

md_parser = subparsers.add_parser('md', help='NPT MD simulation.')
md_parser.add_argument("-t", "--temperature", type=float, help="MD temperature (K).")
md_parser.add_argument("-p", "--pressure", type=float, help="MD pressure (GPa).")
md_parser.add_argument("--dt", type=float, default=2.0, help="MD time step (fs). Default 2.0.")
md_parser.add_argument("--nsteps",  type=int, help="The number of MD steps.")
md_parser.add_argument("--log_interval",  type=int, default=20, help="The step interval between two system snapshots and two log prints. Default 20.")

args = parser.parse_args()

dp_calc = DPCalculator(args.pt)
save_name = args.save_name
atoms = read(args.infile)
atoms.calc = dp_calc 

if args.task == 'opt':
    fcf = FrechetCellFilter(atoms)
    opt = LBFGS(fcf, logfile=f"{save_name}.log", trajectory=f"{save_name}.traj", maxstep=0.1)
    opt.run(fmax=args.fmax, steps=args.nsteps)
else:
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature)
    try:
        from ase.md.nose_hoover_chain import IsotropicMTKNPT
        dyn = IsotropicMTKNPT(atoms=atoms,
                              timestep=args.dt*units.fs,
                              temperature_K=args.temperature,
                              pressure_au=args.pressure * units.GPa,
                              tdamp=100 * units.fs,
                              pdamp=1000 * units.fs,
                              logfile = f"{save_name}.log",
                              loginterval = args.log_interval)
    except ImportError as e:
        from ase.md.nptberendsen import NPTBerendsen
        dyn = NPTBerendsen(
            atoms = atoms,
            timestep = args.dt * units.fs,
            temperature_K = args.temperature,
            pressure_au= args.pressure * units.GPa,
            taut = 100 * units.fs,
            taup = 1000 * units.fs,
            compressibility_au = 4.5e-5/units.bar,
            logfile = f"{save_name}.log",
            loginterval = args.log_interval
        )

    traj = Trajectory(f"{save_name}.traj", 'w', atoms = atoms)
    dyn.attach(traj.write, interval = args.log_interval)
    dyn.run(args.nsteps)

