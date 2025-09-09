1. FFTVIS Installation
	a. Open VSCode
	b. Connect to WSL:Ubuntu for OS compatibility
	c. Generate local environment "python3 -m venv .venv_validation_1" (stored in C:\Users\Rajorshi\Desktop\PostDoc\Vajra_Windows\Saurabh_Mandar_Project_1) 
	d. Activate environment in WSL Terminal in VSCode : source .venv_validation_1/bin/activate
	d. pip install fftvis (on this environment)
	e. Run in a jupyter notebook (kernel running on WSL:Ubuntu), access via "from fftvis import simulate_vis" from code cell
	

	a. Open VSCode
	b. Connect to WSL:Ubuntu for OS compatibility
	c. Install anaconda at : /home/rasch/anaconda3/
	d. conda create -n myenv_validation_1 at : /home/rasch/anaconda3/envs/myenv_validation_1
		Also same name environ on NRAO lustre at : environment location: /lustre/aoc/projects/hera/rchandra/miniconda3/envs/myenv_validation_1
	e. conda activate myenv_validation_1		
	f. python -m pip install fftvis : (force install inside the environment)
	g. python -m pip install git+https://github.com/HERA-Team/hera_sim : (hera_sim part of workflow)
	h. conda install -c conda-forge healpy : (healpy part of workflow)
 
2. Validation Sim Install (Running on WSL terminal through Windows-Ubuntu : install python3, conda on it)
	i. wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
	   sha256sum Anaconda3-2025.06-0-Linux-x86_64.sh
	   bash Anaconda3-2025.06-0-Linux-x86_64.sh
	a. git clone validation_sim into (base) rasch@RRI-D-675:/mnt/c/Users/Rajorshi/Desktop/PostDoc/Vajra_Windows/Saurabh_Mandar_Project_1/Validation_Sim/validation-sim
	b. Comment out scalene in h6c-env.yaml
	c. Comment out nb_conda, python version in h6c-env.yaml
	d. run conda env create -f projects/h6c/h6c-env.yaml
	e. To use : conda activate h6c-sim

3. NRAO Lustre Environ setup : 
	/lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats 
		: stores the project and beam files
	a. Activate environment : conda activate myenv_validation_1
	b. conda install -c conda-forge python=3.13 (updating within existing environ, not creating new one, which is better)
	c. conda update --all (to update everything to work with new python)
	d. python -m pip install fftvis 
	e. python -m pip install git+https://github.com/HERA-Team/hera_sim : (hera_sim part of workflow)
	f. conda install -c conda-forge healpy : (healpy part of workflow)
	g. conda env update -n myenv_validation_1 -f projects/h6c/h6c-env.yaml (do this in folder cd validation-sim from bash, outside myenv_validation_1) 	
		if already inside myenv_validation_1, do : conda env update -f projects/h6c/h6c-env.yaml
	h. Reminder to use the right package versions (like removing scalene) as discussed with Steven and trial and error methods
	i. # download everything under https://data.edu/beams/ into ./beams
		wget -r -np -nH --cut-dirs=1 -e robots=off --reject "index.html*" -P hera-beams https://data.nrao.edu/hera/beams/
	j. 

4. Running validation-sim
	a. (myenv_validation_1) herapost-master$ export VALIDATION_SYSTEM_NAME=NRAO_H6C
		a1. Need to make directory "sky_models/raw/" since it doesn't exist and code only writes if it exists.
	b. (myenv_validation_1) herapost-master$ ./vsim.py sky-model ptsrc --with-confusion
		[10:13:38] INFO     Creating sbatch file: /lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/batch_scripts/skymodel/ptsrc_allfreqs.sbatch        run_sky_model.py:104
		Submitted batch job 3871102
		           DEBUG            																	     run_sky_model.py:114
                    ===Job Script===
                    #!/bin/bash
                    #SBATCH --partition=batch
                    #SBATCH --nodes=1
                    #SBATCH --mem=24GB
                    #SBATCH --ntasks=1
                    #SBATCH --time=0-00:15:00
                    #SBATCH --job-name=ptsrc
                    #SBATCH --output=/lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/logs/skymodel/ptsrc256/%J.out

                    source ~/.bashrc
                    source /home/rasch/anaconda3/bin/activate
                    conda activate myenv_validation_1

                    module load mpi

                    time python vsim.py sky-model ptsrc --local --nside 256 --label '' --channels 0~1536

                    ===END===
	c. ./vsim.py make-obsparams  --layout hera_350_subset --freq-range 100e6 200e6 --channels 0~1023 --sky-model sky_models
	d. ./vsim.py runsim  --layout hera_350_subset --freq-range 100e6 200e6 --channels 0~1023  --sky-model sky_models --n-time-chunks 288 --do-time-chunks 0~3

	e. View generated sbatch files :
		vi  /lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/batch_scripts/skymodel/ptsrc_allfreqs.sbatch

	f. View outfiles : 
		ls -ltrh logs/skymodel/ptsrc256/
		vi logs/skymodel/ptsrc256/3881363.out


/lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats 

conda activate myenv_validation_1

export VALIDATION_SYSTEM_NAME=NRAO_H6C

echo $VALIDATION_SYSTEM_NAME

(freq range in MHz)
./vsim.py sky-model ptsrc --freq-range 100 200 --channels 0~1023 --with-confusion 

./vsim.py sky-model ptsrc --freq-range 100 200 --channels 0~1023 --with-confusion --slurm-override time=0-04:00:00


./vsim.py make-obsparams  --layout H4C --freq-range 100e6 200e6 --channels 0~1023 --sky-model ptsrc256

./vsim.py make-obsparams  --layout H4C --freq-range 100 200 --channels 0~1023 --sky-model ptsrc256


./vsim.py runsim  --layout H4C --freq-range 100 200 --channels 0~1023  --sky-model ptsrc256  --n-time-chunks 288 --do-time-chunks 0~3

./vsim.py runsim  --layout H4C --freq-range 100 200 --channels 0~1023  --sky-model ptsrc256  --n-time-chunks 288 --do-time-chunks 0~3 --simulator simulator-specs/fftvis

           INFO     Working on frequency channel 1022 chunk 2                                                                                                                                                                                                   run_sim.py:127
           INFO     /lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/outputs/ptsrc256/nt17280-00288chunks-H4C-nonred/fch1022_chunk00002.uvh5                                                                                              run_sim.py:134
           INFO     Creating sbatch file: /lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/batch_scripts/vis/ptsrc256/nt17280-00288chunks-H4C-nonred/fch1022_chunk00002                                                                   run_sim.py:194
Submitted batch job 4026112
           DEBUG                                                                                                                                                                                                                                                run_sim.py:206
                    ===Job Script===
                    #!/bin/bash
                    #SBATCH --partition=batch
                    #SBATCH --nodes=1
                    #SBATCH --mem=240GB
                    #SBATCH --ntasks=16
                    #SBATCH --time=0-00:5:00
                    #SBATCH --job-name=ptsrc256/nt17280-00288chunks-H4C-nonred/fch1022_chunk00002
                    #SBATCH --output=/lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/logs/vis/ptsrc256/nt17280-00288chunks-H4C-nonred/fch1022_chunk00002-%J.out

                    source ~/.bashrc
                    source /home/rasch/anaconda3/bin/activate
                    conda activate myenv_validation_1

                    module load mpi

                    hera-sim-vis.py --normalize_beams --fix_autos --log-level INFO --phase-center-name zenith --compress /lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/compression-cache/ch288_H4C.npy
                    /lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/config_files/obsparams/ptsrc256/nt17280-00288chunks-H4C-nonred/fch1022_chunk00002
                    /lustre/aoc/projects/hera/rchandra/H6C_Validation_Stats/validation-sim/fftvis-cpu.yaml




./vsim.py sky-model gleam --freq-range 100e6 200e6 --channels 0~1023 --hpc-config hpc-configs/NRAO_H6C.yaml



5. validation-sim GitHub stuff
	

	a. Forking
		git remote set-url origin https://github.com/RajorshiSChandra/validation-sim.git
		git remote add upstream https://github.com/HERA-Team/validation-sim.git
		git fetch --all --prune
			Local: origin now points to my fork; 
			upstream points to the original. 
			Fetch populates my local .git with the latest refs from both.
	b. Viewing
		git remote -v
			origin  https://github.com/RajorshiSChandra/validation-sim.git (fetch)
			origin  https://github.com/RajorshiSChandra/validation-sim.git (push)
			upstream        https://github.com/HERA-Team/validation-sim.git (fetch)
			upstream        https://github.com/HERA-Team/validation-sim.git (push)

	c. 
		git checkout -b fix/skymodel-time-from-yaml upstream/main
			M       core/run_sky_model.py
			branch 'fix/skymodel-time-from-yaml' set up to track 'upstream/main'.
			Switched to a new branch 'fix/skymodel-time-from-yaml'



	d. Set upstream to original repo :
		git remote add upstream https://github.com/HERA-Team/validation-sim.git
	   Set origin to my forked repo : 
		git remote set-url origin https://github.com/RajorshiSChandra/validation-sim.git
	   Make new branch in my fork, from original repo : 
		git checkout -b practice/config_yaml_priority_timefix
	   Assuming my edits are on a local repo, add the edited files and commit :
		git add core/run_sky_model.py
		git add hpc-configs/NRAO_H6C.yaml
	   	git commit -m "Updated run_sky_model.py to account for hpc-config yaml and prioritize it."
	   Push the edit-commits onto the branch on my fork : 
		git push -u origin practice/config_yaml_priority_timefix
			
			Enumerating objects: 10, done.
			Counting objects: 100% (10/10), done.
			Delta compression using up to 32 threads
			Compressing objects: 100% (6/6), done.
			Writing objects: 100% (6/6), 1.20 KiB | 1.20 MiB/s, done.
			Total 6 (delta 4), reused 0 (delta 0), pack-reused 0
			remote: Resolving deltas: 100% (4/4), completed with 4 local objects.
			remote:
			remote: Create a pull request for 'practice/config_yaml_priority_timefix' on GitHub by visiting:
			remote:      https://github.com/RajorshiSChandra/validation-sim/pull/new/practice/config_yaml_priority_timefix
			remote:
			To github.com:RajorshiSChandra/validation-sim.git
			 * [new branch]      practice/config_yaml_priority_timefix -> practice/config_yaml_priority_timefix
			branch 'practice/config_yaml_priority_timefix' set up to track 'origin/practice/config_yaml_priority_timefix'.
	   Open Pull Request :
		Be sure to confirm which main and branch origin repo is being addressed (defaults to original repo, which is risky; always use own fork first)
	   Merging via CLI (Usually safer to do via GUI on website) Q : Only authorized person can m erge anyway ?
		Step 1 Clone the repository or update your local repository with the latest changes.
			git pull origin main
		Step 2 Switch to the base branch of the pull request.
			git checkout main
		Step 3 Merge the head branch into the base branch.
			git merge practice/config_yaml_priority_timefix
		Step 4 Push the changes.
			git push -u origin main
