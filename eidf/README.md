1. PLEASE CUSTOMISE `launch_jobs.py` AND `pvc.yaml` TO YOUR OWN NEEDS.
MOST IMPORTANTLY, PLEASE CHANGE THE USERNAME AND EMAIL!
2. Install secret env vars for: Github, Wandb, and HuggingFace.
I forgot where but there are tutorials for this somewhere on the EIDF docs.
3. On the EIDF file server, you'll need to create a python installation that has kubejobs installed. 
Unfortunately the version on pip is out of date, so you'll need to clone https://github.com/AntreasAntoniou/kubejobs/tree/main and `pip install .`
4. Run `launch_jobs.py --script some-script.sh` (where `some-script.sh` is in `nanoGPT/scripts`). Additionally, you can specify a branch with `--branch some-branch`.
5. Unfortunately, there is an additional hassle with getting trained models **out of** the pod that trained the model. 
To do this, after training we'll create another pod just for copying with `kubectl pvcsync {user}-mtp-{script_name}`. Then you can login to this with `kubectl exec -it {user}-mtp-{script_name}-rsync-backend -- /bin/bash`. Make sure to `cd /data` to go to the correct directory.
6. To properly download things to your laptop, you'll need to setup `ssh`, see https://git.ecdf.ed.ac.uk/infk8s/getting-started-on-the-eidf-gpu-cluster/-/wikis/Accessing-Persistent-Volume-Claims
