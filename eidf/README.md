1. PLEASE CUSTOMISE `launch_jobs.py` AND `pvc.yaml` TO YOUR OWN NEEDS.
MOST IMPORTANTLY, PLEASE CHANGE THE USERNAME AND EMAIL!
2. Install secret env vars for: Github, Wandb, and HuggingFace.
I forgot where but there are tutorials for this somewhere on the EIDF docs.
3. On the EIDF file server, you'll need to create a python installation that has kubejobs installed. 
Unfortunately the version on pip is out of date, so you'll need to clone https://github.com/AntreasAntoniou/kubejobs/tree/main and `pip install .`
4. Create a PVC with `kubectl apply -f pvc.yaml`
5. Then run `launch_jobs.py`. 