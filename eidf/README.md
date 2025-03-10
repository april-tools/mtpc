PLEASE CUSTOMISE `launch_jobs.py` TO YOUR OWN NEEDS.
MOST IMPORTANTLY, PLEASE CHANGE THE USERNAME AND EMAIL!

You also need to install secret env vars for: Github, Wandb, and HuggingFace.
I forgot where but there are tutorials for this somewhere on the EIDF docs.

Then on the EIDF file server, you'll need to create a python installation that has kubejobs installed. 
Unfortunately the version on pip is out of date, so you'll need to clone https://github.com/AntreasAntoniou/kubejobs/tree/main and `pip install .`

Then run `launch_jobs.py`. 