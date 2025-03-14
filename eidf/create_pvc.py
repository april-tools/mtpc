
from kubejobs.jobs import create_pvc
from .launch_jobs import username
# create some persistent storage to keep around model weights, and perhaps data if you need it
create_pvc(
    pvc_name=f"{username}-mtp-pvc", storage="100Gi", access_modes="ReadWriteOnce"
)
