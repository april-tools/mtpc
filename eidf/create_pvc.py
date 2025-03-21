
def pvc_name(username, script_name):
	return f"{username}-mtp-{script_name.replace('_', '-')}"

def create_pvc(username, script_name, pvc_size):
	pvc_script = f"""
apiVersion: v1
kind: PersistentVolumeClaim
metadata: 
  name: {pvc_name(username, script_name)}
  labels:
    eidf/user: {username}-infk8s
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: {pvc_size}
  storageClassName: csi-rbd-sc
"""
	print(pvc_script)
	with open(f"{pvc_name(username, script_name)}.yaml", "w") as f:
		f.write(pvc_script)
	# Apply the PVC using kubectl
	import subprocess
	
	print(f"Creating PVC {pvc_name(username, script_name)}")
	result = subprocess.run(["kubectl", "apply", "-f", f"{pvc_name(username, script_name)}.yaml"], 
		capture_output=True, text=True)
	
	if result.returncode == 0:
		print(result.stdout)
		print(f"Successfully created PVC: {pvc_name(username, script_name)}")
	else:
		print(f"Failed to create PVC: {result.stderr}")
	