matlab_cmds = []

for camera in range(1, 5):
  for subject in [1] + list(range(5, 9)):
  	for action in range(2, 17):
  		for subaction in range(1, 3):
  			# cmd = "echo {} - {} - {} - {};\n".format(camera, subject, action, subaction)
  			# cmd += "timeout -s KILL 3m /usr/local/bin/matlab -nodisplay -r 'extract_imgs_and_labels({}, {}, {}, {}); exit;';\n".format(camera, subject, action, subaction)
  			cmd = "echo 'extract_imgs_and_labels({}, {}, {}, {}); exit;' > cmd.m;\n".format(camera, subject, action, subaction)
  			cmd += "echo 'Camera:{} Subject:{} Action:{} Subaction:{}'\n".format(camera, subject, action, subaction)
  			cmd += "/usr/local/bin/matlab -nodesktop -nodisplay < cmd.m > out.log &\n"
  			cmd += "MATLAB_PID=$!;\n"
  			cmd += "sleep 10s;\n"
  			cmd += "grep -q -i 'Cleared' <(tail -f out.log);\n"
  			cmd += "sleep 2s;\n"
  			cmd += "echo 'killing Camera:{} Subject:{} Action:{} Subaction:{}\n';\n".format(camera, subject, action, subaction)
  			cmd += "kill $MATLAB_PID;\n"
  			matlab_cmds.append(cmd)

with open('run.sh', 'w') as f:
	f.write('#! /bin/bash\n')
	f.writelines(matlab_cmds)
