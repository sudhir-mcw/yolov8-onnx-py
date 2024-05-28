import sys

preprocess_time = []
postprocess_time = []
post_process_time =[]

fp=open(sys.argv[1],'r')

for line in fp.read().strip().split('\n'):
    if line=='':
        break
    # print(line)
    if "pre process" in line:
        time = float(line.split(":")[1].strip())
        preprocess_time.append(time)
    elif "post process" in line:
        time = float(line.split(":")[1].strip())
        post_process_time.append(time)
fp.close()

print('preprocess',sum(preprocess_time)/len(preprocess_time))
print('postprocess',sum(post_process_time)/len(preprocess_time))
