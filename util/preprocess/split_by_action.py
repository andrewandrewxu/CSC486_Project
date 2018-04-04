import os
import sys

# split set according to the public split set proto
train_set = ['1','5','6','7','8'];
test_set  = ['9','11'];
cameras = ['camera_2','camera_3','camera_4','camera_1']
actions = map(lambda x:str(x),range(1,16))

label_list = [None]*16*2  # 15 actions * train and test

for i in range(len(label_list)):
  label_list[i] = []

#for phase in ['train','val']:
for phase in ['val']:
  for cam_id in cameras:
    label_list_path = 'G:\\CSC-486-Stuff-and-Things\\img_list\\linux_accv_%s_%s_label_cropped_3d.txt' % (phase,cam_id);
    label_list_file = open(label_list_path,'r')
    for lin in label_list_file.readlines():
      item = lin.split(' ')
      elem = item[0]
      elem = elem.split('\\')
      person_id = elem[5]
      print(elem[6])
      #print(elem[7].split('_')[0])
      action_id = int(elem[6].split('_')[0]) - 1
      
      if person_id in train_set:
        label_list[action_id].append(lin)
      elif person_id in test_set:
        label_list[action_id + 16].append(lin)
      else:
        print ("Error")
        print (person_id)

dst_folder = 'G:\\CSC-486-Stuff-and-Things\\new_img_list'
if not os.path.isdir(dst_folder):
  os.mkdir(dst_folder)

for i in range(1,len(label_list)):
  print(dst_label_list_path)
  sys.exit(0)
  if i < 16:
    dst_label_list_path = dst_folder + 'train_' + str(i) + '.txt' 
  elif i == 16:
    continue
  else:
    dst_label_list_path = dst_folder + 'test_' + str(i - 16) + '.txt'

  dst_label_file = open(dst_label_list_path,'w')
  dst_label_file.write(''.join(label_list[i]))

