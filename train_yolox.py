import subprocess
import os 
import os

ROOT = './AH_training_outputs'
lrs = [0.00015625, 0.0015625]
weight_decay = [5e-4, 5e-5]

for lr in lrs: 
    for wd in weight_decay:
        
        small_image_output_dir = os.path.join(ROOT, 'train_Simg_lr{}_wd{}'.format(lr, wd))

        small_cmd = ["python", "train.py", "-f", "S_particulate_yolox_exp.py", "-b", "32", 
                     "-c", "/home/anthony/git_repos/YOLOX-training/yolox_s.pth", "--occupy", "--cache",
                     "basic_lr_per_img", str(lr), "weight_decay", str(wd), "max_epoch", str(200), "output_dir", small_image_output_dir
        ]
        subprocess.run(small_cmd)
