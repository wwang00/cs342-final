import os
from solution import models
import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from PIL import Image
import sys

model = models.Detector()

imgs = [x for x in os.listdir("box10") if x.endswith("_img.png")]
dets = [x for x in os.listdir("solution") if "det" in x]

while True:
	command = input("Command > ")
	if command == "shuffle":
		random.shuffle(imgs)
		print("Done")
	elif command == "list":
		for i, det in enumerate(dets):
			print(i, det)
	elif command.startswith("load"):
		no = int(command[len("load "):])
		print("Loading model", no)
		model.load_state_dict(torch.load(os.path.join("solution", dets[no]), map_location='cpu'))
	elif command == "which":
		print(no, dets[no])
	elif command == "demo":
		with torch.no_grad():
			img = (np.array(Image.open(os.path.join("box10", imgs[0]))) / 255.0)
			fig, axs = plt.subplots(2, 2)
			axs[0,0].imshow(img)
			det, size, depth, is_puck = model(torch.from_numpy(img)[None].permute(0, 3, 1, 2).float())
			sdet = F.sigmoid(det)
			sdepth = F.sigmoid(depth)
			axs[0,1].imshow(det[0,0:3].permute(1, 2, 0))
			axs[1,0].imshow(det[0,3])
			axs[1,1].imshow(depth[0, 0])
			print("Is_puck: ", is_puck.item(), F.sigmoid(is_puck).item())
			plt.show()
	elif command == "box":
		with torch.no_grad():
			img = (np.array(Image.open(os.path.join("box10", imgs[0]))) / 255.0)
			lsts, is_puck = model.detect(torch.from_numpy(img).float().permute(2, 0, 1))
			colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
			for i, lst in enumerate(lsts[0:3]):
				if i == 1 and is_puck < 0:
					continue
				for v in lst:
					print(v)
					s, x, y, w, h = v
					s_x = round(x - w)
					e_x = round(x + w)
					s_y = round(y - h)
					e_y = round(y + h)
					img[s_y:s_y+2,s_x:e_x] = colors[i]
					img[e_y-2:e_y,s_x:e_x] = colors[i]
					img[s_y:e_y,e_x-2:e_x] = colors[i]
					img[s_y:e_y,s_x:s_x+2] = colors[i]
			plt.imshow(img)
			plt.show()
	elif command == "boxvid":
		os.mkdir("boxvid")
		with torch.no_grad():
			for iname in imgs:
				img = (np.array(Image.open(os.path.join("box10", iname))) / 255.0)
				lsts, depth, is_puck = model.detect(torch.from_numpy(img).float().permute(2, 0, 1))
				colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]
				for i, lst in enumerate(lsts):
					if i == 1 and is_puck < 0:
						continue
					for v in lst:
						print(v)
						s, x, y, w, h = v
						s_x = round(x - w)
						e_x = round(x + w)
						s_y = round(y - h)
						e_y = round(y + h)
						img[s_y:s_y+2,s_x:e_x] = colors[i]
						img[e_y-2:e_y,s_x:e_x] = colors[i]
						img[s_y:e_y,e_x-2:e_x] = colors[i]
						img[s_y:e_y,s_x:s_x+2] = colors[i]
				save_image(torch.from_numpy(img).permute(2, 0, 1), os.path.join("boxvid", iname))
				save_image(F.sigmoid(depth), os.path.join("boxvid", iname[:-4] + "_depth.png"))
	elif command == "train":
		model.train()
	elif command == "eval":
		model.eval()
	elif command == "interact":
		import code
		code.interact(local=locals())
	elif command == "exit":
		sys.exit()
	else:
		print("Unknown command")