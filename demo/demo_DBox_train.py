"""
Demonstration of how to generate train DBox network on ODMD.
"""

import sys, os, IPython, torch, tqdm
# import matplotlib.pyplot as plt
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)
sys.path.insert(0,"../")
import odmd, dbox

net_name = "DBox_peper" # to save intermidiate net results.

# Select configuration (more example settings from paper in config directory).
datagen_config = "../config/data_gen/ODMS_paper_cfg.yaml" # standard_data.yaml"
camera_config = "../config/camera/hsr_grasp_camera.yaml" 
train_config = "../config/train/paper_train.yaml"
# train_config = "../config/train/tmp_train.yaml"
dbox_config = "../config/model/DBox.yaml"

# Initiate data generator, model, training parameters, and data loader.
odmd_data = odmd.data_gen.DataGenerator(datagen_config)
odmd_data.initialize_data_gen(camera_config)
net, device, m_params = dbox.load_model(dbox_config, odmd_data.num_pos)
train = dbox.load_training_params(train_config)
bb2net = dbox.BoundingBoxToNetwork(m_params, train["batch_size"])

# Initiate training!
model_dir = os.path.join("../results", "model", net_name)
snps_dir = os.path.join(model_dir, "snps")
status_dir = os.path.join(model_dir, "status")
os.makedirs(snps_dir, exist_ok=True if 'tmp' in train_config else False)
os.makedirs(status_dir, exist_ok=True if 'tmp' in train_config else False)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0) # TODO - move to training cfg

running_loss = 0.0
losses_list = []
# ct = 0
print("Starting training for %s." % net_name)
# while ct < train["train_iter"]:
for ct in tqdm.tqdm(range(train["train_iter"])):

	# Generate examples for ODMD training (repeat for each training iteration).
	bb_3D, bb = odmd_data.generate_object_examples(bb2net.n_bat)
	bb_3D, bb = odmd.data_gen.add_perturbations(bb_3D, bb, odmd_data)

	# Network inputs and labels, forward pass, loss, and gradient.
	bb2net.bb_to_labels(bb_3D, bb)
	inputs, labels = bb2net.inputs.to(device), bb2net.labels.to(device)
	outputs = net(inputs).to(device)
	loss = criterion(outputs, labels)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	running_loss += loss.item()

	# Print progress details and save model at set interval.
	if ct % train["display_iter"] == 0:
		cur_loss = running_loss / train["display_iter"]
		losses_list.append(cur_loss)
		# plt.plot(losses_list) TODO-plot them
		with open(os.path.join(status_dir, "prints.txt"), 'a') as f:
			f.write("[%9d] loss: %.6f\n" % (ct, cur_loss))
		# print("[%9d] loss: %.6f" % (ct, cur_loss))
		running_loss = 0.0
	if ct in train["save_iter"]:
		torch.save(net.state_dict(), "%s/%s_%09d.pt" % (snps_dir, net_name, ct))
		with open(os.path.join(status_dir, "prints.txt"), 'a') as f:
			f.write("[%9d] interval model saved.\n" % ct)
		# print("[%9d] interval model saved." % ct)
