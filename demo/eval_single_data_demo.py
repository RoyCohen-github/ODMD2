"""
Demonstration of how to generate new training data on ODMD.
"""

import yaml

import sys, os, IPython, torch, _pickle as pickle, numpy as np
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)
sys.path.insert(0,"../")
import odmd, dbox

############################################################################################################
#####################################  example #############################################################
############################################################################################################

all_detectron_output = [
    [1664.0343017578125,
     2211.4384765625,
     1846.2708740234375,
     2449.1787109375],

    [1899.4244384765625,
     2256.562744140625,
     2114.703857421875,
     2517.085205078125],

    [1958.7423095703125,
     2262.435546875,
     2191.429931640625,
     2560.0087890625],

    [1757.066162109375,
     2309.724853515625,
     1992.8900146484375,
     2657.20654296875],

    [1552.2760009765625,
     2360.260498046875,
     1825.2176513671875,
     2742.096923828125],

    [1755.489013671875,
     2424.630615234375,
     2066.094970703125,
     2852.80419921875],

    [1604.8935546875,
     2376.311767578125,
     1944.4326171875,
     2869.37255859375],

    [1569.116943359375,
     2450.83349609375,
     1985.792724609375,
     3062.9111328125],

    [1596.1048583984375,
     2702.44384765625,
     2082.6162109375,
     3449.81787109375],

    [1475.8773193359375,
     2818.90771484375,
     2147.44287109375,
     3889.684814453125],

    [1339.128662109375,
    3147.27978515625,
    2170.5,
    4381.2373046875],
]

all_camera_positions = [
    [0, 0, 11],
    [0, 0, 10],
    [0, 0, 9],
    [0, 0, 8],
    [0, 0, 7],
    [0, 0, 6],
    [0, 0, 5],
    [0, 0, 4],
    [0, 0, 3],
    [0, 0, 2],
    [0, 0, 1]
]

all_meas_idx = np.arange(len(all_camera_positions))
# np.random.shuffle(all_meas_idx)
# selected_measurements = sorted(all_meas_idx[:10], reverse=np.random.randint(0, 2))
selected_measurements = sorted(all_meas_idx[0:10], reverse=1)
print(f"selected measurements {selected_measurements}")

detection_output = np.asarray([all_detectron_output[idx] for idx in selected_measurements])
camera_positions = np.asarray([all_camera_positions[idx] for idx in selected_measurements])

# add noise to account for incorrect estimation
# detection_output = detection_output + np.random.normal(scale=(detection_output.max()-detection_output.min())*0.00, size=detection_output.shape)
# camera_positions = camera_positions + camera_positions*np.random.normal(scale=(camera_positions.max()-camera_positions.min())*0.025, size=camera_positions.shape)
# print(camera_positions)
############################################################################################################
############################################################################################################
############################################################################################################

bb = {'bboxes': np.array([[[None], [None], [None], [None], [None]],
                          [[None], [None], [None], [None], [None]],
                          [[None], [None], [None], [None], [None]],
                          [[None], [None], [None], [None], [None]],
                          [[None], [None], [None], [None], [None]],
                          [[None], [None], [None], [None], [None]],
                          [[None], [None], [None], [None], [None]],
                          [[None], [None], [None], [None], [None]],
                          [[None], [None], [None], [None], [None]],
                          [[None], [None], [None], [None], [None]]]),

      'n_positions': 10,
      'n_ex': 1,
      'image_dim': [4624, 3472],  # [y, x]
      'fx_norm': None,
      'fy_norm': None,
      'bbox_format': '[xc_norm; yc_norm; w_norm; h_norm; Z]'}
bb_3D = {'positions': np.array([[[None], [None], [None]],
                                [[None], [None], [None]],
                                [[None], [None], [None]],
                                [[None], [None], [None]],
                                [[None], [None], [None]],
                                [[None], [None], [None]],
                                [[None], [None], [None]],
                                [[None], [None], [None]],
                                [[None], [None], [None]],
                                [[None], [None], [None]]]),
         'camera_movement': np.array([[[None], [None], [None]],  # should be inverse direction of positions
                                      [[None], [None], [None]],
                                      [[None], [None], [None]],
                                      [[None], [None], [None]],
                                      [[None], [None], [None]],
                                      [[None], [None], [None]],
                                      [[None], [None], [None]],
                                      [[None], [None], [None]],
                                      [[None], [None], [None]]]),
         'sizes': [None, None],
         'n_ex': 1,
         'n_positions': 10}


for pos in range(bb_3D['n_positions']):
    bb_3D['positions'][pos][0][0] = camera_positions[pos][0]
    bb_3D['positions'][pos][1][0] = camera_positions[pos][1]
    bb_3D['positions'][pos][2][0] = camera_positions[pos][2]

bb_3D['camera_movement'] = (bb_3D['positions'][-1] - bb_3D['positions'])[:-1]

for pos in range(bb['n_positions']):
    bb['bboxes'][pos][0][0] = (detection_output[pos][0]) / bb['image_dim'][1]
    bb['bboxes'][pos][1][0] = (detection_output[pos][1]) / bb['image_dim'][0]
    bb['bboxes'][pos][2][0] = (detection_output[pos][2] - detection_output[pos][0]) / bb['image_dim'][1]
    bb['bboxes'][pos][3][0] = (detection_output[pos][3] - detection_output[pos][1]) / bb['image_dim'][0]
    bb['bboxes'][pos][4][0] = bb_3D['positions'][pos][2][0]
bb['bboxes'] = bb['bboxes'].astype(type(0.42507987308062717))


############################################################################################################
############################################################################################################
############################################################################################################

"""
Demonstration of how to evaluate DBox network on ODMD.
"""

net_name = "DBox_paper"
#net_name = "DBox_pretrained" # Uncomment to run DBox model from paper.
model_idx = -1 # Can cycle through indices to find best validation performance.

# Select dataset to evaluate.
# dataset = "odmd" # or "odms_detection" for ODMS dataset converted to detection.
# eval_set = "val" # or "test" once model training and development are complete.

# Select configuration (more example settings from paper in config directory).
datagen_config = "../config/data_gen/ODMS_paper_cfg.yaml"
camera_config = "../config/camera/hsr_grasp_camera.yaml"
train_config = "../config/train/paper_train.yaml"
dbox_config = "../config/model/DBox.yaml"

# Initiate data generator, model, data loader, and load weights.
odmd_data = odmd.data_gen.DataGenerator(datagen_config)
odmd_data.initialize_data_gen(camera_config)
net, device, m_params = dbox.load_model(dbox_config, odmd_data.num_pos)
bb2net = dbox.BoundingBoxToNetwork(m_params)
model_dir = os.path.join("..", "results", "model", net_name, "snps")
model_list = sorted([pt for pt in os.listdir(model_dir) if pt.endswith(".pt")])
net = dbox.load_weights(net, os.path.join(model_dir, model_list[model_idx]))


# percent_error=[]; abs_error=[]; predictions_all=[]

with torch.no_grad():
    # Load data for specific set.
    # bb_data = pickle.load(open(os.path.join(set_dir, test), "rb"))
    # bb_3D, bb = bb_data["bb_3D"], bb_data["bb"]

    # Run DBox with correct post-processing for configuration.
    bb2net.set_batch(bb_3D["n_ex"])
    bb2net.bb_to_labels(bb_3D, bb)
    inputs = bb2net.inputs.to(device)
    predictions = net(inputs).cpu().numpy()
    if bb2net.prediction == "normalized":
        predictions[:,0] *= bb2net.norm
    depths = bb["bboxes"][-1][-1]

    percent_error = np.mean( abs(predictions[:,0] - depths) / depths)
    abs_error     = np.mean(abs(predictions[:,0] - depths))
    predictions_all = predictions

# Print out final results.
print("\nResults summary")
print(f"net prediction: {predictions[0, 0]}")
print(f"ground truth  : {bb['bboxes'][-1][-1][0]}")
print(f"Mean Percent  Error: {round(percent_error, 4)*100}%")
print("Mean Absolute Error: %.4f (m)" % abs_error)

# # Generate final results file.
# name = model_list[model_idx].split(".pt")[0]
# data_name = "%s_%s" % (dataset, eval_set)
# print("\nSaving %s results file for %s.\n" % (data_name, name))
# result_data = {"Result Name": name, "Set List": set_list,
# 				"Percent Error": percent_error, "Absolute Error": abs_error,
# 				"Depth Estimates": predictions_all, "Dataset": data_name}
# os.makedirs("../results/", exist_ok=True)
# pickle.dump(result_data, open("../results/%s_%s.pk" % (name, data_name), "wb"))