import math
import numpy as np

import scipy
from scipy.stats import qmc
from smt.sampling_methods import LHS

from sklearn.metrics import mean_squared_error

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import os
import copy
from utils.utils import train_joint, alc_acquisition, run_active_learning, rmspe_sd
from utils.models import Autoencoder, ParametricGP, JointModel

font = {'size'   : 30}
matplotlib.rc('font', **font)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
torch.set_default_dtype(torch.float)
torch.set_default_device(device)
torch.set_printoptions(precision=10)

import argparse
def main():
    parser = argparse.ArgumentParser(description="Run AE-GP-ALC tasks.")
    parser.add_argument('--task', type=str, required=True,
                        choices=['1d', '2d', '3d', 'borehole'],
                        help='Choose the task to run: 1d | 2d | 3d | borehole')
    args = parser.parse_args()
    task = args.task.lower()

    os.makedirs("fig/0/", exist_ok=True)
    os.makedirs("data/0/", exist_ok=True)

    if task == '1d':
        from utils.data_gen import generate_dataset_for_1d as generate_dataset_for_al_gp
        # === 2. 生成数据 ===
        x_train, y_train, x_test, y_test, x_ref, y_ref, x_cand, y_cand = generate_dataset_for_al_gp(
            N_train=10,
            N_test=200,
            N_ref=100,
            N_cand=100,
            noise_scale=0.1,
            device=device,
            seed=None
        )
        np.save('data/0/ALAEmGP-toy-xtrainInitial.npy',x_train.cpu().numpy())
        np.save('data/0/ALAEmGP-toy-ytrainInitial.npy',y_train.cpu().numpy())

        # === 3. 初始化模型 ===
        ae = Autoencoder(input_dim=1, latent_dim=2,hidden_dims=[6,]).to(device)
        gp = ParametricGP(latent_dim=2).to(device)
        model = JointModel(ae, gp).to(device)
        ae_copy = Autoencoder(input_dim=1, latent_dim=2, hidden_dims=[6,]).to(device)
        gp_copy = ParametricGP(latent_dim=2).to(device)
        model_copy = JointModel(ae_copy, gp_copy).to(device)

        model_copy.load_state_dict(copy.deepcopy(model.state_dict()))

        # === 4. 运行主循环 ===
        result = run_active_learning(
            model=model,
            x_train_init=x_train,
            y_train_init=y_train,
            x_cand=x_cand,
            y_cand=y_cand,
            x_ref=x_ref,
            x_test=x_test,
            y_test=y_test,
            T=20,
            save_plot_path="fig/0/ALAEmGP-toy",
            randomAC=False
        )

        np.save('data/0/ALAEmGP-toy-xtrainFinal.npy',result['final_x_train'].cpu().numpy())
        np.save('data/0/ALAEmGP-toy-ytrainFinal.npy',result['final_y_train'].cpu().numpy())
        np.save('data/0/ALAEmGP-toy-xtest.npy',result['x_test'].cpu().numpy())
        np.save('data/0/ALAEmGP-toy-yplot.npy',result['y_plot'].cpu().numpy())
        np.save('data/0/ALAEmGP-toy-testrmse.npy',result['rmse'])
        result_random = run_active_learning(
            model=model_copy,
            x_train_init=x_train,
            y_train_init=y_train,
            x_cand=x_cand,
            y_cand=y_cand,
            x_ref=x_ref,
            x_test=x_test,
            y_test=y_test,
            T=20,
            save_plot_path="fig/0/rALAEmGP-toy",
            randomAC=True
        )
        np.save('data/0/rALAEmGP-toy-xtrainFinal.npy',result_random['final_x_train'].cpu().numpy())
        np.save('data/0/rALAEmGP-toy-ytrainFinal.npy',result_random['final_y_train'].cpu().numpy())
        np.save('data/0/rALAEmGP-toy-yplot.npy',result_random['y_plot'].cpu().numpy())
        np.save('data/0/rALAEmGP-toy-testrmse.npy',result_random['rmse'])
    elif task == '2d':
        from utils.data_gen import generate_dataset_for_2d as generate_dataset_for_al_gp
        # === 2. 生成数据 ===
        x_train, y_train, x_test, y_test, x_ref, y_ref, x_cand, y_cand = generate_dataset_for_al_gp(
            N_train=50,
            N_test=500,
            N_ref=2000,
            N_cand=2000,
            noise_scale=0.,
            device=device,
            seed=None
        )
        np.save('data/0/ALAEmGP-2dSynthetic-xtrainInitial.npy',x_train.cpu().numpy())
        np.save('data/0/ALAEmGP-2dSynthetic-ytrainInitial.npy',y_train.cpu().numpy())

        # === 3. 初始化模型 ===
        ae = Autoencoder(input_dim=2, latent_dim=3,hidden_dims=[10,]).to(device)
        gp = ParametricGP(latent_dim=3).to(device)
        model = JointModel(ae, gp).to(device)
        ae_copy = Autoencoder(input_dim=2, latent_dim=3, hidden_dims=[10,]).to(device)
        gp_copy = ParametricGP(latent_dim=3).to(device)
        model_copy = JointModel(ae_copy, gp_copy).to(device)

        model_copy.load_state_dict(copy.deepcopy(model.state_dict()))

        # === 4. 运行主循环 ===
        result = run_active_learning(
            model=model,
            x_train_init=x_train,
            y_train_init=y_train,
            x_cand=x_cand,
            y_cand=y_cand,
            x_ref=x_ref,
            x_test=x_test,
            y_test=y_test,
            T=50,
            save_plot_path="fig/0/ALAEmGP-2dSynthetic",
            randomAC=False
        )

        np.save('data/0/ALAEmGP-2dSynthetic-xtrainFinal.npy',result['final_x_train'].cpu().numpy())
        np.save('data/0/ALAEmGP-2dSynthetic-ytrainFinal.npy',result['final_y_train'].cpu().numpy())
        np.save('data/0/ALAEmGP-2dSynthetic-xtest.npy',result['x_test'].cpu().numpy())
        np.save('data/0/ALAEmGP-2dSynthetic-testrmse.npy',result['rmse'])
        result_random = run_active_learning(
            model=model_copy,
            x_train_init=x_train,
            y_train_init=y_train,
            x_cand=x_cand,
            y_cand=y_cand,
            x_ref=x_ref,
            x_test=x_test,
            y_test=y_test,
            T=50,
            save_plot_path="fig/0/rALAEmGP-2dSynthetic",
            randomAC=True
        )
        np.save('data/0/rALAEmGP-2dSynthetic-xtrainFinal.npy',result_random['final_x_train'].cpu().numpy())
        np.save('data/0/rALAEmGP-2dSynthetic-ytrainFinal.npy',result_random['final_y_train'].cpu().numpy())
        np.save('data/0/rALAEmGP-2dSynthetic-testrmse.npy',result_random['rmse'])
    elif task == '3d':
        from utils.data_gen import generate_dataset_for_3d as generate_dataset_for_al_gp
        # === 2. 生成数据 ===
        x_train, y_train, x_test, y_test, x_ref, y_ref, x_cand, y_cand = generate_dataset_for_al_gp(
            N_train=50,
            N_test=500,
            N_ref=5000,
            N_cand=5000,
            noise_scale=0.,
            device=device,
            seed=None
        )
        np.save('data/0/ALAEmGP-2dSynthetic-xtrainInitial.npy',x_train.cpu().numpy())
        np.save('data/0/ALAEmGP-2dSynthetic-ytrainInitial.npy',y_train.cpu().numpy())

        # === 3. 初始化模型 ===
        ae = Autoencoder(input_dim=3, latent_dim=2,hidden_dims=[10]).to(device)
        gp = ParametricGP(latent_dim=2).to(device)
        model = JointModel(ae, gp).to(device)
        ae_copy = Autoencoder(input_dim=3, latent_dim=2, hidden_dims=[10]).to(device)
        gp_copy = ParametricGP(latent_dim=2).to(device)
        model_copy = JointModel(ae_copy, gp_copy).to(device)

        model_copy.load_state_dict(copy.deepcopy(model.state_dict()))

        # === 4. 运行主循环 ===
        result = run_active_learning(
            model=model,
            x_train_init=x_train,
            y_train_init=y_train,
            x_cand=x_cand,
            y_cand=y_cand,
            x_ref=x_ref,
            x_test=x_test,
            y_test=y_test,
            T=100,
            save_plot_path="fig/0/ALAEmGP-2dSynthetic",
            randomAC=False
        )

        np.save('data/0/ALAEmGP-2dSynthetic-xtest.npy',result['x_test'].cpu().numpy())
        np.save('data/0/ALAEmGP-2dSynthetic-xtrainFinal.npy',result['final_x_train'].cpu().numpy())
        np.save('data/0/ALAEmGP-2dSynthetic-ytrainFinal.npy',result['final_y_train'].cpu().numpy())
        np.save('data/0/ALAEmGP-2dSynthetic-testrmse.npy',result['rmse'])
        result_random = run_active_learning(
            model=model_copy,
            x_train_init=x_train,
            y_train_init=y_train,
            x_cand=x_cand,
            y_cand=y_cand,
            x_ref=x_ref,
            x_test=x_test,
            y_test=y_test,
            T=100,
            save_plot_path="fig/0/rALAEmGP-2dSynthetic",
            randomAC=True
        )
        np.save('data/0/rALAEmGP-2dSynthetic-xtrainFinal.npy',result_random['final_x_train'].cpu().numpy())
        np.save('data/0/rALAEmGP-2dSynthetic-ytrainFinal.npy',result_random['final_y_train'].cpu().numpy())
        np.save('data/0/rALAEmGP-2dSynthetic-testrmse.npy',result_random['rmse'])

    elif task == 'borehole':
        from utils.data_gen import generate_dataset_for_borehole as generate_dataset_for_al_gp
        latent_dim = 4
        x_train, y_train, x_test, y_test, x_ref, y_ref, x_cand, y_cand = generate_dataset_for_al_gp(
            N_train=50,
            N_test=1000,
            N_ref=5000,
            N_cand=1000,
            noise_scale=0.,
            device=device,
            seed=None
        )
        ae = Autoencoder(input_dim=8, latent_dim=latent_dim,hidden_dims=[30]).to(device)
        gp = ParametricGP(latent_dim=latent_dim).to(device)
        model = JointModel(ae, gp).to(device)
        ae_copy = Autoencoder(input_dim=8, latent_dim=latent_dim, hidden_dims=[30]).to(device)
        gp_copy = ParametricGP(latent_dim=latent_dim).to(device)
        model_copy = JointModel(ae_copy, gp_copy).to(device)

        model_copy.load_state_dict(copy.deepcopy(model.state_dict()))

        result = run_active_learning(
            model=model,
            x_train_init=x_train,
            y_train_init=y_train,
            x_cand=x_cand,
            y_cand=y_cand,
            x_ref=x_ref,
            x_test=x_test,
            y_test=y_test,
            T=200,
            save_plot_path="fig/0/ALAEmGP-Borehole",
            randomAC=False
        )
        np.save('data/0/ALAEmGP-Borehole-xtest.npy',result['x_test'].cpu().numpy())
        np.save('data/0/ALAEmGP-Borehole-xtrainFinal.npy',result['final_x_train'].cpu().numpy())
        np.save('data/0/ALAEmGP-Borehole-ytrainFinal.npy',result['final_y_train'].cpu().numpy())
        np.save('data/0/ALAEmGP-Borehole-testrmse.npy',result['rmse'])
        np.save('data/0/ALAEmGP-Borehole-testrmspesd.npy',result['rmspesd'])
        result_random = run_active_learning(
            model=model_copy,
            x_train_init=x_train,
            y_train_init=y_train,
            x_cand=x_cand,
            y_cand=y_cand,
            x_ref=x_ref,
            x_test=x_test,
            y_test=y_test,
            T=200,
            save_plot_path="fig/0/rALAEmGP-Borehole",
            randomAC=True
        )
        np.save('data/0/rALAEmGP-Borehole-xtrainFinal.npy',result_random['final_x_train'].cpu().numpy())
        np.save('data/0/rALAEmGP-Borehole-ytrainFinal.npy',result_random['final_y_train'].cpu().numpy())
        np.save('data/0/rALAEmGP-Borehole-testrmse.npy',result_random['rmse'])
        np.save('data/0/rALAEmGP-Borehole-testrmspesd.npy',result_random['rmspesd'])
    else:
        raise ValueError(f"Unknown task: {task}")
    


if __name__ == "__main__":
    main()