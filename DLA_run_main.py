from dla import Brownian_Tree
# from dla_19_05_05 import Brownian_Tree
# from dla_19_05_01 import Brownian_Tree
import numpy as np

# # Part 1 - plot the brownian motion walk for a small lattice
# lattice_size = 21  # square lattice size, >= 500
# k = 1.0  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)
# N = 1  # Number of particles for diffusion >=50000
#
# dla = Brownian_Tree(lattice_size=lattice_size, k=k, pad_size=10, log_dir_parent=f'part01_brownian_motion_of_particles/', include_video=True)
# dla.insert_new_particles(N=1, show_walk=True, log_interval=N)
# dla.update_and_log_particle_data()
# dla.print_brownian_tree(add_video=False)
# dla.writer.close()

# # Part 2 - plot the Brownian_Tree simulation for the below configuration, Brownian motion turned off for speedup
# lattice_size = 101  # square lattice size, >= 500
# k = 1.0  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)
# # k = 0.5  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)
# # k = 0.1  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)
# # k = 0.05  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)
# # k = 0.025  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)
# # k = 0.0125  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)
# # k = 0.00625  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)
# # k = 0.003125  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)
# # k = 0.0015625  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)
# # k_s = [1.0, 0.5, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625]
# N = lattice_size**2  # Number of particles for diffusion >=50000
# data_points = N
# log_interval = N//data_points
# print_interval = N//data_points
#
# # for k in k_s:
# dla = Brownian_Tree(lattice_size=lattice_size, k=k, pad_size=1, log_dir_parent=f'part02_stickiness_of_particles/', include_video=True)
# for i in range(N):
#     if dla.Brownian_Tree_possible:  # abort only when the diffusion locus is exhausted
#         dla.insert_new_particles(N=1, show_walk=False, log_interval=log_interval)
#         dla.lattice_boundary_reached = False  # Forcefully tell the dla to continue even after reaching diffusion locus
#     else:
#         dla.update_and_log_particle_data()
#         dla.print_brownian_tree(add_video=False)
#         break
#
#     if dla.num_particles % print_interval == 0:
#         dla.update_and_log_particle_data()
#         dla.print_brownian_tree(add_video=False)
# dla.update_and_log_particle_data()
# dla.print_brownian_tree(add_video=True)
# # dla.writer.export_scalars_to_json(f'{dla.writer.log_dir}/all_scalars.json')
# dla.writer.close()

# # Part 3 - plot the Brownian_Tree simulation for the below configuration, incremental lattice growth and plotting video as well
# import argparse
# import pandas as pd
#
# central_pandas_dataframe_name = None
# central_pandas_dataframe = None
#
# parser = argparse.ArgumentParser()
# parser.add_argument('-ls', '--lattice_sizes', nargs='+', help='list of length of square lattice', default=[101, 201, 301, 401, 501, 701, 901, 1001], type=int)
# parser.add_argument('-ks', '--k_s', nargs='+', help='list of stickiness for Brownian_Tree', default=[1.0, 0.5, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625], type=float)
# args = parser.parse_args()
#
# lattice_sizes = args.lattice_sizes  # square lattice size, >= 500
# k_s = args.k_s  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)
# # lattice_sizes = [101, 201, 301, 401, 501, 701, 901, 1001]  # square lattice size, >= 500
# # k_s = [1.0, 0.5, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625]  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)
#
# # Max_N = np.square(lattice_sizes)  # Number of particles for diffusion
# # Max_N = np.linspace(5e3, 5e4, 10, dtype=np.uint32)  # Number of particles for diffusion
# # # k_s = np.linspace(1e-3, 5e-2, 10)  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)
#
# Max_lattice_size = 1001
# N = Max_lattice_size ** 2
# lattice_steps = np.arange(3, np.ceil(np.sqrt(1001))+1) ** 2
# lattice_steps[lattice_steps % 2 == 0] += 1
# pad_size_dict = {1.0: 1, 0.5: 1, 0.1: 1, 0.05: 1, 0.025: 1, 0.0125: 1, 0.00625: 1, 0.003125: 1, 0.0015625: 1}
#
# exp_num = 0
# init_lattice_size = 0
# for k in k_s:
#     # for lattice_size in lattice_sizes:
#     # N = 5000
#     # num_datapoints = 2000
#     # pad_size = int(3 + np.log10(k))  # smaller the pad_size faster is the tree formation. But approximation increases slightly.
#     # pad_size = 2
#     # pad_size = 1
#     pad_size = pad_size_dict[k]  # get padding size
#     # for i, lattice_size in enumerate(lattice_steps):
#     lattice_size = 2 * init_lattice_size + 1
#     while lattice_size < 2 * Max_lattice_size:
#         dla = Brownian_Tree(lattice_size=lattice_size, k=k, pad_size=pad_size, base_img_path=f'tree_checkpoints/ls{init_lattice_size}/k{k}_checkpoint.png', log_dir_parent=f'part03_brownian_tree_generation/k{k}/')
#         # dla = Brownian_Tree(lattice_size=lattice_size, k=k, pad_size=pad_size, base_img_path='runs/19_05_04_11_18_19_leelavathi_k1.0_ls101_N10201/Brownian_Tree_Images/Brownian_Tree_k1.0_ls101_N1500.png')
#
#         # num_log_datapoints = N
#         # num_plot_datapoints = Max_lattice_size
#         # log_interval = N // num_log_datapoints
#         # plot_interval = N // num_plot_datapoints
#         log_interval = 1
#         plot_interval = int(lattice_size)
#         # log_interval = 1 if log_interval == 0 else log_interval
#
#         # if dla.lattice_boundary_reached:
#         #     break  # increase the lattice size and continue
#
#         # for i in range(N):
#         while dla.Brownian_Tree_possible and not dla.lattice_boundary_reached:
#             dla.insert_new_particles(N=1, show_walk=False)
#             # if dla.Brownian_Tree_possible:
#             #     if dla.num_particles % plot_interval == 0:
#             #         dla.insert_new_particles(N=1, show_walk=False)
#             #     else:
#             #         dla.insert_new_particles(N=1, show_walk=False)
#             # else:
#             #     dla.print_brownian_tree(fps=int(np.sqrt(dla.num_particles)), add_video=False)
#             #     dla.update_and_log_particle_data()
#             #     break
#
#             if dla.num_particles % plot_interval == 0:
#                 dla.print_brownian_tree(fps=int(np.sqrt(dla.num_particles)), add_video=False)
#             if dla.num_particles % log_interval == 0:
#                 dla.update_and_log_particle_data()
#         dla.print_brownian_tree(fps=int(np.sqrt(dla.num_particles)), add_video=True)
#         dla.update_and_log_particle_data()
#         # dla.writer.export_scalars_to_json(f'{dla.writer.log_dir}/all_scalars.json')
#         dla.writer.close()
#
#         # update the central pandas dataframe with the logged data.
#         if central_pandas_dataframe is None:
#             if central_pandas_dataframe_name is None:
#                 central_pandas_dataframe = dla.log_dataframe
#             else:
#                 central_pandas_dataframe = pd.read_csv(central_pandas_dataframe_name)
#         else:
#             central_pandas_dataframe = pd.concat([central_pandas_dataframe, dla.log_dataframe])
#
#         # store all the logs as a single csv file for model fitting. Note : Individual run specific dataframes are stored in the tensorboard logs directory.
#         central_pandas_dataframe.to_csv(f'central_pandas_dataframe_k{k}.csv')
#
#         # update lattice size to twice and continue
#         init_lattice_size = lattice_size
#         lattice_size = 2 * init_lattice_size + 1
#
#     # store all the logs as a single csv file for model fitting. Note : Individual run specific dataframes are stored in the tensorboard logs directory.
#     central_pandas_dataframe.to_csv('central_pandas_dataframe.csv')
#
#     # for N in Max_N:
#     #     dla = Brownian_Tree(lattice_size=lattice_size, k=k)
#     #     for i in range(N):
#     #         if dla.Brownian_Tree_possible:
#     #             dla.insert_particles(N=1)
#     #         else:
#     #             dla.printState()
#     #             dla.logSimData()
#     #             break
#     #
#     #         if i % print_interval == 0:
#     #             dla.printState()
#     #         if i % log_interval == 0:
#     #             dla.logSimData()
#     #     dla.writer.export_scalars_to_json(f'{dla.writer.log_dir}/all_scalars.json')
#     #     dla.writer.close()

# # Part 4 - Run all configurations in one go, for all data logging.
# import argparse
# import pandas as pd

# central_pandas_dataframe_name = None
# central_pandas_dataframe = None

# parser = argparse.ArgumentParser()
# parser.add_argument('-ls', '--lattice_sizes', nargs='+', help='list of length of square lattice', default=[101, 201, 301, 401, 501, 701, 901, 1001], type=int)
# parser.add_argument('-ks', '--k_s', nargs='+', help='list of stickiness for Brownian_Tree', default=[1.0, 0.5, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625], type=float)
# args = parser.parse_args()

# lattice_sizes = args.lattice_sizes  # square lattice size, >= 500
# k_s = args.k_s  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)
# # lattice_sizes = [101, 201, 301, 401, 501, 701, 901, 1001]  # square lattice size, >= 500
# # k_s = [1.0, 0.5, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625]  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)

# # Max_N = np.square(lattice_sizes)  # Number of particles for diffusion
# # Max_N = np.linspace(5e3, 5e4, 10, dtype=np.uint32)  # Number of particles for diffusion
# # # k_s = np.linspace(1e-3, 5e-2, 10)  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)

# Max_lattice_size = max(lattice_sizes)
# # N = Max_lattice_size ** 2
# # lattice_steps = np.arange(3, np.ceil(np.sqrt(1001))+1) ** 2
# # lattice_steps[lattice_steps % 2 == 0] += 1
# pad_size_dict = {1.0: 1, 0.5: 1, 0.1: 1, 0.05: 1, 0.025: 1, 0.0125: 1, 0.00625: 1, 0.003125: 1, 0.0015625: 1}

# exp_num = 0
# for k in k_s:
#     dla = Brownian_Tree(lattice_size=Max_lattice_size, k=k, pad_size=pad_size_dict[k],
#                         base_img_path=f'tree_checkpoints/ls{Max_lattice_size//2}/k{k}_checkpoint.png',
#                         log_dir_parent=f'part04_brownian_tree_generation/k{k}/')
#     log_interval = 1
#     plot_interval = int(Max_lattice_size)

#     while dla.Brownian_Tree_possible and not dla.lattice_boundary_reached:
#         dla.insert_new_particles(N=1, show_walk=False)
#         dla.update_and_log_particle_data()

#         if dla.num_particles % plot_interval == 0:
#             dla.print_brownian_tree(fps=int(np.sqrt(dla.num_particles)), add_video=False)
#     dla.print_brownian_tree(fps=int(np.sqrt(dla.num_particles)), add_video=False)
#     # dla.writer.export_scalars_to_json(f'{dla.writer.log_dir}/all_scalars.json')
#     dla.writer.close()

#     # update the central pandas dataframe with the logged data.
#     if central_pandas_dataframe is None:
#         if central_pandas_dataframe_name is None:
#             central_pandas_dataframe = dla.log_dataframe
#         else:
#             central_pandas_dataframe = pd.read_csv(central_pandas_dataframe_name)
#     else:
#         central_pandas_dataframe = pd.concat([central_pandas_dataframe, dla.log_dataframe])

#     # store all the logs as a single csv file for model fitting. Note : Individual run specific dataframes are stored in the tensorboard logs directory.
#     central_pandas_dataframe.to_csv(f'central_pandas_dataframe_k{k}.csv')

# # store all the logs as a single csv file for model fitting. Note : Individual run specific dataframes are stored in the tensorboard logs directory.
# central_pandas_dataframe.to_csv('central_pandas_dataframe.csv')

# Part 5 - Run all configurations in one go, for all data logging.
import argparse
import pandas as pd

central_pandas_dataframe_name = None
central_pandas_dataframe = None

parser = argparse.ArgumentParser()
parser.add_argument('-ls', '--lattice_sizes', nargs='+', help='list of length of square lattice', default=[101, 201, 301, 401, 501, 701, 901, 1001], type=int)
parser.add_argument('-ks', '--k_s', nargs='+', help='list of stickiness for Brownian_Tree', default=[1.0, 0.5, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625], type=float)
parser.add_argument('-base_img_path', help='Starting image path', default='tree_checkpoints/ls549/k1.0_checkpoint.png')
args = parser.parse_args()

lattice_sizes = args.lattice_sizes  # square lattice size, >= 500
k_s = args.k_s  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)
# lattice_sizes = [101, 201, 301, 401, 501, 701, 901, 1001]  # square lattice size, >= 500
# k_s = [1.0, 0.5, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625]  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)

# Max_N = np.square(lattice_sizes)  # Number of particles for diffusion
# Max_N = np.linspace(5e3, 5e4, 10, dtype=np.uint32)  # Number of particles for diffusion
# # k_s = np.linspace(1e-3, 5e-2, 10)  # stickiness (0.0 <= k <= 1), for us (1e-3, 5e-2)

Max_lattice_size = max(lattice_sizes)
# N = Max_lattice_size ** 2
# lattice_steps = np.arange(3, np.ceil(np.sqrt(1001))+1) ** 2
# lattice_steps[lattice_steps % 2 == 0] += 1
pad_size_dict = {1.0: 500, 0.5: 1, 0.1: 1, 0.05: 1, 0.025: 1, 0.0125: 1, 0.00625: 1, 0.003125: 1, 0.0015625: 1}

exp_num = 0
for k in k_s:
    dla = Brownian_Tree(lattice_size=Max_lattice_size, k=k, pad_size=pad_size_dict[k],
                        # base_img_path=f'tree_checkpoints/ls{Max_lattice_size}/k{k}_checkpoint.png',
                        base_img_path=args.base_img_path,
                        log_dir_parent=f'part05_brownian_tree_generation/k{k}/')
    log_interval = 1
    plot_interval = int(Max_lattice_size)

    while dla.Brownian_Tree_possible and not dla.lattice_boundary_reached:
        dla.insert_new_particles(N=1, show_walk=False)
        dla.update_and_log_particle_data()

        if dla.num_particles % plot_interval == 0:
            dla.print_brownian_tree(fps=int(np.sqrt(dla.num_particles)), add_video=False)
    dla.print_brownian_tree(fps=int(np.sqrt(dla.num_particles)), add_video=False)
    # dla.writer.export_scalars_to_json(f'{dla.writer.log_dir}/all_scalars.json')
    dla.writer.close()

    # update the central pandas dataframe with the logged data.
    if central_pandas_dataframe is None:
        if central_pandas_dataframe_name is None:
            central_pandas_dataframe = dla.log_dataframe
        else:
            central_pandas_dataframe = pd.read_csv(central_pandas_dataframe_name)
    else:
        central_pandas_dataframe = pd.concat([central_pandas_dataframe, dla.log_dataframe])

    # store all the logs as a single csv file for model fitting. Note : Individual run specific dataframes are stored in the tensorboard logs directory.
    central_pandas_dataframe.to_csv(f'central_pandas_dataframe_k{k}.csv')

# store all the logs as a single csv file for model fitting. Note : Individual run specific dataframes are stored in the tensorboard logs directory.
central_pandas_dataframe.to_csv('central_pandas_dataframe.csv')

# # Part 6 - Estimate fractal Dimension for image and also stickiness
# Check DLA_predict_stickiness_k.ipynb jupyter notebook for fitting the model to predict stickiness.
# TODO : Implement the same part inside DLA Brownian tree itself.
# dla = Brownian_Tree(lattice_size=1001, k=0.5, pad_size=2)
# dla.extract_base_image('runs/19_05_04_20_23_27_leelavathi_k1.0_ls101_N10201/Brownian_Tree_Images/Brownian_Tree_k1.0_ls101_N475.png')
# dla.find_fractal_dimension()


