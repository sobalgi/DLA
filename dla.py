# Author : Sourabh Balgi
# Institution : IISc, Bangalore
# Project : DLA - Diffusion Limited Aggregation.

import torch
import numpy as np
# import joblib
import matplotlib

matplotlib.use('Agg')
# matplotlib.use('WXAgg')
# matplotlib.use('GTKAgg')
# matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
# import matplotlib.animation as animation
# from matplotlib import animation
# from IPython import display
# %matplotlib inline
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
import socket
import sys
import os
import datetime
import torchvision.utils as vutils
import torch.nn.functional as F
import cv2
import numpy as np
import imutils
import pandas as pd


# Brownian tree object for creation of brownian tree
class Brownian_Tree():
    def __init__(self, lattice_size=501, k=0.5, pad_size=5, base_img_path=None, log_dir_parent='', include_video=False, log_interval=1):
        self.Brownian_Tree_possible = True  # Set the flag to indicate the possibility of Brownian_Tree
        self.lattice_boundary_reached = False  # Set the flag to indicate the diffusion locus along the border of Max lattice
        self.max_lattice_size = lattice_size  # create a square lattice for Brownian_Tree
        # self.lattice_origin = (self.Max_lattice_size // 2, self.Max_lattice_size // 2)
        self.pad_size = pad_size  # The next valid locations
        # self.base_state = torch.ones((1, 1))  # .type(torch.int16)
        self.pixel_val = 255  # or 1
        self.base_img_path = base_img_path
        self.base_state = torch.ones((1, 1), dtype=torch.uint8) * self.pixel_val  # .type(torch.int16)
        # self.base_state_origin = torch.tensor([0, 0])  # .type(torch.int16) # origin of base_lattice , (r, c) of the numpy array
        self.base_state_origin = np.array(
            [0, 0])  # .type(torch.int16) # origin of base_lattice , (r, c) of the numpy array
        self.base_state_size = self.base_state.shape[0]
        self.coeffs = np.zeros(2)
        self.include_video = include_video
        # curr_neighbours = (self.relative_neighbours + cur_pos) % self.state.shape[0]  # to get current neighbours of cur_pos in the state lattice

        # default values for error handling just for the time being. Needs investigation.
        self.a_bbc = 0  # -1
        self.a_bbe = 0  # -1
        self.cv_centroid_x = 0  # -1
        self.cv_centroid_y = 0  # -1
        self.cv_leftmost_x = 0  # -1
        self.cv_leftmost_y = 0  # -1
        self.cv_rightmost_x = 0  # -1
        self.cv_rightmost_y = 0  # -1
        self.cv_topmost_x = 0  # -1
        self.cv_topmost_y = 0  # -1
        self.cv_bottommost_x = 0  # -1
        self.cv_bottommost_y = 0  # -1
        self.cv_contour_area = 0  # -1
        self.cv_contour_perimeter = 0  # -1
        self.cv_contour_k =  False  # -1
        self.cv_boundingRect_x = 0  # -1
        self.cv_boundingRect_y = 0  # -1
        self.cv_boundingRect_w = 0  # -1
        self.cv_boundingRect_h = 0  # -1
        self.cv_boundingRect_area = 0  # -1
        self.cv_hull_area = 0  # -1
        self.cv_extent = 1  # -1
        self.cv_solidity = 1  # -1
        self.cv_minEnclosingCircle_center_x = 0  # -1
        self.cv_minEnclosingCircle_center_y = 0  # -1
        self.cv_minEnclosingCircle_r = 0  # -1
        self.cv_equi_radius = 0  # -1
        self.coeffs[0] = lattice_size ** 2  # -1
        self.coeffs[1] = 0  # -1

        # if base_img_path is not None:
        #     self.extract_base_image(base_img_path=base_img_path)  # extract base image from the given image of any lattice size
        self.extract_base_image(base_img_path=base_img_path)  # extract base image from the given image of any lattice size

        # self.state = F.pad(self.base_state, pad=(self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='constant', value=0)
        # self.state_size = self.state.shape[0]
        # self.state_origin = self.base_state_origin + self.pad_size  # origin of base_lattice
        # self.num_particles = np.array(np.where(self.state.numpy())).transpose().shape[0]
        # self.walk_iter = 1
        # self.diff_area = 'sq_peri'
        # self.relative_neighbours = np.array(np.where(F.pad(torch.zeros((1, 1)), (1, 1, 1, 1), value=1).numpy())).transpose() - 1  # get the relative neighbour index as a numpy array
        #
        # self.hh_bbe = self.hw_bbe = self.base_state_size//2
        # # self.hw_bbe = max(abs(self.state_origin[1] - self.state_origin[1]), abs(self.state_origin[1] - self.state_origin[1]))
        # # self.hh_bbe = max(abs(self.state_origin[0] - self.state_origin[0]), abs(self.state_origin[0] - self.state_origin[0]))
        # self.l_bbe, self.r_bbe = self.t_bbe, self.b_bbe = np.array([-self.hw_bbe, self.hh_bbe]) + self.state_origin
        # # self.l_bbe, self.r_bbe = self.t_bbe, self.b_bbe = torch.Tensor([-self.hw_bbe, self.hh_bbe]) + self.state_origin
        # # self.r_bbc = torch.Tensor([self.hw_bbe, self.hh_bbe]).norm()
        # self.r_bbc = torch.Tensor([self.hw_bbe, self.hh_bbe]).norm()
        # self.a_bbc = np.pi * (self.r_bbc+1)**2
        # # self.a_bbe = 4.0 * self.hw_bbe * self.hh_bbe
        # self.a_bbe = (2.0 * self.hw_bbe + 1) * (2.0 * self.hh_bbe + 1)
        # self.update_diffusion_locus()

        self.k = k  # set the stickiness probability

        machinename = socket.gethostname()
        # hostname = timestamp = datetime.datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
        # hostname = timestamp = datetime.datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
        hostname = timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")

        absolute_pyfile_path = os.path.abspath(sys.argv[0])

        absolute_base_path = os.path.dirname(absolute_pyfile_path)

        log_dir = os.path.join(absolute_base_path, 'runs',
                               f'{log_dir_parent}{timestamp}_{machinename}_k{self.k}_ls{self.max_lattice_size}_pad{self.pad_size}')

        # self.writer = SummaryWriter('')
        self.writer = SummaryWriter(log_dir=log_dir)

        # storing the images for training and testing later
        os.system(f'mkdir -p {self.writer.log_dir}/Brownian_Tree_Images')

        self.vid_list = []  # Initialize a empty list to
        pad_base_lattice = (
                                       self.max_lattice_size - self.base_state_size) // 2  # find padding size to be done to get full lattice from the base lattice
        if pad_base_lattice < 0:
            pad_base_lattice = 0  # No padding if base lattice is already equal to dev lattice
            pass  # without padding
        elif pad_base_lattice == 0:
            # self.vid_list.append(F.pad(self.base_state, pad=(1, 1, 1, 1), mode='constant', value=0).clone())
            if self.include_video:
                self.vid_list.append(self.base_state.clone())  # append list
            else:  # overwrite last
                self.vid_list = self.base_state.clone()  # Initialize a empty list to
            # self.vid_list.append(self.base_state.clone())
        else:
            if self.include_video:
                self.vid_list.append(
                    F.pad(self.base_state, pad=(pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice),
                          mode='constant', value=0).clone())
            else:  # overwrite last
                # self.vid_list = F.pad(self.base_state, pad=(pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice), mode='constant', value=0).clone()  # Initialize a empty list to
                self.vid_list = self.base_state.clone()  # Initialize a empty list to
            # self.vid_list.append(F.pad(self.base_state, pad=(pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice), mode='constant', value=0).clone())

        self.cols = ['00_parameters/log_dir', '00_parameters/N', '00_parameters/k', '00_parameters/max_ls',
                     '00_parameters/pad_size',
                     '01_bounding_box_data/bounding_circle', '01_bounding_box_data/bounding_square',
                     '01_bounding_box_data/bounding_square_height', '01_bounding_box_data/bounding_square_width',
                     '02_simulation_data/num_diff_locus_points', '02_simulation_data/num_brownian_motion_steps',
                     '03_cv_data/cv_centroid_x', '03_cv_data/cv_centroid_y',
                     '03_cv_data/cv_leftmost_x', '03_cv_data/cv_leftmost_y',
                     '03_cv_data/cv_rightmost_x', '03_cv_data/cv_rightmost_y',
                     '03_cv_data/cv_topmost_x', '03_cv_data/cv_topmost_y',
                     '03_cv_data/cv_bottommost_x', '03_cv_data/cv_bottommost_y',
                     '03_cv_data/cv_contour_area', '03_cv_data/cv_contour_perimeter',
                     '03_cv_data/cv_contour_k', '03_cv_data/cv_boundingRect_x',
                     '03_cv_data/cv_boundingRect_y', '03_cv_data/cv_boundingRect_w',
                     '03_cv_data/cv_boundingRect_h', '03_cv_data/cv_boundingRect_area',
                     '03_cv_data/cv_hull_area', '03_cv_data/cv_extent',
                     '03_cv_data/cv_solidity', '03_cv_data/cv_minEnclosingCircle_center_x',
                     '03_cv_data/cv_minEnclosingCircle_center_y', '03_cv_data/cv_minEnclosingCircle_r',
                     '03_cv_data/cv_equi_radius', '04_fractal_dimension/polyfit_coeff0',
                     '04_fractal_dimension/polyfit_coeff1'
                     ]

        # # default values for error handling just for the time being. Needs investigation.
        # self.a_bbc = -1
        # self.a_bbe = -1
        # self.cv_centroid_x = -1
        # self.cv_centroid_y = -1
        # self.cv_leftmost_x = -1
        # self.cv_leftmost_y = -1
        # self.cv_rightmost_x = -1
        # self.cv_rightmost_y = -1
        # self.cv_topmost_x = -1
        # self.cv_topmost_y = -1
        # self.cv_bottommost_x = -1
        # self.cv_bottommost_y = -1
        # self.cv_contour_area = -1
        # self.cv_contour_perimeter = -1
        # self.cv_contour_k = -1
        # self.cv_boundingRect_x = -1
        # self.cv_boundingRect_y = -1
        # self.cv_boundingRect_w = -1
        # self.cv_boundingRect_h = -1
        # self.cv_boundingRect_area = -1
        # self.cv_hull_area = -1
        # self.cv_extent = -1
        # self.cv_solidity = -1
        # self.cv_minEnclosingCircle_center_x = -1
        # self.cv_minEnclosingCircle_center_y = -1
        # self.cv_minEnclosingCircle_r = -1
        # self.cv_equi_radius = -1
        # self.coeffs[0] = -1
        # self.coeffs[1] = -1

        self.log_dict = {'00_parameters/log_dir': self.writer.log_dir, '00_parameters/N': self.num_particles,
                         '00_parameters/k': self.k,
                         '00_parameters/max_ls': self.max_lattice_size, '00_parameters/pad_size': self.pad_size,
                         '01_bounding_box_data/bounding_circle': self.a_bbc,
                         '01_bounding_box_data/bounding_square': self.a_bbe,
                         '01_bounding_box_data/bounding_square_height': self.hh_bbe * 2 + 1,
                         '01_bounding_box_data/bounding_square_width': self.hh_bbe * 2 + 1,
                         '02_simulation_data/num_diff_locus_points': len(self.next_locations),
                         '02_simulation_data/num_brownian_motion_steps': self.walk_iter,
                         '03_cv_data/cv_centroid_x': self.cv_centroid_x, '03_cv_data/cv_centroid_y': self.cv_centroid_y,
                         '03_cv_data/cv_leftmost_x': self.cv_leftmost_x, '03_cv_data/cv_leftmost_y': self.cv_leftmost_y,
                         '03_cv_data/cv_rightmost_x': self.cv_rightmost_x,
                         '03_cv_data/cv_rightmost_y': self.cv_rightmost_y,
                         '03_cv_data/cv_topmost_x': self.cv_topmost_x, '03_cv_data/cv_topmost_y': self.cv_topmost_y,
                         '03_cv_data/cv_bottommost_x': self.cv_bottommost_x,
                         '03_cv_data/cv_bottommost_y': self.cv_bottommost_y,
                         '03_cv_data/cv_contour_area': self.cv_contour_area,
                         '03_cv_data/cv_contour_perimeter': self.cv_contour_perimeter,
                         '03_cv_data/cv_contour_k': self.cv_contour_k,
                         '03_cv_data/cv_boundingRect_x': self.cv_boundingRect_x,
                         '03_cv_data/cv_boundingRect_y': self.cv_boundingRect_y,
                         '03_cv_data/cv_boundingRect_w': self.cv_boundingRect_w,
                         '03_cv_data/cv_boundingRect_h': self.cv_boundingRect_h,
                         '03_cv_data/cv_boundingRect_area': self.cv_boundingRect_area,
                         '03_cv_data/cv_hull_area': self.cv_hull_area, '03_cv_data/cv_extent': self.cv_extent,
                         '03_cv_data/cv_solidity': self.cv_solidity,
                         '03_cv_data/cv_minEnclosingCircle_center_x': self.cv_minEnclosingCircle_center_x,
                         '03_cv_data/cv_minEnclosingCircle_center_y': self.cv_minEnclosingCircle_center_y,
                         '03_cv_data/cv_minEnclosingCircle_r': self.cv_minEnclosingCircle_r,
                         '03_cv_data/cv_equi_radius': self.cv_equi_radius,
                         '04_fractal_dimension/polyfit_coeff0': self.coeffs[0],
                         '04_fractal_dimension/polyfit_coeff1': self.coeffs[1]
                         }
        self.log_dataframe = pd.DataFrame(columns=self.cols)
        # self.log_data = self.log_data.append({'frame': str(i), 'count': i}, ignore_index=True)

        self.perform_image_processing_and_extract_cv_data()
        self.update_and_log_particle_data()
        self.print_brownian_tree()

        self.writer.add_text('Brownian_tree_log_text',
                             f'k={self.k} | N={self.num_particles} | ps={self.pad_size} | bb_s={self.base_state_size} | bb_r={self.r_bbc}',
                             self.num_particles)  # log in tensorboard
        print(
            f'k={self.k} | N={self.num_particles} | ps={self.pad_size} | bb_s={self.base_state_size} | bb_r={self.r_bbc}')  # print on console

    def update_diffusion_locus(self, cur_pos=None):
        '''
            Get the updated diffusion locus after locus remapping for faster simulations
        '''

        if cur_pos is not None:  # remove the cur_pos from the lattice diffusion locations
            self.next_locations = np.array(list(
                filter(lambda x: tuple(x) != tuple(cur_pos), self.next_locations)))  # get only empty adjacent points
            return

        if self.diff_area == 'sq_peri':
            # get the boundaries of the square lattice for uniformly sampling the next particle for diffusion
            z = np.pad(np.zeros(list(self.state_origin * 2 - 1)), pad_width=1, mode='constant',
                       constant_values=self.pixel_val)
            # z = F.pad(torch.zeros(list((self.state_origin * 2 - 1).numpy())), pad=(1, 1, 1, 1), mode='constant', value=1).numpy()
            next_locations = np.array(np.where(z)).transpose()
        elif self.diff_area == 'cir_peri':
            points = {}

            y1, y2 = self.ycenter, self.ycenter

            for x in range(self.xcenter - self.radius - 10, self.xcenter + 1):
                while (y1 - self.ycenter) ** 2 + (x - self.xcenter) ** 2 <= (self.radius) ** 2:
                    y1 -= 1
                k = y1
                while (k - self.ycenter) ** 2 + (x - self.xcenter) ** 2 <= (self.radius + 1) ** 2:
                    if self.isValid((x, k)):
                        points[(x, k)] = True
                    k -= 1

                while (y2 - self.ycenter) ** 2 + (x - self.xcenter) ** 2 <= (self.radius) ** 2:
                    y2 += 1
                k = y2
                while (k - self.ycenter) ** 2 + (x - self.xcenter) ** 2 <= (self.radius + 1) ** 2:
                    if self.isValid((x, k)):
                        points[(x, k)] = True
                    k += 1

            y1, y2 = self.ycenter, self.ycenter
            for x in range(self.xcenter + self.radius + 10, self.xcenter, -1):
                while (y1 - self.ycenter) ** 2 + (x - self.xcenter) ** 2 <= (self.radius) ** 2:
                    y1 -= 1
                k = y1
                while (k - self.ycenter) ** 2 + (x - self.xcenter) ** 2 <= (self.radius + 1) ** 2:
                    if self.isValid((x, k)):
                        points[(x, k)] = True
                    k -= 1

                while (y2 - self.ycenter) ** 2 + (x - self.xcenter) ** 2 <= (self.radius) ** 2:
                    y2 += 1
                k = y2
                while (k - self.ycenter) ** 2 + (x - self.xcenter) ** 2 <= (self.radius + 1) ** 2:
                    if self.isValid((x, k)):
                        points[(x, k)] = True
                    k += 1

            if any(map(lambda x: self.state[x] == self.pixel_val, points.keys())):
                print('can spawn at marked')

            next_locations = list(points.keys())
        else:
            pass
        self.next_locations = next_locations

    def print_brownian_tree(self, fps=1, add_video=False, is_checkpoint=False):
        '''
            Plot the current brownian tree and save checkpoints.
        '''
        # self.vid_list.append(self.base_state.clone())
        # plt.imshow(1 - self.state, cmap='gist_gray_r', vmin=0, vmax=1)
        # plt.imshow(self.pixel_val - self.vid_list[-1].squeeze(dim=0), cmap='gist_gray_r', vmin=0, vmax=1)
        if self.include_video:
            plt.imshow(self.pixel_val - self.vid_list[-1], cmap='gist_gray_r', vmin=0, vmax=1)
        else:
            plt.imshow(self.pixel_val - self.vid_list, cmap='gist_gray_r', vmin=0, vmax=1)

        plt.gcf().show()
        plt.title(f'LS={self.max_lattice_size}, # Particles={self.num_particles}')
        self.writer.add_figure(f'Images_Brownian_Tree', plt.gcf(), global_step=self.num_particles)
        # self.writer.add_image(f'Brownian_Tree_k{self.k}_ls{self.lattice_size}_N{self.num_particles}', global_step=self.num_particles)  # Tensor
        # vutils.save_image(torch.Tensor(self.state).unsqueeze(dim=0),
        #                   f'{self.writer.log_dir}/Brownian_Tree_images/Brownian_Tree_k{self.k}_ls{self.max_lattice_size}_N{self.num_particles}.png')
        if is_checkpoint:
            # if self.lattice_boundary_reached and is_checkpoint:
            if self.include_video:
                cv2.imwrite(
                    f'{self.writer.log_dir}/Brownian_Tree_Images/Brownian_Tree_k{self.k}_ls{self.base_state_size}_N{self.num_particles}_reached.png',
                    self.vid_list[-1].numpy())
            else:
                cv2.imwrite(
                    f'{self.writer.log_dir}/Brownian_Tree_Images/Brownian_Tree_k{self.k}_ls{self.base_state_size}_N{self.num_particles}_reached.png',
                    self.vid_list.numpy())
            # vutils.save_image(self.vid_list[-1].unsqueeze(dim=0), '{}/Brownian_Tree_Images/Brownian_Tree_k{}_ls{}_N{}_reached.png'.format(self.writer.log_dir, self.k, self.max_lattice_size, self.num_particles), normalize=True)
            # vutils.save_image(vutils.make_grid(self.vid_list[-1].unsqueeze(dim=0)), f'{self.writer.log_dir}/Brownian_Tree_Images/Brownian_Tree_k{self.k}_ls{self.max_lattice_size}_N{self.num_particles}_reached.png', normalize=True)
            # vutils.save_image(self.vid_list[-1].unsqueeze(dim=0), f'{self.writer.log_dir}/Brownian_Tree_Images/Brownian_Tree_k{self.k}_ls{self.max_lattice_size}_N{self.num_particles}_reached.png', normalize=True)
            # os.system(f'mkdir -p tree_checkpoints/ls{self.base_state_size}')
            os.system(f'mkdir -p tree_checkpoints/k{self.k}')
            # os.system(f'mkdir -p tree_checkpoints/ls{self.max_lattice_size*2 + 1}')
            os.system(
                f'cp {self.writer.log_dir}/Brownian_Tree_Images/Brownian_Tree_k{self.k}_ls{self.base_state_size}_N{self.num_particles}_reached.png tree_checkpoints/k{self.k}/k{self.k}_ls{self.base_state_size}_checkpoint.png')
            # os.system(f'cp {self.writer.log_dir}/Brownian_Tree_Images/Brownian_Tree_k{self.k}_ls{self.max_lattice_size}_N{self.num_particles}_reached.png tree_checkpoints/ls{self.max_lattice_size*2 + 1}/k{self.k}_checkpoint.png')
        else:
            if self.include_video:
                cv2.imwrite(
                    f'{self.writer.log_dir}/Brownian_Tree_Images/Brownian_Tree_k{self.k}_ls{self.max_lattice_size}_N{self.num_particles}_good.png',
                    self.vid_list[-1].numpy())
            else:
                cv2.imwrite(
                    f'{self.writer.log_dir}/Brownian_Tree_Images/Brownian_Tree_k{self.k}_ls{self.max_lattice_size}_N{self.num_particles}_good.png',
                    self.vid_list.numpy())
            # vutils.save_image(vutils.make_grid(self.vid_list[-1].unsqueeze(dim=0)), f'{self.writer.log_dir}/Brownian_Tree_Images/Brownian_Tree_k{self.k}_ls{self.max_lattice_size}_N{self.num_particles}_good.png', normalize=True)
        # self.writer.add_figure(f'Brownian_Tree_k{self.k}', plt.gcf(), global_step=self.num_particles)

        # if is_checkpoint:
        #     os.system(f'mkdir -p tree_checkpoints/ls{self.max_lattice_size}')
        #     os.system(f'cp {self.writer.log_dir}/Brownian_Tree_Images/Brownian_Tree_k{self.k}_ls{self.max_lattice_size}_N{self.num_particles}_good.png tree_checkpoints/ls{self.max_lattice_size}/k{self.k}_checkpoint.png')
        #     os.system(f'cp {self.writer.log_dir}/Brownian_Tree_Images/Brownian_Tree_k{self.k}_ls{self.max_lattice_size}_N{self.num_particles}_good.png tree_checkpoints/ls{int(np.sqrt(self.max_lattice_size)**2)}/k{self.k}_checkpoint.png')

        # display.display(plt.gcf())
        # display.clear_output(wait=True)
        # time.sleep(1)
        if self.include_video:
            if add_video:
                # for vid in self.vid_list:
                #     print(vid.shape)
                self.writer.add_video('Videos_Brownian_Tree',
                                      vid_tensor=torch.stack(self.vid_list).unsqueeze_(1).unsqueeze_(0),
                                      fps=int(np.sqrt(self.num_particles)), global_step=self.num_particles)

        self.writer.add_text('Brownian_tree_log_text',
                             f'k={self.k} | N={self.num_particles} | ps={self.pad_size} | bb_s={self.base_state_size} | bb_r={self.r_bbc}',
                             self.num_particles)  # log in tensorboard
        print(
            f'k={self.k} | N={self.num_particles} | ps={self.pad_size} | bb_s={self.base_state_size} | bb_r={self.r_bbc}')  # print on console

    def insert_new_particles(self, N=1, show_walk=False, log_interval=10):
        '''
            Diffuse new particles into the lattice from the selected diffusion locus
        '''
        # fig = plt.figure()
        count = 0
        while count < N:

            if not self.Brownian_Tree_possible:  # Continue if only there are points in the diffusion locus.
                return

            # is_checkpoint = False
            # if self.base_state == self.state[self.t_bbe:self.b_bbe+1, self.l_bbe:self.r_bbe+1]:
            #     pass
            # else:
            #     a = 0
            #     raise ValueError

            cur_pos = self.get_new_particle()  # Get a new particle for diffusion
            next_pos = cur_pos
            if cur_pos[0] == -1 or cur_pos[
                1] == -1:  # np.all(cur_pos == np.array([-1, -1]))  # cur_pos[0] == -1 or cur_pos[1] == -1  # cur_pos == (-1, -1)
                self.Brownian_Tree_possible = False
                print(
                    f'k={self.k}, ls={self.max_lattice_size}, pad_size={self.pad_size}, N={self.num_particles} : Cannot add any new particles from the diffusion location as it is full. N={self.num_particles}')
                self.writer.add_text('Max_N_limit',
                                     f'k={self.k}, ls={self.max_lattice_size}, pad_size={self.pad_size}, N={self.num_particles} : Cannot add any new particles from the diffusion location as it is full. N={self.num_particles}')
                break

            # Increment the counters after adding the particle
            count += 1
            self.num_particles += 1

            if show_walk and (count) % log_interval == 0:
                vid_list = []
                # data = np.copy(self.state)  # create a copy for printing the walk
                data = self.state.clone()  # create a copy for printing the walk
                print(f'count={count}')
                data[tuple(cur_pos)] = self.pixel_val
                if self.include_video:
                    vid_list.append(data.clone())  # append list
                else:  # overwrite last
                    vid_list = data.clone()  # Initialize a empty list to
                # vid_list.append(data.clone())
                os.system(f'mkdir -p {self.writer.log_dir}/Brownian_Motion_Images')

            walk_iter = 0

            while not self.current_particle_sticks(cur_pos):  # if not stuck, keep walking
                next_pos = self.get_next_position(cur_pos)  # Get a new position after a random walk
                # basic_steps = np.abs(next_pos-cur_pos).sum()
                # if basic_steps == 1 or basic_steps == 2:
                #     continue
                # else:
                #     raise ValueError

                if show_walk and (count) % log_interval == 0:
                    print(f'#Walk={walk_iter}, current_pos={cur_pos}, next_pos={next_pos}')
                    data[tuple(cur_pos)] = self.pixel_val
                    plt.imshow(self.pixel_val - data, cmap='gist_gray_r', vmin=0, vmax=1)
                    #         plt.gcf().show()
                    plt.title(f'Brownian motion step={walk_iter} of particle {self.num_particles} ')

                    # plt.gcf().show()
                    # display.display(plt.gcf())
                    # display.clear_output(wait=True)
                    # self.state[cur_pos] = 0
                    # time.sleep(1)
                    # plt.gcf().show()
                    # self.writer.add_figure('brownian_motion_N_{}'.format(self.num_particles+1), plt.gcf(), global_step=walk_iter)
                    # self.writer.add_figure(f'Brownian_motion_image_N{self.num_particles + 1}', plt.gcf(), global_step=walk_iter)
                    self.writer.add_figure('Images_Brownian_Motion_{}'.format(self.num_particles), plt.gcf(),
                                           global_step=walk_iter)
                    self.writer.add_text('Brownian_motion_walk_text',
                                         f'#Walk={walk_iter}, current_pos={cur_pos}, next_pos={next_pos}', walk_iter)

                    # data[cur_pos] = 0
                    data[tuple(cur_pos)] = 0
                    # data[next_pos] = 1
                    data[tuple(next_pos)] = self.pixel_val
                    if self.include_video:
                        vid_list.append(data.clone())  # append list
                        cv2.imwrite(
                            f'{self.writer.log_dir}/Brownian_Motion_Images/Brownian_motion_k{self.k}_ls{self.max_lattice_size}_N{self.num_particles}_walk{walk_iter}.png',
                            vid_list[-1].numpy())
                    else:  # overwrite last
                        vid_list = data.clone()  # Initialize a empty list to
                        cv2.imwrite(
                            f'{self.writer.log_dir}/Brownian_Motion_Images/Brownian_motion_k{self.k}_ls{self.max_lattice_size}_N{self.num_particles}_walk{walk_iter}.png',
                            vid_list.numpy())
                    # vid_list.append(data.clone())
                    # cv2.imwrite(f'{self.writer.log_dir}/Brownian_Motion_Images/Brownian_motion_k{self.k}_ls{self.max_lattice_size}_N{self.num_particles}_walk{walk_iter}.png', vid_list[-1].numpy())

                    # plt.gcf().canvas.draw()
                    # X = torch.from_numpy(np.array(plt.gcf().canvas.renderer._renderer)[:, :, :-1]).permute(2, 0, 1)

                #                     self.printState()

                if np.all(next_pos == cur_pos):  # stuck in bounded area  # next_pos == cur_pos
                    break  # so make it stuck to save time with k < 1.0 case

                cur_pos = next_pos
                walk_iter += 1

            # add the current stuck particle to dev lattice
            # np.array(np.where(self.state.numpy())).transpose()
            self.state[tuple(cur_pos)] = self.pixel_val
            self.walk_iter = walk_iter

            if show_walk and (count) % log_interval == 0:
                print(f'current_pos={cur_pos}, next_pos={next_pos}')
                # data[cur_pos] = 1
                data[tuple(cur_pos)] = self.pixel_val
                plt.imshow(self.pixel_val - data, cmap='gist_gray_r', vmin=0, vmax=1)
                #         plt.gcf().show()
                plt.title(f'Brownian motion step={walk_iter} of particle {self.num_particles} ')

                # plt.gcf().show()
                # display.display(plt.gcf())
                # display.clear_output(wait=True)
                # self.state[cur_pos] = 0
                # time.sleep(1)
                # plt.gcf().show()
                # self.writer.add_figure(f'Brownian_motion_image_N{self.num_particles + 1}', plt.gcf(), global_step=walk_iter)
                self.writer.add_figure('Images_Brownian_Motion_{}'.format(self.num_particles), plt.gcf(),
                                       global_step=walk_iter)
                self.writer.add_text('Brownian_motion_walk_text',
                                     f'#Walk={walk_iter}, current_pos={cur_pos}, next_pos={next_pos}', walk_iter)

                # vid_list.append(data.copy())
                if self.include_video:
                    vid_list.append(data.clone())  # append list
                    cv2.imwrite(
                        f'{self.writer.log_dir}/Brownian_Motion_Images/Brownian_motion_k{self.k}_ls{self.max_lattice_size}_N{self.num_particles}_walk{walk_iter}.png',
                        vid_list[-1].numpy())
                    # self.writer.add_video(f'Brownian_motion_video_N{self.num_particles + 1}', vid_tensor=torch.from_numpy(np.stack(vid_list)).unsqueeze_(1).unsqueeze(0))
                    self.writer.add_video('Videos_Brownian_Motion',
                                          vid_tensor=torch.stack(vid_list).unsqueeze_(1).unsqueeze_(0),
                                          fps=int(np.sqrt(len(vid_list))), global_step=self.num_particles)
                else:  # overwrite last
                    vid_list = data.clone()  # Initialize a empty list to
                    cv2.imwrite(
                        f'{self.writer.log_dir}/Brownian_Motion_Images/Brownian_motion_k{self.k}_ls{self.max_lattice_size}_N{self.num_particles}_walk{walk_iter}.png',
                        vid_list.numpy())
                # vid_list.append(data.clone())
                # cv2.imwrite(f'{self.writer.log_dir}/Brownian_Motion_Images/Brownian_motion_k{self.k}_ls{self.max_lattice_size}_N{self.num_particles}_walk{walk_iter}.png', vid_list[-1].numpy())

                # plt.gcf().canvas.draw()

            # print('*'*55)
            # print(self.base_state_size, self.state_size)
            # print(self.base_state.shape, self.state.shape)
            # print(self.base_state_origin, self.state_origin)
            # print(self.base_state)
            # print(self.state)
            # print('*'*55)

            # after adding if the size of base lattice is increased to update the base lattice and related information
            if np.any(np.abs(
                    cur_pos - self.state_origin) >= self.hw_bbe):  # np.any(np.abs(cur_pos - self.state_origin) >= self.hw_bbe)  # np.abs(cur_pos - self.state_origin) >= self.hw_bbe * 2 + 1
                # update base lattice
                if self.hw_bbe != self.max_lattice_size // 2:
                    self.hw_bbe += 1
                    self.hh_bbe += 1
                    save_lattice_checkpoint = True
                self.l_bbe, self.r_bbe = self.t_bbe, self.b_bbe = np.array(
                    [-self.hw_bbe, self.hh_bbe]) + self.state_origin
                self.base_state = self.state[self.t_bbe:self.b_bbe + 1,
                                  self.l_bbe:self.r_bbe + 1].clone()  # update the dev lattice by one
                self.base_state_size = self.base_state.shape[0]
                # self.r_bbc = torch.Tensor([self.hw_bbe, self.hh_bbe]).norm()
                self.r_bbc = torch.Tensor([self.hw_bbe, self.hh_bbe]).norm().numpy()
                self.a_bbc = np.pi * (self.r_bbc + 1) ** 2
                # self.a_bbe = 4.0 * self.hw_bbe * self.hh_bbe
                self.a_bbe = (2.0 * self.hw_bbe + 1) * (2.0 * self.hh_bbe + 1)

                if self.state_size < self.max_lattice_size:  # update base lattice and since not max lattice reached, pad state by 1
                    self.state = F.pad(self.state, pad=(1, 1, 1, 1), mode='constant',
                                       value=0)  # update the dev lattice by one
                    # np.array(np.where(self.state.numpy())).transpose()
                    self.state_origin += 1
                    self.base_state_origin += 1
                    self.l_bbe, self.r_bbe = self.t_bbe, self.b_bbe = np.array(
                        [-self.hw_bbe, self.hh_bbe]) + self.state_origin
                    self.state_size = self.state.shape[0]
                    self.update_diffusion_locus()
                else:  # point inside base lattice so no problem. just get the updated base lattice.
                    self.base_state = self.state[self.t_bbe:self.b_bbe + 1,
                                      self.l_bbe:self.r_bbe + 1].clone()  # update the dev lattice by one
                    if not self.lattice_boundary_reached and np.any(np.abs(
                            cur_pos - self.state_origin) >= self.max_lattice_size // 2):  # max lattice reached, so set the flag and store the checkpoint for the first time, update the diffusion locus and continue
                        self.lattice_boundary_reached = True

                        self.r_bbc = torch.Tensor([self.hw_bbe, self.hh_bbe]).norm().numpy()
                        self.a_bbc = np.pi * (self.r_bbc + 1) ** 2
                        # self.a_bbe = 4.0 * self.hw_bbe * self.hh_bbe
                        self.a_bbe = (2.0 * self.hw_bbe + 1) * (2.0 * self.hh_bbe + 1)

                        # Pad base lattice to max lattice for video animation of Brownian tree
                        pad_base_lattice = (self.max_lattice_size - self.base_state_size) // 2
                        if pad_base_lattice <= 0:
                            if self.include_video:
                                self.vid_list.append(self.base_state.clone())  # append list
                            else:  # overwrite last
                                self.vid_list = self.base_state.clone()  # Initialize a empty list to
                            # self.vid_list.append(self.base_state.clone())
                        else:
                            if self.include_video:
                                self.vid_list.append(F.pad(self.base_state, pad=(
                                pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice),
                                                           mode='constant', value=0).clone())
                            else:  # overwrite last
                                # self.vid_list = F.pad(self.base_state, pad=(pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice), mode='constant', value=0).clone()  # Initialize a empty list to
                                self.vid_list = self.base_state.clone()  # Initialize a empty list to
                            # self.vid_list.append(F.pad(self.base_state, pad=(pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice), mode='constant', value=0).clone())

                        self.update_and_log_particle_data()
                        self.print_brownian_tree(is_checkpoint=True)
                        self.log_dataframe.to_csv(f'{self.writer.log_dir}/k{self.k}_good.csv')
                        self.update_diffusion_locus(cur_pos=cur_pos)

                        return

                if save_lattice_checkpoint:
                    self.update_and_log_particle_data()
                    self.print_brownian_tree(is_checkpoint=True)
                    self.log_dataframe.to_csv(f'{self.writer.log_dir}/k{self.k}_good.csv')

            else:  # point is a valid point, we sho
                self.base_state = self.state[self.t_bbe:self.b_bbe + 1,
                                  self.l_bbe:self.r_bbe + 1].clone()  # update the dev lattice by one
                if np.any(np.abs(
                        cur_pos - self.state_origin) >= self.max_lattice_size // 2):  # update base lattice and since not max lattice reached, pad state by 1
                    self.update_diffusion_locus(cur_pos=cur_pos)

            # print('-'*55)
            # print(self.base_state_size, self.state_size)
            # print(self.base_state.shape, self.state.shape)
            # print(self.base_state_origin, self.state_origin)
            # print(self.base_state)
            # print(self.state)
            # print('-'*55)

            # Pad base lattice to max lattice for video animation of Brownian tree
            pad_base_lattice = (self.max_lattice_size - self.base_state_size) // 2
            if pad_base_lattice <= 0:
                if self.include_video:
                    self.vid_list.append(self.base_state.clone())  # append list
                else:  # overwrite last
                    self.vid_list = self.base_state.clone()  # Initialize a empty list to
                # self.vid_list.append(self.base_state.clone())
            else:
                if self.include_video:
                    self.vid_list.append(F.pad(self.base_state, pad=(
                    pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice), mode='constant',
                                               value=0).clone())
                else:  # overwrite last
                    # self.vid_list = F.pad(self.base_state, pad=(pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice), mode='constant', value=0).clone()  # Initialize a empty list to
                    self.vid_list = self.base_state.clone()  # Initialize a empty list to
                # self.vid_list.append(F.pad(self.base_state, pad=(pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice), mode='constant', value=0).clone())

            # self.update_and_log_particle_data()

    def get_new_particle(self):
        '''
            Randomly sample a location on the current diffusion locus to diffuse the new particle from.
        '''
        # if self.diff_area == 'sq_peri':  # sample uniformly from the perimeter of lattice
        #     valid_positions = self.next_locations
        # elif self.diff_area == 'cir_area':
        #     valid_positions = list(self.getDiffusionLocus())

        if self.next_locations.shape[0] != 0:
            p = np.random.randint(self.next_locations.shape[0])
            # return tuple(self.next_locations[p])
            return self.next_locations[p]
        else:
            # return (-1, -1)
            return np.array([-1, -1])
            # raise IndexError

    def current_particle_sticks(self, cur_pos):
        '''
            Check if particle in the current position sticks in lattice.
            Both neighbour condition and stickiness conditions are to be met before sticking the particles.

            Input args:
                cur_pos : (x, y) tuple/np array
            Returns True/False
        '''
        # curr_neighbours = self.getAdjacentPoints(cur_pos)  # to get current neighbours of cur_pos in the state lattice
        curr_neighbours = (self.relative_neighbours + cur_pos) % self.state.shape[
            0]  # to get current neighbours of cur_pos in the state lattice
        return any(map(lambda x: self.state[tuple(x)] == self.pixel_val,
                       curr_neighbours)) and np.random.rand() < self.k  # check if any neighbours are stuck and if particles randomly sticks

    def get_next_position(self, cur_pos):
        '''
            Get next position of the particle from the vacant neighbouring positions. Brownian motion is order 2 markov process
            performing 2D random walk.
            Input args:
                cur_pos : (x, y) tuple
            Returns:
                (x, y) coordinate of next position
        '''
        cur_neighbours = (self.relative_neighbours + cur_pos) % self.state.shape[
            0]  # to get current neighbours of cur_pos in the state lattice
        # cur_neighbours = self.get_cur_neighbours(cur_pos)  # List of all adjacent points
        valid_cur_neighbours = np.array(
            list(filter(lambda x: self.state[tuple(x)] == 0, cur_neighbours)))  # get only empty adjacent points
        if valid_cur_neighbours.shape[0] != 0:
            new_pos = np.random.randint(valid_cur_neighbours.shape[0])  # choose a random point for one step walk
            return valid_cur_neighbours[new_pos]
        else:
            return cur_pos  # return the same point again to stick in next walk.

    def get_cur_neighbours(self, cur_pos):
        '''
            get_next_position() takes care of this faster. deprecated!!!
        '''

        y, x = cur_pos  # y-axis=rows, x=-axis=columns
        # adjacentPoints = [(x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
        #                   (x, y - 1),                 (x, y + 1),
        #                   (x + 1, y - 1), (x + 1, y), (x + 1, y + 1)]
        #
        # Remove points outside the image
        # adjacentPoints = filter(lambda x : x[0] > -1 and x[0] < self.lattice_size and \
        #                                    x[1] > -1 and x[1] < self.lattice_size, adjacentPoints)

        #         top = (y + 1) % self.lattice_size
        #         bottom = (y - 1) % self.lattice_size
        #         left = (x - 1) % self.lattice_size
        #         right = (x + 1) % self.lattice_size
        #         adjacentPoints = [(left, bottom), (left, y), (left, top),
        #                           (x, bottom), (x, top),
        #                           (right, bottom), (right, y), (right, top)]
        b = (y + 1) % self.state_size
        t = (y - 1) % self.state_size
        l = (x - 1) % self.state_size
        r = (x + 1) % self.state_size
        #         adjacentPoints = [(l, t), (x, t), (r, t),
        #                           (l, y),         (r, y),
        #                           (l, b), (x, b), (r, b)]
        adjacentPoints = [(t, l), (t, x), (t, r),
                          (y, l), (y, r),
                          (b, l), (b, x), (b, r)]

        return adjacentPoints

    def update_and_log_particle_data(self):
        '''
            Log simulation data.
            first call self.perform_image_processing_and_extract_cv_data(), to process and store, then log data.
        '''
        # self.perform_image_processing_and_extract_cv_data(base_img_path='runs/19_05_04_13_43_29_leelavathi_k1.0_ls101_N10201/Brownian_Tree_Images/Brownian_Tree_k1.0_ls101_N105.png')
        self.perform_image_processing_and_extract_cv_data()
        self.find_fractal_dimension()

        # report the statistics and log the data
        self.writer.add_scalar('00_parameters/N', self.num_particles, self.num_particles)
        self.writer.add_scalar('00_parameters/k', self.k, self.num_particles)
        self.writer.add_scalar('00_parameters/max_ls', self.max_lattice_size, self.num_particles)
        self.writer.add_scalar('00_parameters/pad_size', self.pad_size, self.num_particles)
        self.writer.add_scalar('01_bounding_box_data/bounding_circle', self.a_bbc, self.num_particles)
        self.writer.add_scalar('01_bounding_box_data/bounding_square', self.a_bbe, self.num_particles)
        self.writer.add_scalar('01_bounding_box_data/bounding_square_height', self.hh_bbe * 2 + 1, self.num_particles)
        self.writer.add_scalar('01_bounding_box_data/bounding_square_width', self.hw_bbe * 2 + 1, self.num_particles)
        self.writer.add_scalar('02_simulation_data/num_diff_locus_points', len(self.next_locations), self.num_particles)
        self.writer.add_scalar('02_simulation_data/num_brownian_motion_steps', self.walk_iter, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_centroid_x', self.cv_centroid_x, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_centroid_y', self.cv_centroid_y, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_leftmost_x', self.cv_leftmost_x, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_leftmost_y', self.cv_leftmost_y, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_rightmost_x', self.cv_rightmost_x, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_rightmost_y', self.cv_rightmost_y, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_topmost_x', self.cv_topmost_x, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_topmost_y', self.cv_topmost_y, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_bottommost_x', self.cv_bottommost_x, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_bottommost_y', self.cv_bottommost_y, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_contour_area', self.cv_contour_area, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_contour_perimeter', self.cv_contour_perimeter, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_contour_k', self.cv_contour_k, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_boundingRect_x', self.cv_boundingRect_x, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_boundingRect_y', self.cv_boundingRect_y, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_boundingRect_w', self.cv_boundingRect_w, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_boundingRect_h', self.cv_boundingRect_h, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_boundingRect_area', self.cv_boundingRect_area, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_hull_area', self.cv_hull_area, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_extent', self.cv_extent, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_solidity', self.cv_solidity, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_minEnclosingCircle_center_x', self.cv_minEnclosingCircle_center_x,
                               self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_minEnclosingCircle_center_y', self.cv_minEnclosingCircle_center_y,
                               self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_minEnclosingCircle_r', self.cv_minEnclosingCircle_r, self.num_particles)
        self.writer.add_scalar('03_cv_data/cv_equi_radius', self.cv_equi_radius, self.num_particles)
        self.writer.add_scalar('04_fractal_dimension/polyfit_coeff0', self.coeffs[0], self.num_particles)
        self.writer.add_scalar('04_fractal_dimension/polyfit_coeff1', self.coeffs[1], self.num_particles)

        log_dict = {'00_parameters/log_dir': self.writer.log_dir, '00_parameters/N': self.num_particles,
                    '00_parameters/k': self.k, '00_parameters/max_ls': self.max_lattice_size,
                    '00_parameters/pad_size': self.pad_size,
                    '01_bounding_box_data/bounding_circle': self.a_bbc,
                    '01_bounding_box_data/bounding_square': self.a_bbe,
                    '01_bounding_box_data/bounding_square_height': self.hh_bbe * 2 + 1,
                    '01_bounding_box_data/bounding_square_width': self.hh_bbe * 2 + 1,
                    '02_simulation_data/num_diff_locus_points': len(self.next_locations),
                    '02_simulation_data/num_brownian_motion_steps': self.walk_iter,
                    '03_cv_data/cv_centroid_x': self.cv_centroid_x, '03_cv_data/cv_centroid_y': self.cv_centroid_y,
                    '03_cv_data/cv_leftmost_x': self.cv_leftmost_x, '03_cv_data/cv_leftmost_y': self.cv_leftmost_y,
                    '03_cv_data/cv_rightmost_x': self.cv_rightmost_x, '03_cv_data/cv_rightmost_y': self.cv_rightmost_y,
                    '03_cv_data/cv_topmost_x': self.cv_topmost_x, '03_cv_data/cv_topmost_y': self.cv_topmost_y,
                    '03_cv_data/cv_bottommost_x': self.cv_bottommost_x,
                    '03_cv_data/cv_bottommost_y': self.cv_bottommost_y,
                    '03_cv_data/cv_contour_area': self.cv_contour_area,
                    '03_cv_data/cv_contour_perimeter': self.cv_contour_perimeter,
                    '03_cv_data/cv_contour_k': self.cv_contour_k,
                    '03_cv_data/cv_boundingRect_x': self.cv_boundingRect_x,
                    '03_cv_data/cv_boundingRect_y': self.cv_boundingRect_y,
                    '03_cv_data/cv_boundingRect_w': self.cv_boundingRect_w,
                    '03_cv_data/cv_boundingRect_h': self.cv_boundingRect_h,
                    '03_cv_data/cv_boundingRect_area': self.cv_boundingRect_area,
                    '03_cv_data/cv_hull_area': self.cv_hull_area, '03_cv_data/cv_extent': self.cv_extent,
                    '03_cv_data/cv_solidity': self.cv_solidity,
                    '03_cv_data/cv_minEnclosingCircle_center_x': self.cv_minEnclosingCircle_center_x,
                    '03_cv_data/cv_minEnclosingCircle_center_y': self.cv_minEnclosingCircle_center_y,
                    '03_cv_data/cv_minEnclosingCircle_r': self.cv_minEnclosingCircle_r,
                    '03_cv_data/cv_equi_radius': self.cv_equi_radius,
                    '04_fractal_dimension/polyfit_coeff0': self.coeffs[0],
                    '04_fractal_dimension/polyfit_coeff1': self.coeffs[1]
                    }
        self.log_dataframe = self.log_dataframe.append(log_dict, ignore_index=True)

        # https: // docs.opencv.org / 3.0 - beta / doc / py_tutorials / py_imgproc / py_contours / py_contour_properties / py_contour_properties.html
        # https: // docs.opencv.org / 3.0 - beta / cntdoc / py_tutorials / py_imgproc / py_contours / py_contour_properties / py_contour_properties.html
        # https: // www.pyimagesearch.com / 2016 / 04 / 11 / finding - extreme - points - in -contours -
        # with-opencv /
        # test_img = 'runs/19_05_03_18_26_39_leelavathi_k1.0_ls501_N251001/Brownian_Tree_images/Brownian_Tree_k1.0_ls501_N1.png'
        # test_img = 'runs/19_05_02_12_59_20_leelavathi_k0.1_ls1001_N1002001/DLA_images/DLA_k0.1_ls1001_N2001.png'
        # test_img = 'runs/19_05_02_12_59_19_leelavathi_k1.0_ls501_N251001/DLA_images/DLA_k1.0_ls501_N2001.png'
        # img = 'runs/19_05_03_19_48_05_leelavathi_k1.0_ls11_N121/Brownian_Tree_Images/Brownian_Tree_k1.0_ls11_N2.png'

    def save_brownian_tree(self):
        '''
            Save the base lattice as tensor and image for loading later.
        '''
        # storing the images for training and testing later
        os.system(f'mkdir -p {self.writer.log_dir}/Base_lattice')
        torch.save(self.base_state, f'{self.writer.log_dir}/Data/base_lattice_k{self.k}_N{self.num_particles}.pt')
        cv2.imwrite(f'{self.writer.log_dir}/Data/base_lattice_k{self.k}_N{self.num_particles}.png', self.base_state)
        # vutils.save_image(self.base_state.unsqueeze(0), f'{self.writer.log_dir}/Data/base_lattice_k{self.k}_N{self.num_particles}.png', normalize=True)
        # image = cv2.imread(f'{self.writer.log_dir}/Data/base_lattice_k{self.k}_N{self.num_particles}.png')
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(gray, 127, 255, 0)
        #
        # # find contours in thresholded image, then grab the largest one
        # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cnts = imutils.grab_contours(cnts)
        # c = max(cnts, key=cv2.contourArea)
        #
        # # determine the most extreme points along the contour
        # extLeft = tuple(c[c[:, :, 0].argmin()][0])
        # extRight = tuple(c[c[:, :, 0].argmax()][0])
        # extTop = tuple(c[c[:, :, 1].argmin()][0])
        # extBot = tuple(c[c[:, :, 1].argmax()][0])

    def load_brownian_tree(self):
        '''
            Load the base lattice from saved tensor. for image use perform_image_processing_and_extract_cv_data()
        '''
        self.base_state = torch.load(f'{self.writer.log_dir}/Data/base_lattice_k{self.k}_N{self.num_particles}.pt')
        self.base_state_size = self.base_state.shape[0]
        self.base_state_origin = np.array([self.base_state_size // 2,
                                           self.base_state_size // 2])  # .type(torch.int16) # origin of base_lattice , (r, c) of the numpy array
        self.state = F.pad(self.base_state, pad=(self.pad_size, self.pad_size, self.pad_size, self.pad_size),
                           mode='constant', value=0)
        self.state_size = self.state.shape[0]
        self.state_origin = self.base_state_origin + self.pad_size  # origin of base_lattice
        self.num_particles = self.base_state.sum()
        self.walk_iter = 1

        # bounding box edge coordinates wrt actual coordinates of base lattice
        self.hw_bbe = max(abs(self.state_origin[1] - self.state_origin[1]),
                          abs(self.state_origin[1] - self.state_origin[1]))
        self.hh_bbe = max(abs(self.state_origin[0] - self.state_origin[0]),
                          abs(self.state_origin[0] - self.state_origin[0]))
        self.l_bbe, self.r_bbe = self.t_bbe, self.b_bbe = np.array([-self.hw_bbe, self.hh_bbe]) + self.state_origin
        # self.l_bbe, self.r_bbe = self.t_bbe, self.b_bbe = torch.Tensor([-self.hw_bbe, self.hh_bbe]) + self.state_origin
        # self.r_bbc = torch.Tensor([self.hw_bbe, self.hh_bbe]).norm()
        self.r_bbc = torch.Tensor([self.hw_bbe, self.hh_bbe]).norm().numpy()
        self.a_bbc = np.pi * (self.r_bbc + 1) ** 2
        # self.a_bbe = 4.0 * self.hw_bbe * self.hh_bbe
        self.a_bbe = (2.0 * self.hw_bbe + 1) * (2.0 * self.hh_bbe + 1)
        self.update_diffusion_locus()

        machinename = socket.gethostname()
        # hostname = timestamp = datetime.datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
        # hostname = timestamp = datetime.datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
        hostname = timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")

        absolute_pyfile_path = os.path.abspath(sys.argv[0])

        absolute_base_path = os.path.dirname(absolute_pyfile_path)

        log_dir = os.path.join(absolute_base_path, 'runs',
                               f'{timestamp}_{machinename}_k{self.k}_ls{self.max_lattice_size}_N{self.max_lattice_size ** 2}')

        # self.writer = SummaryWriter('')
        self.writer = SummaryWriter(log_dir=log_dir)

        # storing the images for training and testing later
        os.system(f'mkdir -p {self.writer.log_dir}/Brownian_Tree_Images')

        self.vid_list = []  # Initialize a empty list to
        pad_base_lattice = (
                                       self.max_lattice_size - self.base_state_size) // 2  # find padding size to be done to get full lattice from the base lattice
        if pad_base_lattice < 0:
            pad_base_lattice = 0  # No padding if base lattice is already equal to dev lattice
            pass  # without padding
        elif pad_base_lattice == 0:
            # self.vid_list.append(F.pad(self.base_state, pad=(1, 1, 1, 1), mode='constant', value=0).clone())
            if self.include_video:
                self.vid_list.append(self.base_state.clone())  # append list
            else:  # overwrite last
                self.vid_list = self.base_state.clone()  # Initialize a empty list to
            # self.vid_list.append(self.base_state.clone())
        else:
            if self.include_video:
                self.vid_list.append(
                    F.pad(self.base_state, pad=(pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice),
                          mode='constant', value=0).clone())
            else:  # overwrite last
                # self.vid_list = F.pad(self.base_state, pad=(pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice), mode='constant', value=0).clone()  # Initialize a empty list to
                self.vid_list = self.base_state.clone()  # Initialize a empty list to
            # self.vid_list.append(F.pad(self.base_state, pad=(pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice), mode='constant', value=0).clone())

        self.update_and_log_particle_data()
        self.print_brownian_tree()

    def reset_brownian_tree(self):
        '''
            reset state to restart from a single particle.
        '''
        self.Brownian_Tree_possible = True  # Set the flag to indicate the possibility of Brownian_Tree
        self.base_state = torch.ones((1, 1), dtype=torch.uint8) * self.pixel_val  # .type(torch.int16)
        # self.base_state_origin = torch.tensor([0, 0])  # .type(torch.int16) # origin of base_lattice , (r, c) of the numpy array
        self.base_state_origin = np.array(
            [0, 0])  # .type(torch.int16) # origin of base_lattice , (r, c) of the numpy array
        self.base_state_size = self.base_state.shape[0]
        self.state = F.pad(self.base_state, pad=(self.pad_size, self.pad_size, self.pad_size, self.pad_size),
                           mode='constant', value=0)
        self.state_size = self.state.shape[0]
        self.state_origin = self.base_state_origin + self.pad_size  # origin of base_lattice
        self.num_particles = 1
        self.walk_iter = 1
        self.diff_area = 'sq_peri'
        self.relative_neighbours = np.array(np.where(F.pad(torch.zeros((1, 1)), (1, 1, 1, 1),
                                                           value=1).numpy())).transpose() - 1  # get the relative neighbour index as a numpy array
        # curr_neighbours = (self.relative_neighbours + cur_pos) % self.state.shape[0]  # to get current neighbours of cur_pos in the state lattice

        # bounding box edge coordinates wrt actual coordinates of base lattice
        self.hw_bbe = max(abs(self.state_origin[1] - self.state_origin[1]),
                          abs(self.state_origin[1] - self.state_origin[1]))
        self.hh_bbe = max(abs(self.state_origin[0] - self.state_origin[0]),
                          abs(self.state_origin[0] - self.state_origin[0]))
        self.l_bbe, self.r_bbe = self.t_bbe, self.b_bbe = np.array([-self.hw_bbe, self.hh_bbe]) + self.state_origin
        # self.l_bbe, self.r_bbe = self.t_bbe, self.b_bbe = torch.Tensor([-self.hw_bbe, self.hh_bbe]) + self.state_origin
        # self.r_bbc = torch.Tensor([self.hw_bbe, self.hh_bbe]).norm()
        self.r_bbc = torch.Tensor([self.hw_bbe, self.hh_bbe]).norm().numpy()
        self.a_bbc = np.pi * (self.r_bbc + 1) ** 2
        # self.a_bbe = 4.0 * self.hw_bbe * self.hh_bbe
        self.a_bbe = (2.0 * self.hw_bbe + 1) * (2.0 * self.hh_bbe + 1)
        self.update_diffusion_locus()

        machinename = socket.gethostname()
        # hostname = timestamp = datetime.datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
        # hostname = timestamp = datetime.datetime.now().strftime("%Y_%b_%d_%H_%M_%S")
        hostname = timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")

        absolute_pyfile_path = os.path.abspath(sys.argv[0])

        absolute_base_path = os.path.dirname(absolute_pyfile_path)

        log_dir = os.path.join(absolute_base_path, 'runs',
                               f'{timestamp}_{machinename}_k{self.k}_ls{self.max_lattice_size}_N{self.max_lattice_size ** 2}')

        # self.writer = SummaryWriter('')
        self.writer = SummaryWriter(log_dir=log_dir)

        # storing the images for training and testing later
        os.system(f'mkdir -p {self.writer.log_dir}/Brownian_Tree_Images')

        self.vid_list = []  # Initialize a empty list to
        pad_base_lattice = (
                                       self.max_lattice_size - self.base_state_size) // 2  # find padding size to be done to get full lattice from the base lattice
        if pad_base_lattice < 0:
            pad_base_lattice = 0  # No padding if base lattice is already equal to dev lattice
            pass  # without padding
        elif pad_base_lattice == 0:
            # self.vid_list.append(F.pad(self.base_state, pad=(1, 1, 1, 1), mode='constant', value=0).clone())
            if self.include_video:
                self.vid_list.append(self.base_state.clone())  # append list
            else:  # overwrite last
                self.vid_list = self.base_state.clone()  # Initialize a empty list to
            # self.vid_list.append(self.base_state.clone())
        else:
            if self.include_video:
                self.vid_list.append(
                    F.pad(self.base_state, pad=(pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice),
                          mode='constant', value=0).clone())
            else:  # overwrite last
                # self.vid_list = F.pad(self.base_state, pad=(pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice), mode='constant', value=0).clone()  # Initialize a empty list to
                self.vid_list = self.base_state.clone()  # Initialize a empty list to
            # self.vid_list.append(F.pad(self.base_state, pad=(pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice), mode='constant', value=0).clone())

        self.update_and_log_particle_data()
        self.print_brownian_tree()

    # def get_unreachable_island_area(self):
    #     '''
    #         Get surface area of all unreachable points.
    #         Presently O(m**2), can be done better
    #         Returns integer
    #     '''
    #     free_surface_area = []
    #     for i in range(self.max_lattice_size):
    #         for j in range(self.max_lattice_size):
    #             if self.state[i, j] == 1:
    #                 adj = self.get_cur_neighbours((i, j))
    #                 adj = list(filter(lambda x: self.state[x] == 0, adj))
    #                 free_surface_area.extend(adj)
    #     area = len(list(set(free_surface_area)))

    #     return area

    # def getNeighbourCount(self):
    #     '''
    #         For each cell with 1, count neghbouring cells with 1
    #         Returns integer
    #     '''

    #     count = 0
    #     for i in range(self.max_lattice_size):
    #         for j in range(self.max_lattice_size):
    #             if self.state[i, j] == 1:
    #                 adj = self.get_cur_neighbours((i, j))
    #                 adj = filter(lambda x: self.state[x] == 1, adj)
    #                 count += len(adj)

    #     return count

    # def perform_image_processing_and_log_data(self, base_img_path=None):
    #     if base_img_path is not None:  # if no img_path specified to load, use the base lattice as the image
    #         base_img_path = 'runs/19_05_03_19_48_05_leelavathi_k1.0_ls11_N121/Brownian_Tree_Images/Brownian_Tree_k1.0_ls11_N2.png'
    #         base_img_path = 'runs/19_05_02_12_59_20_leelavathi_k0.1_ls1001_N1002001/DLA_images/DLA_k0.1_ls1001_N2001.png'
    #         # base_img_path = 'runs/19_05_03_19_48_05_leelavathi_k1.0_ls11_N121/Brownian_Tree_Images/Brownian_Tree_k1.0_ls11_N2.png'
    #         base_img = cv2.imread(base_img_path)
    #         base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    #         # cv2.imshow("Base Image original", base_img)
    #         # cv2.waitKey(0)
    #
    #         # find minimum base image and remove outer layer.
    #         all_pixels = np.array(np.where(base_img)).transpose()[:, (0, 1)]
    #         left_top_most = all_pixels.min()
    #         right_bottom_most = all_pixels.max()
    #         center = base_img.shape[0] // 2
    #         self.hh_bbe = self.hw_bbe = max(right_bottom_most - center, center - left_top_most)
    #         self.base_state = base_state = torch.from_numpy(
    #             base_img[center - self.hh_bbe:center + self.hh_bbe + 1, center - self.hh_bbe:center + self.hh_bbe + 1])
    #         self.base_state_size = self.base_state.shape[0]
    #         self.base_state_origin = np.array([center, center])
    #         self.num_particles = all_pixels.shape[0]
    #         self.base_state[self.base_state == 255] = self.pixel_val
    #         # self.base_state_origin = torch.tensor([0, 0])  # .type(torch.int16) # origin of base_lattice , (r, c) of the numpy array
    #         self.state = F.pad(self.base_state, pad=(self.pad_size, self.pad_size, self.pad_size, self.pad_size),
    #                            mode='constant', value=0)
    #         self.state_size = self.state.shape[0]
    #         self.state_origin = self.base_state_origin + self.pad_size  # origin of base_lattice
    #
    #         try:
    #             # find contours in thresholded image, then grab the largest one
    #             ret, thresh = cv2.threshold(base_img, 127, 255, 0)  # Thresholding for binary map
    #             contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #             contours_imutils = imutils.grab_contours((contours, hierarchy))
    #             cnt_imutils = max(contours_imutils, key=cv2.contourArea)
    #
    #             # determine the most extreme points along the contour
    #             extLeft = tuple(cnt_imutils[cnt_imutils[:, :, 0].argmin()][0])
    #             extRight = tuple(cnt_imutils[cnt_imutils[:, :, 0].argmax()][0])
    #             extTop = tuple(cnt_imutils[cnt_imutils[:, :, 1].argmin()][0])
    #             extBot = tuple(cnt_imutils[cnt_imutils[:, :, 1].argmax()][0])
    #
    #             # draw the outline of the object, then draw each of the
    #             # extreme points, where the left-most is red, right-most
    #             # is green, top-most is blue, and bottom-most is teal
    #             cv2.drawContours(base_img, [cnt_imutils], -1, (0, 255, 255), 2)
    #             cv2.circle(base_img, extLeft, 8, (0, 0, 255), -1)
    #             cv2.circle(base_img, extRight, 8, (0, 255, 0), -1)
    #             cv2.circle(base_img, extTop, 8, (255, 0, 0), -1)
    #             cv2.circle(base_img, extBot, 8, (255, 255, 0), -1)
    #
    #             # show the output image
    #             # cv2.imshow("Image", base_img)
    #             # cv2.waitKey(0)
    #
    #             cnt = contours[0]
    #             M = cv2.moments(cnt)
    #             self.cv_centroid_x = int(M['m10'] / M['m00'])
    #             self.cv_centroid_y = int(M['m01'] / M['m00'])
    #
    #             self.cv_leftmost_x, self.cv_leftmost_y = leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    #             self.cv_rightmost_x, self.cv_rightmost_y = rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    #             self.cv_topmost_x, self.cv_topmost_y = topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    #             self.cv_bottommost_x, self.cv_bottommost_y = bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    #             # print(topmost, leftmost, bottommost, rightmost)
    #
    #             # draw the outline of the object, then draw each of the
    #             # extreme points, where the left-most is red, right-most
    #             # is green, top-most is blue, and bottom-most is teal
    #             # cv2.drawContours(base_img, [cnt_imutils], -1, (0, 255, 255), 2)
    #             # cv2.circle(base_img, leftmost, 8, (0, 0, 255), -1)
    #             # cv2.circle(base_img, rightmost, 8, (0, 255, 0), -1)
    #             # cv2.circle(base_img, topmost, 8, (255, 0, 0), -1)
    #             # cv2.circle(base_img, bottommost, 8, (255, 255, 0), -1)
    #
    #             # show the output image
    #             # cv2.imshow("Image", base_img)
    #             # cv2.waitKey(0)
    #
    #             # base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    #             # ret, thresh = cv2.threshold(base_img, 127, 255, 0)
    #             # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #             # contours_imutils = imutils.grab_contours((contours, hierarchy))
    #             # cnt_imutils = max(contours_imutils, key=cv2.contourArea)
    #             #
    #             # cnt = contours[0]
    #             # M = cv2.moments(cnt)
    #             # cx = int(M['m10'] / M['m00'])
    #             # cy = int(M['m01'] / M['m00'])
    #             #
    #             # leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    #             # rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    #             # topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    #             # bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    #             # print(topmost, leftmost, bottommost, rightmost)
    #
    #             # cv2.drawContours(base_img, [cnt_imutils], -1, (0, 255, 255), 2)
    #             # cv2.circle(base_img, leftmost, 8, (0, 0, 255), -1)
    #             # cv2.circle(base_img, rightmost, 8, (0, 255, 0), -1)
    #             # cv2.circle(base_img, topmost, 8, (255, 0, 0), -1)
    #             # cv2.circle(base_img, bottommost, 8, (255, 255, 0), -1)
    #             #
    #             # # show the output image
    #             # cv2.imshow("Image", base_img)
    #             # cv2.waitKey(0)
    #
    #             self.cv_contour_area = cv2.contourArea(cnt)
    #             self.cv_contour_perimeter = cv2.arcLength(cnt, True)
    #             epsilon = 0.1 * cv2.arcLength(cnt, True)
    #             approx = cv2.approxPolyDP(cnt, epsilon, True)
    #             hull = cv2.convexHull(cnt)
    #             self.cv_contour_k = cv2.isContourConvex(cnt)
    #
    #             self.cv_boundingRect_x, self.cv_boundingRect_y, self.cv_boundingRect_w, self.cv_boundingRect_h = x, y, w, h = cv2.boundingRect(
    #                 cnt)
    #             # smallest_base_img = cv2.rectangle(base_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    #             # rect = cv2.minAreaRect(cnt)
    #             # box = cv2.boxPoints(rect)
    #             # box = np.int0(box)
    #             # base_img = cv2.drawContours(base_img, [box], 0, (0, 0, 255), 2)
    #
    #             (x, y), radius = cv2.minEnclosingCircle(cnt)
    #             self.cv_minEnclosingCircle_center_x = x
    #             self.cv_minEnclosingCircle_center_y = y
    #             self.cv_minEnclosingCircle_r = radius
    #             # center = (int(x), int(y))
    #             # radius = int(radius)
    #             # base_img = cv2.circle(base_img, center, radius, (0, 255, 0), 2)
    #
    #         except:
    #             # print(e)
    #             pass
    #
    #     else:
    #         base_img = self.base_state.numpy()
    #
    #     try:
    #         ret, thresh = cv2.threshold(base_img, 127, 255, 0)
    #         contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #         cnt = contours[0]
    #
    #         # https: // docs.opencv.org / 3.0 - beta / doc / py_tutorials / py_imgproc / py_contours / py_contour_properties / py_contour_properties.html
    #         # # Aspect Ratio
    #         # x, y, w, h = cv2.boundingRect(cnt)  # Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.
    #         # aspect_ratio = float(w) / h
    #         # print(aspect_ratio)
    #         # self.writer.add_scalar('04_contour_data/bounding_circle', self.a_bbc, self.num_particles)
    #         # self.writer.add_scalar('00_areas/bounding_square', self.a_bbe, self.num_particles)
    #         # self.writer.add_scalar('01_dimensions/bounding_square_height', self.hh_bbe * 2 + 1, self.num_particles)
    #         # self.writer.add_scalar('01_dimensions/bounding_square_width', self.hw_bbe * 2 + 1, self.num_particles)
    #         # self.writer.add_scalar('02_parameters/k', self.k, self.num_particles)
    #         # self.writer.add_scalar('02_parameters/max_ls', self.max_lattice_size, self.num_particles)
    #         # self.writer.add_scalar('02_parameters/N', self.num_particles, self.num_particles)
    #         # self.writer.add_scalar('02_parameteres/num_diff_locus_points', len(self.next_locations), self.num_particles)
    #         # self.writer.add_scalar('03_brownian_motion/num_walk_steps', self.walk_iter, self.num_particles)
    #         #
    #         # area = cv2.contourArea(cnt)
    #         # x, y, w, h = cv2.boundingRect(cnt)
    #         # rect_area = w * h
    #         # extent = float(area) / rect_area
    #         # print(extent)
    #         #
    #         # area = cv2.contourArea(cnt)
    #         # hull = cv2.convexHull(cnt)
    #         # hull_area = cv2.contourArea(hull)
    #         # solidity = float(area) / hull_area
    #         # print(solidity)
    #         #
    #         # leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    #         # rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    #         # topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    #         # bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    #         # print(topmost, leftmost, bottommost, rightmost)
    #         #
    #         # area = cv2.contourArea(cnt)
    #         # equi_diameter = np.sqrt(4 * area / np.pi)
    #         # print(equi_diameter)
    #         #
    #         # # Rotated Rectangle # https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
    #         # rect = cv2.minAreaRect(cnt)
    #         # box = cv2.boxPoints(rect)
    #         # box = np.int0(box)
    #         # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    #         #
    #         # # Minimum Enclosing Circle
    #         # (x, y), radius = cv2.minEnclosingCircle(cnt)
    #         # center = (int(x), int(y))
    #         # radius = int(radius)
    #         # cv2.circle(img, center, radius, (0, 255, 0), 2)
    #         #
    #         # # Checking Convexity
    #         # k = cv2.isContourConvex(cnt)
    #         #
    #         # # Contour Approximation
    #         # epsilon = 0.1 * cv2.arcLength(cnt, True)
    #         # approx = cv2.approxPolyDP(cnt, epsilon, True)
    #         #
    #         # # Moments
    #         # cnt = contours[0]
    #         # M = cv2.moments(cnt)
    #         # print(M)
    #         #
    #         # # Contour Area
    #         # area = cv2.contourArea(cnt)
    #         #
    #         # # Contour Perimeter
    #         # perimeter = cv2.arcLength(cnt, True)
    #         #
    #         base_img = cv2.imread(base_img_path)
    #         gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    #         ret, thresh = cv2.threshold(gray, 127, 255, 0)
    #
    #         # find contours in thresholded image, then grab the largest one
    #         cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #         cnts = imutils.grab_contours(cnts)
    #         c = max(cnts, key=cv2.contourArea)
    #
    #         # determine the most extreme points along the contour
    #         extLeft = tuple(c[c[:, :, 0].argmin()][0])
    #         extRight = tuple(c[c[:, :, 0].argmax()][0])
    #         extTop = tuple(c[c[:, :, 1].argmin()][0])
    #         extBot = tuple(c[c[:, :, 1].argmax()][0])
    #
    #         # draw the outline of the object, then draw each of the
    #         # extreme points, where the left-most is red, right-most
    #         # is green, top-most is blue, and bottom-most is teal
    #         cv2.drawContours(base_img, [c], -1, (0, 255, 255), 2)
    #         cv2.circle(base_img, extLeft, 8, (0, 0, 255), -1)
    #         cv2.circle(base_img, extRight, 8, (0, 255, 0), -1)
    #         cv2.circle(base_img, extTop, 8, (255, 0, 0), -1)
    #         cv2.circle(base_img, extBot, 8, (255, 255, 0), -1)
    #
    #         # show the output image
    #         cv2.imshow("Image", base_img)
    #         cv2.waitKey(0)
    #     except:
    #         # print(e)
    #         pass

    def perform_image_processing_and_extract_cv_data(self, base_img_path=None):
        '''
        :param base_img_path: path of image to be used as base lattice to continue DLA from.
        :return: updated self
        '''
        # base_img_path = 'runs/19_05_02_12_59_20_leelavathi_k0.1_ls1001_N1002001/DLA_images/DLA_k0.1_ls1001_N2001.png'
        # base_img_path = 'runs/19_05_03_19_48_05_leelavathi_k1.0_ls11_N121/Brownian_Tree_Images/Brownian_Tree_k1.0_ls11_N2.png'
        # base_img_path = 'runs/19_05_03_19_48_05_leelavathi_k1.0_ls11_N121/Brownian_Tree_Images/Brownian_Tree_k1.0_ls11_N2.png'

        if base_img_path is not None:  # if no img_path specified to load, use the base lattice as the image
            base_img_filename = base_img_path.split('/')[-1]
            os.system(f'cp {base_img_path} {self.writer.log_dir}/base_image_{base_img_filename}')
            base_img_rgb = cv2.imread(base_img_path)
            base_img = cv2.cvtColor(base_img_rgb, cv2.COLOR_BGR2GRAY)  # convert to gray scale
            # base_img_rgb = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)  # convert to gray scale

            # find minimum base image and remove outer layer.
            all_pixels = np.array(np.where(base_img)).transpose()[:, (0, 1)]
            left_top_most = all_pixels.min()
            right_bottom_most = all_pixels.max()
            center = base_img.shape[0] // 2
            hh_bbe = hw_bbe = max(right_bottom_most - center, center - left_top_most)
            base_img = torch.from_numpy(
                base_img[center - hh_bbe:center + hh_bbe + 1, center - hh_bbe:center + hh_bbe + 1])
        else:
            base_img_filename = None
            # base_img = self.state[self.t_bbe:self.b_bbe+1, self.l_bbe:self.r_bbe+1].clone().numpy()  # get the current base lattice
            base_img = self.base_state.clone().numpy()  # get the current base lattice
            base_img_rgb = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)  # convert to gray scale
            # base_img_filename = f'Brownian_Tree_k{self.k}_N{self.num_particles}.png'

        try:
            # find contours in thresholded image, then grab the largest one
            # base_img = self.state.numpy()
            ret, thresh = cv2.threshold(base_img, 127, 255, 0)  # Thresholding for binary map
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours_imutils = imutils.grab_contours((contours, hierarchy))
            cnt_imutils = max(contours_imutils, key=cv2.contourArea)

            # get moments of contour
            M = cv2.moments(cnt_imutils)
            try:
                self.cv_centroid_x = int(M['m10'] / M['m00'])
                self.cv_centroid_y = int(M['m01'] / M['m00'])
            except:
                # print(e)
                self.cv_centroid_x = -1
                self.cv_centroid_y = -1

            cnt = cnt_imutils
            # cnt = contours[0]

            self.cv_leftmost_x, self.cv_leftmost_y = leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
            self.cv_rightmost_x, self.cv_rightmost_y = rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
            self.cv_topmost_x, self.cv_topmost_y = topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
            self.cv_bottommost_x, self.cv_bottommost_y = bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

            # get contour area, perimeter, convexity, rect_bounding box
            self.cv_contour_area = contour_area = cv2.contourArea(cnt)
            self.cv_contour_perimeter = cv2.arcLength(cnt, True)
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            hull = cv2.convexHull(cnt)
            self.cv_contour_k = cv2.isContourConvex(cnt)

            # boundingRect
            self.cv_boundingRect_x, self.cv_boundingRect_y, self.cv_boundingRect_w, self.cv_boundingRect_h = x, y, w, h = cv2.boundingRect(
                cnt)
            self.cv_boundingRect_area = rect_area = w * h
            self.cv_extent = extent = float(self.cv_contour_area) / rect_area
            # print(extent)

            hull = cv2.convexHull(cnt)
            try:
                self.cv_hull_area = cv2.contourArea(hull)
                self.cv_solidity = float(contour_area) / self.cv_hull_area
            except:
                # print(e)
                self.cv_hull_area = 0.0
                self.cv_solidity = -1
            # print(solidity)

            # minEnclosingCircle
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            self.cv_minEnclosingCircle_center_x = x
            self.cv_minEnclosingCircle_center_y = y
            self.cv_minEnclosingCircle_r = radius

            self.cv_equi_radius = np.sqrt(self.cv_contour_area / np.pi)

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            base_img = cv2.drawContours(base_img, [box], 0, (0, 0, 255), 2)

            # draw the outline of the object, then draw each of the
            # extreme points, where the left-most is red, right-most
            # is green, top-most is blue, and bottom-most is teal
            cv2.drawContours(base_img_rgb, [cnt_imutils], -1, (0, 255, 255), 2)
            cv2.circle(base_img, leftmost, 2, (0, 0, 255), -1)
            cv2.circle(base_img, rightmost, 2, (0, 255, 0), -1)
            cv2.circle(base_img, topmost, 2, (255, 0, 0), -1)
            cv2.circle(base_img, bottommost, 2, (255, 255, 0), -1)

            if base_img_filename is not None:
                cv2.imwrite(f'{self.writer.log_dir}/base_image_{base_img_filename[:-4]}_contour.png', base_img)
            # cv2.imshow(f'{self.writer.log_dir}/base_image_{base_img_filename[:-4]}_contour.png', base_img)
            # cv2.waitKey(0)
        except:
            # print(e)
            pass

    def extract_base_image(self, base_img_path=None):
        # base_img_path = 'runs/19_05_03_19_48_05_leelavathi_k1.0_ls11_N121/Brownian_Tree_Images/Brownian_Tree_k1.0_ls11_N2.png'
        # base_img_path = 'runs/19_05_02_12_59_20_leelavathi_k0.1_ls1001_N1002001/DLA_images/DLA_k0.1_ls1001_N2001.png'
        # base_img_path = 'runs/19_05_03_19_48_05_leelavathi_k1.0_ls11_N121/Brownian_Tree_Images/Brownian_Tree_k1.0_ls11_N2.png'
        if base_img_path is not None:  # if no img_path specified to load, use the base lattice as the image
            try:
                # print(base_img_path)
                base_img = None
                Max_lattice_size = self.max_lattice_size
                while True:  # base_img is None and Max_lattice_size > 1:
                    # print(base_img_path)
                    base_img = cv2.imread(base_img_path)
                    # print(base_img)
                    if base_img is None and Max_lattice_size > 1:
                        # print('looking for lower lattice')
                        # print(Max_lattice_size)
                        Max_lattice_size -= 2
                        # print(Max_lattice_size)
                        # print(base_img_path)
                        # base_img_path = 'tree_checkpoints/ls{}/k{}_checkpoint.png'.format(Max_lattice_size, self.k)
                        base_img_path = 'tree_checkpoints/k{}/k{}_ls{}_checkpoint.png'.format(self.k, self.k, Max_lattice_size)
                        # print(base_img_path)
                    else:
                        break
                    # print(base_img is None and Max_lattice_size > 1)
                # print('extracting image')
                base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)  # convert to gray scale
                # cv2.imshow("Base Image original", base_img)
                # cv2.waitKey(0)

                # find minimum base image and remove outer layer.
                all_pixels = np.array(np.where(base_img)).transpose()[:, (0, 1)]
                left_top_most = all_pixels.min()
                right_bottom_most = all_pixels.max()
                center = base_img.shape[0] // 2
                hh_bbe = hw_bbe = max(right_bottom_most - center, center - left_top_most)
                self.base_state = torch.from_numpy(
                    base_img[center - hh_bbe:center + hh_bbe + 1, center - hh_bbe:center + hh_bbe + 1])
                self.base_state_size = self.base_state.shape[0]
                self.base_state_origin = np.array([center, center])
                print(f'Used image {base_img_path} as base lattice')
            except:
                # print(e)
                print(f'Specified image {base_img_path} is absent!!! Start with single particle')
        self.state = F.pad(self.base_state, pad=(self.pad_size, self.pad_size, self.pad_size, self.pad_size),
                           mode='constant', value=0)
        self.state_size = self.state.shape[0]
        self.state_origin = self.base_state_origin + self.pad_size  # origin of base_lattice
        self.num_particles = np.array(np.where(self.state.numpy())).transpose().shape[0]
        self.walk_iter = 1
        self.diff_area = 'sq_peri'
        self.relative_neighbours = np.array(np.where(F.pad(torch.zeros((1, 1)), (1, 1, 1, 1),
                                                           value=1).numpy())).transpose() - 1  # get the relative neighbour index as a numpy array

        self.hh_bbe = self.hw_bbe = self.base_state_size // 2
        # self.hw_bbe = max(abs(self.state_origin[1] - self.state_origin[1]), abs(self.state_origin[1] - self.state_origin[1]))
        # self.hh_bbe = max(abs(self.state_origin[0] - self.state_origin[0]), abs(self.state_origin[0] - self.state_origin[0]))
        self.l_bbe, self.r_bbe = self.t_bbe, self.b_bbe = np.array([-self.hw_bbe, self.hh_bbe]) + self.state_origin
        # self.l_bbe, self.r_bbe = self.t_bbe, self.b_bbe = torch.Tensor([-self.hw_bbe, self.hh_bbe]) + self.state_origin
        # self.r_bbc = torch.Tensor([self.hw_bbe, self.hh_bbe]).norm()
        self.r_bbc = torch.Tensor([self.hw_bbe, self.hh_bbe]).norm().numpy()
        self.a_bbc = np.pi * (self.r_bbc + 1) ** 2
        # self.a_bbe = 4.0 * self.hw_bbe * self.hh_bbe
        self.a_bbe = (2.0 * self.hw_bbe + 1) * (2.0 * self.hh_bbe + 1)

        self.vid_list = []  # Initialize a empty list to
        pad_base_lattice = (
                                       self.max_lattice_size - self.base_state_size) // 2  # find padding size to be done to get full lattice from the base lattice
        if pad_base_lattice < 0:
            pad_base_lattice = 0  # No padding if base lattice is already equal to dev lattice
            pass  # without padding
        elif pad_base_lattice == 0:
            # self.vid_list.append(F.pad(self.base_state, pad=(1, 1, 1, 1), mode='constant', value=0).clone())
            if self.include_video:
                self.vid_list.append(self.base_state.clone())  # append list
            else:  # overwrite last
                self.vid_list = self.base_state.clone()  # Initialize a empty list to
            # self.vid_list.append(self.base_state.clone())
        else:
            if self.include_video:
                self.vid_list.append(
                    F.pad(self.base_state, pad=(pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice),
                          mode='constant', value=0).clone())
            else:  # overwrite last
                # self.vid_list = F.pad(self.base_state, pad=(pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice), mode='constant', value=0).clone()  # Initialize a empty list to
                self.vid_list = self.base_state.clone()  # Initialize a empty list to
            # self.vid_list.append(F.pad(self.base_state, pad=(pad_base_lattice, pad_base_lattice, pad_base_lattice, pad_base_lattice), mode='constant', value=0).clone())

        self.update_diffusion_locus()

    def find_fractal_dimension(self):
        '''
        Calculate the fractal dimension for the given base lattice using box counting method
        :return: updated self
        '''
        lattice_size_p2 = int(
            2 ** np.ceil(np.log2(self.base_state_size)))  # make the lattice as a power of 2 for computational purpose
        total_pad_size = lattice_size_p2 - self.base_state_size
        left_top_pad_size = total_pad_size // 2
        bottom_right_pad_size = total_pad_size - left_top_pad_size
        padded_state = F.pad(self.base_state,
                             pad=(left_top_pad_size, bottom_right_pad_size, left_top_pad_size, bottom_right_pad_size),
                             mode='constant', value=0)

        box_sizes = np.arange(int(np.log2(lattice_size_p2))) + 1
        N_box_sizes = np.zeros(int(np.log2(lattice_size_p2)))
        for box_size in range(int(np.log2(lattice_size_p2))):
            N_box_sizes[box_size] = \
            np.where(F.max_pool2d(padded_state.type(torch.float32).unsqueeze(0), box_size + 1).numpy())[0].shape[0]

        try:
            self.coeffs = np.polyfit(box_sizes * -1, np.log2(N_box_sizes), 1)
            self.poly1d_model = np.poly1d(self.coeffs)
        except:
            # print(e)
            self.poly1d_model = None

        # print("======= coeffs :", self.coeffs)

        # pixels = np.array(np.where(self.base_state.numpy())).transpose()
        #
        # # computing the fractal dimension
        # # considering only scales in a logarithmic list
        # scales = np.logspace(0.01, 1, num=100, endpoint=False, base=2)
        # Ns = []
        # # looping over several scales
        # for scale in scales:
        #     print("======= Scale :", scale)
        #     # computing the histogram
        #     H, edges = np.histogramdd(pixels, bins=(np.arange(0, self.state.shape[1], scale), np.arange(0, self.state.shape[0], scale)))
        #     Ns.append(np.sum(H > 0))
        #
        # # linear fit, polynomial of degree 1
        # coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
        # print("======= coeffs :", coeffs)
        #
        # self.coeffs = coeffs

    def estimate_stickiness(self):
        '''
        Estimate stickiness from the base lattice.

        Not yet implemented fully. look at DLA_predict_k.ipynb file.
        Should be incorporated later . TODO : TBD
        :return:
        '''
        lattice_size_p2 = int(
            2 ** np.ceil(np.log2(self.base_state_size)))  # make the lattice as a power of 2 for computational purpose
        total_pad_size = lattice_size_p2 - self.base_state_size
        left_top_pad_size = total_pad_size // 2
        bottom_right_pad_size = total_pad_size - left_top_pad_size
        padded_state = F.pad(self.base_state,
                             pad=(left_top_pad_size, bottom_right_pad_size, left_top_pad_size, bottom_right_pad_size),
                             mode='constant', value=0)

        box_sizes = np.arange(int(np.log2(lattice_size_p2))) + 1
        N_box_sizes = np.zeros(int(np.log2(lattice_size_p2)))
        for box_size in range(int(np.log2(lattice_size_p2))):
            N_box_sizes[box_size] = \
            np.where(F.max_pool2d(padded_state.type(torch.float32).unsqueeze(0), box_size + 1).numpy())[0].shape[0]

        try:
            self.coeffs = np.polyfit(box_sizes * -1, np.log2(N_box_sizes), 1)
            self.poly1d_model = np.poly1d(self.coeffs)
        except:
            # print(e)
            self.poly1d_model = None

        # print("======= coeffs :", self.coeffs)

        # pixels = np.array(np.where(self.base_state.numpy())).transpose()
        #
        # # computing the fractal dimension
        # # considering only scales in a logarithmic list
        # scales = np.logspace(0.01, 1, num=100, endpoint=False, base=2)
        # Ns = []
        # # looping over several scales
        # for scale in scales:
        #     print("======= Scale :", scale)
        #     # computing the histogram
        #     H, edges = np.histogramdd(pixels, bins=(np.arange(0, self.state.shape[1], scale), np.arange(0, self.state.shape[0], scale)))
        #     Ns.append(np.sum(H > 0))
        #
        # # linear fit, polynomial of degree 1
        # coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
        # print("======= coeffs :", coeffs)
        #
        # self.coeffs = coeffs

