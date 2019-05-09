#!/usr/bin/env bash
#nohup python DLA_run_main.py -ks 1.0 0.5 0.1 0.05 0.025 0.0125 0.00625 0.003125 0.0015625 -ls 101 201 301 401 501 701 901 1001
#nohup python DLA_run_main.py -ks 1.0 -ls 1001 -base_img_path /home/sourabh/prj/DLA_gro/Diffusion-Limited-Aggregation/runs/part05_brownian_tree_generation/k1.0/19_05_07_00_15_53_grothendieck_k1.0_ls1001_pad1/Brownian_Tree_Images/Brownian_Tree_k1.0_ls549_N8131_reached.png &
#nohup python DLA_run_main.py -ks 0.5 -ls 1001 -base_img_path /home/sourabh/prj/DLA_gro/Diffusion-Limited-Aggregation/runs/part05_brownian_tree_generation/k0.5/19_05_07_00_16_27_grothendieck_k0.5_ls1001_pad1/Brownian_Tree_Images/Brownian_Tree_k0.5_ls543_N7416_reached.png &
#nohup python DLA_run_main.py -ks 0.1 -ls 1001 -base_img_path /home/sourabh/prj/DLA_gro/Diffusion-Limited-Aggregation/runs/part05_brownian_tree_generation/k0.1/19_05_07_00_16_39_grothendieck_k0.1_ls1001_pad1/Brownian_Tree_Images/Brownian_Tree_k0.1_ls447_N14046_reached.png &
#nohup python DLA_run_main.py -ks 0.05 -ls 1001 -base_img_path /home/sourabh/prj/DLA_gro/Diffusion-Limited-Aggregation/runs/part05_brownian_tree_generation/k0.05/19_05_07_00_16_49_grothendieck_k0.05_ls1001_pad1/Brownian_Tree_Images/Brownian_Tree_k0.05_ls411_N15873_reached.png &
#nohup python DLA_run_main.py -ks 0.025 -ls 1001 -base_img_path /home/sourabh/prj/DLA_gro/Diffusion-Limited-Aggregation/runs/part05_brownian_tree_generation/k0.025/19_05_07_00_17_02_grothendieck_k0.025_ls1001_pad1/Brownian_Tree_Images/Brownian_Tree_k0.025_ls377_N18122_reached.png &
#nohup python DLA_run_main.py -ks 0.0125 -ls 1001 -base_img_path /home/sourabh/prj/DLA_gro/Diffusion-Limited-Aggregation/runs/part05_brownian_tree_generation/k0.0125/19_05_07_00_17_12_grothendieck_k0.0125_ls1001_pad1/Brownian_Tree_Images/Brownian_Tree_k0.0125_ls329_N24216_reached.png &
#nohup python DLA_run_main.py -ks 0.00625 -ls 1001 -base_img_path /home/sourabh/prj/DLA_gro/Diffusion-Limited-Aggregation/runs/part05_brownian_tree_generation/k0.00625/19_05_07_00_17_24_grothendieck_k0.00625_ls1001_pad1/Brownian_Tree_Images/Brownian_Tree_k0.00625_ls271_N21208_reached.png &
#nohup python DLA_run_main.py -ks 0.003125 -ls 1001 -base_img_path /home/sourabh/prj/DLA_gro/Diffusion-Limited-Aggregation/runs/part05_brownian_tree_generation/k0.003125/19_05_07_00_17_39_grothendieck_k0.003125_ls1001_pad1/Brownian_Tree_Images/Brownian_Tree_k0.003125_ls223_N14204_reached.png &
#nohup python DLA_run_main.py -ks 0.0015625 -ls 1001 -base_img_path /home/sourabh/prj/DLA_gro/Diffusion-Limited-Aggregation/runs/part05_brownian_tree_generation/k0.0015625/19_05_07_00_17_49_grothendieck_k0.0015625_ls1001_pad1/Brownian_Tree_Images/Brownian_Tree_k0.0015625_ls193_N14096_reached.png &
nohup python DLA_run_main.py -ks 1.0 -ls 1001 &
nohup python DLA_run_main.py -ks 0.5 -ls 1001 &
nohup python DLA_run_main.py -ks 0.1 -ls 1001 &
nohup python DLA_run_main.py -ks 0.05 -ls 1001 &
nohup python DLA_run_main.py -ks 0.025 -ls 1001 &
nohup python DLA_run_main.py -ks 0.0125 -ls 1001 &
nohup python DLA_run_main.py -ks 0.00625 -ls 1001 &
nohup python DLA_run_main.py -ks 0.003125 -ls 1001 &
nohup python DLA_run_main.py -ks 0.0015625 -ls 1001 &
#wait
# rm -rf runs/part05_brownian_tree_generation/k0*/19_05_07_1*