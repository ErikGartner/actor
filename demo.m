% This file generates the visualizations from Figure 5 in main paper.
% The script assumes that the dataset used is the one described in the
% paper.

% Variables:
MODEL_WEIGHTS_PATH = 'actor.caffemodel';


%%
% First row, example from: 160422_ultimatum1_1

scene_name = '160422_ultimatum1_1';
frame_nbr = 20;
camera_nbr = 13;
run_visualize_model('val', MODEL_WEIGHTS_PATH, scene_name, frame_nbr, camera_nbr);
clear env;

%%
% Second row, example from: 160224_ultimatum2_4

scene_name = '160224_ultimatum2_4';
frame_nbr = 73;
camera_nbr = 14;
run_visualize_model('test', MODEL_WEIGHTS_PATH, scene_name, frame_nbr, camera_nbr);
clear env;

%%
% Third row, example from: 160226_mafia2_8

scene_name = '160226_mafia2_8';
frame_nbr = 55;
camera_nbr = 4;
run_visualize_model('test', MODEL_WEIGHTS_PATH, scene_name, frame_nbr, camera_nbr);
clear env;