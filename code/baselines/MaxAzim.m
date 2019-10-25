classdef MaxAzim < Baseline
% Max-Azim baseline. Tries to maximize "spread" in the azimuth for diverse
% viewpoint selection.

    methods

        function obj = MaxAzim(baselines)
        	obj = obj@Baseline(baselines, 'max_azim');
        end

        function [recon_err, recon_err_hip_aligned, preds, cams_visited, out_camera, ...
                    recon_errs_all, recon_errs_hip_aligned_all] = run(obj, env, tracker, tri_poses, nbr_cameras)

            % Specify max elev spread for this scene
            elev_angle_max = env.scene().camera_rig_individual.elev_angle_span_half;

            % Initialize containers
            nbr_persons = tracker.nbr_persons;
            preds_all = nan(15, 3, obj.max_traj_len, nbr_persons);
            preds_cam_all = nan(15, 2, obj.max_traj_len, nbr_persons);
            pose_idxs_all = nan(obj.max_traj_len, nbr_persons);
            cams_visited = nan(1, obj.max_traj_len);

            % Get initial state and camera index
            pose_idxs = tracker.get_detections();
            pose_idxs_all(1, :) = pose_idxs;
            P_mat = env.scene().camera_calibrations{env.camera_idx}.P;

            cam_idx = env.camera_idx;
            cams_visited(1) = env.camera_idx;
            for j = 1 : nbr_persons

                % Extract current pose index
                pose_idx = pose_idxs(j);

                % Goto current person
                env.goto_person(j);

                % Get 2D pose prediction and data blob
                [pred_camera, ~] = env.get_current_predictor(pose_idx);

                % Add to triangulated pose
                tri_poses.tri_poses{j}.add(pred_camera, P_mat, env.camera_idx);

                state = env.get_state(pose_idx, nan, pred_camera, tri_poses.tri_poses{j});
                preds_all(:, :, 1, j) = state.pred;
                preds_cam_all(:, :, 1, j) = state.pred_camera;
            end

            % Find the global azim angle of the given camera
            angle_global = env.scene().global_angles_cam(cam_idx);
			azim_global = angle_global(1);

            i = 2;
            while i <= obj.max_traj_len

                % If more than 8 spread cameras, sample rest randomly
                if i > 8
                    next_cam_idx = randi(env.scene().nbr_cameras);
                    while i <= env.scene().nbr_cameras && any(cams_visited == next_cam_idx)
                        next_cam_idx = randi(env.scene().nbr_cameras);
                    end
                    env.goto_cam(next_cam_idx)
                else
                
                    % Depending on how many cameras seen, the azim step-size
                    % varies
                    if i <= 4 || i == 6 || i == 7 || i == 8
                        azim_step = 2 * pi / 4; % 90 degrees
                    elseif i == 5
                        azim_step = 2 * pi / 8; % 45 degrees
                    end

                    % Update azim angle (unidistant steps)
                    azim_global = azim_global + azim_step;
                    azim_global = angle(cos(azim_global) + 1i * sin(azim_global));

                    % Update elev angle (randomly)
                    elev_local = elev_angle_max * 2 * (rand() - 0.5);
                    [~, elev_global] = env.agent_angles_to_global(nan, elev_local);

                    % Go to new (azim, elev) angles
                    env.goto_cam_mises(azim_global, elev_global);

                    % Ensure new camera location!
                    while any(cams_visited == env.camera_idx)
                        try_cam_idx = randi(env.scene().nbr_cameras);
                        env.goto_cam(try_cam_idx);
                    end
                end
                    
                tracker.next(env.frame_idx, env.camera_idx);
                pose_idxs = tracker.get_detections();
                pose_idxs_all(i, :) = pose_idxs;
                P_mat = env.scene().camera_calibrations{env.camera_idx}.P;

                cams_visited(i) = env.camera_idx;
                for j = 1 : nbr_persons

                    % Extract current pose index
                    pose_idx = pose_idxs(j);

                    % Goto current person
                    env.goto_person(j);

                    % Get 2D pose prediction and data blob
                    [pred_camera, ~] = env.get_current_predictor(pose_idx);

                    % Add to triangulated pose
                    tri_poses.tri_poses{j}.add(pred_camera, P_mat, env.camera_idx);

                    state = env.get_state(pose_idx, nan, pred_camera, tri_poses.tri_poses{j});
                    preds_all(:, :, i, j) = state.pred;
                    preds_cam_all(:, :, i, j) = state.pred_camera;
                end

                % Check for termination
                if ~obj.stop_tri && i == nbr_cameras
                    break;
                end
                if obj.stop_tri
					all_triangulated = true;
					for jj = 1 : nbr_persons
					    all_triangulated = all_triangulated && ...
											all(tri_poses.tri_poses{jj}.joints_triangulated);
					end
					if all_triangulated
					    break;
					end
                end
                i = i + 1;
            end
            cam_counter = min(i, obj.max_traj_len);
            [recon_err, recon_err_hip_aligned, preds, cams_visited, out_camera] = ...
                obj.compute_cum_recon_err(preds_all, env, tracker, ...
                                          pose_idxs_all, cam_counter, cams_visited);

            recon_errs_all = zeros(1, cam_counter);
            recon_errs_hip_aligned_all = zeros(1, cam_counter);
            if obj.compute_all_fixations
                for i = 1 : cam_counter
                    [recon_err_i, recon_err_hip_aligned_i, ~, ~, ~] = ...
                        obj.compute_cum_recon_err(preds_all, env, tracker, ...
                                                  pose_idxs_all, cam_counter, cams_visited, i);
                    recon_errs_all(i) = recon_err_i;
                    recon_errs_hip_aligned_all(i) = recon_err_hip_aligned_i;
                end
            end
        end
    end
end
