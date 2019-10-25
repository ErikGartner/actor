classdef Random < Baseline
% Baseline which samples random camera locations

    properties
    end

    methods

        function obj = Random(baselines)
        	obj = obj@Baseline(baselines, 'random');
        end

        function [recon_err, recon_err_hip_aligned, preds, cams_visited, out_camera, ...
                    recon_errs_all, recon_errs_hip_aligned_all] = run(obj, env, tracker, tri_poses, nbr_cameras)

            nbr_persons = tracker.nbr_persons;
            preds_all = nan(15, 3, obj.max_traj_len, nbr_persons);
            preds_cam_all = nan(15, 2, obj.max_traj_len, nbr_persons);
            pose_idxs_all = nan(obj.max_traj_len, nbr_persons);
            cams_visited = nan(1, obj.max_traj_len);
            i = 1;
            while i <= obj.max_traj_len
                if i > 1
                    env.goto_cam(next_cam_idx);
                    tracker.next(env.frame_idx, env.camera_idx);
                end
                cams_visited(i) = env.camera_idx;
                next_cam_idx = randi(env.scene().nbr_cameras);
                while i <= env.scene().nbr_cameras && any(cams_visited == next_cam_idx)
                    next_cam_idx = randi(env.scene().nbr_cameras);
                end
                pose_idxs = tracker.get_detections();
                pose_idxs_all(i, :) = pose_idxs;
                P_mat = env.scene().camera_calibrations{env.camera_idx}.P;

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
