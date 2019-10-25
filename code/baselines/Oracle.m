classdef Oracle < Baseline
% Oracle baseline. Cheats by using 3d ground-truth.

    properties
    end

    methods

        function obj = Oracle(baselines)
        	obj = obj@Baseline(baselines, 'oracle');
        end

        function [recon_err, recon_err_hip_aligned, preds, cams_visited, out_camera, ...
                    recon_errs_all, recon_errs_hip_aligned_all] = run(obj, env, tracker, tri_poses, nbr_cameras)

            % Initialize containers
            nbr_persons = tracker.nbr_persons;
            preds_all = nan(15, 3, obj.max_traj_len, nbr_persons);
            preds_cam_all = nan(15, 2, obj.max_traj_len, nbr_persons);
            pose_idxs_all = nan(obj.max_traj_len, nbr_persons);
            cams_visited = nan(1, obj.max_traj_len);

            % Extract things relevant from the environment
            best_cam_ord = env.scenes{env.scene_idx}.best_cam_ord(env.frame_idx, :);

            % Run algorithm
            i = 1;
            while i <= obj.max_traj_len
                if i > 1

                    % Greedily add camera which reduces avearge 3D error the most
                    min_err = 999999;
                    min_idx = 0;
                    best_cams_setdiff = setdiff(best_cam_ord, cams_visited, 'stable');
                    for j = 1 : numel(best_cams_setdiff)

                        % Goto candidate camera
                        camera_idx = best_cams_setdiff(j);
                        env.goto_cam(camera_idx);
                        tracker.next(env.frame_idx, env.camera_idx);
                        pose_idxs = tracker.get_detections();
                        pose_idxs_candidate = pose_idxs_all;
                        pose_idxs_candidate(i, :) = pose_idxs;
                        P_mat = env.scene().camera_calibrations{env.camera_idx}.P;

                        preds_candidate = preds_all;
                        preds_cam_candidate = preds_cam_all;

                        for jj = 1 : nbr_persons

                            % Extract current pose index
                            pose_idx = pose_idxs(jj);

                            % Goto current person
                            env.goto_person(jj);

                            % Get 2D pose prediction and data blob
                            [pred_camera, ~] = env.get_current_predictor(pose_idx);

                            % Add to triangulated pose
                            tri_poses.tri_poses{jj}.add(pred_camera, P_mat, env.camera_idx);

                            state = env.get_state(pose_idx, nan, pred_camera, tri_poses.tri_poses{jj});
                            preds_candidate(:, :, i, jj) = state.pred;
                            preds_cam_candidate(:, :, i, jj) = state.pred_camera;
                        end

                        % Check if new best camera
                        [recon_err, ~, ~, ~, ~] = ...
                            obj.compute_cum_recon_err(preds_candidate, env, tracker, ...
                                                      pose_idxs_candidate, i, cams_visited, i, 0);

                        if recon_err < min_err
                            min_err = recon_err;
                            min_idx = camera_idx;
                        end

                        % Revert
                        tracker.remove();
                        tri_poses.remove();
                    end

                    % Goto winning camera
                    camera_idx = min_idx;
                    env.goto_cam(camera_idx);
                    tracker.next(env.frame_idx, env.camera_idx);
                else
                    camera_idx = env.camera_idx;
                end

                % Get 2D pose prediction and data blob
                pose_idxs = tracker.get_detections();
                pose_idxs_all(i, :) = pose_idxs;
                P_mat = env.scene().camera_calibrations{env.camera_idx}.P;

                cams_visited(i) = camera_idx;
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
