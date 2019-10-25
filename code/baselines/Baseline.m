classdef Baseline < handle
% Super-class of all baselines
    
    properties
        name
        id
        baselines
        helper
        max_traj_len
        stop_tri
        always_reset
        prev_cum_preds
        prev_cum_preds_all
        compute_all_fixations
        max_missed_curr
    end

    methods

        function obj = Baseline(baselines, name)
            global CONFIG
            obj.baselines = baselines;
            obj.name = name;
            obj.helper = Helpers();
            obj.max_traj_len = CONFIG.agent_max_traj_len;
            obj.stop_tri = CONFIG.baselines_stop_triangulated;
            obj.always_reset = CONFIG.always_reset_reconstruction;
            obj.compute_all_fixations = CONFIG.evaluation_compute_all_fixations;
        end

        function [recon_err, preds, cams_visited, prev_cum_pred] = ...
                    run(obj, env, nbr_steps, prev_cum_pred) %#ok<*INUSL,*INUSD>
            recon_err = nan;
            preds = nan;
            cams_visited = nan;
        end

        function reset_prev_cums(obj, tracker)
            obj.prev_cum_preds = cell(obj.max_traj_len, tracker.nbr_persons);
            obj.prev_cum_preds_all = cell(obj.max_traj_len, tracker.nbr_persons);
            for traj = 1 : obj.max_traj_len
                for pid = 1 : tracker.nbr_persons
                    obj.prev_cum_preds{traj, pid} = nan;
                    obj.prev_cum_preds_all{traj, pid} = nan;
                end
            end
        end

        function [recon_err, recon_err_hip_aligned, preds, cams_visited, out_camera] = ...
                compute_cum_recon_err(obj, preds_all, env, tracker, ...
                                      pose_idxs_all, cam_counter, cams_visited, ...
                                      nbr_steps, update_prev_cum)

            % Default args
            if ~exist('nbr_steps', 'var')
                nbr_steps = cam_counter;
            end
            if ~exist('update_prev_cum', 'var')
                update_prev_cum = 1;
            end

            nbr_persons = tracker.nbr_persons;
            cams_visited = cams_visited(1 : cam_counter);
            recon_errs = nan(1, nbr_persons);
            recon_errs_hip_aligned = nan(1, nbr_persons);
            for i = 1 : nbr_persons
                env.goto_person(i);
                prev_cum_pred = obj.prev_cum_preds{nbr_steps, i};
                preds = preds_all(:, :, 1 : nbr_steps, i);
                pose_idxs = pose_idxs_all(1 : nbr_steps, i);
                if ~isscalar(prev_cum_pred)
                    preds = cat(3, preds, prev_cum_pred);
                end
                qualifier = (pose_idxs ~= -1);
                current_is_garbage = ~isempty(qualifier) && all(qualifier == 0);
                if current_is_garbage
                    % Ensure non-collapse
                    qualifier(1) = 1;
                end
                last_sane_idx = find(qualifier, 1, 'last');
                if ~obj.always_reset && ~isscalar(prev_cum_pred)
                    if current_is_garbage
                        % Only set current to previous cum-estimate, as the
                        % current is garbage --- qualfier 0s
                        qualifier = [0 * qualifier; 1];
                    else
                        % Ensure that the previous pose estimate is fused
                        % together with current
                        qualifier = [qualifier; 1]; %#ok<*AGROW>
                    end
                    preds(:, :, ~qualifier) = nan;

                    pred_curr = preds(:, :, last_sane_idx);
                    pred_prev = preds(:, :, end);
                    prev_idxs = isnan(pred_curr(:, 1));
                    recon3D = pred_curr;
                    recon3D(prev_idxs, :) = pred_prev(prev_idxs, :);

                else
                    recon3D = preds(:, :, last_sane_idx);
                end
                recon3D_nonan = obj.helper.infer_missing_joints(recon3D);
                cum_recon_err = env.get_recon_error(recon3D_nonan);

                % Also compute hip-aligned error
                [gt_pose, ~] = env.get_annot_calib();
                recon3D_nonan_hip_aligned = obj.helper.trans_hip_coord(recon3D_nonan, gt_pose(:, 1 : 3));
                cum_recon_err_hip_aligned = env.get_recon_error(recon3D_nonan_hip_aligned);

                recon_errs(i) = cum_recon_err;
                recon_errs_hip_aligned(i) = cum_recon_err_hip_aligned;
                if update_prev_cum
                    obj.prev_cum_preds{nbr_steps, i} = recon3D;
                end
            end
            preds = preds_all(:, :, 1 : nbr_steps, :);
            recon_err = mean(recon_errs);
            recon_err_hip_aligned = mean(recon_errs_hip_aligned);
            out_camera = cams_visited(end);
        end
    end
end
