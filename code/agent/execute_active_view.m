function out = execute_active_view(env, agent, tracker, tri_poses, stats, greedy, ...
                                   ep, recorder, do_record)
% Produce active-view using ACTOR

global CONFIG

% Keep track of environment's initial camera index
init_cam_idx = env.camera_idx;

% Initialize some things
all_recon_errs = cell(1, tracker.nbr_persons);
all_recon_errs_hip_aligned = cell(1, tracker.nbr_persons);
all_reproj_errs = cell(1, tracker.nbr_persons);
all_joint_fracs = cell(1, tracker.nbr_persons);
traj_len = 0;
action.binary = 0;
elev_angle_max = env.scene().camera_rig_individual.elev_angle_span_half;
agent.set_elev_angle_max(elev_angle_max);
% At initial time-step, set agent angle canvas at azimuth angle 0 and
% initial global elevation angle to indicate to agent where it starts
global_angles_cam = env.global_angles_cam();
agent.update_angle_canvas(0, global_angles_cam(2) ...
							 - env.scene().camera_rig_individual.elev_angle_mean);

% Initialize camera rig canvas
agent.init_rig_canvas(env);

% Reset triangulated pose for current time-freeze
tri_poses.reset();

% Begin agent-environment interaction
while ~action.binary && traj_len < CONFIG.agent_max_traj_len

    % Get next state and triangulate the persons
    pose_idxs = tracker.get_detections();
    P_mat = env.scene().camera_calibrations{env.camera_idx}.P;
    state = cell(1, tracker.nbr_persons);

    % Used for episode recorder to track 2dpreds
    preds = cell(tracker.nbr_persons, 1);
    for i = 1 : tracker.nbr_persons

        % Extract current pose index
        pose_idx = pose_idxs(i);

        % Goto current person
        env.goto_person(i);

        % Get 2D pose prediction and data blob
        [pred, blob] = env.get_current_predictor(pose_idx);
        if i > 1
            % One blob to rule them all ...
            blob = nan;
        end

        % Add to triangulated pose
        tri_poses.tri_poses{i}.add(pred, P_mat, env.camera_idx);
        preds{i} = pred;

        % Update state
        state{i} = env.get_state(pose_idx, blob, pred, tri_poses.tri_poses{i});
    end
    agent.pose_idxs = [agent.pose_idxs; pose_idxs];
    agent.visited_cams = [agent.visited_cams, env.camera_idx];

    % Take next action based on state
    [action, data_in] = agent.take_action(state, greedy, env);
    agent.update_hist(ep, data_in, action)

	% Insert prediction error
    if do_record || traj_len == 0 || (action.binary || traj_len >= CONFIG.agent_max_traj_len - 1)
        [agent_recons_nonan, ~] = agent.get_reconstruction();
        for i = 1 : tracker.nbr_persons
            env.goto_person(i);
            all_recon_errs{i} = [all_recon_errs{i}, env.get_recon_error(...
                                 agent_recons_nonan{i})]; %#ok<*AGROW>
            if do_record
                all_reproj_errs{i} = [all_reproj_errs{i}, env.get_reproj_error(...
                                     agent_recons_nonan{i})];
            end
            frac_joints_triangulated = nnz(agent.joint_counters{i} >= 2) / ...
                                        numel(agent.joint_counters{i});
            all_joint_fracs{i} = [all_joint_fracs{i}, frac_joints_triangulated];
            if do_record
                [gt_pose, ~] = env.get_annot_calib();
                pose_hip_centered = ...
                    agent.helpers.trans_hip_coord(agent_recons_nonan{i}, gt_pose(:, 1 : 3));
                all_recon_errs_hip_aligned{i} = [all_recon_errs_hip_aligned{i}, ...
                                                 env.get_recon_error(...
                                                 pose_hip_centered)];
            end
        end
    end

	% Go to sampled camera
    old_cam_idx = env.camera_idx;
    if do_record
        old_frame_annots = cell(1, tracker.nbr_persons);
        gt_pose_idxs = nan(1, tracker.nbr_persons);
        for i = 1 : tracker.nbr_persons
            env.goto_person(i);
            old_frame_annots{i} = env.get_frame_annot();
            [gt_pose_idx, ~] = env.scene().get_gt_detection(env.frame_idx, ...
                                                           env.camera_idx, i);
			gt_pose_idxs(i) = gt_pose_idx;
        end
        old_img = env.get_current_img();
        agent_angles_global = env.global_angles_cam();
    end
	[azim_angle_global, elev_angle_global] = ...
		env.agent_angles_to_global(action.azim_angle, ...
								   action.elev_angle);
    env.goto_cam_mises(azim_angle_global, elev_angle_global);

    % Strategy for what to do if visiting same camera twice during TESTING
    if greedy
        if strcmp(CONFIG.agent_eval_same_cam_strategy, 'random')
            while any(agent.visited_cams == env.camera_idx)
                env.goto_cam(randi(env.scene().nbr_cameras))
            end
        elseif strcmp(CONFIG.agent_eval_same_cam_strategy, 'continue')
            if agent.visited_cams == env.camera_idx
                action.binary = 1;
            end
        end
    end

    next_cam_angles = env.global_angles_cam();
    if (action.binary || traj_len + 1 == CONFIG.agent_max_traj_len)
        env.goto_cam(old_cam_idx);
    else
        % We actually chose new camera, update tracking
        tracker.next(env.frame_idx, env.camera_idx);
    end

	% Record the episode steps (if final trajectory)
    if do_record
	    recorder.record_step(action.azim_angle, action.elev_angle, azim_angle_global, ...
	                         elev_angle_global, agent_angles_global(1), ...
	                         agent_angles_global(2), next_cam_angles(1), ...
	                         next_cam_angles(2), ...
	                         env.scene().camera_rig.cam_coords(old_cam_idx, :), ...
	                         all_recon_errs, all_recon_errs_hip_aligned, ...
                             all_reproj_errs, old_frame_annots, old_img, ...
	                         env.scene(), old_cam_idx, env.frame_idx, ...
                             pose_idxs, gt_pose_idxs, preds);
    end

    % Collect the camera chosen by the agent
    traj_len = traj_len + 1;
    stats.s('Selected cameras dist.').collect(env.camera_idx);

    % End of step updates
    stats.next_step();
end

% Perform reward computation at very end of trajectory
reward_input = all_joint_fracs;

% Reward is that of the person whose final recon errors was highest /
% worst OR if using the joint-triangulation mode we stimply feed
% the element-wise minimum
reward_input_mat = cell2mat(reward_input');
reward_input = min(reward_input_mat, [], 1);
rewards = get_reward(reward_input, agent.visited_cams, tracker.nbr_persons);
agent.update_hist_reward(ep, rewards);
episode_reward = sum(rewards.mises(~isnan(rewards.mises)));

% The output may vary a bit depending on the mode
sum_cum_recon_error = 0;
sum_init_recon_errs = 0;
sum_cum_recon_error_hip_centered = 0;
sum_cum_recon_error_all = zeros(1, traj_len);
sum_cum_recon_error_hip_centered_all = zeros(1, traj_len);

for i = 1 : tracker.nbr_persons
    env.goto_person(i);
    all_rec_errs_i = all_recon_errs{i};
    preds_seq = agent.predictions3d_seq{i};
    sum_init_recon_errs = sum_init_recon_errs + all_rec_errs_i(1);
    pose_reconstruction = agent.helpers.infer_missing_joints(preds_seq(:, :, 1));
    cum_recon_error = env.get_recon_error(pose_reconstruction);
    sum_cum_recon_error = sum_cum_recon_error + cum_recon_error;

    % Also track hip-centered error
    [gt_pose, ~] = env.get_annot_calib();
    pose_hip_centered = agent.helpers.trans_hip_coord(pose_reconstruction, gt_pose(:, 1 : 3));
    sum_cum_recon_error_hip_centered = sum_cum_recon_error_hip_centered + ...
                                        env.get_recon_error(pose_hip_centered);

    % Potentially do it for 1, ..., N-1 too (for speed-boost in eval)
    if CONFIG.evaluation_compute_all_fixations
        for j = 1 : traj_len
            preds_seq_j = agent.predictions3d_seq_all{j}{i};
            pose_reconstruction = agent.helpers.infer_missing_joints(preds_seq_j(:, :, 1));
            cum_recon_error_j = env.get_recon_error(pose_reconstruction);
            sum_cum_recon_error_all(j) = sum_cum_recon_error_all(j) + cum_recon_error_j;

            % Also track hip-centered error
            [gt_pose, ~] = env.get_annot_calib();
            pose_hip_centered = agent.helpers.trans_hip_coord(pose_reconstruction, gt_pose(:, 1 : 3));
            sum_cum_recon_error_hip_centered_all(j) = sum_cum_recon_error_hip_centered_all(j) + ...
                                                        env.get_recon_error(pose_hip_centered);
        end
    end
end
stats.collect({'Reconstruction error RL', sum_cum_recon_error / tracker.nbr_persons, ...
               'Reward', episode_reward, ...
               'Hip-aligned error RL', sum_cum_recon_error_hip_centered / ...
                                       tracker.nbr_persons});
stats.collect({'Traj len', traj_len - 1});
stats.collect({'Hist TLen', traj_len - 1});
out = struct('init_cam_idx', init_cam_idx, 'episode_reward', episode_reward, ...
             'traj_len', traj_len, 'recorder', recorder, ...
             'cum_recon_error', sum_cum_recon_error / tracker.nbr_persons, ...
             'init_recon_error', sum_init_recon_errs / tracker.nbr_persons, ...
             'cum_recon_error_hip_aligned', sum_cum_recon_error_hip_centered / tracker.nbr_persons, ...
             'cum_recon_error_all', sum_cum_recon_error_all / tracker.nbr_persons, ...
             'cum_recon_error_hip_aligned_all', sum_cum_recon_error_hip_centered_all / tracker.nbr_persons);
end
