 classdef EpisodeRecorder < handle
    % Records the actions taken for an episode + visulizes them

    properties
        env
        agent
        frames
        helper
    end

    methods

        function obj = EpisodeRecorder(env, agent)
            obj.env = env;
            obj.agent = agent;
            obj.helper = Helpers();
            obj.reset()
        end

        function reset(obj)
            obj.frames = {};
            obj.next_frame();
        end

        function next_frame(obj)
            frame = struct;
            frame.recon_errors = [];
            frame.recon_errors_hip_aligned = [];
            frame.recon_errors_reproj = [];
            frame.azimuths = [];
            frame.elevations = [];
            frame.global_azimuths = [];
            frame.global_elevations = [];
            frame.cams_visited = [];
            frame.prev_azims_cam = [];
            frame.next_azims_cam = [];
            frame.prev_elevs_cam = [];
            frame.next_elevs_cam = [];
            frame.cum_recons = {};
            frame.joint_counters = {};
            frame.frame_annots = {};
            frame.views = [];
            frame.scene = nan;
            frame.frame_idx = nan;
            frame.camera_idx = [];
            frame.pose_idxs = [];
            frame.gt_pose_idxs = [];
            frame.preds2d = {};
            obj.frames{end + 1} = frame;
        end

        function record_step(obj, azim_angle, elev_angle, azim_global, elev_global, ...
                             prev_azim_cam, prev_elev_cam, next_azim_cam, ...
                             next_elev_cam, cam_coords, recon_errors, ...
                             recon_errors_hip_aligned, recon_errors_reproj, ...
                             frame_annots, img, scene, cam_idx, frame_idx, ...
                             pose_idxs, gt_pose_idxs, preds2d)
            frame = obj.frames{end};
            frame.azimuths = [frame.azimuths; azim_angle];
            frame.global_azimuths = [frame.global_azimuths; azim_global];
            frame.elevations = [frame.elevations; elev_angle];
            frame.global_elevations = [frame.global_elevations; elev_global];
            frame.prev_azims_cam = [frame.prev_azims_cam; prev_azim_cam];
            frame.prev_elevs_cam = [frame.prev_elevs_cam; prev_elev_cam];
            frame.next_azims_cam = [frame.next_azims_cam; next_azim_cam];
            frame.next_elevs_cam = [frame.next_elevs_cam; next_elev_cam];
            frame.cams_visited = [frame.cams_visited; cam_coords];
            curr_recon_errs = nan(1, numel(recon_errors));
            curr_recon_errs_hip_aligned = nan(1, numel(recon_errors));
            curr_recon_errs_reproj = nan(1, numel(recon_errors));
            for i = 1 : numel(recon_errors)
                curr_recon_errs(i) = recon_errors{i}(end);
                curr_recon_errs_hip_aligned(i) = recon_errors_hip_aligned{i}(end);
                curr_recon_errs_reproj(i) = recon_errors_reproj{i}(end);
            end
            frame.recon_errors = [frame.recon_errors; curr_recon_errs];
            frame.recon_errors_hip_aligned = [frame.recon_errors_hip_aligned; curr_recon_errs_hip_aligned];
            frame.recon_errors_reproj = [frame.recon_errors_reproj; curr_recon_errs_reproj];
            [~, cum_recon] = obj.agent.get_reconstruction();
            frame.cum_recons{end + 1} = cum_recon;
            frame.joint_counters{end + 1} = obj.agent.joint_counters;
            frame.frame_annots{end + 1} = frame_annots;
            frame.views{end + 1} = img;
            frame.scene = scene;
            frame.frame_idx = frame_idx;
            frame.camera_idx = [frame.camera_idx; cam_idx];
            frame.pose_idxs = [frame.pose_idxs; pose_idxs];
            frame.gt_pose_idxs = [frame.gt_pose_idxs; gt_pose_idxs];
            frame.preds2d{end + 1} = preds2d;
            obj.frames{end} = frame;
        end

        function save_path = plot(obj, extra_path)
            global CONFIG

            if ~CONFIG.use_recorder
                % Unless we should save; we do nothing.
                return;
            end
            if ~exist('extra_path', 'var')
                extra_path = '';
            end

            % Prepare saving full video
            save_video = CONFIG.recorder_save_video;
            if save_video
                save_path = strcat(CONFIG.output_dir, ...
                                    'episode-recorder/agent/');
                mkdir(save_path);
                video_path = strcat(save_path, 'full_episode.avi');
                full_video_file = VideoWriter(video_path);
                full_video_file.FrameRate = CONFIG.recorder_frame_rate;
                full_video_file.Quality = 100;
                open(full_video_file);

                full_video_frames = {};
            end

            % Settings
            fontsize_label = 15;
            fontsize_title = 20;
            fontsize_tick = fontsize_label;
            linewidth = 1.5;
            markersize = 7;

            % Create colors for persons
            colors = [0.0000    0.4470    0.7410;
                      0.8500    0.3250    0.0980;
                      0.9290    0.6940    0.1250;
                      0.4940    0.1840    0.5560;
                      0.4660    0.6740    0.1880;
                      0.3010    0.7450    0.9330;
                      0.6350    0.0780    0.1840;
                      1.0000    0.4000    0.6980];

            total_frame_data = struct;
            total_frame_data.error = [];
            total_frame_data.error_hip_aligned = [];
            total_frame_data.error_reproj = [];
            total_frame_data.joints = [];
            total_frame_data.lines = [];

            for active_view_idx = 1:numel(obj.frames)

                % Get current frame
                frame = obj.frames{active_view_idx};

                % Create save directory
                if numel(extra_path) == 0
                    save_path = strcat(CONFIG.output_dir, ...
                                'episode-recorder/agent/frame_', num2str(active_view_idx), '/');
                else
                    save_path = strcat(CONFIG.output_dir, ...
                                'episode-recorder/agent/', extra_path, '/frame_', num2str(active_view_idx), '/');
                end
                if ~exist(save_path, 'dir')
                    mkdir(save_path);
                end

                % Save info
                fileID = fopen(strcat(save_path, '/info.txt'), 'w');
                fprintf(fileID,'Scene: %s, frame: %d, start cam: %d\n', frame.scene.scene_name, frame.frame_idx, frame.camera_idx(1));
                fclose(fileID);
                errorFile = fopen(strcat(save_path, '/errors.txt'), 'w');

                % Prepare saving video file for current frame
                if save_video
                    video_path = strcat(save_path, '/episode.avi');
                    video_file = VideoWriter(video_path);
                    video_file.FrameRate = CONFIG.recorder_frame_rate;
                    video_file.Quality = 100;
                    open(video_file);
                end

                % Regarless of whether we want to save video or not we create
                % the video data structure, this is used for pictures too
                nbr_video_frames = size(frame.cams_visited, 1);
                video_frames = cell(nbr_video_frames, 1);

                if nbr_video_frames > 2
                    x_max = nbr_video_frames;
                else
                    x_max = Inf;
                end
                ticks_hip = [25 50 100 250 500 750];
                ticks_reproj = [25 50 100 250 500 750];
                xlim_cams = 2;

                for f = 1:nbr_video_frames
                    video_frames{f}.figure = figure('visible', 'off', 'Renderer', 'painters', 'Position', [0 0 1920 1080]);
                    video_frames{f}.img = subplot(2, 4, 1); title('2d Pose Estimate', 'FontSize', fontsize_title); axis off; obj.resize_fig(gca, 1.2);hold on;
                    video_frames{f}.recon = subplot(2, 4, [2, 3]); title('3d Pose Reconstruction', 'FontSize', fontsize_title); axis off; obj.resize_fig(gca, 1.2); hold on;
                    video_frames{f}.rig = subplot(2, 4, 4); title('Camera Locations', 'FontSize', fontsize_title); set(gca,'FontSize', fontsize_tick); axis off; obj.resize_fig(gca, 1.2); hold on;
                    %video_frames{f}.error = subplot(2, 4, 5); xlim([xlim_cams, x_max]); ylim([50, 2000]); set(gca, 'YTick', ticks_err); set(gca, 'YScale', 'log'); title('3d Reconstruction Error', 'FontSize', fontsize_title, 'Units', 'normalized', 'Position', [0.5, 1.1, 0]); xlabel('# camera views', 'FontSize', fontsize_label, 'FontWeight','bold'), ylabel('Error (mm / joint)', 'FontSize', fontsize_label, 'FontWeight','bold'); set(gca,'FontSize', fontsize_tick); set(gca,'linewidth',linewidth); hold on;
                    video_frames{f}.error_hip_aligned = subplot(2, 4, [6, 7]); xlim([xlim_cams, x_max]);  ylim([25, 750]); set(gca, 'YTick', ticks_hip); set(gca, 'YScale', 'log'); title({'Hip-Aligned', '3d Reconstruction Error'}, 'FontSize', fontsize_title, 'Units', 'normalized', 'Position', [0.5, 1.1, 0]); xlabel('# camera views', 'FontSize', fontsize_label, 'FontWeight','bold'), ylabel('Error (mm / joint)', 'FontSize', fontsize_label, 'FontWeight', 'bold'); set(gca,'FontSize', fontsize_tick); set(gca,'linewidth',linewidth); hold on;
                    video_frames{f}.error_reproj = subplot(2, 4, 5); xlim([xlim_cams, x_max]); ylim([25, 750]);  set(gca, 'YTick', ticks_reproj); set(gca, 'YScale', 'log'); title('2d Reprojection Error', 'FontSize', fontsize_title, 'Units', 'normalized', 'Position', [0.5, 1.1, 0]); xlabel('# camera views', 'FontSize', fontsize_label, 'FontWeight','bold'), ylabel('Error (pixels / joint)', 'FontSize', fontsize_label, 'FontWeight','bold'); set(gca,'FontSize', fontsize_tick); set(gca,'linewidth',linewidth); hold on;
                    video_frames{f}.joints = subplot(2, 4, 8); xlim([1, nbr_video_frames]); ylim([0, 15]); title('Triangulated Joints', 'FontSize', fontsize_title, 'Units', 'normalized', 'Position', [0.5, 1.1, 0]); xlabel('# camera views', 'FontSize', fontsize_label, 'FontWeight', 'bold'); ylabel('# joints', 'FontSize', fontsize_label, 'FontWeight','bold'); set(gca,'FontSize', fontsize_tick); set(gca,'linewidth',linewidth); hold on;
                end

                % Compute nbr joints triangulated
                [nbr_steps, nbr_persons] = size(frame.recon_errors);
                nbr_triangulateds = nan(nbr_steps, nbr_persons);
                for i = 1 : nbr_steps
                    for j = 1 : nbr_persons
                        nbr_triangulateds(i, j) = nnz(frame.joint_counters{i}{j} >= 2);
                    end
                end

                % Create video frames
                for f = 1:nbr_video_frames

                    % Plot hip-aligned errors
                    ax = video_frames{f}.error_hip_aligned;
                    errors = frame.recon_errors_hip_aligned(1:f, :);
                    if active_view_idx == 1 && xlim_cams == 2
                        errors(1, :) = nan;
                    end

                    total_frame_data.error_hip_aligned = [total_frame_data.error_hip_aligned; errors(f, :)];
                    for i = 1 : size(frame.recon_errors_hip_aligned, 2)
                        plot(total_frame_data.error_hip_aligned(:, i), '-*', 'color', colors(i, :), 'Parent', ax, 'LineWidth', linewidth, 'MarkerSize', markersize);
                    end
                    plot(mean(total_frame_data.error_hip_aligned, 2), '--o', 'color', 'k', 'Parent', ax, 'LineWidth', linewidth, 'MarkerSize', markersize);
                    obj.plot_ep_lines(ax, total_frame_data.lines, max(max(total_frame_data.error_hip_aligned)) * 1.1, linewidth);

                    % Plot OpenPose reprojection errors
                    ax = video_frames{f}.error_reproj;
                    errors = frame.recon_errors_reproj(1:f, :);
                    if active_view_idx == 1 && xlim_cams == 2
                        errors(1, :) = nan;
                    end

                    total_frame_data.error_reproj = [total_frame_data.error_reproj; errors(f, :)];
                    for i = 1 : size(frame.recon_errors_reproj, 2)
                        plot(total_frame_data.error_reproj(:, i), '-*', 'color', colors(i, :), 'Parent', ax, 'LineWidth', linewidth, 'MarkerSize', markersize);
                    end
                    plot(mean(total_frame_data.error_reproj, 2), '--o', 'color', 'k', 'Parent', ax, 'LineWidth', linewidth, 'MarkerSize', markersize);
                    obj.plot_ep_lines(ax, total_frame_data.lines, max(max(total_frame_data.error_reproj)) * 1.1, linewidth);

                    % Plot nbr triangulated joints
                    ax = video_frames{f}.joints;
                    total_frame_data.joints = [total_frame_data.joints; nbr_triangulateds(f, :)];
                    for i = 1 : nbr_persons
                        plot(total_frame_data.joints(:, i), '-*', 'color', colors(i, :), 'Parent', ax, 'LineWidth', linewidth, 'MarkerSize', markersize);
                    end
                    mean_joints_triangulated = mean(total_frame_data.joints, 2);
                    plot(mean_joints_triangulated, '--o', 'color', 'k', 'Parent', ax, 'LineWidth', linewidth, 'MarkerSize', markersize);

                    % Plot camera choices on rig
					save_path_curr = strcat(save_path, 'cam_choices/');
					if ~exist(save_path_curr, 'dir')
						mkdir(save_path_curr);
					end
                    set(0, 'currentfigure', video_frames{f}.figure);
                    set(video_frames{f}.figure, 'currentaxes', video_frames{f}.rig);
                    cameras = frame.camera_idx(1:f);
                    frame.scene().camera_rig.show_cam_choices(cameras, video_frames{f}.figure);
                    fig = frame.scene().camera_rig.show_cam_choices(cameras);
                    obj.save_plot(fig, sprintf('%s/cam_choices_%d', save_path_curr, f));

                    % Plot RGB image + open pose projections
                    set(0, 'currentfigure', video_frames{f}.figure);
                    set(video_frames{f}.figure, 'currentaxes', video_frames{f}.img);
                    imshow(frame.views{f});
                    pose2Ds = frame.preds2d{f};
                    for j = 1 : numel(pose2Ds)
                        obj.helper.plot_skeleton(pose2Ds{j}, colors(j, :), 20, 3);
                    end

                    % Plot 3d reconstruction
                    set(0, 'currentfigure', video_frames{f}.figure);
                    set(video_frames{f}.figure, 'currentaxes', video_frames{f}.recon);
                    pose3Ds = frame.cum_recons{f};
                    for j = 1 : numel(pose3Ds)
                        obj.helper.plot_skeleton(pose3Ds{j}, colors(j, :));
                    end

                    % Write frame number
                    annotation('textbox', [0.85, 0.04, 0.15, 0.0], 'string', sprintf('Active-view: %d', active_view_idx), 'Parent', video_frames{f}.figure, 'FontSize', fontsize_title, 'FontWeight', 'bold', 'FitBoxToText', 'on', 'LineStyle', 'none')
                end

                % Save errors to text
                obj.print_errors(errorFile, frame.recon_errors, '3d Reconstruction Errors');
                obj.print_errors(errorFile, frame.recon_errors_hip_aligned, '3d Hip-Algined Reconstruction Errors');
                obj.print_errors(errorFile, frame.recon_errors_reproj, '2d Reprojection Errors');
                obj.print_errors(errorFile, nbr_triangulateds, '# joints triangulated');
                fclose(errorFile);

                % Hip-aligned reconstruction error
                f = figure('visible', 'off');
                h = copyobj(video_frames{nbr_video_frames}.error_hip_aligned, f);
                set(h, 'pos', [0.23162 0.2233 0.72058 0.63107])
                obj.save_plot(f, strcat(save_path, '/errors_hip'));
                close(f);

                % 2d reprojection error
                f = figure('visible', 'off');
                h = copyobj(video_frames{nbr_video_frames}.error_reproj, f);
                set(h, 'pos', [0.23162 0.2233 0.72058 0.63107])
                obj.save_plot(f, strcat(save_path, '/errors_2d_reproj'));
                close(f);

                % Triangulated joints
                f = figure('visible', 'off');
                h = copyobj(video_frames{nbr_video_frames}.joints, f);
                set(h, 'pos', [0.23162 0.2233 0.72058 0.63107])
                obj.save_plot(f, strcat(save_path, '/joints'));
                close(f);

                % 3d reconstruction -- redo from scratch for optimal graphics
                for f = 1:nbr_video_frames
                    fig = figure('visible', 'off');
                    pose3Ds = frame.cum_recons{f};
                    for j = 1 : numel(pose3Ds)
                        obj.helper.plot_skeleton(pose3Ds{j}, colors(j, :));
                    end
					save_path_curr = strcat(save_path, 'recons_no_plane/');
					if ~exist(save_path_curr, 'dir')
						mkdir(save_path_curr);
					end
                    obj.save_plot(fig, sprintf('%s/recon_%d', save_path_curr, f));
                    obj.helper.plot_ground_plane(frame.cum_recons{nbr_video_frames}, 'red');
					save_path_curr = strcat(save_path, 'recons_with_plane/');
					if ~exist(save_path_curr, 'dir')
						mkdir(save_path_curr);
					end
                    obj.save_plot(fig, sprintf('%s/recon_%d_plane', save_path_curr, f));
                    close(fig);
                end

                % Write rgb to output
                for f = 1:nbr_video_frames

                    % Write untouched RGB
					save_path_curr = strcat(save_path, 'views/');
					if ~exist(save_path_curr, 'dir')
						mkdir(save_path_curr);
					end
                    imwrite(frame.views{f}, sprintf('%s/view_%d.png', save_path_curr, f));

                    % Write RGB with openpose joints
                    fig = figure('visible', 'off');
                    imshow(frame.views{f}); hold on;
                    pose2Ds = frame.preds2d{f};
                    for j = 1 : numel(pose2Ds)
                        obj.helper.plot_skeleton(pose2Ds{j}, colors(j, :), 20, 3);
                    end
					save_path_curr = strcat(save_path, 'views_openpose/');
					if ~exist(save_path_curr, 'dir')
						mkdir(save_path_curr);
					end
                    obj.save_plot(fig, sprintf('%s/view_openpose_%d', save_path_curr, f));
                    close(fig);


                    % Write RGB with FULL BODY openpose joints
                    if CONFIG.recorder_full_body
                        fig = figure('visible', 'off');
                        imshow(frame.views{f}); hold on;
                        for person_idx = 1 : size(frame.pose_idxs, 2)
                             pose_idx = frame.pose_idxs(f, person_idx);
                             if pose_idx == -1
                                continue
                             end
                             camera_index = frame.camera_idx(f);
                             full_pose = frame.scene.get_full_pose(frame.frame_idx, camera_index, pose_idx);
                             obj.helper.plot_fullbody(0, full_pose, colors(person_idx, :));
                        end
                        save_path_curr = strcat(save_path, 'views_full_openpose/');
                        if ~exist(save_path_curr, 'dir')
                            mkdir(save_path_curr);
                        end
                        obj.save_plot(fig, sprintf('%s/view_openpose_%d', save_path_curr, f));
                        close(fig);
                    end

                    % Write RGB with projected 3d reconstruction
                    fig = figure('visible', 'off');
                    imshow(frame.views{f}); hold on;
                    pose3Ds = frame.cum_recons{f};
                    for j = 1 : numel(pose3Ds)
                        pose = frame.scene.project_frame_pose(pose3Ds{j}, frame.camera_idx(f));
                        obj.helper.plot_skeleton(pose, colors(j, :), 20, 3);
                    end
					save_path_curr = strcat(save_path, 'views_3d_proj/');
					if ~exist(save_path_curr, 'dir')
						mkdir(save_path_curr);
					end
                    obj.save_plot(fig, sprintf('%s/view_3d_proj_%d', save_path_curr, f));
                    close(fig);

                    if CONFIG.recorder_full_body
                        % Write RGB with full-body projected 3d reconstruction
                        save_path_curr = strcat(save_path, 'views_full_3d_proj/');
                        if ~exist(save_path_curr, 'dir')
                            mkdir(save_path_curr);
                        end
                        fig = figure('visible', 'off');
                        imshow(frame.views{f}); hold on;
                        pose3Ds = obj.triangulate_full_pose(frame, f);
                        for j = 1 : numel(pose3Ds)
                            projected = struct;
                            projected.body = frame.scene.project_frame_pose(pose3Ds{j}.body, frame.camera_idx(f));
                            projected.face = frame.scene.project_frame_pose(pose3Ds{j}.face, frame.camera_idx(f));
                            projected.left_hand = frame.scene.project_frame_pose(pose3Ds{j}.left_hand, frame.camera_idx(f));
                            projected.right_hand = frame.scene.project_frame_pose(pose3Ds{j}.right_hand, frame.camera_idx(f));

                            obj.helper.plot_fullbody(0, projected, colors(j, :));
                        end
                        obj.save_plot(fig, sprintf('%s/view_full_3d_proj_%d', save_path_curr, f));
                        close(fig);
                        
                        % 3d reconstruction -- full body
                        fig = figure('visible', 'off');
                        for  j = 1 : numel(pose3Ds)
                             obj.helper.plot_fullbody(1, pose3Ds{j}, colors(j, :));
                        end
                        final_poses3d_full = obj.triangulate_full_pose(frame, nbr_video_frames);
                        obj.helper.plot_ground_plane_full(final_poses3d_full, 'red');
                        save_path_curr = strcat(save_path, 'recons_with_plane_full_body/');
                        if ~exist(save_path_curr, 'dir')
                            mkdir(save_path_curr);
                        end
                        obj.save_plot(fig, sprintf('%s/recon_%d_plane', save_path_curr, f));
                        close(fig);        

                    end

                    % Write RGB with projected hip-aligned 3d reconstruction
                    fig = figure('visible', 'off');
                    imshow(frame.views{f}); hold on;
                    pose3Ds = frame.cum_recons{f};
                    for j = 1 : numel(pose3Ds)
                        [gt_pose, ~] = frame.scene.get_annot_calib(frame.frame_idx, j);
                        pose_hip_centered = ...
                            obj.helper.trans_hip_coord(pose3Ds{j}, gt_pose(:, 1 : 3));
                        pose_hip_centered = frame.scene.project_frame_pose(pose_hip_centered, frame.camera_idx(f));
                        obj.helper.plot_skeleton(pose_hip_centered, colors(j, :), 20, 3);
                    end
					save_path_curr = strcat(save_path, 'views_aligned_3d_proj/');
					if ~exist(save_path_curr, 'dir')
						mkdir(save_path_curr);
					end
                    obj.save_plot(fig, sprintf('%s/view_aligned_3d_proj_%d', save_path_curr, f));
                    close(fig);
                end

                % Write video to file
                if save_video
                    for f = 1:nbr_video_frames
                        video_frame = getframe(video_frames{f}.figure);
                        writeVideo(video_file, video_frame);
                        close(video_frames{f}.figure);

                        % Append to full video
                        full_video_frames{end + 1} = video_frame; %#ok<*AGROW>
                    end

                    % Write last frame twice
                    writeVideo(video_file, video_frame);
                    close(video_file);

                    % Reset ep recorder data between frames if requested
                    if CONFIG.recorder_reset_video
                        total_frame_data = struct;
                        total_frame_data.error = [];
                        total_frame_data.error_hip_aligned = [];
                        total_frame_data.error_reproj = [];
                        total_frame_data.joints = [];
                        total_frame_data.lines = [];
                    else
                        total_frame_data.lines = [total_frame_data.lines, numel(full_video_frames)];
                    end
                end
            end

            % Write full video to file
            if save_video
                for f = 1:numel(full_video_frames)
                   writeVideo(full_video_file, full_video_frames{f});
                end
                % Write last frame twice
                writeVideo(full_video_file, full_video_frames{f});
                close(full_video_file);
            end

            try
                delete(findall(0));
            catch

            end
        end

        function resize_fig(~, fig, factor)

            pos = get(fig, 'Position');
            width = pos(3) * factor;
            height = pos(4) * factor;
            left = pos(1) - (width - pos(3)) / 2;
            bottom = pos(2) - (height - pos(4)) / 2;
            new_pos = [left, bottom, width, height];
            set(fig, 'Position', new_pos);

        end

        function plot_ep_lines(~, ax, line_indices, y_max, linewidth)
            for i = 1:numel(line_indices)
                ep = line_indices(i) + 0.5;
                plot([ep, ep], [0, y_max], '-', 'color', [0.5, 0.5, 0.5], 'LineWidth', linewidth, 'Parent', ax);
                ylim(ax, [-Inf, y_max]);
            end
        end

        function print_errors(~, fileId, errors, error_name)
            fprintf(fileId, '%s:\n', error_name);
            for pid = 1:size(errors, 2);
                fprintf(fileId, 'Person %d: ', pid);
                for t = 1:size(errors, 1)
                    fprintf(fileId, '%0.2f ,', errors(t, pid));
                end
                fprintf(fileId, '\n');
            end
            mean_errors = mean(errors, 2);
            fprintf(fileId, 'Mean: ');
            for t = 1:size(mean_errors, 1)
                fprintf(fileId, '%0.2f ,', mean_errors(t));
            end
            fprintf(fileId, '\n\n');
        end

        function save_plot(~, fig, path_no_extension)
            print(fig, strcat(path_no_extension, '.png'), '-dpng','-r600');
%             if CONFIG.recorder_save_plot_fig_format
                 saveas(fig, strcat(path_no_extension, '.fig'), 'fig');
%                 saveas(fig, strcat(path_no_extension, '.svg'), 'svg');
%             end
        end

        function full_poses = triangulate_full_pose(~, frame, nbr_cams)

            % Collect all triangulated joints
            full_poses = cell(size(frame.pose_idxs, 2), 1);

            for person_idx = 1 : size(frame.pose_idxs, 2)
                triag_body = TriangulatedPose(25);
                triag_face = TriangulatedPose(70);
                triag_left_hand = TriangulatedPose(21);
                triag_right_hand = TriangulatedPose(21);
                for c = 1 : nbr_cams
                    pose_idx = frame.pose_idxs(c, person_idx);
                    if pose_idx == -1
                        continue
                    end

                    camera_index = frame.camera_idx(c);
                    full_pose = frame.scene.get_full_pose(frame.frame_idx, camera_index, pose_idx);

                    P = frame.scene.camera_calibrations{camera_index}.P;
                    triag_body.add(full_pose.body(:, 1:2), P, camera_index);
                    triag_face.add(full_pose.face(:, 1:2), P, camera_index);
                    triag_left_hand.add(full_pose.left_hand(:, 1:2), P, camera_index);
                    triag_right_hand.add(full_pose.right_hand(:, 1:2), P, camera_index);
                end
                full_poses{person_idx}.body = triag_body.get();
                full_poses{person_idx}.face = triag_face.get();
                full_poses{person_idx}.left_hand = triag_left_hand.get();
                full_poses{person_idx}.right_hand = triag_right_hand.get();
            end

        end

    end
 end
