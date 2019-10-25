classdef Panoptic < handle
    % Representation of Panoptic. This is the environment where the agents
    % and baselines operate.

    properties
        scenes
        scene_idx
        frame_idx
        camera_idx
        person_idx
        dataset_path
        dataset_cache
        elev_angle_span_half
        helpers
        recon_err_type
    end

    methods
        function obj = Panoptic(dataset_path, dataset_cache, predictor)
            global CONFIG
            obj.dataset_path = dataset_path;
            obj.dataset_cache = dataset_cache;
            obj.helpers = Helpers();
            obj.recon_err_type = CONFIG.panoptic_error_type;

            % Check if scenes are given or setup scenes here
            obj.load_scenes(predictor);

            % Load pre-computed cache
            for i = 1 : numel(obj.scenes)
                curr_scene = obj.scenes{i};
                fprintf('Creating predictor cache for scene %s\n', curr_scene.scene_name);
                obj.scenes{i}.load_predictor_cache();
                obj.scenes{i}.init_pred_ordering();

                if CONFIG.predictor_limited_caching
                    % Unload the cache after creating it and computing
                    % the ordering.
                    obj.scenes{i}.unload_predictor_cache();
                end
            end

            % Set max elevation span
            max_elev_span = 0;
            for i = 1 : numel(obj.scenes)
                max_elev_span = max(max_elev_span, obj.scenes{i}.camera_rig.elev_angle_span_half);
            end
            obj.elev_angle_span_half = max_elev_span;

            % Reset to random state
            obj.reset()
        end

        function reset(obj)
            obj.goto_scene(randi(numel(obj.scenes)));
            obj.goto_frame(randi(obj.scene().nbr_frames));
            obj.goto_cam(randi(obj.scene().nbr_cameras));
        end

        function s = scene(obj)
            s = obj.scenes{obj.scene_idx};
        end

        function toggle_scene_predictor_cache(obj, scene_idx)
            global CONFIG

            if ~CONFIG.predictor_limited_caching
                % Only toggle cache if we have limited caching on.
                return
            end

            for i = 1 : length(obj.scenes)
                if scene_idx == i
                    obj.scenes{i}.load_predictor_cache();
                else
                    obj.scenes{i}.unload_predictor_cache();
                end
            end
        end

        function [frame_annot, calib] = get_annot_calib(obj)
            % This function returns all 3D information for the current
            % time-freeze, e.g. for plotting of the 3D scene
            [frame_annot, calib] = obj.scene().get_annot_calib(obj.frame_idx, obj.person_idx);
        end

        function goto_scene(obj, scene_idx)
            if scene_idx < 1 || scene_idx > numel(obj.scenes)
                error('scene_idx out of bounds')
            end
            if isempty(obj.scene_idx) || scene_idx ~= obj.scene_idx
                obj.scene_idx = scene_idx;
                obj.toggle_scene_predictor_cache(obj.scene_idx);
            end
        end

        function goto_person(obj, person_idx)
            if person_idx < 1 || person_idx > obj.scene().nbr_persons
                error('person_idx out of bounds')
            end
            obj.person_idx = person_idx;
        end

        function goto_frame(obj, frame_idx)
            if frame_idx < 1 || frame_idx > obj.scene().nbr_frames
                error('frame_idx out of bounds')
            end
            obj.frame_idx = frame_idx;
        end

        function goto_cam(obj, camera_idx)
            if camera_idx < 1 || camera_idx > obj.scene().nbr_cameras
                error('camera_idx out of bounds')
            end
            obj.camera_idx = camera_idx;
        end

        function [pose_idx, iou] = get_gt_detection(obj)
            [pose_idx, iou] = obj.scene().get_gt_detection(obj.frame_idx, obj.camera_idx, obj.person_idx);
        end

        function state = get_state(~, pose_idx, blob, pred_camera, tri_pose)
            % Get 3D pose prediction and data blob. Note that the 3D pose
            % prediction is returned in hip-centered and frame-level perspective

            % Get current triangulated 3D pose
            pred = tri_pose.get();
            pred_nonan = pred;
            pred_nonan(isnan(pred_nonan)) = 0;

            % Infer which joints were seen
            visible_joint_idxs = ~isnan(pred_camera(:, 1));

            % Insert to state
            pred_camera(isnan(pred_camera)) = 0;
            state.blob = blob;
            state.pred = pred;
            state.pred_camera = pred_camera;
            state.pose_idx = pose_idx;
            state.pred_nonan = pred_nonan;
            state.visible_joint_idxs = visible_joint_idxs;
            state.joints_triangulated = tri_pose.joints_triangulated;
        end

        function [pred, blob] = get_current_predictor(obj, pose_idx)
            % Returns pred, blob in non-translated and non-rotated view
            [pred, blob] = obj.scene().get_predictor(obj.frame_idx, obj.camera_idx, pose_idx);
        end

        function min_dist_cam_idx = goto_cam_mises(obj, azim_angle, elev_angle)
            % Move to a new camera / view based on predicted azimuth
            % angle azim_angle and elevation angle elev_angle

            % "Theoretical point / view" and the index of the camera
            % closest to that one in 3D
            theo_chosen_3D = obj.get_theoretical_3Dpoint(azim_angle, elev_angle, 0);
            cam_coords = obj.scene().camera_rig.cam_coords;
            [~, min_dist_cam_idx] ...
                = min(sum((bsxfun(@minus, cam_coords, theo_chosen_3D).^2), 2));
            % Now go to the actual camera
            obj.goto_cam(min_dist_cam_idx);
        end

        function theo_chosen_3D = get_theoretical_3Dpoint(obj, azim_angle, elev_angle, to_individual)
            if ~exist('to_individual', 'var')
                to_individual = 1;
            end
            if to_individual
                r = obj.scene().camera_rig_individual.radius;
                c = obj.scene().camera_rig_individual.center;
            else
                r = obj.scene().camera_rig.radius;
                c = obj.scene().camera_rig.center;
            end
            % note the minus sign in y-dir, due to upside-down rig
            theo_chosen_3D = [r * sin(elev_angle) .* cos(azim_angle) + c(1), ...
                              -r * cos(elev_angle) + c(2), ...
                              r * sin(elev_angle) .* sin(azim_angle) + c(3)];
        end

        function img = get_current_img(obj)
            % Returns current image
            img = obj.scene().get_img(obj.frame_idx, obj.camera_idx);
        end

        function frame_annot = get_frame_annot(obj)
            % Get global (frame-level) annotation for the current scene
            % (thus this needs to be adapted to camera space for the
            % specific camera currently indexed by obj.camera_idx)
            [frame_annot, ~] = obj.scene().get_annot_calib(obj.frame_idx, obj.person_idx);
        end

        function camera_annot = get_camera_annot(obj)
            % Returns the camera space annotation associated with the frame
            % indexed by obj.frame_idx (thus it rotates and translates the
            % frame-level annotation according to the camera parameters of
            % the camera indexed by obj.camera_idx)
            camera_annot = obj.scene().get_camera_annot(obj.frame_idx, obj.camera_idx, obj.person_idx);
        end

        function frame_perspective_coords = rotate_to_normal_view(obj, coords)
            % This function rotates and translates to global (frame-level)
            frame_perspective_coords = obj.scene().rotate_to_normal_view(coords, obj.camera_idx);
        end

		function pose_pred_error = get_recon_error(obj, pred, frame_idx)
            % Reconstruction error (depending on chosen metric)
            if ~exist('frame_idx', 'var')
                frame_idx = obj.frame_idx;
            end
            if strcmp(obj.recon_err_type, '3d')
                pose_pred_error = obj.get_recon_error_3d(pred, frame_idx);
            elseif strcmp(obj.recon_err_type, '2d_gt')
                pose_pred_error = obj.get_reproj_error(pred, frame_idx);
            elseif strcmp(obj.recon_err_type, '2d_op')
                pose_pred_error = obj.get_reproj_error_op(pred, frame_idx);
            end
        end

        function pose_pred_error = get_recon_error_3d(obj, pred, frame_idx)
            % This function computes the mean per-joint error in mm
            %
            % Input:
            %    - pred is a 15 x 3 matrix, where each row is (x,y,z)
            %      prediction for the resp. joint
            if exist('frame_idx', 'var')
                pose_pred_error = obj.scene().get_recon_error(pred, frame_idx, obj.person_idx);
            else
                pose_pred_error = obj.scene().get_recon_error(pred, obj.frame_idx, obj.person_idx);
            end
        end

        function pose_pred_errors = get_recon_error_perjoint(obj, pred, frame_idx)
            % This function computes the errors per joint in mm
            %
            % Input:
            %    - pred is a 15 x 3 matrix, where each row is (x,y,z)
            %      prediction for the resp. joint
            if exist('frame_idx', 'var')
                pose_pred_errors = obj.scene().get_recon_error_perjoint(pred, frame_idx, obj.person_idx);
            else
                pose_pred_errors = obj.scene().get_recon_error_perjoint(pred, obj.frame_idx, obj.person_idx);
            end
        end

        function reproj_error = get_reproj_error(obj, pred, frame_idx)
            % This function computes the mean per-joint error in mm
            %
            % Input:
            %    - pred is a 15 x 3 matrix, where each row is (x,y,z)
            %      prediction for the resp. joint
            if exist('frame_idx', 'var')
                reproj_error = obj.scene().get_reproj_error(pred, frame_idx, obj.person_idx);
            else
                reproj_error = obj.scene().get_reproj_error(pred, obj.frame_idx, obj.person_idx);
            end
        end

        function reproj_error = get_reproj_error_op(obj, pred, frame_idx)
            % This function computes the mean per-joint error in mm
            %
            % Input:
            %    - pred is a 15 x 3 matrix, where each row is (x,y,z)
            %      prediction for the resp. joint
            if exist('frame_idx', 'var')
                reproj_error = obj.scene().get_reproj_error_op(pred, frame_idx, obj.person_idx);
            else
                reproj_error = obj.scene().get_reproj_error_op(pred, obj.frame_idx, obj.person_idx);
            end
        end

        function load_scenes(obj, predictor)
            % Loads all the scenes
            global CONFIG

            % List all scenes
            [all_scene_names, nbr_all_scenes] = obj.helpers.list_filenames(obj.dataset_path);

            % Filter scenes
            if ~isempty(strfind(obj.dataset_path, 'train'))
                scene_filter = CONFIG.panoptic_scene_filter_train;
            else
                scene_filter = CONFIG.panoptic_scene_filter_eval;
            end
            scene_names = {};
            nbr_scenes = 0;
            for i = 1 : nbr_all_scenes
                ok = true;
                for j = 1 : numel(scene_filter)
                    if numel(strfind(all_scene_names{i}, scene_filter{j})) > 0
                        ok = false;
                    end
                end
                if ok
                    scene_names{end + 1} = all_scene_names{i}; %#ok<*AGROW>
                    nbr_scenes = nbr_scenes + 1;
                end
            end
            obj.scenes = cell(1, nbr_scenes);

            % Create all the scene objects
            fprintf('\nCreating scene objects ... \n\n')
            start_time = tic;
            for i = 1 : nbr_scenes
                scene_name = scene_names{i};
                fprintf('Creating scene object %d, with name %s\n', i, scene_name);
                obj.scenes{i} = Scene(scene_name, obj.dataset_path, obj.dataset_cache, predictor);
            end
            end_time = toc(start_time);
            fprintf('\nDone creating scene objects! Elapsed time: %.2f seconds\n\n', end_time);
        end

        function global_angles = global_angles_cam(obj, camera_idx)
            if ~exist('camera_idx', 'var')
                global_angles = obj.scene().global_angles_cam(obj.camera_idx);
            else
                global_angles = obj.scene().global_angles_cam(camera_idx);
            end
        end

        function [azim_angle_global, elev_angle_global] = agent_angles_to_global(obj, azim_angle, elev_angle, to_individual)
            if ~exist('to_individual', 'var')
                to_individual = 1;
            end
            agent_angles_global = obj.global_angles_cam();
            azim_angle_global = agent_angles_global(1) + azim_angle;
            azim_angle_global = angle(cos(azim_angle_global) + 1i * sin(azim_angle_global));
            if to_individual
                elev_angle_global = elev_angle + obj.scene().camera_rig_individual.elev_angle_mean;
            else
                elev_angle_global = elev_angle + obj.scene().camera_rig.elev_angle_mean;
            end
        end
    end
end
