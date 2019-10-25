classdef Helpers < handle
% Various utility functions
    
    methods
        function [files, nbr_files] = list_filenames(obj, path) %#ok<*INUSL>
            % Lists and counts all files in a directory
            files = dir(path);
            files = struct2cell(files);
            files = files(1, 3 : end);
            nbr_files = numel(files);
        end

        function setup_caffe(~)
            global CONFIG
            % WARNING! Adjust the path to your caffe accordingly!
            caffepath = './caffe/matlab';
            fprintf('You set your caffe in caffePath.cfg to: %s\n', caffepath);
            addpath(caffepath);
            caffe.reset_all();
            if CONFIG.use_gpu
                caffe.set_mode_gpu();
                caffe.set_device(CONFIG.gpu_id);
            else
                caffe.set_mode_cpu();
            end
        end

        function set_agent_m_values(~)
            % Want linear scheme for increasing m. Therefore we first
            % calculate how many batch gradient updates will be in total
            global CONFIG

            % Initialize container of m-parameter updates (values)
            nbr_batch_updates = floor(CONFIG.training_agent_nbr_eps / ...
                                      CONFIG.training_agent_eps_per_batch);
            CONFIG.agent_m_values = nan(2, nbr_batch_updates);
            agent_ms_all = CONFIG.agent_ms;
            for i = 1 : numel(agent_ms_all)
                agent_ms = agent_ms_all{i};
                ms_start = agent_ms{1};
                ms_end = agent_ms{2};
                interval = agent_ms{3};
                if numel(interval) == 1
                    % Attach end of training episode to interval
                    interval = [interval, CONFIG.training_agent_nbr_eps];
                end
                curr_nbr_batch_updates = ceil((interval(2) - interval(1) + 1) / ...
                                               CONFIG.training_agent_eps_per_batch);
                interval = ceil(interval / CONFIG.training_agent_eps_per_batch);
                curr_azims = linspace(ms_start(1), ms_end(1), curr_nbr_batch_updates);
                curr_elevs = linspace(ms_start(2), ms_end(2), curr_nbr_batch_updates);
                CONFIG.agent_m_values(:, interval(1) : interval(2)) = ...
										[curr_azims; curr_elevs];
            end

            % Fill in the m-constant parts
            for i = 1 : nbr_batch_updates
				if isnan(CONFIG.agent_m_values(1, i))
					CONFIG.agent_m_values(:, i) = CONFIG.agent_m_values(:, i - 1);
				end
            end
        end

        function [n,V,p] = affine_fit(obj, X)
            %Computes the plane that fits best (lest square of the normal distance
            %to the plane) a set of sample points.
            % source: https://www.mathworks.com/matlabcentral/fileexchange/43305-plane-fit
            %INPUTS:
            %
            %X: a N by 3 matrix where each line is a sample point
            %
            %OUTPUTS:
            %
            %n : a unit (column) vector normal to the plane
            %V : a 3 by 2 matrix. The columns of V form an orthonormal basis of the
            %plane
            %p : a point belonging to the plane
            %
            %NB: this code actually works in any dimension (2,3,4,...)
            %Author: Adrien Leygue
            %Date: August 30 2013

            %the mean of the samples belongs to the plane
            p = mean(X,1);

            %The samples are reduced:
            R = bsxfun(@minus,X,p);
            %Computation of the principal directions if the samples cloud
            [V, ~] = eig(R'*R);
            %Extract the output from the eigenvectors
            n = V(:,1);
            V = V(:,2:end);
        end

        function plot_skeleton(obj, pose, jointColor, joint_size, lineWidth, perspective)
            % pose3D - 15x3 matrix of 3D joints or 15x2 for 2d joints
            % colorOption - color to plot (default='r')
            if(nargin <= 2)
                jointColor = 'r';
            end
            if(nargin <= 3)
                joint_size = 100;
            end
            if(nargin <= 4)
                lineWidth = 5;
            end
            if(nargin <= 5)
                perspective = [-3, -56];
            end

            hold on;
            is_3d = size(pose, 2) >= 3;

            % configure plot
            set(gcf,'Color',[1,1,1]);
            axis equal;
            ax = gca;               % get the current axis
            ax.Clipping = 'off';    % turn clipping off
            axis off;
            grid on;

            if is_3d
                % only rotate image if we show 3d plot
                view(perspective);
            end

            neck = 1;
            head = 2;
            center = 3;
            lshoulder = 4;
            lelbow = 5;
            lwrist = 6;
            lhip = 7;
            lknee = 8;
            lankle = 9;
            rshoulder = 10;
            relbow = 11;
            rwrist = 12;
            rhip = 13;
            rknee = 14;
            rankle = 15;

            order = [1 2 3];

            connections = [
                head neck;
                neck center;

                lshoulder neck;
                rshoulder neck;

                lshoulder lelbow;
                lelbow lwrist;

                rshoulder relbow;
                relbow rwrist;

                rhip rknee;
                rknee rankle;

                lhip lknee;
                lknee lankle;

                lhip center;
                rhip center;
            ];


            pose = pose';
            % plot limbs
            for i = 1:size(connections, 1)
                if any(isnan(pose(:, connections(i, :))))
                    % Skip nan joints
                    continue;
                end
                c = jointColor;
                if is_3d
                    plot3(pose(order(1), connections(i, :)), pose(order(2), connections(i, :)), ...
                          pose(order(3), connections(i, :)), '-', 'Color', c, 'LineWidth', lineWidth);
                else
                    plot(pose(order(1), connections(i, :)), pose(order(2), connections(i, :)), ...
                         '-', 'Color', c, 'LineWidth', lineWidth);
                end
            end

            pose = pose';
            % plot joints
            colors = parula(size(pose, 1));
            if is_3d
                scatter3(pose(:, order(1)), pose(:, order(2)), pose(:, order(3)), ...
                         joint_size, colors, 'filled');
            else
                scatter(pose(:, order(1)), pose(:, order(2)), ...
                        joint_size, colors, 'filled');
            end
        end

        function plot_ground_plane(obj, poses, color, size)
            if(nargin <= 2)
                color = 'red';
            end
            if(nargin <= 3)
                size = 200;
            end

            % Takes a cell array of 3d poses
            rankle = 15;
            lankle = 9;
            feet = [];
            for i = 1:numel(poses)
                pose = poses{i};
                f = pose([rankle, lankle], :);
                f = f(~isnan(f(:, 1)), :);
                feet = [feet; f];
            end

            if numel(feet) > 0
                % We want to make sure the axis isn't affected by plane
                h = gca;
                hold on
                h.YLimMode='manual';
                h.XLimMode='manual';
                h.ZLimMode='manual';

                % Fit plane
                [~, V_2, p_2] = obj.affine_fit(feet);
                steps = [-1:1:1];
                num_pts = numel(steps);
                [S1,S2] = meshgrid(steps * size);
                %generate the pont coordinates
                X = p_2(1)+[S1(:) S2(:)]*V_2(1,:)';
                Y = p_2(2)+[S1(:) S2(:)]*V_2(2,:)';
                Z = p_2(3)+[S1(:) S2(:)]*V_2(3,:)';
                %plot the plane
                s = surf(reshape(X,num_pts, num_pts),reshape(Y, num_pts, num_pts),reshape(Z, num_pts, num_pts),'facecolor', color,'facealpha', 0.1);
                %s.EdgeColor = [0.3, 0.3, 0.3];
                %set(h, 'LineWidth', 0.8);
                s.EdgeColor = 'none';
            end
        end
        
         function plot_ground_plane_full(obj, poses, color, size)
            if(nargin <= 2)
                color = 'red';
            end
            if(nargin <= 3)
                size = 200;
            end

            % Takes a cell array of 3d poses
            %rfoot = 22;
            %lfoot = 19;
            feet_indices = [22, 24, 21, 19] + 1;
            feet = [];
            for i = 1:numel(poses)
                pose = poses{i}.body;
                f = pose(feet_indices, :);
                f = f(~isnan(f(:, 1)), :);
                feet = [feet; f];
            end

            if numel(feet) > 0
                % We want to make sure the axis isn't affected by plane
                h = gca;
                hold on
                h.YLimMode='manual';
                h.XLimMode='manual';
                h.ZLimMode='manual';

                % Fit plane
                [~, V_2, p_2] = obj.affine_fit(feet);
                steps = [-1:1:1];
                num_pts = numel(steps);
                [S1,S2] = meshgrid(steps * size);
                %generate the pont coordinates
                X = p_2(1)+[S1(:) S2(:)]*V_2(1,:)';
                Y = p_2(2)+[S1(:) S2(:)]*V_2(2,:)';
                Z = p_2(3)+[S1(:) S2(:)]*V_2(3,:)';
                %plot the plane
                s = surf(reshape(X,num_pts, num_pts),reshape(Y, num_pts, num_pts),reshape(Z, num_pts, num_pts),'facecolor', color,'facealpha', 0.1);
                %s.EdgeColor = [0.3, 0.3, 0.3];
                %set(h, 'LineWidth', 0.8);
                s.EdgeColor = 'none';
            end
        end

        function time_freezes = get_time_freezes(obj, env, mode)
            global CONFIG

            time_freezes = [];
            counter = 0;

            for i = 1 : numel(env.scenes)

                scene_name = strcat('scene_', env.scenes{i}.scene_name);

                if strcmp(mode, 'train')
                    sequence_length = CONFIG.sequence_length_train;
					if ~isempty(strfind(scene_name, 'pose'))
						frame_step_length = max(1, sequence_length - ceil(0.5 * sequence_length));
					elseif ~isempty(strfind(scene_name, 'mafia'))
						frame_step_length = max(1, sequence_length - ceil(0.5 * sequence_length));
					elseif ~isempty(strfind(scene_name, 'ultimatum'))
						frame_step_length = max(1, sequence_length - ceil(0.5 * sequence_length));
					else
						frame_step_length = 1;
					end
                else
                    sequence_length = CONFIG.sequence_length_eval;
					frame_step_length = 1;
                end
                if frame_step_length > 1
                    % used so that each time we may get slightly different
                    % data shuffle
                    if ~isfield(CONFIG, 'trainset_seed')
                        CONFIG.trainset_seed = 0;
                    else
                        CONFIG.trainset_seed = CONFIG.trainset_seed + 1;
                    end
                    rng(CONFIG.trainset_seed);
                    offset = randi(10) - 1;
                    rng(CONFIG.rng_seed);
                else
                    offset = 0;
                end
                for j = 1 + offset : frame_step_length : env.scenes{i}.nbr_frames - CONFIG.sequence_step * (sequence_length - 1)
                    % If quality of time_freeze is good enough, then we
                    % will allow it to be part of the training set
                    ncams = env.scenes{i}.nbr_cameras;
                    time_freezes = [time_freezes; [i * ones(ncams, 1), j * ones(ncams, 1), (1 : ncams)']];
                    counter = counter + 1;
                end
            end
        end

        % Below functions useful for chaning prototxt files

        function proto_name = set_proto_name(~, type)
            global CONFIG
            proto_name = strcat(CONFIG.agent_proto_folder, type);
            if numel(CONFIG.agent_proto_name) > 0
                proto_name = strcat(proto_name, '_', CONFIG.agent_proto_name);
            end
            proto_name = strcat(proto_name, '.prototxt');
        end

        function set_solver_proto(obj)

            global CONFIG

            % Read and get interesting lines (and then also close orig file)
            file_id = fopen(CONFIG.agent_solver_proto);
            [file_content, idxs_nonempty_noncommented] = obj.read_line_by_line(file_id);

            % Change values of desired fields
            fields_to_process = {'base_lr', 'gamma', 'random_seed', 'lr_policy'};

            new_vals = {CONFIG.training_agent_lr, CONFIG.training_agent_lr_update_factor, ...
                        CONFIG.rng_seed_caffe, sprintf('"%s"', CONFIG.caffe_lr_policy)};

            for i = 1 : numel(fields_to_process)
                field = fields_to_process{i};
                id = find(~cellfun(@isempty, strfind(file_content, field)));
                id = id(idxs_nonempty_noncommented(id)); id = id(end);
                line = file_content{id};
                line1_split = strsplit(line, ':');
                new_val = new_vals{i};
                new_line = strcat(line1_split{1}, ':', {' '}, num2str(new_val));
                file_content{id} = new_line{1};
            end

            % Replace in solver.prototxt
            file_id = fopen(CONFIG.agent_solver_proto, 'w');
            obj.replace_line_by_line(file_id, file_content);

            % Update caffe stepvalues
            obj.set_caffe_multistep_values();
        end

        function set_caffe_multistep_values(~)
            global CONFIG

            % Read and get interesting lines (and then also close orig file)
            file_content = fileread(CONFIG.agent_solver_proto);

            % Remove all old step values
            file_content = regexprep(file_content, 'stepvalue:\s*\d*', '');

            steps = CONFIG.training_agent_lr_update_steps;
            for i = 1 : numel(steps)
                file_content = strcat(file_content, sprintf('\nstepvalue: %d', steps(i)));
            end

            % Replace in train.prototxt
            file_id = fopen(CONFIG.agent_solver_proto, 'w');
            fprintf(file_id, file_content);
            fclose(file_id);
        end

        function set_train_proto(obj)

            global CONFIG

            % Read and get interesting lines (and then also close orig file)
            file_id = fopen(CONFIG.agent_train_proto);
            [file_content, ~] = obj.read_line_by_line(file_id);

            % Extract the desired batch size to be set
            batch_size = num2str(CONFIG.training_agent_batch_size);

            % Change values of desired rows
            line_counter = 1;
            while line_counter <= numel(file_content)

                % Extract current line
                line = file_content{line_counter};
                line_counter_start = line_counter;

                % 1) Find row with  <<type: "Input">>
                if sum(strfind(line, '"Input"')) > 0 || sum(strfind(line, '"DummyData"')) > 0

                    % 2) Find the closest row with <<shape: >>
                    str_find = strfind(file_content{line_counter}, 'shape');
                    while isempty(str_find) || ~str_find
                        line_counter = line_counter + 1;
                        str_find = strfind(file_content{line_counter}, 'shape');
                    end

                    % 3) Find first, second occurrence of <<dim >>
                    line = file_content{line_counter};
                    idxs_dim_string = strfind(line, 'dim');

                    % 4) Replace all space to next occurrence of <<dim >> with new data
                    idx_start = idxs_dim_string(1); idx_next = idxs_dim_string(2);
                    line_part1 = line(1 : idx_start + 2);
                    line_part2 = line(idx_next : end);
                    new_line = strcat(line_part1, ':', {' '}, batch_size, {' '}, line_part2);
                    new_line = new_line{1};
                    file_content{line_counter} = new_line;
                elseif sum(strfind(line, 'name: "canvas"')) > 0
                    line_counter = line_counter + 3;
                    line = file_content{line_counter};
                    last_dim = strfind(line, 'dim:');
                    last_dim = last_dim(end);
                    if CONFIG.agent_use_angle_canvas
                        canvas_dim = CONFIG.agent_nbr_bins_azim * CONFIG.agent_nbr_bins_elev;
                        line_rest = strcat(num2str(canvas_dim), '} }');
                    else
                        line_rest = '1} }';
                    end
                    new_line = line;
                    new_line = strcat(new_line(1 : last_dim + 4), {' '}, line_rest);
                    file_content{line_counter} = new_line{1};
                elseif sum(strfind(line, 'name: "rig"')) > 0
                    line_counter = line_counter + 3;
                    line = file_content{line_counter};
                    last_dim = strfind(line, 'dim:');
                    last_dim = last_dim(end);
                    if CONFIG.agent_use_rig_canvas
                        canvas_dim = CONFIG.agent_nbr_bins_azim * CONFIG.agent_nbr_bins_elev;
                        line_rest = strcat(num2str(canvas_dim), '} }');
                    else
                        line_rest = '1} }';
                    end
                    new_line = line;
                    new_line = strcat(new_line(1 : last_dim + 4), {' '}, line_rest);
                    file_content{line_counter} = new_line{1};
                elseif sum(strfind(line, 'name: "aux"')) > 0
                    line = file_content{line_counter + 3};
                    last_dim = strfind(line, 'dim:');
                    last_dim = last_dim(end);
                    new_line = strcat(line(1 : last_dim + 4), {' '}, ...
                                num2str(5 + CONFIG.agent_use_joints_triangulated * 15), '} }');
                    file_content{line_counter + 3} = new_line{1};
                elseif sum(strfind(line, 'name: "data"')) > 0
                    line = file_content{line_counter + 3};
                    last_dim = strfind(line, 'dim:');
                    last_dim = last_dim(end - 1);
                    new_line = line;

                    new_line(last_dim : end) = 'dim: 41 dim: 23} }';

                    file_content{line_counter + 3} = new_line;
                end

                % 2) Make input to fc1 appropriate based on config
                if sum(strfind(line, 'name: "fc1_input"')) > 0

                    use_canvas = CONFIG.agent_use_angle_canvas;
                    use_rig_canvas = CONFIG.agent_use_rig_canvas;
                    while ~sum(strfind(line, 'bottom:'))
                        line_counter = line_counter + 1;
                        line = file_content{line_counter};
                    end
                    end_idx = strfind(line, ':');
                    new_line = strcat(line(1 : end_idx));
                    if use_canvas && use_rig_canvas
                        new_line = strcat(new_line, ' "data_canvas_rig"');
                    elseif use_canvas
                        new_line = strcat(new_line, ' "data_canvas"');
					elseif use_rig_canvas
                        new_line = strcat(new_line, ' "data_rig"');
                    else
                        new_line = strcat(new_line, ' "data_flat"');
                    end
                    file_content{line_counter} = new_line;
                end

                % Continue to next line
                line_counter = line_counter_start + 1;
            end

            % Replace in train.prototxt
            file_id = fopen(CONFIG.agent_train_proto, 'w');
            obj.replace_line_by_line(file_id, file_content);
        end

        % Two helpers for read / write file below
        function [file_content, idxs_nonempty_noncommented] = read_line_by_line(obj, file_id)
            file_content = {};
            idxs_nonempty_noncommented = logical([]);
            while 1
                line = fgetl(file_id);
                if ~ischar(line);
                    break;
                end
                space_check = isspace(line);
                if (sum(space_check) == numel(space_check) || strcmp(line(1), '#'))
                    idxs_nonempty_noncommented = [idxs_nonempty_noncommented, false]; %#ok<*AGROW>
                else
                    idxs_nonempty_noncommented = [idxs_nonempty_noncommented, true];
                end
                file_content{end+1} = line; %#ok<*SAGROW>
            end
            fclose(file_id);
        end

        function replace_line_by_line(obj, file_id, file_content)
            for i = 1 : numel(file_content)
                if i == numel(file_content)
                    fprintf(file_id,'%s', file_content{i});
                    break
                else
                    fprintf(file_id,'%s\n', file_content{i});
                end
            end
            fclose(file_id);
        end

        function nbr_params = count_network_params(obj, net)
            layers_list = net.layer_names;
            % for those layers which have parameters, count them
            nbr_params = 0;
            for j = 1:length(layers_list),
                if ~isempty(net.layers(layers_list{j}).params)
                    feat = net.layers(layers_list{j}).params(1).get_data();
                    nbr_params = nbr_params + numel(feat);
                end
            end
        end

        % Hyperdock helper functions
        function tf = in_hyperdock(~)
            tf = exist('/hyperdock/params.json', 'file');
        end

        function serie = hyperdock_serie(~, label, x_data, y_data)
           serie = struct('label', label, 'x', x_data, 'y', y_data);
        end

        function plot = hyperdock_plot(~, name, x_label, y_label, serie_array)
           plot = struct('name', name, 'x_axis', x_label, 'y_axis', y_label, ...
                         'series', serie_array);
        end

        function json= write_hyperdock_graph(obj, plot_array)
            % Saves a struct of hyperdock to disk. Returns the created
            % json for debugging purposes.
            if obj.in_hyperdock()
                fprintf('Writing Hyperdock graph\n');
                addpath('code/panoptic/json-matlab');
                path = '/hyperdock/graphs.json';
                json = savejson('', plot_array, struct('SingletArray', 1, ...
                                                    'SingletCell', 1));
                if ~strcmp(json(1), '[')
                    json = sprintf('[%s]', json);
                end

                fileID = fopen(path, 'w');
                fprintf(fileID, json);
                fclose(fileID);
            end
        end

        function write_hyperdock_loss(obj, loss, epsiode)
            if obj.in_hyperdock()
                fprintf('Writing Hyperdock loss\n');
                fileID = fopen('last_loss.json', 'w');
                fprintf(fileID,'{"loss": %f, "state": "ok", "ep": %d}', ...
                        loss, epsiode);
                fclose(fileID);
            end
        end

        function d = iou(~, in1, in2)
            % Much faster than Matlab's built-in version bboxOverlapRatio
            % x1 y1 x2 y2
            %
            % IOU Intersection over union score.
            %   The inputs can be a matrix Nx4 and a 1x4 vector. The output
            %   is a vector with the IOU score of mask2 with each one
            %   of the bounding boxes in mask1
            %
            %   d = iou(in1,in2)
            %
            %   Stavros Tsogkas, <stavros.tsogkas@ecp.fr>
            %   Last update: October 2014
            intersectionBox = [max(in1(:,1), in2(:,1)), max(in1(:,2), in2(:,2)),...
                               min(in1(:,3), in2(:,3)), min(in1(:,4), in2(:,4))];
            iw = intersectionBox(:,3)-intersectionBox(:,1)+1;
            ih = intersectionBox(:,4)-intersectionBox(:,2)+1;
            unionArea = bsxfun(@minus, in1(:,3), in1(:,1)-1) .*...
                        bsxfun(@minus, in1(:,4), in1(:,2)-1) +...
                        bsxfun(@minus, in2(:,3), in2(:,1)-1) .*...
                        bsxfun(@minus, in2(:,4), in2(:,2)-1) - iw.*ih;
            d = iw .* ih ./ unionArea;
            d(iw <= 0 | ih <= 0) = 0;
        end

        function coords = trans_hip_coord(~, in_coords, annot)
            % Translate hip position to Panoptic format
            chip = in_coords(3, :);
            chip_annot = annot(3, :);
            coords = bsxfun(@plus, in_coords, chip_annot - chip);
        end

        function [recon3D, current_is_garbage] = ...
			get_recon_with_prev(obj, predictions3d, pose_idxs, idx, visited_cams, nbr_persons, ...
                                nbr_steps, predictions3d_seq, predictions3d_seq_ok, ...
                                predictions3d_seq_all, predictions3d_seq_ok_all, use_cum, ...
                                intermediate_step)

            if ~exist('intermediate_step', 'var')
                intermediate_step = 0;
            end

			[~, unique_idxs] = unique(visited_cams(1 : nbr_steps));
            use_indices = true(nbr_steps, 1);
            use_indices(setdiff(1 : nbr_steps, unique_idxs)) = 0;
			preds = predictions3d{idx};
			p_idxs = pose_idxs(:, idx);
			qualifier = (p_idxs(1 : nbr_steps) ~= -1) .* use_indices;
			current_is_garbage = ~isempty(qualifier) && all(qualifier == 0);
			if current_is_garbage
				% Ensure non-collapse
				qualifier(1) = 1;
			end
			qualifier = [qualifier; false(numel(visited_cams) - numel(qualifier), 1)]; %#ok<*AGROW>
			last_sane_idx = find(qualifier, 1, 'last');
			preds(:, :, ~qualifier) = nan;

            if ~use_cum
                recon3D = preds(:, :, last_sane_idx);
                return;
            end

            if ~intermediate_step
                preds_seq_ok = predictions3d_seq_ok;
                preds_seq = predictions3d_seq;
            else
				preds_seq_ok = predictions3d_seq_ok_all{nbr_steps};
				preds_seq = predictions3d_seq_all{nbr_steps};
            end

            if any(cellfun(@isempty, preds_seq))
                recon3D = preds(:, :, last_sane_idx);
            else

                % Get pose reconstruction from previous frame (if any)
				if numel(preds_seq_ok) == nbr_persons
					preds_seq_ok = preds_seq_ok{idx};
					preds_seq = preds_seq{idx};
				else
					preds_seq_ok = [];
					preds_seq = [];
				end
                if ~isempty(preds_seq_ok) && preds_seq_ok(1)
                    pose_recon_prev = preds_seq(:, :, 1);
                else
                    pose_recon_prev = [];
                end

                if current_is_garbage && ~isempty(pose_recon_prev)
                    % Current is garbage, and previous thing exists --
                    % just propagate the previous thing
                    recon3D = pose_recon_prev;
                else
                    pred_prev = pose_recon_prev;
                    if ~isempty(pred_prev)
                        pred_curr = preds(:, :, last_sane_idx);
                        prev_idxs = isnan(pred_curr(:, 1));
                        new_pred_with_prev = pred_curr;
                        new_pred_with_prev(prev_idxs, :) = pred_prev(prev_idxs, :);
                    else
                        new_pred_with_prev = preds(:, :, last_sane_idx);
                    end
                    recon3D = new_pred_with_prev;
                end
            end
        end

        function pose_reconstruction = infer_missing_joints(obj, pose_reconstruction)
            nan_idxs = isnan(pose_reconstruction(:, 1));
            mean_nonan = mean(pose_reconstruction(~nan_idxs, :), 1);
            if any(isnan(mean_nonan))
                mean_nonan = zeros(size(mean_nonan));
            end
            pose_reconstruction(nan_idxs, :) = repmat(mean_nonan, nnz(nan_idxs), 1);
        end

        function pose_coco = convert_body25(~, pose25)
            % first convert to coco, then from coco to mpii
            % https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md

            pose_coco = [
                pose25(1, :); % 1
                pose25(2, :); % 2
                pose25(3, :);
                pose25(4, :);
                pose25(5, :);
                pose25(6, :);
                pose25(7, :);
                pose25(8, :);
                pose25(10, :);
                pose25(11, :);
                pose25(12, :); % 10
                pose25(13, :);
                pose25(14, :);
                pose25(15, :); % 14
                pose25(16, :);
                pose25(17, :);
                pose25(18, :);
                pose25(19, :);
            ];
        end


        function plot_fullbody(~, is_3d, fullbody, jointColor, joint_size, lineWidth, perspective)
            % pose3D - 15x3 matrix of 3D joints or 15x2 for 2d joints
            % colorOption - color to plot (default='r')
            if(nargin <= 3)
                jointColor = 'r';
            end
            if(nargin <= 4)
                joint_size = 100;
            end
            if(nargin <= 5)
                lineWidth = 5;
            end
            if(nargin <= 6)
                perspective = [-3, -56];
            end

            hold on;

            % configure plot
            set(gcf,'Color',[1,1,1]);
            axis equal;
            ax = gca;               % get the current axis
            ax.Clipping = 'off';    % turn clipping off
            axis off;
            grid on;

            if is_3d
                % only rotate image if we show 3d plot
                view(perspective);
            end

            order = [1 2 3];

            % BODY:
            % https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
            % 1 - 25 (0-24 in source document)

            connections = [
                % head
                0 1;
                0 15;
                15 17;
                0 16;
                16 18;

                % left arm
                1 2;
                2 3;
                3 4;

                % right arm
                1 5;
                5 6;
                6 7;

                % hip
                1 8;
                8 9;
                8 12;

                % left leg
                9 10;
                10 11;
                11 24;
                11 22;
                22 23;

                % right leg
                12 13;
                13 14;
                14 21;
                14 19;
                19 20;
            ] + 1;
        
            pose = fullbody.body';
            pose(:, pose(end, :) == 0) = nan;
            
%             if ~isnan(fullbody.face)
%                % remove "skeleton face"
%                pose(:, 1) = nan;
%                pose(:, 16) = nan;
%                pose(:, 17) = nan;
%                pose(:, 18) = nan;
%                pose(:, 19) = nan;
%             end
            
            % plot limbs
            for i = 1:size(connections, 1)
                if any(isnan(pose(:, connections(i, :))))
                    % Skip nan joints
                    continue;
                end
                
                curr_line_width = lineWidth;
                if i < 6
                    % set smaller line width for the face
                    curr_line_width = curr_line_width / 2;
                end

                c = jointColor;
                if is_3d
                    plot3(pose(order(1), connections(i, :)), pose(order(2), connections(i, :)), ...
                          pose(order(3), connections(i, :)), '-', 'Color', c, 'LineWidth', curr_line_width);
                else
                    plot(pose(order(1), connections(i, :)), pose(order(2), connections(i, :)), ...
                         '-', 'Color', c, 'LineWidth', curr_line_width);
                end
            end
            
            pose = pose';
        
            face_indices = [1, 16, 17, 18, 19];
            % plot joints
            colors = parula(size(pose, 1));
            if is_3d
                % head
                scatter3(pose(face_indices, order(1)), pose(face_indices, order(2)), pose(face_indices, order(3)), ...
                    joint_size / 20, colors(face_indices, :), 'filled');
                % body
                pose(face_indices, :) = [];
                colors(face_indices, :) = [];
                scatter3(pose(:, order(1)), pose(:, order(2)), pose(:, order(3)), ...
                         joint_size, colors, 'filled');
            else
                % head
                scatter(pose(face_indices, order(1)), pose(face_indices, order(2)), ...
                        joint_size / 20, colors(face_indices, :), 'filled');
                % body
                pose(face_indices, :) = [];
                colors(face_indices, :) = [];
                scatter(pose(:, order(1)), pose(:, order(2)), ...
                        joint_size, colors, 'filled');
            end

            % FACE
           connections = [
                (0:15)', (1:16)'   % chin

                (17:20)', (18:21)'  % right eyebrow

                (22:25)', (23:26)'  % left eyebrow

                (27:29)', (28:30)'  % nose

                (31:34)', (32:35)'  % nose bottom

                (36:40)', (37:41)'  % right eye
                41 36;

                (42:46)', (43:47)'  % left eye
                47 42;

                (48:58)', (49:59)'  % outer mouth
                59 48;

                (60:66)', (61:67)'  % outer mouth
                67 60;
           ] + 1;


            pose = fullbody.face';
            pose(:, pose(end, :) == 0) = nan;
            % plot limbs
            for i = 1:size(connections, 1)
                if any(isnan(pose(:, connections(i, :))))
                    continue;
                end

                c = jointColor;
                if is_3d
                    plot3(pose(order(1), connections(i, :)), pose(order(2), connections(i, :)), ...
                          pose(order(3), connections(i, :)), '-', 'Color', c, 'LineWidth', lineWidth / 2);
                else
                    plot(pose(order(1), connections(i, :)), pose(order(2), connections(i, :)), ...
                         '-', 'Color', c, 'LineWidth', lineWidth / 2);
                end
            end

            pose = pose';
            % plot joints
            colors = parula(size(pose, 1));
            if is_3d
                scatter3(pose(:, order(1)), pose(:, order(2)), pose(:, order(3)), ...
                         joint_size / 20, colors, 'filled');
            else
                scatter(pose(:, order(1)), pose(:, order(2)), ...
                        joint_size / 20, colors, 'filled');
            end


           % HANDS
           connections = [

                % thumb
                0 1;
                1 2;
                2 3;
                3 4;

                % index
                0 5;
                5 6;
                6 7;
                7 8;

                % middle
                0 9;
                9 10;
                10 11;
                11 12;

                % ring
                0 13;
                13 14;
                14 15;
                15 16;

                % little
                0 17;
                17 18;
                18 19;
                19 20;
           ] + 1;


            pose = fullbody.right_hand';
            pose(pose(end, :) == 0, :) = nan;
            
            % align hand with arm, since OP sometimes fails with this
            arm = fullbody.body';
            if ~isnan(arm(:,  8))
                pose = pose - repmat(pose(:,1), 1, size(pose, 2)) + repmat(arm(:,  8), 1, size(pose, 2));
            end
            
            % plot limbs
            for i = 1:size(connections, 1)
                if any(isnan(pose(:, connections(i, :))))
                    % Skip nan joints
                    continue;
                end
                c = jointColor;
                if is_3d
                    plot3(pose(order(1), connections(i, :)), pose(order(2), connections(i, :)), ...
                          pose(order(3), connections(i, :)), '-', 'Color', c, 'LineWidth', lineWidth / 2);
                else
                    plot(pose(order(1), connections(i, :)), pose(order(2), connections(i, :)), ...
                         '-', 'Color', c, 'LineWidth', lineWidth / 2);
                end
            end

            pose = pose';
            % plot joints
            colors = parula(size(pose, 1));
            if is_3d
                scatter3(pose(:, order(1)), pose(:, order(2)), pose(:, order(3)), ...
                         joint_size / 20, colors, 'filled');
            else
                scatter(pose(:, order(1)), pose(:, order(2)), ...
                        joint_size / 20, colors, 'filled');
            end

            pose = fullbody.left_hand';
            pose(pose(end, :) == 0, :) = nan;
            
            % align hand with arm, since OP sometimes fails with this
            arm = fullbody.body';
            if ~isnan(arm(:,  5))
                pose = pose - repmat(pose(:,1), 1, size(pose, 2)) + repmat(arm(:,  5), 1, size(pose, 2));
            end
            
            % plot limbs
            for i = 1:size(connections, 1)
                if any(isnan(pose(:, connections(i, :))))
                    % Skip nan joints
                    continue;
                end
                c = jointColor;
                if is_3d
                    plot3(pose(order(1), connections(i, :)), pose(order(2), connections(i, :)), ...
                          pose(order(3), connections(i, :)), '-', 'Color', c, 'LineWidth', lineWidth / 2);
                else
                    plot(pose(order(1), connections(i, :)), pose(order(2), connections(i, :)), ...
                         '-', 'Color', c, 'LineWidth', lineWidth / 2);
                end
            end

            pose = pose';
            % plot joints
            colors = parula(size(pose, 1));
            if is_3d
                scatter3(pose(:, order(1)), pose(:, order(2)), pose(:, order(3)), ...
                         joint_size / 20, colors, 'filled');
            else
                scatter(pose(:, order(1)), pose(:, order(2)), ...
                        joint_size / 20, colors, 'filled');
            end
        end
    end
end