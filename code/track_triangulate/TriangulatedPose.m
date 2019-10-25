classdef TriangulatedPose < handle
    % Triangulates a set of 2d poses for a person into a 3d reconstruction
    
    properties
        poses2d
        triangulated_poses
        final_poses
        cams_visited
        img_height
        img_width
        max_reproj_err
        joints_triangulated
        nbr_joints
    end
    
    methods

        function obj = TriangulatedPose(nbr_joints)
            if nargin < 1
                nbr_joints = 15;
            end
            
            global CONFIG
            obj.img_height = CONFIG.dataset_img_dimensions(1);
            obj.img_width = CONFIG.dataset_img_dimensions(2);
            obj.max_reproj_err = CONFIG.triangulation_max_reproj_err;
            obj.nbr_joints = nbr_joints;
            obj.reset();
        end
        
        function reset(obj)
            obj.poses2d = {};
            obj.triangulated_poses = {};
            obj.final_poses = {};
            obj.cams_visited = [];
            obj.joints_triangulated = false(obj.nbr_joints, 1);
        end
            
        function add(obj, joints, P, camera_idx)
            
            if size(joints, 1) ~= obj.nbr_joints || size(joints, 2) ~= 2
                error('Incorrect dimension of joints');
            end
            
            % If already visited this cam, nothing new happens
            if any(obj.cams_visited == camera_idx)
                obj.final_poses{end + 1} = obj.final_poses{end};
                obj.cams_visited(end + 1) = camera_idx;
                return;
            end
            obj.cams_visited(end + 1) = camera_idx;
            
            % Triangulate the new image with respect to existing ones
            nbr_poses = numel(obj.poses2d);
            curr_tri_poses = nan(obj.nbr_joints, 3, nbr_poses);
            for pose_idx = 1 : nbr_poses
                joints2 = obj.poses2d{pose_idx}.joints;
                P2 = obj.poses2d{pose_idx}.P;
                curr_tri_poses(:, :, pose_idx) = obj.triangulate(joints, joints2, P, P2);
            end
            obj.triangulated_poses{end + 1} = curr_tri_poses;

            % Attach to container of final poses through median fusion
            obj.final_poses{end + 1} = nanmedian(cat(3, obj.triangulated_poses{:}), 3);
            
            % Adds a 2d pose and adds it to the fused, triangulated pose
            obj.poses2d{end + 1} = struct('joints', joints, 'P', P);
        end
        
        function remove(obj)
            % Reverts last add operation
            obj.final_poses = obj.final_poses(1 : end - 1);
            cam_latest = obj.cams_visited(end);
            obj.cams_visited = obj.cams_visited(1 : end - 1);
            if ~any(obj.cams_visited == cam_latest)
                obj.triangulated_poses = obj.triangulated_poses(1 : end - 1);
                obj.poses2d = obj.poses2d(1 : end - 1);
            end
        end
        
        function pose3D = get(obj, idx)
            if ~exist('idx', 'var')
                idx = numel(obj.final_poses);
            end
            
            if numel(obj.final_poses) > 0
                pose3D = obj.final_poses{idx};
            else
                pose3D = nan(obj.nbr_joints, 3);
            end
            
            % This is for the case where we have only one view -- no
            % reasonable triangulated 3D pose available
            if numel(pose3D) == 0
                pose3D = nan(obj.nbr_joints, 3);
            end
        end
        
        function pose3D = triangulate(obj, joints1, joints2, P1, P2)
            
            % Extract mutually existing joints
            idxs_joints = ~isnan(joints1(:, 1) + joints2(:, 1));
            joints1 = joints1(idxs_joints, :)';
            joints2 = joints2(idxs_joints, :)';
            
            % DLT triangulation
            pose3D = nan(numel(idxs_joints), 3);
            nnz_joints = nnz(idxs_joints);
            if nnz_joints > 0
                
                % Triangulate joints
                tri_out = stereoReconsPts(P1, P2, joints1, joints2, ...
                                            [obj.img_height, obj.img_height; ...
                                             obj.img_width, obj.img_width]);

                % Check that the re-projection errors are sane
                tri_out_one = [tri_out; ones(1, nnz_joints)];
                reproj1 = P1 * tri_out_one;
                reproj1 = bsxfun(@rdivide, reproj1, reproj1(end, :));
                reproj1 = reproj1(1 : 2, :);
                reproj2 = P2 * tri_out_one;
                reproj2 = bsxfun(@rdivide, reproj2, reproj2(end, :));
                reproj2 = reproj2(1 : 2, :);
                reproj_errs1 = sqrt(sum((reproj1 - joints1).^2, 1));
                reproj_errs2 = sqrt(sum((reproj2 - joints2).^2, 1));
                ok_idxs = max(reproj_errs1, reproj_errs2) < obj.max_reproj_err;
                idxs_joints = find(idxs_joints);
                idxs_joints = idxs_joints(ok_idxs);
                
                % Insert only sanely triangulated joints
                pose3D(idxs_joints, :) = tri_out(:, ok_idxs)';
                obj.joints_triangulated(idxs_joints) = true;
            end
        end
    end
end

