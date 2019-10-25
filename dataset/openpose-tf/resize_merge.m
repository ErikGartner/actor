function resize_merge(path)
    % Path should be to cache/<split>/

    path = strcat(path, '/openpose/');
    [scenes, nbr_scenes] = list_filenames(path);

    for scene_id = 1:nbr_scenes
        scene = scenes{scene_id};
        [cameras, ~] = list_filenames(strcat(path, '/', scene));

        % Filter only for camera folders
        out = cell2mat(regexp(cameras,'^\d+_\d+$'));
        cameras = cameras(out > 0);
        nbr_cameras = numel(cameras);

        conv4_4_CPM_small = cell(0,0);

        for camera_id = 1:nbr_cameras
            camera = cameras{camera_id};
            big_feats = strcat(path, '/', scene, '/', camera, '/conv4_4_CPM.mat');
            fprintf('Resizing %s - %s\n', scene, camera);


            S = load(big_feats);
            data = S.('conv4_4_CPM');

            for frame_id = 1:size(data, 1)
                if size(conv4_4_CPM_small, 1) == 0
                   conv4_4_CPM_small = cell(nbr_cameras,  size(data, 1));
                end

                conv4_4_CPM_small{camera_id, frame_id} = imresize(squeeze(data(frame_id, :, :, :)), [23, 41]);
            end
        end

        save_path = strcat(path, '/', scene, '/conv4_4_CPM_small.mat');
        save(save_path, 'conv4_4_CPM_small');
        [warn_msg, ~] = lastwarn;
        if ~isempty(warn_msg)
            fprintf('Saving as -v7.3 instead due to size issue!\n')
            save(save_path, 'conv4_4_CPM_small', '-v7.3');
        end

        % Delete old big files
        for camera_id = 1:nbr_cameras
            camera = cameras{camera_id};
            big_feats = strcat(path, '/', scene, '/', camera, '/conv4_4_CPM.mat');
            fprintf('Deleting big cache %s - %s\n', scene, camera);
            delete(big_feats);
        end
    end
end


function [files, nbr_files] = list_filenames(path) %#ok<*INUSL>
    % Lists and counts all files in a directory
    files = dir(path);
    files = struct2cell(files);
    files = files(1, 3 : end);
    nbr_files = numel(files);
end
