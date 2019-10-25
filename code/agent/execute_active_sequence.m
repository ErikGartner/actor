function [out_sequence, recorder] = ...
            execute_active_sequence(env, agent, stats, greedy, ep, ...
                                    sequence_length, recorder, do_record)                             
% Produce active-sequence using ACTOR
                                
global CONFIG
                                    
% Default args
if ~exist('do_record', 'var')
    do_record = 0;
end

% Reset agent
agent.reset_episode();
tracker = Tracker(env.scene());
tri_poses = MultiPeopleTriPose(env.scene().nbr_persons);

% Store environment starting point to reverse to that after running sequence
init_frame_idx = env.frame_idx;
init_camera_idx = env.camera_idx;
tracker.start(init_frame_idx, init_camera_idx);

% Run agent over video sequence
out_sequence = cell(1, sequence_length);
for i = 1 : sequence_length
    
    % Go to next frame
    frame_idx = init_frame_idx + CONFIG.sequence_step * (i - 1);
    if frame_idx ~= env.frame_idx
        env.goto_frame(frame_idx);
        tracker.next(frame_idx, env.camera_idx);
    end
        
    % Produce active-view in current time-freeze
    out = execute_active_view(env, agent, tracker, tri_poses, stats, ...
                              greedy, ep, recorder, do_record);
    out_sequence{i} = out;
                               
    % Prepare for next frame
    if i < sequence_length
        agent.next_frame();
        recorder.next_frame();
    end
end
% Go back to environment start state
env.goto_frame(init_frame_idx);
env.goto_cam(init_camera_idx);

% Track stats for init frame
stats.s('Recon err RL init frame').collect(out_sequence{1}.cum_recon_error);
