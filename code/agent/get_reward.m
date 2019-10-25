function rewards = get_reward(fracs_triangulated, visited_cams, nbr_persons)
% Computes reward for ACTOR

% Load global config
global CONFIG

% Initialize von Mises rewards
rewards_mises = zeros(numel(visited_cams), 1);

% NaN added at the, to prevent "learning" from the end of an active-view.
% We have one less action taken than the number of visited views, as
% a trajectory looks like "view-1 --> view-2 --> ... --> view-N", i.e.,
% the arrows are one less than #views, and arrows represent the actions
% taken. But due to caffe input formatting the reward-holder must have
% equally many entries as #views, hence we set the final=NaN, and our code
% is such that we ignore gradients from this NaN-reward.
rewards_mises(end) = nan; 

% Add penalties for same cam choices
[~, unique_idxs] = unique(visited_cams);
same_penalties = ones(size(rewards_mises)) * CONFIG.agent_same_cam_penalty;
same_penalties(unique_idxs) = 0;
same_penalties = circshift(same_penalties, [-1, 0]);
rewards_mises = rewards_mises + same_penalties;

% Now we need to propagate the final reward and add step penalties
end_reward = fracs_triangulated(end); % reward = (minimum of) final fraction of covered joints
rewards_mises(end - 1) = rewards_mises(end - 1) + end_reward;
if CONFIG.agent_stop_triangulated
    rewards_mises(end - 1) = rewards_mises(end - 1) + ...
                                (numel(rewards_mises) - 1) / ...
                                 nbr_persons * CONFIG.agent_step_penalty;
end
rewards_mises(1 : end - 1) = discount_rewards(rewards_mises(1 : end - 1));

% Bundle rewards
rewards = struct('mises', rewards_mises(:));
end
function discounted = discount_rewards(rewards)
    nan_idxs_rewards = isnan(rewards);
    rewards(nan_idxs_rewards) = 0;
    discounted = flipud(cumsum(flipud(rewards)));
    discounted(nan_idxs_rewards) = nan;
end