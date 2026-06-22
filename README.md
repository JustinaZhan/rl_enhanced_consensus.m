function [decision_weights, final_ranking] = rl_enhanced_consensus(data_file)
%RL_ENHANCED_CONSENSUS Reinforcement-learning-enhanced consensus model
%
%   [decision_weights, final_ranking] = rl_enhanced_consensus(data_file)
%
%   This function implements a reinforcement-learning-enhanced consensus
%   reaching process for group decision making. The model includes:
%
%       1. Rating data validation and normalization
%       2. Preference relation construction
%       3. Reinforcement-learning-enhanced K-means clustering
%       4. Intra-cluster and inter-cluster consensus improvement
%       5. Dynamic decision-maker weight adjustment
%       6. Final group ranking generation
%
%   Input:
%       data_file        Excel file containing the rating matrix.
%                        Rows correspond to decision makers.
%                        Columns correspond to alternatives/movies.
%                        Rating values should be in [0, 5].
%
%   Output:
%       decision_weights Final weights of decision makers
%       final_ranking    Final ranking of movies, where each row is:
%                        [movie_id, final_score]
%
%   Example:
%       [weights, ranking] = rl_enhanced_consensus("2.xlsx");
%
%   ---------------------------------------------------------------------
%   Author: Your Name
%   Date  : YYYY-MM-DD
%   ---------------------------------------------------------------------

    %% 1. Data Loading and Validation

    if nargin < 1 || isempty(data_file)
        error("Please provide the Excel data file path.");
    end

    ratings_matrix = readmatrix(data_file);

    validateattributes(ratings_matrix, {'numeric'}, {'2d', 'nonempty'});
    assert(all(ratings_matrix(:) >= 0 & ratings_matrix(:) <= 5), ...
        'Rating values must be in the range [0, 5].');
    assert(~any(isnan(ratings_matrix(:))), ...
        'The rating matrix contains NaN values.');
    assert(~any(isinf(ratings_matrix(:))), ...
        'The rating matrix contains Inf values.');

    normalized_ratings = ratings_matrix / 5;

    [numDecisionMakers, numMovies] = size(normalized_ratings);

    fprintf('System initialized: %d decision makers, %d movies.\n', ...
        numDecisionMakers, numMovies);

    %% 2. Construct Preference Matrices

    preference_matrices = cell(numDecisionMakers, 1);

    for dm = 1:numDecisionMakers
        pref_matrix = zeros(numMovies);

        for i = 1:numMovies
            for j = 1:numMovies
                if i == j
                    pref_matrix(i, j) = 0.5;
                else
                    pref_value = ...
                        (normalized_ratings(dm, i) - normalized_ratings(dm, j) + 1) / 2;
                    pref_matrix(i, j) = max(0, min(1, pref_value));
                end
            end
        end

        preference_matrices{dm} = pref_matrix;
    end

    %% 3. Reinforcement-Learning-Based K-means Initialization

    rng(123);

    numClusters = 3;
    maxEpisodes = 50;

    learningRate = 0.1;
    discountFactor = 0.9;
    explorationRate = 0.3;

    featureDim = numel(preference_matrices{1});
    X = zeros(numDecisionMakers, featureDim);

    for dm = 1:numDecisionMakers
        X(dm, :) = preference_matrices{dm}(:)';
    end

    Q = zeros(numDecisionMakers, numDecisionMakers);

    for episode = 1:maxEpisodes

        current_centers = randperm(numDecisionMakers, numClusters);

        [~, ~, ~, D] = kmeans(X, numClusters, ...
            'Start', X(current_centers, :));

        current_reward = 1 / sum(min(D, [], 2));

        if rand() < explorationRate
            new_centers = randperm(numDecisionMakers, numClusters);
        else
            [~, sorted_indices] = sort(sum(Q, 2), 'descend');
            new_centers = sorted_indices(1:numClusters)';
        end

        [~, ~, ~, D_new] = kmeans(X, numClusters, ...
            'Start', X(new_centers, :));

        new_reward = 1 / sum(min(D_new, [], 2));

        for i = 1:numClusters
            for j = 1:numClusters
                old_state = current_centers(i);
                new_state = new_centers(j);

                Q(old_state, new_state) = Q(old_state, new_state) + ...
                    learningRate * ...
                    (new_reward + ...
                    discountFactor * max(Q(new_state, :)) - ...
                    Q(old_state, new_state));
            end
        end

        explorationRate = explorationRate * 0.95;
    end

    [~, best_centers] = maxk(sum(Q, 2), numClusters);

    fprintf('\nRL-selected initial centers: %s\n', mat2str(best_centers'));

    [final_idx, ~] = kmeans(X, numClusters, ...
        'Start', X(best_centers, :));

    %% 4. Display Clustering Results

    fprintf('\n=== Clustering Results ===\n');

    for c = 1:numClusters
        members = find(final_idx == c);

        fprintf('Cluster %d: Number of decision makers = %d, Members: ', ...
            c, numel(members));
        fprintf('%d ', members);
        fprintf('\n');
    end

    %% 5. Initial Consensus Calculation

    initial_intra_consensus = zeros(numClusters, 1);

    for c = 1:numClusters
        members = find(final_idx == c);
        cluster_preferences = preference_matrices(members);

        cluster_mean = mean(cat(3, cluster_preferences{:}), 3);

        consensus = 1 - mean( ...
            abs(cat(3, cluster_preferences{:}) - cluster_mean), ...
            'all');

        initial_intra_consensus(c) = consensus;
    end

    fprintf('\n=== Initial Intra-Cluster Consensus ===\n');

    for c = 1:numClusters
        fprintf('Cluster %d = %.4f\n', c, initial_intra_consensus(c));
    end

    all_preferences = cat(3, preference_matrices{:});
    global_mean = mean(all_preferences, 3);

    global_consensus = 1 - mean(abs(all_preferences - global_mean), 'all');

    fprintf('\nInitial global consensus = %.4f, total members = %d\n', ...
        global_consensus, numDecisionMakers);

    print_movie_ranking('Initial Movie Ranking', global_mean, numMovies, false);

    %% 6. Reinforcement-Learning-Based Consensus Improvement

    consensusThreshold = 0.98;
    maxIterations = 200;

    rlLearningRate = 0.2;
    rlDiscount = 0.95;
    rlExplore = 0.25;

    cluster_means = cell(numClusters, 1);
    intra_consensus_history = cell(numClusters, 1);
    intra_iterations = zeros(numClusters, 1);
    intra_Q_tables = cell(numClusters, 1);

    %% 6.1 Intra-Cluster Consensus Improvement

    for c = 1:numClusters

        members = find(final_idx == c);
        numMembers = numel(members);

        cluster_preferences = preference_matrices(members);

        intra_Q_tables{c} = zeros(11, 3);

        current_mean = mean(cat(3, cluster_preferences{:}), 3);
        consensus = initial_intra_consensus(c);

        intra_consensus_history{c} = consensus;

        iter = 0;

        while consensus < consensusThreshold && iter < maxIterations

            state = min(floor(consensus * 10) + 1, 11);

            if rand() < rlExplore
                action = randi(3);
            else
                [~, action] = max(intra_Q_tables{c}(state, :));
            end

            adjusted_preferences = cell(numMembers, 1);

            switch action
                case 1
                    % Mild adjustment
                    for i = 1:numMembers
                        adjusted_preferences{i} = ...
                            (cluster_preferences{i} + current_mean) / 2;
                    end

                case 2
                    % Aggressive adjustment
                    for i = 1:numMembers
                        adjusted_preferences{i} = ...
                            (cluster_preferences{i} + 2 * current_mean) / 3;
                    end

                case 3
                    % Selective adjustment
                    differences = zeros(numMembers, 1);

                    for i = 1:numMembers
                        differences(i) = sum(abs( ...
                            cluster_preferences{i}(:) - current_mean(:)));
                    end

                    diff_threshold = mean(differences) + 0.5 * std(differences);

                    for i = 1:numMembers
                        if differences(i) > diff_threshold
                            adjusted_preferences{i} = ...
                                (cluster_preferences{i} + current_mean) / 2;
                        else
                            adjusted_preferences{i} = cluster_preferences{i};
                        end
                    end
            end

            new_mean = mean(cat(3, adjusted_preferences{:}), 3);

            new_consensus = 1 - mean( ...
                abs(cat(3, adjusted_preferences{:}) - new_mean), ...
                'all');

            reward = (new_consensus - consensus) * 100;

            new_state = min(floor(new_consensus * 10) + 1, 11);

            intra_Q_tables{c}(state, action) = ...
                intra_Q_tables{c}(state, action) + ...
                rlLearningRate * ...
                (reward + ...
                rlDiscount * max(intra_Q_tables{c}(new_state, :)) - ...
                intra_Q_tables{c}(state, action));

            consensus = new_consensus;
            current_mean = new_mean;
            cluster_preferences = adjusted_preferences;

            iter = iter + 1;

            intra_consensus_history{c} = ...
                [intra_consensus_history{c}; consensus];

            if iter > 20 && ...
                    abs(intra_consensus_history{c}(end) - ...
                    intra_consensus_history{c}(end - 1)) < 1e-4
                break;
            end
        end

        intra_iterations(c) = iter;
        cluster_means{c} = current_mean;
    end

    %% 6.2 Inter-Cluster Consensus Improvement

    global_mean = mean(cat(3, cluster_means{:}), 3);

    inter_consensus = 1 - mean( ...
        abs(cat(3, cluster_means{:}) - global_mean), ...
        'all');

    inter_consensus_history = inter_consensus;
    inter_Q_table = zeros(11, 3);

    inter_iteration = 0;

    while inter_consensus < consensusThreshold && inter_iteration < maxIterations

        state = min(floor(inter_consensus * 10) + 1, 11);

        if rand() < rlExplore
            action = randi(3);
        else
            [~, action] = max(inter_Q_table(state, :));
        end

        adjusted_means = cell(numClusters, 1);

        switch action
            case 1
                % Uniform adjustment
                for c = 1:numClusters
                    adjusted_means{c} = ...
                        (cluster_means{c} + global_mean) / 2;
                end

            case 2
                % Size-weighted adjustment
                cluster_sizes = zeros(numClusters, 1);

                for c = 1:numClusters
                    cluster_sizes(c) = sum(final_idx == c);
                end

                total_size = sum(cluster_sizes);

                for c = 1:numClusters
                    weight = cluster_sizes(c) / total_size;
                    adjusted_means{c} = ...
                        (cluster_means{c} + weight * global_mean) / ...
                        (1 + weight);
                end

            case 3
                % Selective adjustment
                differences = zeros(numClusters, 1);

                for c = 1:numClusters
                    differences(c) = sum(abs( ...
                        cluster_means{c}(:) - global_mean(:)));
                end

                diff_threshold = mean(differences) + 0.5 * std(differences);

                for c = 1:numClusters
                    if differences(c) > diff_threshold
                        adjusted_means{c} = ...
                            (cluster_means{c} + global_mean) / 2;
                    else
                        adjusted_means{c} = cluster_means{c};
                    end
                end
        end

        new_global_mean = mean(cat(3, adjusted_means{:}), 3);

        new_consensus = 1 - mean( ...
            abs(cat(3, adjusted_means{:}) - new_global_mean), ...
            'all');

        reward = (new_consensus - inter_consensus) * 100;

        new_state = min(floor(new_consensus * 10) + 1, 11);

        inter_Q_table(state, action) = inter_Q_table(state, action) + ...
            rlLearningRate * ...
            (reward + ...
            rlDiscount * max(inter_Q_table(new_state, :)) - ...
            inter_Q_table(state, action));

        inter_consensus = new_consensus;
        global_mean = new_global_mean;
        cluster_means = adjusted_means;

        inter_iteration = inter_iteration + 1;

        inter_consensus_history = ...
            [inter_consensus_history; inter_consensus];

        print_movie_ranking( ...
            sprintf('Inter-Cluster Iteration %d Movie Ranking', inter_iteration), ...
            global_mean, numMovies, false);

        if inter_iteration > 20 && ...
                abs(inter_consensus_history(end) - ...
                inter_consensus_history(end - 1)) < 1e-4
            break;
        end
    end

    %% 7. Dynamic Decision-Maker Weight Adjustment

    contribution = zeros(numDecisionMakers, 1);

    for dm = 1:numDecisionMakers
        c = final_idx(dm);

        diff_value = sum(abs( ...
            preference_matrices{dm}(:) - cluster_means{c}(:)).^2);

        contribution(dm) = exp(-diff_value / numel(preference_matrices{dm}));
    end

    baseAlpha = 0.15;
    minWeight = 0.2;
    maxReward = 3.5;

    [~, sorted_dm_indices] = sort(contribution, 'descend');

    decision_weights = ones(numDecisionMakers, 1);

    top_n = floor(0.3 * numDecisionMakers);
    middle_n = floor(0.4 * numDecisionMakers);
    bottom_n = floor(0.3 * numDecisionMakers);

    % Reward top decision makers
    for i = 1:top_n
        dm = sorted_dm_indices(i);
        c = final_idx(dm);

        max_q = max(intra_Q_tables{c}(end, :));
        reward = baseAlpha * (1 + max_q / 100);

        decision_weights(dm) = min(maxReward, 1 + 2 * reward);
    end

    % Penalize bottom decision makers
    for i = numDecisionMakers - bottom_n + 1:numDecisionMakers
        dm = sorted_dm_indices(i);

        penalty = baseAlpha * ...
            (3 - contribution(dm) / mean(contribution));

        decision_weights(dm) = max(minWeight, 1 - penalty);
    end

    % Fine-tune middle decision makers
    for i = top_n + 1:numDecisionMakers - bottom_n
        dm = sorted_dm_indices(i);

        if contribution(dm) > 1.2 * mean(contribution)
            decision_weights(dm) = 1 + baseAlpha;
        elseif contribution(dm) < 0.8 * mean(contribution)
            decision_weights(dm) = 1 - baseAlpha * 0.5;
        end
    end

    decision_weights = decision_weights * ...
        numDecisionMakers / sum(decision_weights);

    %% 8. Weighted Consensus Calculation

    weighted_intra_consensus = zeros(numClusters, 1);
    weighted_cluster_means = cell(numClusters, 1);
    weighted_intra_iterations = zeros(numClusters, 1);

    for c = 1:numClusters

        members = find(final_idx == c);

        cluster_weights = decision_weights(members);
        cluster_weights = cluster_weights / sum(cluster_weights);

        weighted_mean = zeros(numMovies);

        for i = 1:numel(members)
            weighted_mean = weighted_mean + ...
                cluster_weights(i)^2 * preference_matrices{members(i)};
        end

        weighted_mean = weighted_mean / sum(cluster_weights.^2);

        weighted_diffs = zeros(numel(members), 1);

        for i = 1:numel(members)
            weighted_diffs(i) = ...
                sum(abs(preference_matrices{members(i)}(:) - ...
                weighted_mean(:)).^2) * cluster_weights(i)^2;
        end

        consensus = 1 - sqrt(sum(weighted_diffs)) / numMovies^2;

        weighted_intra_history = consensus;
        iter = 0;

        while consensus < consensusThreshold && iter < maxIterations

            new_mean = zeros(numMovies);

            for i = 1:numel(members)

                state = min(floor(consensus * 10) + 1, 11);
                [~, best_action] = max(intra_Q_tables{c}(state, :));

                switch best_action
                    case 1
                        adjusted_pref = ...
                            (preference_matrices{members(i)} + weighted_mean) / 2;

                    case 2
                        adjusted_pref = ...
                            (preference_matrices{members(i)} + ...
                            2 * weighted_mean) / 3;

                    case 3
                        diff_value = sum(abs( ...
                            preference_matrices{members(i)}(:) - ...
                            weighted_mean(:)));

                        if diff_value > mean(weighted_diffs) + ...
                                0.5 * std(weighted_diffs)
                            adjusted_pref = ...
                                (preference_matrices{members(i)} + ...
                                weighted_mean) / 2;
                        else
                            adjusted_pref = preference_matrices{members(i)};
                        end
                end

                new_mean = new_mean + ...
                    cluster_weights(i)^2 * adjusted_pref;
            end

            weighted_mean = new_mean / sum(cluster_weights.^2);

            weighted_diffs = zeros(numel(members), 1);

            for i = 1:numel(members)
                weighted_diffs(i) = ...
                    sum(abs(preference_matrices{members(i)}(:) - ...
                    weighted_mean(:)).^2) * cluster_weights(i)^2;
            end

            consensus = 1 - sqrt(sum(weighted_diffs)) / numMovies^2;

            iter = iter + 1;

            weighted_intra_history = ...
                [weighted_intra_history; consensus];

            print_movie_ranking( ...
                sprintf('Weighted Cluster %d Iteration %d Movie Ranking', c, iter), ...
                weighted_mean, numMovies, false);

            if iter > 20 && ...
                    abs(weighted_intra_history(end) - ...
                    weighted_intra_history(end - 1)) < 1e-4
                break;
            end
        end

        weighted_intra_iterations(c) = iter;
        weighted_intra_consensus(c) = consensus;
        weighted_cluster_means{c} = weighted_mean;
    end

    %% 9. Weighted Inter-Cluster Consensus

    cluster_weights = zeros(numClusters, 1);

    for c = 1:numClusters
        cluster_weights(c) = ...
            sum(decision_weights(final_idx == c)) * ...
            weighted_intra_consensus(c);
    end

    cluster_weights = cluster_weights / sum(cluster_weights);

    weighted_global_mean = zeros(numMovies);

    for c = 1:numClusters
        weighted_global_mean = weighted_global_mean + ...
            cluster_weights(c) * weighted_cluster_means{c};
    end

    weighted_inter_diffs = zeros(numClusters, 1);

    for c = 1:numClusters
        weighted_inter_diffs(c) = ...
            sum(abs(weighted_cluster_means{c}(:) - ...
            weighted_global_mean(:)).^2) * cluster_weights(c);
    end

    weighted_inter_consensus = ...
        1 - sqrt(sum(weighted_inter_diffs)) / numMovies^2;

    weighted_inter_history = weighted_inter_consensus;
    weighted_inter_iteration = 0;

    while weighted_inter_consensus < consensusThreshold && ...
            weighted_inter_iteration < maxIterations

        new_global_mean = zeros(numMovies);

        for c = 1:numClusters

            state = min(floor(weighted_inter_consensus * 10) + 1, 11);
            [~, best_action] = max(inter_Q_table(state, :));

            switch best_action
                case 1
                    adjusted_cluster = ...
                        (weighted_cluster_means{c} + weighted_global_mean) / 2;

                case 2
                    adjusted_cluster = ...
                        (weighted_cluster_means{c} + ...
                        cluster_weights(c) * weighted_global_mean) / ...
                        (1 + cluster_weights(c));

                case 3
                    diff_value = sum(abs( ...
                        weighted_cluster_means{c}(:) - ...
                        weighted_global_mean(:)));

                    if diff_value > mean(weighted_inter_diffs) + ...
                            0.5 * std(weighted_inter_diffs)
                        adjusted_cluster = ...
                            (weighted_cluster_means{c} + weighted_global_mean) / 2;
                    else
                        adjusted_cluster = weighted_cluster_means{c};
                    end
            end

            new_global_mean = new_global_mean + ...
                cluster_weights(c) * adjusted_cluster;
        end

        weighted_global_mean = new_global_mean;

        weighted_inter_diffs = zeros(numClusters, 1);

        for c = 1:numClusters
            weighted_inter_diffs(c) = ...
                sum(abs(weighted_cluster_means{c}(:) - ...
                weighted_global_mean(:)).^2) * cluster_weights(c);
        end

        weighted_inter_consensus = ...
            1 - sqrt(sum(weighted_inter_diffs)) / numMovies^2;

        weighted_inter_iteration = weighted_inter_iteration + 1;

        weighted_inter_history = ...
            [weighted_inter_history; weighted_inter_consensus];

        print_movie_ranking( ...
            sprintf('Weighted Inter-Cluster Iteration %d Movie Ranking', ...
            weighted_inter_iteration), ...
            weighted_global_mean, numMovies, false);

        if weighted_inter_iteration > 20 && ...
                abs(weighted_inter_history(end) - ...
                weighted_inter_history(end - 1)) < 1e-4
            break;
        end
    end

    %% 10. Result Summary

    fprintf('\n=== Iteration Summary ===\n');
    fprintf('Stage              | Cluster 1 | Cluster 2 | Cluster 3 | Inter-cluster\n');
    fprintf('-------------------+-----------+-----------+-----------+---------------\n');

    fprintf('RL consensus       | %9d | %9d | %9d | %13d\n', ...
        intra_iterations(1), intra_iterations(2), ...
        intra_iterations(3), inter_iteration);

    fprintf('Weighted consensus | %9d | %9d | %9d | %13d\n', ...
        weighted_intra_iterations(1), weighted_intra_iterations(2), ...
        weighted_intra_iterations(3), weighted_inter_iteration);

    fprintf('\n=== Consensus Comparison ===\n');
    fprintf('Item              | RL Consensus | Weighted Consensus | Change\n');
    fprintf('------------------+--------------+--------------------+--------\n');

    for c = 1:numClusters
        fprintf('Cluster %-9d | %-12.4f | %-18.4f | %+6.4f\n', ...
            c, ...
            intra_consensus_history{c}(end), ...
            weighted_intra_consensus(c), ...
            weighted_intra_consensus(c) - intra_consensus_history{c}(end));
    end

    fprintf('Inter-cluster     | %-12.4f | %-18.4f | %+6.4f\n', ...
        inter_consensus_history(end), ...
        weighted_inter_consensus, ...
        weighted_inter_consensus - inter_consensus_history(end));

    %% 11. Final Movie Ranking

    movie_scores = zeros(numMovies, 1);

    for i = 1:numMovies
        movie_scores(i) = mean(weighted_global_mean(i, :));
    end

    final_scores = movie_scores * 5;

    [sorted_scores, sorted_indices] = sort(final_scores, 'descend');

    fprintf('\n=== Final Group Movie Ranking ===\n');
    fprintf('Rank | Movie ID | Score [0, 5]\n');
    fprintf('-----+----------+-------------\n');

    for i = 1:numMovies
        fprintf('%4d | %8d | %11.4f\n', ...
            i, sorted_indices(i), sorted_scores(i));
    end

    final_ranking = [sorted_indices, sorted_scores];

    %% 12. Display Weight Categories

    fprintf('\n=== Rewarded Decision Makers ===\n');
    fprintf('%d ', sorted_dm_indices(1:top_n));
    fprintf('\n');

    fprintf('\n=== Fine-Tuned Decision Makers ===\n');
    fprintf('%d ', sorted_dm_indices(top_n + 1:top_n + middle_n));
    fprintf('\n');

    fprintf('\n=== Penalized Decision Makers ===\n');
    fprintf('%d ', sorted_dm_indices(numDecisionMakers - bottom_n + 1:numDecisionMakers));
    fprintf('\n');

end


%% ========================================================================
%  Local Utility Function
%  ========================================================================

function print_movie_ranking(title_text, preference_matrix, numMovies, scale_to_five)
%PRINT_MOVIE_RANKING Print movie ranking based on a preference matrix.
%
%   title_text        Title displayed before the ranking table
%   preference_matrix Group or cluster preference matrix
%   numMovies         Number of movies
%   scale_to_five     Whether to scale scores to [0, 5]

    if nargin < 4
        scale_to_five = false;
    end

    movie_scores = zeros(numMovies, 1);

    for i = 1:numMovies
        movie_scores(i) = mean(preference_matrix(i, :));
    end

    if scale_to_five
        movie_scores = movie_scores * 5;
        score_label = 'Score [0, 5]';
    else
        score_label = 'Preference Score';
    end

    [sorted_scores, sorted_indices] = sort(movie_scores, 'descend');

    fprintf('\n=== %s ===\n', title_text);
    fprintf('Rank | Movie ID | %s\n', score_label);
    fprintf('-----+----------+----------------\n');

    for i = 1:numMovies
        fprintf('%4d | %8d | %14.4f\n', ...
            i, sorted_indices(i), sorted_scores(i));
    end

end
