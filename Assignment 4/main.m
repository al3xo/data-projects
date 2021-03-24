%% Reset Workspace
close all; clear; clc;

%% Training Data Setup 
path_to_digits = "train-images.idx3-ubyte";
path_to_labels = "train-labels.idx1-ubyte";

[images, labels] = mnist_parse(path_to_digits, path_to_labels);

data = zeros(28*28, 60000);
for i = 1:60000
    data(:, i) = reshape(images(:, :, i), [28*28, 1]);
end

avg_scaling = mean(data, 2);

for i = 1:728
    data(i,:) = data(i,:) - avg_scaling(i);
end

%% Test Data Setup
path_to_t_digits = "t10k-images.idx3-ubyte";
path_to_t_labels = "t10k-labels.idx1-ubyte";

[t_images, t_labels] = mnist_parse(path_to_t_digits, path_to_t_labels);


test_data = zeros(28*28, length(t_images));
for i = 1:length(t_images)
    test_data(:, i) = reshape(t_images(:, :, i), [28*28, 1]);
end

for i = 1:728
    test_data(i,:) = test_data(i,:) - avg_scaling(i);
end

%% Analysis of SVD

% rank of S matrix
[U, S, V] = svd(data, 'econ');
r_digit = rank(S);
sig = diag(S);

% singular value space
figure(1)
plot(1:length(sig), sig, 'co', r_digit, 0, 'k+'); 
legend('Singular Values (SV)', 'Last Non-Zero SV');
title('SV of Images SVD');
xlabel('SV Index')
ylabel('SV')

% energy
energy = zeros(1, r_digit);
for i = 1:r_digit
    energy(i) = sum(sig(1:i).^2)/sum(sig.^2);
end

%% feature determination:
figure(2)
sv_10s = zeros(1, 10);
for i = 1:10
    mode = find(energy >= i/10, 1, 'first');
    M_rank = U(:,1:mode)*S(1:mode,1:mode)*V(:,1:mode).';
    
    subplot(2, 5, i) % plot first image
    
    image(reshape(M_rank(:, 1), [28, 28]))
    
    axis image;
    title(['Image at ', num2str(i*10), '% Energy (M = ',num2str(mode),')'])
    set(gca, 'FontSize', 14)
    axis off;
    
    sv_10s(i) = mode;
end
feature = sv_10s(9); %87
%%
figure(3)
plot(1:length(energy), energy, 'go', feature, energy(feature), 'k+');
legend('Energy', '90% Threshold');
title('Energy Contained per SV')
xlabel('SV Index')
ylabel('Energy (%)')

%% Projection onto 3 V-modes
figure(4)
modes = [4 5 8];
for label = 0:9
    label_indices = find(labels == label);
    plot3(V(label_indices, modes(1)), V(label_indices, modes(2)),... 
    V(label_indices, modes(3)),'o', 'DisplayName', sprintf('%i',label))
    hold on;
end
xlabel('Mode 4'), ylabel('Mode 5'), zlabel('Mode 8');
title('Projection onto V-modes 4, 5, and 8');
legend;
hold off;

%% LDA Model for two digits 
[U, ~, ~, t, w, ~, ~, p] = two_index_trainer(data, labels, 6, 7, feature);

a = 6;
b = 7;
% refine test_data for digits 6 and 7
index_abt = find(t_labels == a | t_labels == b);
TestSet = test_data(:, index_abt);
labels_abt = t_labels(index_abt);
labels_abt = double(labels_abt == a);

% test LDA model on test data
TestNum = size(TestSet,2);
TestMat = U'*TestSet; % PCA projection
pval = w'*TestMat;

% calculate accuracy
results = (pval > t);
err = abs(results - labels_abt');
errNum = sum(err);

successRates_LDA_6_7 = 1 - errNum/TestNum;
successRates_LDA_6_7_TRAINING = p;


%% LDA Model Construction and Testing => successRate for all unique pairs of a and b (0:9)

successRates_LDA = zeros(10, 10);
successRates_LDA_TRAINING = zeros(10, 10);
for a = 0:9
    for b = a+1:9
        [U, ~, ~, t, w, ~, ~, p] = two_index_trainer(data, labels, a, b, feature);
        
        % refine test_data for digits a and b
        index_abt = find(t_labels == a | t_labels == b);
        TestSet = test_data(:, index_abt);
        labels_abt = t_labels(index_abt);
        labels_abt = double(labels_abt == a);
        

        % test LDA model on data
        TestNum = size(TestSet,2);
        TestMat = U'*TestSet; % PCA projection
        pval = w'*TestMat;

        % calculate accuracy
        results = (pval > t);
        err = abs(results - labels_abt');
        errNum = sum(err);
        
        successRates_LDA(a+1, b+1) = 1 - errNum/TestNum;
        
        successRates_LDA_TRAINING(a+1, b+1) = p;
    end
end

%% successRate Analysis
max_val = zeros(1, 3);
min_val = ones(1, 3);

for a = 0:9
    for b = a+1:9
        if max_val(1) < successRates_LDA(a+1, b+1)
            max_val = [successRates_LDA(a+1, b+1) a+1 b+1];
        end
        
        if min_val(1) > successRates_LDA(a+1, b+1)
            min_val = [successRates_LDA(a+1, b+1) a+1 b+1];
        end
    end
end

easy = max_val(2:3) - 1; % easiest to seperate
hard = min_val(2:3) - 1; % hardest to seperate

%% SVM - 10 Digits
[~, S, V] = svd(data, 'econ');
scale = max(S*V', [], 'all');
normalizedTrainingData = S*V' / scale;

% train model
SVMmodel = fitcecoc(normalizedTrainingData', labels');

% test on training data
labelPredictions = predict(SVMmodel, normalizedTrainingData');
results = labels == labelPredictions;
success_10digits_SVM_TRAINING = sum(results) / length(labels);

% test on test data
[~, St, Vt] = svd(test_data, 'econ');
normalizedTestData = St*Vt' / scale;

labelPredictions = predict(SVMmodel, normalizedTestData');

results = labelPredictions == t_labels;
success_10digits_SVM = sum(results) / length(t_labels);

%% SVM - 2 Digits
targets = [easy, hard];
success_2digits_SVM = zeros(1, 2);
success_2digits_SVM_TRAINING = zeros(1, 2);

for i = 1:2:4
    a = targets(i);
    b = targets(i+1);

    % Training Data Set Filtration
    ind_ab = find(labels == a | labels == b);
    data_ab = data(:, ind_ab);
    labels_ab = labels(ind_ab);

    % SVD of binary data
    [U,S,V] = svd(data_ab, 'econ');
    scale = max(S*V', [], 'all');
    normalizedTrainingData = S*V' / scale;

    % Train SVM Model
    SVMmodel = fitcsvm(normalizedTrainingData', labels_ab);
    
    % Test on Training Data
    labelPredictions = predict(SVMmodel, normalizedTrainingData');
    results = labels_ab == labelPredictions;
    success_2digits_SVM_TRAINING(ceil(i/2)) = sum(results) / length(labels_ab);

    % Test Data Set Filtration
    ind_ab_t = find(t_labels == a | t_labels == b);
    TestSet = test_data(:, ind_ab_t);
    labels_ab_t = t_labels(ind_ab_t);

    % SVD of binary test data
    [Ut, St, Vt] = svd(TestSet);
    normalizedTestData = St*Vt' / scale;

    % SVM Model Predictions and Success Rate
    labelPredictions = predict(SVMmodel, normalizedTestData');

    results = (labelPredictions == labels_ab_t);
    
    % first is easy, second is hard
    success_2digits_SVM(ceil(i/2)) = sum(results) / length(labels_ab_t); 
end

%% Decision Trees - 10 digits
[~, S, V] = svd(data, 'econ');
scale = max(S*V', [], 'all');
normalizedTrainingData = S*V' / scale;

tree = fitctree(normalizedTrainingData', labels');

% test on training data
labelPredictions = predict(tree, normalizedTrainingData');
results = labels == labelPredictions;
success_10digits_tree_TRAINING = sum(results) / length(labels);

% test on test data
[~, St, Vt] = svd(test_data, 'econ');
normalizedTestData = St*Vt' / scale;

labelPredictions = predict(tree, normalizedTestData');

results = labelPredictions == t_labels;
success_10digits_tree = sum(results) / length(t_labels);



%% Decision Trees - 2 Digits
targets = [easy, hard];
success_2digits_tree = zeros(1, 2);

for i = 1:2:4
    a = targets(i);
    b = targets(i+1);

    % Training Data Set Filtration
    ind_ab = find(labels == a | labels == b);
    data_ab = data(:, ind_ab);
    labels_ab = labels(ind_ab);

    % SVD of binary data
    [U,S,V] = svd(data_ab, 'econ');
    scale = max(S*V', [], 'all');
    normalizedTrainingData = S*V' / scale;

    % Train SVM Model
    tree = fitctree(normalizedTrainingData', labels_ab);
    
    % Test on Training Data
    labelPredictions = predict(tree, normalizedTrainingData');
    results = labels_ab == labelPredictions;
    success_2digits_tree_TRAINING(ceil(i/2)) = sum(results) / length(labels_ab);

    % Test Data Set Filtration
    ind_ab_t = find(t_labels == a | t_labels == b);
    TestSet = test_data(:, ind_ab_t);
    labels_ab_t = t_labels(ind_ab_t);

    % SVD of binary test data
    [Ut, St, Vt] = svd(TestSet);
    normalizedTestData = St*Vt' / scale;

    % SVM Model Predictions and Success Rate
    labelPredictions = predict(tree, normalizedTestData');

    results = (labelPredictions == labels_ab_t);
    
    % first is easy, second is hard
    success_2digits_tree(ceil(i/2)) = sum(results) / length(labels_ab_t); 
end