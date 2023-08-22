%% Synthetic Data Generation by Markov Chain Monte Carlo (MCMC)
% MCMC methods, such as Metropolis-Hastings and Gibbs sampling, generate samples from a target 
% distribution by iteratively updating states based on conditional probabilities.

%% Load the dataset
clear;
load fisheriris.mat;
original_data=meas;
Classes=3; % Number of Classes
NF=size(original_data); NF=NF(1,2); % Number of Features
Target(1:50)=1;Target(51:100)=2;Target(101:150)=3;Target=Target'; % Original labels

%% MCMC Body
for i=1:NF
original_data=meas (:,i);
num_original_data = numel(original_data);
% Define the target distribution (you can adjust this based on your data)
target_mean = mean(original_data);
target_std = std(original_data);
% Parameters for the Metropolis-Hastings algorithm

num_samples = 800; % Number of synthetic data points to generate

burn_in = 1000;      % Number of burn-in iterations
proposal_std = 0.9;  % Standard deviation of proposal distribution
% Initialize the chain
current_state = mean(original_data); % Start from the mean of the original data
% Preallocate array for storing generated samples
synthetic_data = zeros(1, num_samples);
% Metropolis-Hastings algorithm
for t = 1:num_samples + burn_in
% Propose a new state from a Gaussian distribution
proposed_state = current_state + proposal_std * randn();
% Calculate acceptance ratio
current_likelihood = normpdf(current_state, target_mean, target_std);
proposed_likelihood = normpdf(proposed_state, target_mean, target_std);
acceptance_ratio = min(1, proposed_likelihood / current_likelihood);
% Accept or reject the proposed state
if rand() < acceptance_ratio
current_state = proposed_state;
end
% Store samples after burn-in
if t > burn_in
synthetic_data(t - burn_in) = current_state;
end
end
Syn(:,i)=synthetic_data;
end

%% Getting labels of synthetic generated data by K-means clustering
[Lbl,C,sumd,D] = kmeans(Syn,Classes,'MaxIter',10000,...
    'Display','final','Replicates',10);

%% Plot data and classes
Feature1=2;
Feature2=3;
f1=meas(:,Feature1); % feature1
f2=meas(:,Feature2); % feature 2
ff1=Syn(:,Feature1); % feature1
ff2=Syn(:,Feature2); % feature 2
figure('units','normalized','outerposition',[0 0 1 1])
subplot(3,2,1)
area(meas, 'linewidth',1); title('Original Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(3,2,2)
area(Syn, 'linewidth',1); title('Synthetic Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(3,2,3)
gscatter(f1,f2,Target,'rkg','.',20); title('Original');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(3,2,4)
gscatter(ff1,ff2,Lbl,'rkg','.',20); title('Synthetic');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(3,2,[5 6])
histogram(meas, 'Normalization', 'probability', 'DisplayName', 'Original Data');
hold on;
histogram(Syn, 'Normalization', 'probability', 'DisplayName', 'Synthetic Data');
legend('Original','Synthetic')
%% Train and Test
% Training Synthetic dataset by SVM
Mdlsvm  = fitcecoc(Syn,Lbl); CVMdlsvm = crossval(Mdlsvm); 
SVMError = kfoldLoss(CVMdlsvm);
SVMAccAugTrain = (1 - SVMError)*100;

% Predict new samples (the whole original dataset)
[label5,score5,cost5] = predict(Mdlsvm,meas);

% % Test error and accuracy calculations
DataSize=size(meas);DataSize=DataSize(1,1);
a=0;b=0;c=0;
for i=1:DataSize
if label5(i)== 1
a=a+1;
elseif label5(i)==2
b=b+1;
else
label5(i)==3
c=c+1;
end;end;
erra=abs(a-50);errb=abs(b-50);errc=abs(c-50);
err=erra+errb+errc;TestErr=err*100/DataSize;
SVMAccAugTest=100-TestErr; % Test Accuracy

% Train and Test Accuracy Results
AugRessvm = [' SDG Train SVM "',num2str(SVMAccAugTrain),'" SDG Test SVM"', num2str(SVMAccAugTest),'"'];
disp(AugRessvm);


