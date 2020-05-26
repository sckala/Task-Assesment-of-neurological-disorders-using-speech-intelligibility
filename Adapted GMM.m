clear all
clc
folder='G:\phd\code and implementation\TRAIN_22(EDITED)\';
files = dir(strcat(folder,'*.wav'));
for i = 1:length(files)
    %fprintf('i \n ',i);
%for i=1:2
fname = strcat(folder,files(i,1).name);
[d sr] = audioread(fname);
%features{i} = melcepst(d,sr,'C',13);
features4{i} = melcepst1(d,sr,'D',26);
% features{i} = melcepst(d,sr);
fprintf('i= %d\n',i);
a3= cellfun(@transpose,features4,'UniformOutput',false); %feature extraction
end

load('trainfeaturesfor100-words20-dimensionaldateapril13''19.mat');
ncha=2;
   nWorkers=1;
%  nmix = 4;           % In this case, we know the # of mixtures needed
 %nmix=2;%75
 nmix=128;
 %nmix=256;
final_niter = 10;
ds_factor = 1;

features5={features{:};
features3= {features4{1:4},features1{1:300},features2{1:300},features4{1:300}};
 for i=1:4400
 gmm{i} = gmm_em(a(:,i), nmix, final_niter, ds_factor, nWorkers);
 end
 for i=1:11
 gmm2{i} = gmm_em(a3(:,i+11), nmix, final_niter, ds_factor, nWorkers);
 end
 for i = 1:11
 gmm3{i} = gmm_em(a3(:,i+22), nmix, final_niter, ds_factor, nWorkers);
 end
 for i = 1:22
 gmm4{i} = gmm_em(a3(:,i+33), nmix, final_niter, ds_factor, nWorkers);
 end
gmmClsfull = { gmm{:}, gmm2{:}, gmm3{:}, gmm4{:}};
map_tau = 10; %relavance factor 10
 config = 'mvw';
 gmm1=cell(1,1200);

map_tau = 10; %relavance factor 10
 config = 'mvw';
 gmm1=cell(1,1200);

folder='G:\phd\code and implementation\uncommon words_intelligible\fold 1\train 1200\';
files = dir(strcat(folder,'*.wav'));
for i = 1:length(files)
fname = strcat(folder,files(i,1).name);
[d sr] = audioread(fname);
features3{i} = melcepst(d,sr,'C',13);
fprintf('i= %d\n',i);
a2= cellfun(@transpose,features,'UniformOutput',false); %feature extraction
end

for i = 1:1
    trainData{i} = features4{i};
   t2 = cellfun(@transpose,trainData,'UniformOutput',false);%testdata
 end



trainData=[];
for i = 1:4400
    trainData{i} = features4{i};
   t1 = cellfun(@transpose,trainData,'UniformOutput',false);%testdata
end

for z=1:1
gmm1{z} = mapAdapt(a2(:,z), gmmClsfull{z}, map_tau, config); %
end

logLikT = zeros(4060,1200);
logLikTT = zeros(1740,1200);


logLikT = zeros(1,1);
for i = 1:1 %no of train examples
fprintf('i= %d\n',i);
for j = 1:11    %no of sub models
gmm_llk1 = compute_llk(t1{i}, gmm{j}.mu, gmm{j}.sigma, gmm{j}.w(:));
gmm1_llk1 = compute_llk(t1{i}, gmm1{j}.mu, gmm1{j}.sigma, gmm1{j}.w(:));
logLikT(i,j) = mean(gmm1_llk1 - gmm_llk1);
end
end

map_tau = 10; %relavance factor 10
 config = 'mvw';
 gmm1=cell(1,1200);

folder='G:\phd\code and implementation\uncommon words_intelligible\fold 1\train 1200\';
files = dir(strcat(folder,'*.wav'));
for i = 1:length(files)
fname = strcat(folder,files(i,1).name);
[d sr] = audioread(fname);
features3{i} = melcepst(d,sr,'C',13);
fprintf('i= %d\n',i);
a2= cellfun(@transpose,features3,'UniformOutput',false); %feature extraction
end

for i = 1:1200
    trainData{i} = features3{i};
   t2 = cellfun(@transpose,trainData,'UniformOutput',false);%testdata
end



trainData=[];
for i = 1:4200
    trainData{i} = features1{i};
   t1 = cellfun(@transpose,trainData,'UniformOutput',false);%testdata
end

for z=1:1200
gmm1{z} = mapAdapt(a2(:,z), gmm2{z}, map_tau, config); %
end

logLikT = zeros(29000,512);
logLikTT = zeros(8700,512);
for i = 1:4200 %no of train examples
fprintf('i= %d\n',i);
for j = 1:1200    %no of sub models
gmm_llk1 = compute_llk(t1{i}, gmm2{j}.mu, gmm2{j}.sigma, gmm2{j}.w(:));
gmm1_llk1 = compute_llk(t1{i}, gmm1{j}.mu, gmm1{j}.sigma, gmm1{j}.w(:));
logLikT(i,j) = mean(gmm1_llk1 - gmm_llk1);
end
end

map_tau = 10; %relavance factor 10
 config = 'mvw';
 gmm1=cell(1,1200);

folder='G:\phd\code and implementation\uncommon words_intelligible\fold 1\train 1200\';
files = dir(strcat(folder,'*.wav'));
for i = 1:length(files)
fname = strcat(folder,files(i,1).name);
[d sr] = audioread(fname);
features3{i} = melcepst(d,sr,'C',13);
fprintf('i= %d\n',i);
a2= cellfun(@transpose,features3,'UniformOutput',false); %feature extraction
end

for i = 1:1200
    trainData{i} = features3{i};
   t2 = cellfun(@transpose,trainData,'UniformOutput',false);%testdata
    %t=testData';
end



trainData=[];
for i = 1:4200
    trainData{i} = features2{i};
   t1 = cellfun(@transpose,trainData,'UniformOutput',false);%testdata
end

for z=1:1200
gmm1{z} = mapAdapt(a2(:,z), gmm3{z}, map_tau, config); %
end

logLikT = zeros(4200,1200);
for i = 1:4200 %no of train examples
fprintf('i= %d\n',i);
for j = 1:1200    %no of sub models
gmm_llk1 = compute_llk(t1{i}, gmm3{j}.mu, gmm3{j}.sigma, gmm3{j}.w(:));
gmm1_llk1 = compute_llk(t1{i}, gmm1{j}.mu, gmm1{j}.sigma, gmm1{j}.w(:));
logLikT(i,j) = mean(gmm1_llk1 - gmm_llk1);
end
end

map_tau = 10; %relavance factor 10
 config = 'mvw';
 gmm1=cell(1,1200);

folder='G:\phd\code and implementation\uncommon words_intelligible\fold 1\train 1200\';
files = dir(strcat(folder,'*.wav'));
for i = 1:length(files)
fname = strcat(folder,files(i,1).name);
[d sr] = audioread(fname);
features3{i} = melcepst(d,sr,'C',13);
fprintf('i= %d\n',i);
a2= cellfun(@transpose,features3,'UniformOutput',false); %feature extraction
end

for i = 1:1200
    trainData{i} = features3{i};
 %trainData{i} = features3{i};
   t2 = cellfun(@transpose,trainData,'UniformOutput',false);%testdata
    %t=testData';
end



trainData=[];
for i = 1:7500
    trainData{i} = features5{i};
   t1 = cellfun(@transpose,trainData,'UniformOutput',false);%testdata
    %t=testData';
end

for z=1:1200
gmm1{z} = mapAdapt(a2(:,z), gmm4{z}, map_tau, config); %
end

logLikT = zeros(7500,1200);
%train=[];
for i = 1:7500 %no of train examples
fprintf('i= %d\n',i);
for j = 1:1200    %no of sub models
gmm_llk1 = compute_llk(t1{i}, gmm4{j}.mu, gmm4{j}.sigma, gmm4{j}.w(:));
gmm1_llk1 = compute_llk(t1{i}, gmm1{j}.mu, gmm1{j}.sigma, gmm1{j}.w(:));
logLikT(i,j) = mean(gmm1_llk1 - gmm_llk1);
%logLikT(i,j) = mean(gmm1_llk1);
end
end

trainxv = zeros(6900,512);
testxv = zeros(2800,512);


logLikT = zeros(100,1200);
%train=[];
for i = 1:55 %no of train examples
fprintf('i= %d\n',i);
for j = 1:40    %no of sub models
gmm_llk1 = compute_llk(t1{i}, gmmClsfull{j}.mu, gmmClsfull{j}.sigma, gmmClsfull{j}.w(:));
gmm1_llk1 = compute_llk(t1{i}, gmm1{j}.mu, gmm1{j}.sigma, gmm1{j}.w(:));
logLikT(i,j) = mean(gmm1_llk1 - gmm_llk1);
%logLikT(i,j) = mean(gmm1_llk1);
end
end

%M = logLikT(5,:);
%plot(:,M(:),'p');
%plot(M(1:3,1:3)');

folder='G:\phd\code and implementation\TEST_11\';
%folder='G:\phd\code and implementation\kaldi_all common3\common_intelligible\fold 2\test_f2\';
files = dir(strcat(folder,'*.wav'));
for i = 1:length(files)
    %fprintf('i \n ',i);
%for i=1:2
fname = strcat(folder,files(i,1).name);
[d sr] = audioread(fname);
%features{i} = melcepst(d,sr,'C',13);
features2{i} = melcepst1(d,sr,'D',26);
% features{i} = melcepst(d,sr);
fprintf('i= %d\n',i);
a11= cellfun(@transpose,features2,'UniformOutput',false); %feature extraction
end

testData=[];
for i = 1:8700
fprintf('i=%d\n',i);
testData{i} = features7{i};
t2 = cellfun(@transpose,testData,'UniformOutput',false);%testdata
%t=testData';
end
logLikTT = zeros(8700,1200);
for i = 1:8700
fprintf('i=%d\n',i);
for j = 1:1200
gmm2_llk1 = compute_llk(t2{i}, gmmClsfull{j}.mu, gmmClsfull{j}.sigma, gmmClsfull{j}.w(:));
gmm3_llk1 = compute_llk(t2{i}, gmm1{j}.mu, gmm1{j}.sigma, gmm1{j}.w(:));
logLikTT(i,j) = mean(gmm3_llk1 - gmm2_llk1);
%logLikTT(i,j) = mean(gmm3_llk1);
end
end

 %trainLabel=[ones(4400,1);2*ones(4200,1);3*ones(4200,1);4*ones(7500,1)];
 %testLabel=[ones(2100,1);2*ones(1800,1);3*ones(1800,1);4*ones(3000,1)];
 %trainLabel=[ones(880,1);2*ones(840,1);3*ones(840,1);4*ones(1500,1)];
 %testLabel=[ones(420,1);2*ones(360,1);3*ones(360,1);4*ones(600,1)];
 trainLabel=[ones(1600,1);2*ones(1400,1);3*ones(1400,1);4*ones(2500,1)];
 testLabel=[ones(600,1);2*ones(600,1);3*ones(600,1);4*ones(1000,1)];
d1 = normalize(d,'range',[0.1,0.8])
trainData=d1;



[bestc, bestg, bestcv] = automaticParameterSelection(trainLabel, trainData,10);
 bestc=0.0019531; 
 bestg=0.00048828;
 bestc=0.0028;%90 0.0028

cmd = ['-t 2 -c ', num2str(bestc), ' -g ', num2str(bestg),' -b 1'];%multiclass svm
numLabels = 4;
model = cell(numLabels,1);
for k=1:numLabels
model{k} = svmtrainLib(double(trainLabel==k), trainData, cmd);
end
numTest = 2800;
prob = zeros(numTest,numLabels); %#ok<NASGU>
testData=normalize(logLikTT,1);

for k=1:numLabels
[x,b,p] = svmpredict(double(testLabel==k), testData, model{k}, '-b 1');
prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
end
[~,pred] = max(prob,[],2);
acc = sum(pred == testLabel) ./ numel(testLabel) * 100 ;  %# accuracy
C = confusionmat(testLabel, pred)

