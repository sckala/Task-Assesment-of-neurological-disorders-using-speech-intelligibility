clear all
clc
folder='G:\phd\code and implementation\digits_intelligible\train_inte1\';
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
a1= cellfun(@transpose,features4,'UniformOutput',false); %feature extraction
end

%load('trainfeaturesfor100-words20-dimensionaldateapril13''19.mat');
ncha=2;
   nWorkers=1;
%  nmix = 4;           % In this case, we know the # of mixtures needed
 %nmix=2;%75
 nmix=512;
 %nmix=256;
final_niter = 10;
ds_factor = 1;
%ubm=cell(1,numCls);
%gmm=cell(1,4);
%featCls=0;
%for i=1:670
 % fprintf('i= %d\n',i);
%traindata{i}=featCls{j}';
%gmm{i} = gmm_em(mat2cell(a{1,i}(:)), nmix, final_niter, ds_factor, nWorkers);%scene-specific GMM
ubm = gmm_em(a1(:), nmix, final_niter, ds_factor, nWorkers);%creation of UBM For all features
 %gmm{i} = gmm_em(a(:,i), nmix, final_niter, ds_factor, nWorkers);%class specific GMM
%end
map_tau = 7; %relavance factor 10
 config = 'mvw';

for z=1:140
gmm1{z} = mapAdapt(a1(:,z), ubm, map_tau, config); %
end
for z=1:140
gmm3{z} = mapAdapt(a1(:,z+140), ubm, map_tau, config); %
end
for z=1:140
gmm4{z} = mapAdapt(a1(:,z+280), ubm, map_tau, config); %
end
for z=1:250
gmm5{z} = mapAdapt(a1(:,z+420), ubm, map_tau, config); %
end



gmmClsfull = { gmm1{:}, gmm3{:}, gmm4{:}, gmm5{:}};

folder='G:\phd\code and implementation\digits_intelligible\test_inte1\';
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
a3= cellfun(@transpose,features2,'UniformOutput',false); %feature extraction
end
for i = 1:300
    for j = 1:4
         %testData{i} = features{i+80};
        logllk = compute_llk(a3{i},gmmClsfull{j}.mu,gmmClsfull{j}.sigma,gmmClsfull{j}.w(:));
        logLik(i,j) = mean(logllk);
    end
 end
 for i = 1:300
 [llkVal(i), llkLabel(i)]=max(logLik(i,:),[],2);
 end
actualLabel=[ones(80,1);2*ones(60,1);3*ones(60,1);4*ones(100,1)];
C=confusionmat(actualLabel,llkLabel)
Accuracy =mean(actualLabel==llkLabel)*100;