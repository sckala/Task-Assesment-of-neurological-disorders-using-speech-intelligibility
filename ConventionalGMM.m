
clear all
clc
folder='G:\phd\code and implementation\digits_intelligible\train_inte1\';
files = dir(strcat(folder,'*.wav'));
for i = 1:length(files)
    %fprintf('i \n ',i);
%for i=1:2
fname = strcat(folder,files(i,1).name);
[d sr] = audioread(fname);
features{i} = melcepst(d,sr,'C',13);
%features4{i} = melcepst1(d,sr,'D',26);
% features{i} = melcepst(d,sr);
fprintf('i= %d\n',i);
a1= cellfun(@transpose,features,'UniformOutput',false); %feature extraction
end

%load('trainfeaturesfor100-words20-dimensionaldateapril13''19.mat');
ncha=2;
   nWorkers=1;
%  nmix = 4;           % In this case, we know the # of mixtures needed
 %nmix=2;%75
 nmix=32;
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
%gmm{i} = gmm_em(a1(:,i), nmix, final_niter, ds_factor, nWorkers);%creation of UBM For all features
 %gmm{i} = gmm_em(a(:,i), nmix, final_niter, ds_factor, nWorkers);%class specific GMM
%end

 for i=1:140
%     %traindata{i}=featCls{i}';
%gmm{i} = gmm_em(mat2cell(a{1,i}(:)), nmix, final_niter, ds_factor, nWorkers);
 gmm1{i} = gmm_em(a1(:,i), nmix, final_niter, ds_factor, nWorkers);
 end
 for i=1:140
%     %traindata{i}=featCls{i}';
%gmm{i} = gmm_em(mat2cell(a{1,i}(:)), nmix, final_niter, ds_factor, nWorkers);
 gmm2{i} = gmm_em(a1(:,i+140), nmix, final_niter, ds_factor, nWorkers);
 end
 for i = 1:140
%     %traindata{i}=featCls{i}';
%gmm{i} = gmm_em(mat2cell(a{1,i}(:)), nmix, final_niter, ds_factor, nWorkers);
 gmm3{i} = gmm_em(a1(:,i+280), nmix, final_niter, ds_factor, nWorkers);
 end
for i = 1:250
%     %traindata{i}=featCls{i}';
%gmm{i} = gmm_em(mat2cell(a{1,i}(:)), nmix, final_niter, ds_factor, nWorkers);
 gmm4{i} = gmm_em(a1(:,i+420), nmix, final_niter, ds_factor, nWorkers);
 end



gmmClsfull = { gmm1{:}, gmm2{:}, gmm3{:}, gmm4{:}};

folder='G:\phd\code and implementation\digits_intelligible\test_inte1\';
files = dir(strcat(folder,'*.wav'));
for i = 1:length(files)
    %fprintf('i \n ',i);
%for i=1:2
fname = strcat(folder,files(i,1).name);
[d sr] = audioread(fname);
features2{i} = melcepst(d,sr,'C',13);
%features2{i} = melcepst1(d,sr,'D',26);
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
llk=llkVal';