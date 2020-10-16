%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This is an examplar file on how the KELM-AE [1] program could be used.
%
% [1] CHENG Yu-sheng, ZHAO Da-wei, WANG Yi-bin, PEI Geng-shen.
%     Multi-label Learning with Kernel Extreme Learning Machine AutoEncoder.
%     Knowledge-Based Systems(Accepted).
%
% Please feel free to contact me (zhaodwahu@163.com), if you have any problem about this programme.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc
%%load data
addpath(genpath('.'));
load('emotion.mat');
%% set parameter
s=1;%Suggest set to 1. Smoothing parameter.
alpha=0.4;%Suggest set to [0.1-0.5]. Non-equilibrium parameter.
kernel_type='RBF_kernel';%Suggest set to 'RBF_kernel'. This is a kernel type.

%% the non-equilibrium label completion matrix construction
Conf= NeLC(train_target,alpha,s);
%First ELM module
Ytrain=(train_target'*Conf);
[num_class,num_testing] = size(test_target);
Xtrain=[train_data,sum(Ytrain,2)/num_class];

C1=1;%Suggest set to 1. This is a regularization parameter of first ELM module.
kernel_para1=20;%Suggest set to 1. This is a kernel parameter of first ELM module.
[X,TX] = felm_kernel(test_data,Xtrain,train_data,C1, kernel_type,kernel_para1);

Xtrain_data=X;
Xtest_data=TX;
%Second ELM module
C2 = 1;             %Suggest set to 1. This is a regularization parameter of second ELM module.
kernel_para2 = 1.5; %Suggest set to 1. This is a kernel parameter of second ELM module.
[Y,TY] = selm_kernel(Xtest_data,Xtrain_data,train_target,C2, kernel_type,kernel_para2,Conf);

%prediction
Outputs = TY';
Pre_Labels=sign(Outputs);
ret.HL=Hamming_loss(Pre_Labels,test_target);
ret.RL=Ranking_loss(Outputs,test_target);
ret.OE=One_error(Outputs,test_target);
ret.CV=coverage(Outputs,test_target);
ret.AP=Average_precision(Outputs,test_target);


