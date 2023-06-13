clc;
clear 
close all

%% GRU预测
tic
load origin_data.mat
load rlmd_data.mat

disp('…………………………………………………………………………………………………………………………')
disp('单一的GRU预测')
disp('…………………………………………………………………………………………………………………………')

num_samples = length(X);       % 样本个数 
kim = 5;                      % 延时步长（kim个历史数据作为自变量）
zim =  1;                      % 跨zim个时间点进行预测
or_dim = size(X,2);

%  重构数据集
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X(i: i + kim - 1,:), 1, kim*or_dim), X(i + kim + zim - 1,:)];
end


% 训练集和测试集划分
outdim = 1;                                  % 最后一列为输出
num_size = 0.85;                              % 训练集占数据集比例
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度


P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%  格式转换
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);

    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end

%  创建GRU网络，
layers = [ ...
    sequenceInputLayer(f_)              % 输入层
    gruLayer(70)                      % gru层
    reluLayer                           % Relu激活层
    fullyConnectedLayer(outdim)         % 回归层
    regressionLayer];

%  参数设置 
options = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', 70, ...                            % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', 0.01, ...         % 初始学习率
    'LearnRateSchedule', 'piecewise', ...             % 学习率调整
    'LearnRateDropPeriod', 60, ...                   % 训练850次后开始调整学习率
    'LearnRateDropFactor',0.2, ...                    % 学习率调整因子
    'L2Regularization', 0.01, ...         % 正则化参数
    'ExecutionEnvironment', 'gpu',...                 % 训练环境
    'Verbose', 0, ...                                 % 关闭优化过程
    'Plots', 'training-progress');                    % 画出曲线

%  训练
net = trainNetwork(vp_train, vt_train, layers, options);
%analyzeNetwork(net);% 查看网络结构
%  预测
t_sim1 = predict(net, vp_train); 
t_sim2 = predict(net, vp_test); 

%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_train1 = T_train;
T_test2 = T_test;

%  数据格式转换
T_sim1 = cell2mat(T_sim1);% cell2mat将cell元胞数组转换为普通数组
T_sim2 = cell2mat(T_sim2);

% 指标计算
disp('训练集误差指标')
[mae1,rmse1,mape1,error1]=calc_error(T_train1,T_sim1');
fprintf('\n')

disp('测试集误差指标')
[mae2,rmse2,mape2,error2]=calc_error(T_test2,T_sim2');
fprintf('\n')
toc


tic
disp('…………………………………………………………………………………………………………………………')
disp('RLMD-GRU预测')
disp('…………………………………………………………………………………………………………………………')

imf=u;
c=size(imf,1);
%% 对每个分量建模
for d=1:c
disp(['第',num2str(d),'个分量建模'])

X_imf=[X(:,1:end-1) imf(d,:)'];
num_samples = length(X_imf);  % 样本个数 

%  重构数据集
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X_imf(i: i + kim - 1,:), 1, kim*or_dim), X_imf(i + kim + zim - 1,:)];
end


% 训练集和测试集划分
outdim = 1;                                  % 最后一列为输出
num_size = 0.85;                              % 训练集占数据集比例
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度


P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';


P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';


%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%  格式转换
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);
    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end

%  创建GRU网络，
layers = [ ...
    sequenceInputLayer(f_)              % 输入层
    gruLayer(70)                      % gru层
    reluLayer                           % Relu激活层
    fullyConnectedLayer(outdim)         % 回归层
    regressionLayer];

%  参数设置
options = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', 70, ...                            % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', 0.01, ...         % 初始学习率
    'LearnRateSchedule', 'piecewise', ...             % 学习率调整
    'LearnRateDropPeriod', 60, ...                   % 训练850次后开始调整学习率
    'LearnRateDropFactor',0.2, ...                    % 学习率调整因子
    'L2Regularization', 0.01, ...         % 正则化参数
    'ExecutionEnvironment', 'gpu',...                 % 训练环境
    'Verbose', 0, ...                                 % 关闭优化过程
    'Plots', 'training-progress');                    % 画出曲线

%  训练
net = trainNetwork(vp_train, vt_train, layers, options);
%analyzeNetwork(net);% 查看网络结构
%  预测
t_sim5 = predict(net, vp_train); 
t_sim6 = predict(net, vp_test); 

%  数据反归一化
T_sim5_imf = mapminmax('reverse', t_sim5, ps_output);
T_sim6_imf = mapminmax('reverse', t_sim6, ps_output);

%  数据格式转换
T_sim5(d,:) = cell2mat(T_sim5_imf);% cell2mat将cell元胞数组转换为普通数组
T_sim6(d,:) = cell2mat(T_sim6_imf);
T_train5(d,:)= T_train;
T_test6(d,:)= T_test;
end

% 各分量预测的结果相加
T_sim5=sum(T_sim5);
T_sim6=sum(T_sim6);
T_train5=sum(T_train5);
T_test6=sum(T_test6);

% 指标计算
disp('训练集误差指标')
[mae5,rmse5,mape5,error5]=calc_error(T_train5,T_sim5);
fprintf('\n')

disp('测试集误差指标')
[mae6,rmse6,mape6,error6]=calc_error(T_test6,T_sim6);
fprintf('\n')
toc

%% RLMD-SMA-GRU预测
tic
disp('…………………………………………………………………………………………………………………………')
disp('RLMD-SMA-GRU预测')
disp('…………………………………………………………………………………………………………………………')

% SMA参数
pop=3; % 种群数量
Max_iter=5; % 最大迭代次数
dim=3; % 优化GRU的3个参数
lb = [50,50,0.001];%下边界
ub = [300,300,0.01];%上边界
numFeatures=f_;
numResponses=outdim;
fobj = @(x) fun(x,numFeatures,numResponses,X);
[Best_pos,Best_score,curve,BestNet]=SSA(pop,Max_iter,lb,ub,dim,fobj);

% 绘制进化曲线
figure
plot(curve,'r-','linewidth',3)
xlabel('进化代数')
ylabel('均方根误差RMSE')
legend('最佳适应度')
disp('')
disp(['最优隐藏单元数目为   ',num2str(round(Best_pos(1)))]);
disp(['最优最大训练周期为   ',num2str(round(Best_pos(2)))]);
disp(['最优初始学习率为   ',num2str((Best_pos(3)))]);

%% 对每个分量建模
for d=1:c
disp(['第',num2str(d),'个分量建模'])

X_imf=[X(:,1:end-1) imf(d,:)'];

%  重构数据集
for i = 1: num_samples - kim - zim + 1
    res(i, :) = [reshape(X_imf(i: i + kim - 1,:), 1, kim*or_dim), X_imf(i + kim + zim - 1,:)];
end


% 训练集和测试集划分
outdim = 1;                                  % 最后一列为输出
num_size = 0.85;                              % 训练集占数据集比例
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度


P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%  格式转换
for i = 1 : M 
    vp_train{i, 1} = p_train(:, i);
    vt_train{i, 1} = t_train(:, i);
end

for i = 1 : N 
    vp_test{i, 1} = p_test(:, i);
    vt_test{i, 1} = t_test(:, i);
end

% 最佳参数的GRU预测
layers = [ ...
    sequenceInputLayer(f_)              % 输入层
    gruLayer(round(Best_pos(1)))                      % gru层
    reluLayer                           % Relu激活层
    fullyConnectedLayer(outdim)         % 回归层
    regressionLayer];


options = trainingOptions('adam', ...                 % 优化算法Adam
    'MaxEpochs', round(Best_pos(2)), ...                            % 最大训练次数
    'GradientThreshold', 1, ...                       % 梯度阈值
    'InitialLearnRate', Best_pos(3), ...         % 初始学习率
    'LearnRateSchedule', 'piecewise', ...             % 学习率调整
    'LearnRateDropPeriod', round(Best_pos(2)*0.9), ...                   % 训练850次后开始调整学习率
    'LearnRateDropFactor',0.2, ...                    % 学习率调整因子
    'L2Regularization', 0.001, ...         % 正则化参数
    'ExecutionEnvironment', 'gpu',...                 % 训练环境
    'Verbose', 0, ...                                 % 关闭优化过程
    'Plots', 'training-progress');                    % 画出曲线

%  训练
net = trainNetwork(vp_train, vt_train, layers, options);
%analyzeNetwork(net);% 查看网络结构
%  预测
t_sim7 = predict(net, vp_train); 
t_sim8 = predict(net, vp_test); 

%  数据反归一化
T_sim7_imf = mapminmax('reverse', t_sim7, ps_output);
T_sim8_imf = mapminmax('reverse', t_sim8, ps_output);

%  数据格式转换
T_sim7(d,:) = cell2mat(T_sim7_imf);% cell2mat将cell元胞数组转换为普通数组
T_sim8(d,:) = cell2mat(T_sim8_imf);
T_train7(d,:)= T_train;
T_test8(d,:)= T_test;
end

% 各分量预测的结果相加
T_sim7=sum(T_sim7);
T_sim8=sum(T_sim8);
T_train7=sum(T_train7);
T_test8=sum(T_test8);

% 指标计算
disp('训练集误差指标')
[mae7,rmse7,mape7,error7]=calc_error(T_train7,T_sim7);
fprintf('\n')

disp('测试集误差指标')
[mae8,rmse8,mape8,error8]=calc_error(T_test8,T_sim8);
fprintf('\n')
toc

%%模型测试集结果绘图对比

figure
plot(T_test2,'k','linewidth',3);
hold on;
plot(T_sim2,'y','linewidth',3);
hold on;
plot(T_sim6,'g','linewidth',3);
hold on;
plot(T_sim8,'r','linewidth',3);
legend('Target','GRU','RLMD-GRU','RLMD-SMA-GRU');
title('三种模型预测结果对比图');
xlabel('Sample Index');
ylabel('Values');
grid on;

figure
plot(error2,'k','linewidth',3);
hold on
plot(error6,'g','linewidth',3);
hold on
plot(error8,'r','linewidth',3);
legend('GRU','RLMD-GRU','RLMD-SMA-GRU');
title('三种模型预测误差对比图');
grid on;