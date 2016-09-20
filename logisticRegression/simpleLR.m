global X;
global Y;
global lamda;
global mu;
mu = 1;
lamda = 0.02;
X =[-15.6738    1.4885
    -15.0079   -2.6139
    -14.3576   -5.5521
    -13.9617    0.9660
    -13.1034    2.6069
    -12.4349    2.7076
    -11.7355   -0.2656
    -11.4862    0.0170
    -11.4615    1.3360
    -10.8039    2.5671
    -10.3526    1.3873
    -10.2719   -0.6463
    -10.1338    3.1736
    -10.1126    1.6368
    -9.4429    0.7760
    -9.0523   -0.1393
    -7.2429    0.5406
    -0.8606   14.6276
    -0.7727   10.0132
    2.1016   13.2808
    2.4757    9.8281
    3.4253    7.1400
    3.5665   13.7207
    3.8459   10.3861
    4.0317    4.2259
    4.1311   10.2018
    4.6003   12.1438
    6.6906   10.7475
    7.5823   12.4356
    9.3363    4.8559]';
Y =[ -1    -1    -1    -1    -1    -1    -1    -1    -1    -1    -1    -1    -1    -1    -1    -1    -1     1     1     1     1     1     1     1   1     1     1     1     1     1]';

% A = generate(2,18,2);
% X = A(1:2,:);
% Y = A(3, :)';
% [R,IA,IC] = unique(X','rows');
% X = R';
% Y = Y(IA,:);
tic
[xx,loss,ite1] = lr(0);
t1 = toc;
X = X(2:end,:);
tic
[~,~,ite2] = lr(1);
t2 = toc;
w = xx(1:end,size(xx,2));
disp(['Objective: ',num2str(loss(size(loss,2)))]);
disp('Decision boundary coefficient:');
disp(w');
disp(['Normal Gradient Descent solved in ',num2str(t1),' seconds, ',num2str(ite1),' iterations']);
disp(['Accelerated Gradient Descent solved in ',num2str(t2),' seconds, ',num2str(ite2),' iterations']);