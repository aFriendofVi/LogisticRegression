clear;clc;
%print the result statistic or not
print = 1;
%normalize data or not
ns = 0;
%load data
traindata = csvread('tra.csv');
siz = size(traindata);
X = traindata(:,1:siz(2)-1)';
Y = traindata(:,siz(2));
%only take unique columns of X (it is a big deal for SVM
[R,IA,IC] = unique(X','rows');
X = R';
%normalize data if needed
if ns == 1
    X = (X-8)/16;
end
%process data
Y = Y(IA,:);
Y = (Y==5);
Y=Y*2-1;
[n,m] = size(X);
X = [ones(1,m);X];
size(X)
testdata = csvread('tes.csv');
[mt,nt] = size(testdata);
Xt = [ones(1,mt);testdata(:, 1:nt-1)'];
if ns == 1
    Xt = (Xt-8)/16;
end
Yt = testdata(:,nt);
Yt = (Yt==5);
Yt=Yt*2-1;
mu = 4;
%list the initial Tolerence(s) and lambda(s) you want here
tollist = [1e-2 1e-3 1e-4 1e-5 1e-10];
lambdalist = [0.001 0.002 0.004 0.008];
%initial cells to store the statistics for the grid search
munum = size(tollist,2);
lamnum = size(lambdalist,2);
im = zeros(munum,lamnum);
rec = {{[im],[im],[im]},{[im],[im],[im]}};
%grid search starts here
count = 0;
for acc = 0:1
    for i = 1:size(tollist,2)
        for j = 1:size(lambdalist,2)
            tol = tollist(i);
            lambda = lambdalist(j);
            tic
            [w,loss,ite,~] = solveLogReg(X,Y,mu,lambda,acc,tol);
            rec{acc+1}{1}(i,j) = toc;
            rec{acc+1}{2}(i,j) = ite;  
            pret = [];
            func_pred = @(x) 1/(exp(-x'*w)+1);
            for l = 1:mt
                pret = cat(1,pret,func_pred(Xt(:,l)));
            end
            binloss = (sign(pret-0.5)~=Yt);
            binlosssum = sum(binloss);
            hingeloss = sum(log(exp(-Y.*(X'*w))+1));
            accuracy = (1-binlosssum/mt)*100;
            rec{acc+1}{3}(i,j) = accuracy;
            count = count+1;
            clc;
            disp(['Completed: ',num2str(50*count/(size(tollist,2)*size(lambdalist,2))),'%']);
            esttime = 2*size(tollist,2)*size(lambdalist,2)*sum(sum(rec{1}{1}+rec{2}{1}))/count;
            usedtime = sum(sum(rec{1}{1}+rec{2}{1}));
            disp(['Estimated Time left: ',num2str(esttime-usedtime),' seconds.']);
        end
    end
end
    logtol = log10(tollist);
    table1 = [0 lambdalist;[logtol' rec{1}{1}]];
    table2 = [0 lambdalist;[logtol' rec{1}{2}]];
    table3 = [0 lambdalist;[logtol' rec{1}{3}]];
    table4 = [0 lambdalist;[logtol' rec{2}{1}]];
    table5 = [0 lambdalist;[logtol' rec{2}{2}]];
    table6 = [0 lambdalist;[logtol' rec{2}{3}]];    
    dt = datestr(now,'yyyy-mm-dd_HH-MM-SS');
    
    disp('Time Consumed without Acceleration');
    disp(table1);
    disp('Iteration without Acceleration');
    disp(table2);
    disp('Accuracy without Acceleration');
    disp(table3);
    
    disp('Time Consumed with Acceleration');
    disp(table4);
    disp('Iteration with Acceleration');
    disp(table5);
    disp('Accuracy with Acceleration');
    disp(table6);
if print ==1
    filename = ['BGD_Statistics',dt,'.xlsx'];
    writetable(table(table1),filename,'Sheet',1,'WriteVariableNames',false);
    writetable(table(table2),filename,'Sheet',2,'WriteVariableNames',false);
    writetable(table(table3),filename,'Sheet',3,'WriteVariableNames',false);
    filename = ['AGD_Statistics',dt,'.xlsx'];
    writetable(table(table4),filename,'Sheet',1,'WriteVariableNames',false);
    writetable(table(table5),filename,'Sheet',2,'WriteVariableNames',false);
    writetable(table(table6),filename,'Sheet',3,'WriteVariableNames',false);
end
