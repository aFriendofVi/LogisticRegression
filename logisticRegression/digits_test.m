clear;clc;
global X;
global Y;
global lamda;
global mu;
print = 1;
ns = 0;
traindata = csvread('tra.csv');
siz = size(traindata);
X = traindata(:,1:siz(2)-1)';
Y = traindata(:,siz(2));
[R,IA,IC] = unique(X','rows');
X = R';
if ns == 1
    X = (X-8)/16;
end
Y = Y(IA,:);
Y = (Y==5);
Y=Y*2-1;
mulist = [1 2 3 4 5];
lamdalist = [0.02 0.2 2 6];
aa = size(mulist,2);
bb = size(lamdalist,2);
im = zeros(aa,bb);
rec = {{[im],[im],[im]},{[im],[im],[im]}};
count = 0;
for acc = 0:1
    for i = 1:size(mulist,2)
        for j = 1:size(lamdalist,2)
            mu = mulist(i);
            lamda = lamdalist(j);
            tic
            [xx,loss,ite] = lr(acc);
            rec{acc+1}{1}(i,j) = toc;
            rec{acc+1}{2}(i,j) = ite;
            w = xx(:,size(xx,2));         
            testdata = csvread('tes.csv');
            sizt = size(testdata);
            Xt = testdata(:, 1:sizt(2)-1)';
            if ns == 1
                Xt = (Xt-8)/16;
            end
            Yt = testdata(:,sizt(2));
            Yt = (Yt==5);
            Yt=Yt*2-1;
            func_pred = @(x) 1/(exp((-[1;x]'*w))+1);
            pret = [];
            for l = 1:sizt(1)
                pret = cat(1,pret,func_pred(Xt(:,l)));
            end
            binloss = (sign(pret-0.5)~=Yt);
            binlosssum = sum(binloss);
            hingeloss = sum(log(exp(-Y.*(X'*w))+1));
            accuracy = (1-binlosssum/sizt(1))*100;
            rec{acc+1}{3}(i,j) = accuracy;
            X = X(2:end,:);
            count = count+1;
            clc;
            disp(['Completed: ',num2str(50*count/(size(mulist,2)*size(lamdalist,2))),'%']);
            esttime = 2*size(mulist,2)*size(lamdalist,2)*sum(sum(rec{1}{1}+rec{2}{1}))/count;
            usedtime = sum(sum(rec{1}{1}+rec{2}{1}));
            disp(['Estimated Time left: ',num2str(esttime-usedtime),' seconds.']);
        end
    end
end
if print ==1
    table1 = [0 lamdalist;[mulist' rec{1}{1}]];
    table2 = [0 lamdalist;[mulist' rec{1}{2}]];
    table3 = [0 lamdalist;[mulist' rec{1}{3}]];
    table4 = [0 lamdalist;[mulist' rec{2}{1}]];
    table5 = [0 lamdalist;[mulist' rec{2}{2}]];
    table6 = [0 lamdalist;[mulist' rec{2}{3}]];
    dt = datestr(now,'yyyy-mm-dd_HH-MM-SS');
    filename = ['BGD_Statistics',dt,'.xlsx'];
    disp('Time Consumed without Acceleration');
    disp(table1);
    disp('Iteration without Acceleration');
    disp(table2);
    disp('Accuracy without Acceleration');
    disp(table3);
    writetable(table(table1),filename,'Sheet',1,'WriteVariableNames',false);
    writetable(table(table2),filename,'Sheet',2,'WriteVariableNames',false);
    writetable(table(table3),filename,'Sheet',3,'WriteVariableNames',false);
    filename = ['AGD_Statistics',dt,'.xlsx'];
    disp('Time Consumed with Acceleration');
    disp(table4);
    disp('Iteration with Acceleration');
    disp(table5);
    disp('Accuracy with Acceleration');
    disp(table6);
    writetable(table(table4),filename,'Sheet',1,'WriteVariableNames',false);
    writetable(table(table5),filename,'Sheet',2,'WriteVariableNames',false);
    writetable(table(table6),filename,'Sheet',3,'WriteVariableNames',false);
end
