global X;
global Y;
global lamda;
global mu;

mu = 1;
lamda = 0.02;
acc = 1;
ns = 0;
traindata = csvread('tra.csv');
siz = size(traindata);
X = traindata(:,1:siz(2)-1)';
Y = traindata(:,siz(2));
[R,IA,IC] = unique(X','rows');
X = R';
Y = Y(IA,:);
Y = (Y==5);
Y=Y*2-1;
if ns == 1
    X = (X-8)/16;
end
tic
[xx,loss,ite] = lr(acc);
disp(['Solved in ',num2str(toc),' seconds, ',num2str(ite),' iterations.' ]);
w = xx(:,size(xx,2));
% disp('The normal vector of the decision boundary is:');
% disp(xx(:,size(xx,2))');
disp(['Objective: ',num2str(loss(size(loss,2)))]);

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
for j = 1:sizt(1)
    pret = [pret;func_pred(Xt(:,j))];
end
binloss = (sign(pret-0.5)~=Yt);
binlosssum = sum(binloss);
accuracy = (1-binlosssum/sizt(1))*100;
disp(['Prediction binary loss: ',num2str(binlosssum)]);
disp(['Prediction accuracy: ',num2str(accuracy), '%']);