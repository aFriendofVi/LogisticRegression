function [w_rec,j_rec,ite] = lr(acc,tol)
global X;
global Y;
global mu;
global lamda;
if nargin <2
    tol = 1e-3;
end
siz = size(X);
n = siz(1);
m = siz(2);
X = cat(1,ones(1,m),X);
w = zeros(n+1,1);
w_rec = cat(2,w,w-mu*gradient(w));
u = w;
u_rec = u;
j_rec = computeloss(w);
itemax = 2000;
t_rec = 0;
mu_rec = [mu; mu];
mut = mu;
recover = 1.1;
dis = 0.1;
ite = 0 ;
for i= 1:itemax
    grad = gradient(w);
    mut = mut*recover;
    if acc == 0
        wtmp = w-mut*grad;
        imp = 0;
        while computeloss(wtmp) > prox(w,wtmp,mut)
            imp = imp+1;
            mut = mut*dis;
            wtmp = w-mut*grad;
        end
%         disp(imp);
        w = wtmp;
        w_rec = cat(2,w_rec, w);
    else
        k = size(t_rec,2);
        t = (1+(1+4*t_rec(k))^(0.5))/2;
        
        gamma = (1-t_rec(k))/t;
        u = (1-gamma)*w_rec(:,k+1)+gamma*w_rec(:,k);

        wtmp = u-mut*gradient(u);
        while computeloss(wtmp) > prox(u,wtmp,mut)
            mut = mut*dis;
            wtmp = u-mut*gradient(u);
        end
        w = wtmp;
        w_rec = cat(2,w_rec,w);
        u_rec = cat(2,u_rec,u);
        t_rec = cat(2,t_rec,t);
        
        
    end
    
    j_rec = cat(2,j_rec, computeloss(w));
    if norm(grad) <= tol
        disp(['Lamda: ',num2str(lamda)]);
        disp(['Mu: ', num2str(mut)]);
        ite = i;
%         disp(['Iterations: ', num2str(i)]);
        break;
    end
    if i == itemax
        disp(['Lamda: ',num2str(lamda)]);
        disp(['Mu: ', num2str(mut)]);
        ite = itemax;
%         disp(['Iterations: ', num2str(i)]);
    end
end

    
    train_pred = 1./(1+exp(-X'*w));
    train_pred(train_pred>=0.5) = 1;
    train_pred(train_pred< 0.5) = -1;
    train_loss = sum(train_pred ~= Y);
%     disp('training loss is:');
%     disp(train_loss);
    
griddot = 100;
if n == 2
    stp1 = (0.1+range(X(2,:)))/griddot;
    stp2 = (0.1+range(X(3,:)))/griddot;
    x1 = min(min(X))-0.1*griddot*stp1:stp1:max(max(X))+0.1*griddot*stp1;
    x2 = min(min(X))-0.1*griddot*stp2:stp2:max(max(X))+0.1*griddot*stp2;
    [x1,x2] = meshgrid(x1,x2);
    predict = @(x_,y_) 1/(1+exp(-[1;x_;y_]'*w));
    z = arrayfun(predict,x1,x2);
    figure();subplot(2,2,1); hold on;
    contour(x1,x2,z,[0.5 0.5],'k-');
    xpt1 = X(2,:)';
    xp1 = xpt1(Y==1);
    xpt2 = X(3,:)';
    xp2 = xpt2(Y==1);
    xnt1 = X(2,:)';
    xn1 = xnt1(Y==-1);
    xnt2 = X(3,:)';
    xn2 = xnt2(Y==-1);
    scatter(xp1,xp2,16,[0.4 0.3 0.6],'filled')
    scatter(xn1,xn2,16,[1 0.6 0.2],'filled')
    title('Decision boundary');
    xlabel('w1');
    ylabel('w2');
    subplot(2,2,2);
    line(1:size(j_rec,2),j_rec);
    title('Loss function value');
    xlabel('iteration');
    ylabel('loss');
% else
% %     disp('Dimension is not 2');
%     figure();
%     line(1:size(j_rec,2),j_rec);
%     title('Loss function value');
%     xlabel('iteration');
%     ylabel('loss');
end
end