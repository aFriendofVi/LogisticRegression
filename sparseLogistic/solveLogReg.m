function [w,trainLoss,ite,w_rec] = solveLogReg(X,Y, initialMu, lambda,acc, Tol)
if nargin <6
    Tol = 1e-10;
    if nargin <5
        acc =1;
    end
end
%n dimension feature, m training samples
[n,~] = size(X);
%initialize all-zeors w
w = zeros(n,1);
w_rec = [w,w-initialMu*gradient(X,Y,w)];
u = w;
u_rec = u;
j_rec = computeLoss(X,Y,w,lambda);
itemax = 2000;
t_rec = 0;
t = 1;
mut = initialMu;
recover = 1.2;
discount = 0.1;
for ite = 1:itemax
    mut = mut*recover; 
    grad = gradient(X,Y,w);
    subgrad = grad;
    subgrad(any(w,2)) = subgrad(any(w,2)) + lambda*sign(grad(any(w,2)));
    subgrad(~any(w,2)) = sign(grad(~any(w,2))).*max(abs(grad(~any(w,2)))-lambda,0);
    %the intercept term is not regularized and we dont want it to be sparse
    subgrad(1,:) = grad(1,:);
    if acc == 0
        wtmp = w-mut*subgrad;
        while computeLoss(X,Y,wtmp,lambda) > prox(X,Y,w,wtmp,mut,lambda)
            mut = mut*discount;
            wtmp = w-mut*subgrad;
        end
    else
        k = size(t_rec,2);
        t = (1+(1+4*t_rec(k))^(0.5))/2;
        
        gamma = (1-t_rec(k))/t;
        u = (1-gamma)*w_rec(:,k+1)+gamma*w_rec(:,k);
        gradu = gradient(X,Y,u);
        subgradu = gradu;
        subgradu(any(u,2)) = subgradu(any(u,2)) + lambda*sign(gradu(any(u,2)));
        subgradu(~any(u,2)) = sign(gradu(~any(u,2))).*max(abs(gradu(~any(u,2)))-lambda,0);
        wtmp = u-mut*subgradu;
        while computeLoss(X,Y,wtmp,lambda) > prox(X,Y,u,wtmp,mut,lambda)
            mut = mut*discount;
            wtmp = u-mut*subgradu;
%             disp(computeLoss(X,Y,wtmp,lambda) > prox(X,Y,u,wtmp,mut,lambda));
        end
    end
    w = wtmp;
    w_rec = cat(2,w_rec,w);
    u_rec = cat(2,u_rec,u);
    t_rec = cat(2,t_rec,t);
    j_rec = cat(2,j_rec,computeLoss(X,Y,w,lambda));
    
    if norm(gradient(X,Y,w),inf) <= lambda*(1+Tol)
        disp(norm(gradient(X,Y,w),inf));
        break;
    end
end
trainLoss = computeLoss(X,Y,w,lambda);
plot(1:size(j_rec,2),j_rec);
disp(size(j_rec))
end
        