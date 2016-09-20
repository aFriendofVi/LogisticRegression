function qp = prox(X,Y,ww,wtmp,mut,lambda)

qp = computeLoss(X,Y,ww,lambda)+gradient(X,Y,ww)'*(wtmp-ww)+(0.5/mut)*norm(wtmp-ww)^2+lambda*sum(abs([ww(2:end,:)]));
end