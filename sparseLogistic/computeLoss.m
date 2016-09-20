function loss = computeLoss(X,Y,w,lambda)
p = log(exp(-bsxfun(@times,Y,(X'*w)))+1);
loss = mean(p)+lambda*sum(abs([w(2:end,:)]));
end

