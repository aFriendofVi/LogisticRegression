function loss = computeloss(w)
global X;
global Y;
global lamda;
p = log(exp(-bsxfun(@times,Y,(X'*w)))+1);
loss = mean(p)+lamda*([0;w(2:end,:)]' * [0;w(2:end,:)]);
end

