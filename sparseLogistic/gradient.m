function grd = gradient(X,Y,ww)
[n,~] = size(X);
YX = bsxfun(@times,Y',X);
grd = (1/n)*(-YX)*(1-1./(exp(-YX'*ww)+1))/size(Y,1);%+lambda*([0;ww(2:end,:)]<0);
end