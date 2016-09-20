function grd = gradient(ww)
global X;
global Y;
global lamda;
YX = bsxfun(@times,Y',X);
grd = (-YX)*(1-1./(exp(-YX'*ww)+1))/size(Y,1)+2*lamda*[ww(1:end,:)];
end