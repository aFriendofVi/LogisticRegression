function qp = prox(ww,wtmp,mut)

qp = computeloss(ww)+gradient(ww)'*(wtmp-ww)+(0.5/mut)*norm(wtmp-ww)^2;
end