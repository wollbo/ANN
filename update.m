function [] = update(W,V,eta,alpha,delta)

  dw = (dw .* alpha) - (delta_h * pat) .* (1-alpha);
  dv = (dv .* alpha) - (delta_o * hout) .* (1-alpha);
  W = w + dw .* eta;
  V = v + dv .* eta;

end

