%
%3.1 Test
clear all
close all

x_input = 0:0.1:pi;
f_sin = sin(x_input);
f_square = sign(f_sin);

n_nodes = 100;
epochs = 100;
eta = 0.005;

w_r = 0.1 *randn(1,n_nodes)';
w_s = 0.1 *randn(1,n_nodes)';

% mu_r = zeros(1,n_nodes)';
% mu_s = zeros(1,n_nodes)';
mu_r = linspace(0,pi,n_nodes)'; 
mu_s = linspace(0,pi,n_nodes)';

sigma_r = ones(1,n_nodes)';
sigma_s = ones(1,n_nodes)';

rbf_nodes_r = exp(-(x_input-mu_r).^2./(2*sigma_r));
rbf_nodes_s = exp(-(x_input-mu_s).^2./(2*sigma_s));

for k = 1:epochs

output_r = sum(w_r.*rbf_nodes_r);
output_s = sum(w_s.*rbf_nodes_s);

e_r = 0.5*(output_r - f_square).^2;
e_s = 0.5*(output_s - f_sin).^2;

delta_r = eta*e_r.*rbf_nodes_r;
delta_s = eta*e_s.*rbf_nodes_s;

w_r = w_r + delta_r;
w_s = w_s + delta_s;


diff_r = sum((output_r - f_square).^2);
diff_s = sum((output_s - f_sin).^2);

output_r_update = sum(w_r.*rbf_nodes_r);
output_s_update = sum(w_s.*rbf_nodes_s);

plot(x_input,output_s_update)
hold on
plot(x_input,f_sin)
hold off

max(e_s)

drawnow
pause(0.1)

end

plot(x_input,output_r)
hold on
plot(x_input,f_square)

figure()
plot(x_input,output_s)
hold on
plot(x_input,f_sin)