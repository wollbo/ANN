%
%3.1 Test
clear all
close all

x_input = 0:0.1:2*pi;
x_test = 6:5:61;

% f_sin = (sin(2*x_input));
% f_square = sign(2*f_sin);

%With added noise
f_sin = (sin(2*x_input))+ randn(1,length(x_input))*0.1;
f_square = sign(2*f_sin)+ randn(1,length(x_input))*0.1;

n_nodes = 5;
epochs = 500;
eta = 0.005;
holdout = 5;

w_r = 0.1 *randn(1,n_nodes)';
w_s = 0.1 *randn(1,n_nodes)';

% mu_r = zeros(1,n_nodes)';
% mu_s = zeros(1,n_nodes)';
mu_r = linspace(0,2*pi,n_nodes)'; 
mu_s = linspace(0,2*pi,n_nodes)';

sigma_r = ones(1,n_nodes)'*1;
sigma_s = ones(1,n_nodes)'*1;

rbf_nodes_r = exp(-(x_input-mu_r).^2./(2*sigma_r));
rbf_nodes_s = exp(-(x_input-mu_s).^2./(2*sigma_s));

residual_r = [];
residual_s = [];

for k = 1:epochs

output_r = sum(w_r.*rbf_nodes_r);
output_s = sum(w_s.*rbf_nodes_s);

ksi_r = 0.5*(output_r - f_square).^2;
ksi_s = 0.5*(output_s - f_sin).^2;
    
e_r = (f_square - output_r);
e_s = (f_sin - output_s);

delta_r = eta*e_r.*rbf_nodes_r;
delta_s = eta*e_s.*rbf_nodes_s;

w_r = w_r + delta_r;
w_s = w_s + delta_s;


diff_r = mean((output_r - f_square).^2);
diff_s = mean((output_s - f_sin).^2);

output_r_update = sum(w_r.*rbf_nodes_r);
output_s_update = sum(w_s.*rbf_nodes_s);

residual_r = [residual_r,mean(abs(output_r_update(x_test) - f_square(x_test)))];
residual_s = [residual_s,mean(abs(output_s_update(x_test) - f_sin(x_test)))];

plot(x_input,output_r_update)
hold on
plot(x_input,f_square)
for i = 1:size(rbf_nodes_r,1)
    plot(x_input,rbf_nodes_r(i,:),'linestyle','-.')
end
hold off
drawnow

pause(0.1)

end

plot(x_input,output_r)
hold on
plot(x_input,f_square)
for i = 1:size(rbf_nodes_r,1)
    plot(x_input,rbf_nodes_r(i,:),'linestyle','-.')
end

figure()
plot(x_input,output_s)
hold on
plot(x_input,f_sin)
for i = 1:size(rbf_nodes_s,1)
    plot(x_input,rbf_nodes_s(i,:),'linestyle','-.')
end

% residual_r_step = mean(abs(sign(output_r_update(x_test)) - f_square(x_test)))
% residual_s = mean(abs(output_s_update(x_test) - f_sin(x_test)))