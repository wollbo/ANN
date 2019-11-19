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

n_nodes = 30;
epochs = 300;
eta = 0.01;%You can use larger step size with on line learning, as it updates more slowly
holdout = 5;

w_r = 0.1 *randn(1,n_nodes)';
w_s = 0.1 *randn(1,n_nodes)';

%Linearly spaced nodes
mu_r = linspace(0,2*pi,n_nodes)'; 
mu_s = linspace(0,2*pi,n_nodes)';

%Randomly placed nodes
mu_r = rand(n_nodes,1)*2*pi; 
mu_s = rand(n_nodes,1)*2*pi;

%Tailor placed nodes
% mu_r = linspace(0,2*pi,n_nodes)'; 
% mu_s = linspace(0,2*pi,n_nodes)';

sigma_r = ones(1,n_nodes)'*0.1*0.5; %Largers variance increases variance, duh, and minimizes bias
sigma_s = ones(1,n_nodes)'*0.1*0.5;

rbf_nodes_r = exp(-(x_input-mu_r).^2./(2*sigma_r));
rbf_nodes_s = exp(-(x_input-mu_s).^2./(2*sigma_s));

residual_r = [];
residual_s = [];

output_r_update = zeros(1,length(x_input));
output_s_update = zeros(1,length(x_input));

for k = 1:epochs
    for j = 1:length(x_input)
    output_r = sum(w_r.*rbf_nodes_r(:,j));
    output_s = sum(w_s.*rbf_nodes_s(:,j));

    ksi_r = 0.5*(output_r - f_square(j)).^2;
    ksi_s = 0.5*(output_s - f_sin(j)).^2;

    e_r = (f_square(j) - output_r);
    e_s = (f_sin(j) - output_s);

    delta_r = eta*e_r.*rbf_nodes_r(:,j);
    delta_s = eta*e_s.*rbf_nodes_s(:,j);

    w_r = w_r + delta_r;
    w_s = w_s + delta_s;


%     diff_r = mean((output_r - f_square).^2);
%     diff_s = mean((output_s - f_sin).^2);

    output_r_update(j) = sum(w_r.*rbf_nodes_r(:,j));
    output_s_update(j) = sum(w_s.*rbf_nodes_s(:,j));

%     residual_r = [residual_r,mean(abs(output_r_update(x_test) - f_square(x_test)))];
%     residual_s = [residual_s,mean(abs(output_s_update(x_test) - f_sin(x_test)))];

%     plot(x_input,output_r_update)
%     hold on
%     plot(x_input,f_square)
%     for i = 1:size(rbf_nodes_r,1)
%         plot(x_input,rbf_nodes_r(i,:),'linestyle','-.')
%     end
%     hold off
%     drawnow

    % pause(0.1)    
    end
end

subplot(1,2,1)
plot(x_input,output_r_update, 'linewidth',1)
hold on
plot(x_input,f_square, 'linewidth',1)
% for i = 1:size(rbf_nodes_r,1)
%     plot(x_input,rbf_nodes_r(i,:),'linestyle','-.')
% end

subplot(1,2,2)
plot(x_input,output_s_update, 'linewidth',1)
hold on
plot(x_input,f_sin, 'linewidth',1)
% for i = 1:size(rbf_nodes_s,1)
%     plot(x_input,rbf_nodes_s(i,:),'linestyle','-.')
% end

residual_r_step = mean(abs(sign(output_r_update(x_test)) - f_square(x_test)))
residual_s = mean(abs(output_s_update(x_test) - f_sin(x_test)))
residual_r = mean(abs(output_r_update(x_test) - f_square(x_test)))