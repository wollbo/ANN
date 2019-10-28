clear all 
close all

b = load('ballist.dat');

input_var = [b(:,1) b(:,2)];
output_var = [b(:,3) b(:,4)];

epochs = 10;
eta = 0.2;
n_nodes = 20;
sigma = 0.1;

nodes = rand(n_nodes,2)';
w_a = 0.1 *randn(1,n_nodes);

%CL
for i = 1:1000
   list = input_var(randperm(size(input_var,1)),[1:2]);
   for j = 1:length(list);
     sample = list(j,:)';
     d = diag(sqrt(((sample-nodes)'*(sample-nodes))));
     [distance winner] = min(d);
     nodes(:,winner) = nodes(:,winner) + eta*(sample-nodes(:,winner));
   end
end

% rbf = exp(-(x_input-nodes).^2./(2*sigma));
rbf_angle = exp(-(input_var(:,1)-nodes(1,:)).^2./(2*sigma));
rbf_velocity = exp(-(input_var(:,1)-nodes(1,:)).^2./(2*sigma));


%% Kanske borde vara online
for k = 1:epochs
    output_angle = sum(w_a.*rbf_angle);
    e_a = (input_var(:,1) - output_angle);
    delta = eta*e_a'*rbf_angle;
    w_a = w_a + delta;
    output_update = sum(w_a.*rbf_angle');
end

%%

rbf = exp(-(input_var-mu_r).^2./(2*sigma));

scatter(nodes(1,1:end),nodes(2,1:end))
hold on
scatter(output_var(:,1),output_var(:,2))

