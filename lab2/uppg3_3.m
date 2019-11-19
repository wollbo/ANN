clear all 
close all

b = load('ballist.dat');

input_var = [b(:,1) b(:,2)];
output_var = [b(:,3) b(:,4)];

n_nodes = 2;
epochs = 200;
epochs_CL = 100;
eta_CL = 0.2;
eta = 0.01/n_nodes;
sigma = 0.25;

nodes = rand(n_nodes,2)';
w = 0.01 *randn(1,n_nodes);
w_a = w;
w_v = w;

%CL
for i = 1:epochs_CL
   list = input_var(randperm(size(input_var,1)),[1:2]);
   for j = 1:length(list);
     sample = list(j,:)';
     d = diag(sqrt(((sample-nodes)'*(sample-nodes))));
%         d = abs(sample - nodes);
     [distance winner] = min(d);
     nodes(:,winner) = nodes(:,winner) + eta_CL*(sample-nodes(:,winner));
   end
   
%    scatter(input_var(:,1),input_var(:,2))
%    hold on
%    scatter(nodes(1,:),nodes(2,:),100)
%    legend('Input Data Points','RBF Node Placement')
%    hold off
%    drawnow
   
end

rbf = zeros(length(input_var),2,length(nodes));

for i = 1:length(nodes)
    d = output_var-nodes(:,i)';  
    rbf(:,:,i) = exp(-(d).^2./(2*sigma));
end

output = zeros(2,length(output_var))';
delta = zeros(n_nodes,2);
% err = [0 0];
% f = sum(rbf,3);

%% Kanske borde vara online
for k = 1:epochs
    for i = 1:length(input_var)
%         output(i,:) = w*reshape(rbf(i,:,:),[2,n_nodes])';
        f_temp =  reshape(rbf(i,:,:),[2,n_nodes]);
%         f = w*f_temp';
        output(i,:) = w*f_temp';
    end
    
%     output = f;
    e = (output_var - output);
%     e = (output_var(:,1) - output(:,1));
    
    for j = 1:length(output_var)
%         delta = eta*reshape(sum(e(j)*rbf(j,:,:)),[1 length(rbf(j,:,:))]);
        rbf_temp = reshape(rbf(j,:,:),[2 length(rbf(j,:,:))]);
        e_temp = (e(j,:));
        delta = eta*e_temp*rbf_temp;
        w = w + delta;
    end
    err(k) = sum(sum(abs(e(:,1))));
     
%     scatter(output(:,1),output(:,2))
%     hold on
%     scatter(output_var(:,1),output_var(:,2))
%     hold off
%     drawnow

%     pause(0.2)
end
% figure()
plot(err)
title('Error Curve')
xlabel('Epochs')
ylabel('Total Error')
figure()

w_test = zeros(1,n_nodes);
w_test(1) = 0.75;
% w = w_test;

for i = 1:length(input_var)
    output(i,:) = w*reshape(rbf(i,:,:),[2,n_nodes])';
end

figure()
[xq,yq]=meshgrid(0:0.05:1,0:0.05:1);
vq = griddata(input_var(:,1),input_var(:,2),output(:,1),xq,yq);
mesh(xq,yq,vq,'edgecolor',[0 0.4470 0.7410])
hold on
vq = griddata(input_var(:,1),input_var(:,2),output_var(:,1),xq,yq);
mesh(xq,yq,vq,'edgecolor',[0.8500 0.3250 0.0980])
scatter3(input_var(:,1),input_var(:,2),output(:,1),'*')
scatter3(input_var(:,1),input_var(:,2),output_var(:,1),'*')
axis equal
legend('RBF','Real Data','Estimated Data Points','Real Data Points')
title('RBF - Distance')
xlabel('Angle')
ylabel('Velocity')
zlabel('Distance')
view([-1 1 1])


figure()
[xq,yq]=meshgrid(0:0.05:1,0:0.05:1);
vq = griddata(input_var(:,1),input_var(:,2),output(:,2),xq,yq);
mesh(xq,yq,vq,'edgecolor',[0 0.4470 0.7410])
hold on
vq = griddata(input_var(:,1),input_var(:,2),output_var(:,2),xq,yq);
mesh(xq,yq,vq,'edgecolor',[0.8500 0.3250 0.0980])
scatter3(input_var(:,1),input_var(:,2),output(:,2),'*')
scatter3(input_var(:,1),input_var(:,2),output_var(:,2),'*')
axis equal
title('RBF - Height')
xlabel('Angle')
ylabel('Velocity')
zlabel('Height')
legend('RBF','Real Data','Estimated Data Points','Real Data Points')
view([1 -1 1])

figure()
scatter(input_var(:,1),input_var(:,2))
hold on
scatter(nodes(1,:),nodes(2,:),100)
legend('Input Data Points','RBF Node Placement')

