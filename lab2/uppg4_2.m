%Data Preprocessing
clear all
close all

load cities.dat;

%SOM
max_neighbour = 3;
epochs = 100;
eta = 0.2;
w = rand(10,2);    

for i = 1:epochs
   list = randperm(size(cities,1));
   n_func = floor(max_neighbour-(i-1)*(max_neighbour/(epochs-1)));
   for j = list
       w_temp = w;
       index = [];
       distance = [];
%        for k = 1:max(1,n_func)
%            k;
           [distance,index] = min(sum((abs(cities(j,:) - w_temp).^2),2));
%            w_temp(k,:) = [];
%        end
       update_index = (1+mod([index-floor(n_func):index+floor(n_func)]-1,size(w,1)));
%        update_index
%        distance = sum(abs(cities(j,:) - w(update_index,:))./(abs(cities(j,:) - w(index,:)))/2,2)
       distance = [fliplr(2:(n_func+1)) 1 2:(n_func+1)];
       w(update_index,:) = w(update_index,:) + eta*(cities(j,:) - w(update_index,:))./distance';
       1
   end
   scatter(cities(:,1),cities(:,2))
   hold on
   scatter(w(:,1),w(:,2))
   hold off
   drawnow
end

for k = 1:size(cities,1)
   [distance_sort(k) index_sort(k)] = min(sum(abs(((cities(k,:) - w).^2)),2));
end
%%



[sorted order] = sort(index_sort);
order_wrap = [order(end) order order(1)];


scatter(cities(:,1),cities(:,2))
hold on

line([cities(order_wrap(1:end-2),1) cities(order,1)],[cities(order_wrap(1:end-2),2) cities(order,2)],'Color','k')

distances = (abs(cities(order_wrap(1:end-1),:)-cities(order_wrap(2:end),:)));
distances = distances(:,1).^2 + distances(:,2).^2
total_distance = sum(distances)








