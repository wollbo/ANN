%Data Preprocessing
clear all
close all

animal_data = load('animals.dat');
animal_data = reshape(animal_data,[32 84]);

fid = fopen('animalnames.txt');
tline = fgetl(fid);
animal_names = [];
while ischar(tline)
    animal_names = [animal_names erase(string(tline),'')];
    tline = fgetl(fid);
end
fclose(fid);

fid = fopen('animalattributes.txt');
tline = fgetl(fid);
animal_attr = [];
while ischar(tline)
    animal_attr = [animal_attr string(tline)];
    tline = fgetl(fid);
end
fclose(fid);

%SOM
max_neighbour = 50;
epochs = 200;
eta = 0.2;
w = randi([0 1],100,84);    

for i = 1:epochs
   list = randperm(size(animal_data,1));
   n_func = max([1 floor(max_neighbour-(i-1)*(max_neighbour/(epochs-1)))]);
   for j = list
       [distance index] = min(sum(((animal_data(j,:) - w).^2),2));
       update_index = (1+mod([index-floor(n_func/2):index+floor(n_func/2)]-1,size(w,1)));
       w(update_index,:) = w(update_index,:) + eta*(animal_data(j,:) - w(update_index,:));
   end
end

for k = 1:size(animal_data,1)
   [distance(k) index(k)] = min(sum(abs(((animal_data(k,:) - w).^2)),2));
end



[sorted order] = sort(index);
order_wrap = [order(end) order order(1)];

for k = 1:size(animal_data,1)
   disp([sum(abs(animal_data(order(k),:) - animal_data(order_wrap(k),:)))/length(animal_data) animal_names(order(k)) sum(abs(animal_data(order(k),:) - animal_data(order_wrap(k+2),:)))/length(animal_data)])
end



for t = 1:size(animal_data,1)
    [sorted_corr index_corr] = sort(sum(abs(animal_data(t,:) - animal_data),2));
    most_similar = find(sorted_corr(2:end)==min(sorted_corr(2:end)))+1;
    disp([animal_names(t) animal_names((index_corr(most_similar)))]);
end


