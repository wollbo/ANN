% %Data Preprocessing
clear all
close all

load votes.dat;
load mpparty.dat;
load mpsex.dat;
load mpdistrict.dat

votes = reshape(votes,[349 31]);

%SOM
max_neighbour = 2;
epochs = 500;
w = randi([2],100,31)/2;
eta = 0.02;
sigma_start = 1.5;
sigma_end = 0.75;
tau = -epochs^2/log(sigma_end/sigma_start);
% w = rand(100,31)
%%
for i = 1:epochs
   n_func = max(0, round(max_neighbour*exp(-i/(epochs/2))));
%     n_func = 
   for j = randperm(349)
        update_index = [];
        
        [distance index] = min(sum(abs(votes((j),:) - w),2));
        
        distance = [];
%         index = 1;
        
        for l = 1:length(w)
            distance(l)  = max(abs((mod(abs(index-1),10)-mod(abs(l-1),10))),...
               abs(((index - mod(index,10))/10)-((l+1 - mod(l+1,10))/10)));
        end
        sigma =sigma_start*exp(-i^2/tau);
        distance_update = exp(-distance/sigma^2);
        
        w = w + distance_update'.*eta.*(votes(j,:) - w);
        
        
   end
   w_logg1(i) = w(1,1);
   w_logg2(i) = w(1,2);
   w_logg3(i) = w(50,1);
   w_logg4(i) = w(50,2);
    imagesc(reshape((sum(w,2)),[10 10])')
    colorbar
    drawnow
    sigma
end

% Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
% Use some color scheme for these different groups
for k = 1:length(votes)
        [distance_ultimate(k) index_ultimate(k)] = min(sum(abs(votes(k,:) - w),2));
end
%%
imagesc(reshape((sum(w,2)),[10 10])')
% [sorted index_ultimate] = sort(index_ultimate);

color_code = [0 0 0;0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250; ...
    0.4940 0.1840 0.5560; 0.4660 0.6740 0.1880; 0.3010 0.7450 0.9330; 0.6350 0.0780 0.1840];
title_names = ["No Party";"M"; "Fp";  "S"; "V"; "Mp"; "Kd"; "C"];

coord = zeros(100,2);
for i = 1:100
    num = i-1;
    coord(i,:) = [(num-mod(num,10))/10 mod(num,10)];
end


for i = 0:7
    figure()
    ind = index_ultimate(find(mpparty==i));
    uni = unique(ind);
    cnt = histc(ind,uni);
    scatter(coord(uni,2),coord(uni,1),cnt*100,'markerfacecolor',[color_code(1+i,:)],'markeredgecolor',[color_code(1+i,:)],'LineWidth',2)
    hold on
    axis([-1 10 -1 11])
    title(title_names(i+1,:))
end
    
% legend('No Party','M', 'Fp', 'S', 'V', 'Mp', 'Kd', 'C','location','northeastoutside')  
% axis([-0.1 1 -0.1 1])


