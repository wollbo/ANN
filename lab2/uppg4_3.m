%Data Preprocessing
clear all
close all

load votes.dat;
load mpparty.dat;
load mpsex.dat;
load mpdistrict.dat

votes = reshape(votes,[349 31]);

%SOM
max_neighbour = 0;
epochs = 500;
eta = 0.2;
w = randi([2],100,31)/2;
% w = rand(100,31)

for i = 1:epochs
   n_func = max(0, round(max_neighbour*exp(-i/(epochs/2))));
%     n_func = 0;
   for j = randperm(349)
        update_index = [];
        [distance index] = min(sum((votes((j),:) - w).^2,2));
        for k = -n_func:n_func
            update_index = [update_index find((abs((1:100)-(index + k*10)) <= n_func).*(1:100))]; %H�R �R FELET, SE UPP S� INTE FLER RADER UPPDATERAS
        end
        w(update_index,:) = w(update_index,:) + eta*(votes(j,:) - w(update_index,:));
%         w(end,end,end)
%         winner((i-1)*epochs+j)=index;
%         z = zeros(100,31);
%         z(update_index,:) = w(update_index,:)*10;
%         w_test = z + w;
%         imagesc(reshape((sum(w_test,2)),[10 10]))
%         colorbar
%         drawnow
%         pause(0.5)
   end
    imagesc(reshape((sum(w,2)),[10 10]))
    colorbar
    drawnow
end
%%
% Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
% Use some color scheme for these different groups
for k = 1:length(votes)
        [distance_ultimate(k) index_ultimate(k)] = min(sum(abs(votes(k,:) - w),2));
end

imagesc(reshape((sum(w,2)),[10 10])')
% [sorted index_ultimate] = sort(index_ultimate);

color_code = [0 0 0;0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250; ...
    0.4940 0.1840 0.5560; 0.4660 0.6740 0.1880; 0.3010 0.7450 0.9330; 0.6350 0.0780 0.1840];
title_names = ["No Party";"M"; "Fp";  "S"; "V"; "Mp"; "Kd"; "C"];

coord = zeros(100,2);
for i = 1:100
    num = i-1;
    coord(i,:) = [(num-mod(num,10))/10 mod(num,10)]/10;
end


for i = 0:7
    figure()
    ind = index_ultimate(find(mpparty==i));
    uni = unique(ind);
    cnt = histc(ind,uni);
    scatter(coord(uni,2),coord(uni,1),cnt*100,'markerfacecolor',[color_code(1+i,:)],'markeredgecolor',[color_code(1+i,:)],'LineWidth',2)
    hold on
    axis([-0.1 1 -0.1 1])
    title(title_names(i+1,:))
end
    
% legend('No Party','M', 'Fp', 'S', 'V', 'Mp', 'Kd', 'C','location','northeastoutside')  
axis([-0.1 1 -0.1 1])


