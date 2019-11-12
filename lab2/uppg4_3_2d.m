% %Data Preprocessing
clear all
close all

load votes.dat;
load mpparty.dat;
load mpsex.dat;
load mpdistrict.dat

votes = reshape(votes,[349 31]);

%SOM
epochs = 200;
w = randi([2],10,10,31)/2;
% w = rand(100,31);
eta = 0.02;
sigma_start = 2.5;
sigma_end = 0.75;
tau = -epochs^2/log(sigma_end/sigma_start);
n_func = 2;
% w = rand(100,31)
%%
w_list = zeros(100,2);
for j = 1:10
   for k = 1:10
       w_list((j-1)*10 + k,:) = [j k];
   end
end
%%
for i = 1:epochs
   for j = randperm(length(votes))
        for k = 1:length(w_list)
            w_temp = reshape(w(w_list(k,1),w_list(k,2),:),[length(w),1])';
            distance(k) = sum((votes(j,:)-w_temp).^2,2);
        end
        [~,winner] = min(distance);
        winner_coord = w_list(winner,:);
        winner_coord(1) = winner_coord(1)-1;
        n_func =round(sigma_start*exp(-i^2/tau));
        
        for t = max(1,winner_coord(1)-n_func):min(10,winner_coord(1)+n_func)
            for r = max(1,winner_coord(2)-n_func):min(10,winner_coord(2)+n_func)
                if (abs(winner_coord(2)-r) + abs(winner_coord(1)-t)) <=n_func
                     w_delta = eta*(reshape(votes(j,:),[1,1,31])-w(t,r,:));
                     w(t,r,:) = w(t,r,:) + w_delta;
                   
                end
            end
        end
   end
   imagesc(sum(w,3),[19 24])
   colorbar
   drawnow
end
%%
% Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
% Use some color scheme for these different groups

winner_coord = zeros(length(votes),2);
for j = randperm(length(votes))
        for k = 1:length(w_list)
            w_temp = reshape(w(w_list(k,1),w_list(k,2),:),[length(w),1])';
            distance(k) = sum((votes(j,:)-w_temp).^2,2);
        end
        [~,winner(j)] = min(distance);
        winner_coord(j,:) = w_list(winner(j),:);
end

%%
% [sorted sorted_index] = sort(distance_ultimate);

imagesc(sum(w,3))
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
    ind = (find(mpparty==i));
%     uni = find(mpparty(index_ultimate)==i);
    
    uni = unique(ind);
    cnt = histc(ind,uni);
%     scatter(coord(uni,2),coord(uni,1),cnt*100,'markerfacecolor',[color_code(1+i,:)],...
%         'markeredgecolor',[color_code(1+i,:)],'LineWidth',2)

    scatter(winner_coord(uni,1)-1,winner_coord(uni,2)-1,cnt*100,'markerfacecolor',[color_code(1+i,:)],...
        'markeredgecolor',[color_code(1+i,:)],'LineWidth',2)

%     imagesc(reshape(sum(votes(ind,:)*w',1),[10 10])')
    hold on
    axis([-1 10 -1 11])
    title(title_names(i+1,:))
end
%%
% title_sex = ["Men","Women"]
% 
% for i = 0:1
%     figure()
%     ind = index_ultimate(find(mpsex==i));
%     
%     uni = unique(ind);
%     cnt = histc(ind,uni);
%     scatter(coord(uni,2),coord(uni,1),cnt*100,'markerfacecolor',[color_code(1+i,:)],...
%         'markeredgecolor',[color_code(1+i,:)],'LineWidth',2)
%     hold on
%     axis([-1 10 -1 11])
%     title(title_sex(i+1))
% end
% 
% 